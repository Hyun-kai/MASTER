"""
src/bakers/pipeline/sampling_runner.py

[설명]
Monomer 및 Dimer 시스템을 아우르는 범용 적응형 샘플링(Adaptive Sampling) 파이프라인.
멀티프로세싱 워커를 통해 좌표계를 조작하고(align.py), 초기 포인트를 배포하며(sampler.py),
AIMNet2 연산을 통합하여 볼츠만 탐색 루프를 총괄합니다.

[최적화 내역]
- 워커 초기화 시(init_worker) 이면각 회전에 필요한 원자 클러스터(moving_indices)를 단 1회 탐색 및 캐싱하여, 
  샘플링 시 수십만 번 반복되는 RDKit BFS 그래프 탐색 오버헤드를 완전히 제거했습니다.
"""

import os
import itertools
import numpy as np
import tqdm
import argparse
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional, Any

from ase import Atoms
from rdkit import Chem
from rdkit.Chem import SDMolSupplier

# 역할에 맞게 분리된 모듈에서 기능 이식 (Import)
from bakers.chem import topology
from bakers.chem.align import merge_residues, rotate_dihedral
from bakers.sim.calculator import EnsembleAIMNet2
from bakers.sim.sampler import BoltzmannAdaptiveSampler, get_sobol_points
from bakers.utils import io, visual, safety

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

_WORKER_DATA: Dict[str, Any] = {}

def init_worker(residues: List[str], rotamers: List[int], rotamer_dir: str, params_file: str) -> None:
    """각 워커 프로세스 초기화: 분자 객체, Topology 파라미터, 회전 그룹 캐싱"""
    global _WORKER_DATA
    _WORKER_DATA['molecules'] = {}
    _WORKER_DATA['params'] = {}
    _WORKER_DATA['moving_indices'] = {} # 성능 최적화를 위한 캐싱 딕셔너리
    
    full_params = topology.load_residue_params(params_file)
    unique_requests = set(zip(residues, rotamers))
    
    for res, rot in unique_requests:
        if res in full_params:
            _WORKER_DATA['params'][res] = full_params[res]
        
        key = f"{res}_{rot}"
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        
        if os.path.exists(sdf_path):
            suppl = SDMolSupplier(sdf_path, removeHs=False)
            try:
                mol = suppl[int(rot)]
                if mol:
                    _WORKER_DATA['molecules'][key] = mol
                    
                    # [최적화 적용] 초기화 단계에서 각 DOF에 대한 회전 그룹을 1회 탐색 후 캐싱
                    if res in _WORKER_DATA['params']:
                        dofs = _WORKER_DATA['params'][res].get('dofs', [])
                        dof_moving_indices = []
                        for dof in dofs:
                            u, v = int(dof[1]), int(dof[2])
                            visited = set()
                            queue = [v]
                            while queue:
                                curr = queue.pop(0)
                                if curr not in visited:
                                    visited.add(curr)
                                    for nbr in mol.GetAtomWithIdx(curr).GetNeighbors():
                                        n_idx = nbr.GetIdx()
                                        if n_idx != u and n_idx not in visited:
                                            queue.append(n_idx)
                            dof_moving_indices.append(list(visited))
                        _WORKER_DATA['moving_indices'][key] = dof_moving_indices

            except IndexError as e:
                raise IndexError(f"[Error] Rotamer index {rot} out of range for {res}.sdf") from e
        else:
            raise FileNotFoundError(f"[Error] SDF file not found: {sdf_path}")

def build_task(args: Tuple[List[str], List[int], np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """이면각 배열 기반 3D 구조 조립 태스크 (Monomer/Dimer 분기 처리)"""
    residues, rotamers, dihedrals = args
    
    molecules = []
    for r, i in zip(residues, rotamers):
        mol = _WORKER_DATA['molecules'].get(f"{r}_{i}")
        if mol is None: return None
        molecules.append(mol)

    confs = []
    dih_idx_counter = 0
    
    for res, rot, mol_obj in zip(residues, rotamers, molecules):
        key = f"{res}_{rot}"
        p = _WORKER_DATA['params'].get(res)
        if not p: return None
        
        # 워커 초기화 시 단 1회 계산해둔 moving_indices 리스트 캐시 로드
        moving_indices_list = _WORKER_DATA['moving_indices'].get(key, [])
        
        mol_copy = Chem.Mol(mol_obj)
        coords = mol_copy.GetConformer().GetPositions()
        
        for dof_idx, dof in enumerate(p.get('dofs', [])):
            if dih_idx_counter >= len(dihedrals): break
            angle = float(dihedrals[dih_idx_counter])
            
            # 캐싱된 원자 그룹 할당
            mov_idx = moving_indices_list[dof_idx] if dof_idx < len(moving_indices_list) else None
            
            # RDKit 탐색 없이 순수 수학 연산만으로 구조 회전 (병목 제거)
            coords = rotate_dihedral(
                coords, mol_copy, 
                int(dof[0]), int(dof[1]), int(dof[2]), int(dof[3]), 
                angle,
                moving_indices=mov_idx
            )
            dih_idx_counter += 1
            
        conf = Chem.Conformer(mol_copy.GetNumAtoms())
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, pos.tolist())
        mol_copy.RemoveAllConformers()
        mol_copy.AddConformer(conf)
        confs.append(mol_copy)
        
    final_coords = None
    final_nums = None
    
    # 1. Monomer 구조 반환
    if len(molecules) == 1:
        mol = confs[0]
        final_coords = mol.GetConformer().GetPositions()
        final_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

    # 2. Dimer 구조 병합(Align) 후 반환
    elif len(molecules) == 2:
        mol1, mol2 = confs[0], confs[1]
        pos1, pos2 = mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()
        p1, p2 = _WORKER_DATA['params'][residues[0]], _WORKER_DATA['params'][residues[1]]
        
        merged_mol, merged_coords = merge_residues(mol1, pos1, p1, mol2, pos2, p2)
        if merged_mol is None: return None
        
        final_coords = merged_coords
        final_nums = np.array([a.GetAtomicNum() for a in merged_mol.GetAtoms()])
    else:
        return None

    # 원자 번호에 따라 Heavy Atom을 앞쪽으로 재정렬
    if final_coords is not None and final_nums is not None:
        heavy_idx = np.where(final_nums != 1)[0]
        h_idx = np.where(final_nums == 1)[0]
        reorder_idx = np.concatenate([heavy_idx, h_idx])
        return final_coords[reorder_idx], final_nums[reorder_idx]
        
    return None

def run_sampling_pipeline(args: argparse.Namespace, project_root: str) -> None:
    """적응형 샘플링 파이프라인의 메인 실행 함수"""
    name = '-'.join(f'{r}_{i}' for r, i in zip(args.residues, args.rotamers))
    
    mode_type = "dimer" if len(args.residues) >= 2 else "monomer"
    output_dir = os.path.join(project_root, '1_data', f'{mode_type}s')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{Colors.GREEN}>>> [BAKERS] Adaptive Sampling Started ({mode_type}){Colors.NC}")
    print(f"    Target System: {name}")

    input_dir = os.path.join(project_root, '0_inputs')
    rotamer_dir = os.path.join(input_dir, 'rotamers')
    params_path = os.path.join(input_dir, 'residue_params.py')
    model_dir = os.path.join(input_dir, 'models')
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory missing: {model_dir}")
        
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.jpt')]
    if not model_files:
        raise FileNotFoundError("No AIMNet2 models (.jpt) found in the model directory.")
    
    device = 'cuda' if args.use_gpu else 'cpu'
    print(f"    [Device] AIMNet2 running on {device.upper()}")
    
    calc = EnsembleAIMNet2(model_files, device=device)
    main_params = topology.load_residue_params(params_path)
    topo_mask = topology.build_topological_mask(args.residues, main_params)

    pool = Pool(
        processes=args.threads, 
        initializer=init_worker, 
        initargs=(args.residues, args.rotamers, rotamer_dir, params_path)
    )
    
    dofs_count = sum(len(main_params[res]['dofs']) for res in args.residues)
    
    # 시스템 무결성 확인
    dummy_input = np.zeros(dofs_count)
    dummy_res = pool.map(build_task, [(args.residues, args.rotamers, dummy_input)])[0]
    if dummy_res is None: 
        pool.close(); pool.join()
        raise ValueError("Structure build returned None during initialization check.")
        
    atom_count = dummy_res[0].shape[0]
    print(f"    [System Info] Atoms: {atom_count}, DOFs: {dofs_count}")

    def evaluate_batch(points: np.ndarray, save_xyz: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) == 0: 
            return np.array([]), np.empty((0, atom_count, 3))

        batch_size = args.batch_size
        results_scores = []
        results_coords = []
        
        debug_dir = os.path.join(project_root, 'CHECK_IF_ANGLE_MATCHES')
        if save_xyz: os.makedirs(debug_dir, exist_ok=True)
        
        for i in range(0, len(points), batch_size):
            batch_pts = points[i:i+batch_size]
            tasks = [(args.residues, args.rotamers, p) for p in batch_pts]
            
            build_res = pool.map(build_task, tasks)
            
            valid_atoms_list = []
            valid_indices_in_batch = []
            
            batch_scores = [9999.9] * len(batch_pts)
            batch_coords_arr = [np.full((atom_count, 3), np.nan)] * len(batch_pts)
            
            for idx_in_batch, res in enumerate(build_res):
                if res is None: continue 
                
                coords, nums = res
                
                # Steric Clash 사전 차단
                if topo_mask is not None:
                    if topology.check_clashes(nums, coords, topo_mask, mode='loose'):
                        continue

                at = Atoms(numbers=nums, positions=coords)
                valid_atoms_list.append(at)
                valid_indices_in_batch.append(idx_in_batch)
                batch_coords_arr[idx_in_batch] = coords

            # 첫 초기화 단계일 경우 디버그용 xyz 구조 저장
            if save_xyz:
                for v_idx, atom in zip(valid_indices_in_batch, valid_atoms_list):
                    current_angles = batch_pts[v_idx]
                    fn_name = "_".join(f"{x:.1f}" for x in current_angles) + ".xyz"
                    io.write_xyz(atom.get_chemical_symbols(), atom.get_positions(), fn=os.path.join(debug_dir, fn_name))

            # 유효한 구조에 한해 AIMNet2 에너지 연산 수행 (Batch GPU 처리)
            if valid_atoms_list:
                for v_idx, at in zip(valid_indices_in_batch, valid_atoms_list):
                    at.calc = calc
                    batch_scores[v_idx] = at.get_potential_energy()

            results_coords.extend(batch_coords_arr)
            results_scores.extend(batch_scores)
            
        return np.array(results_scores), np.array(results_coords)

    # 샘플링 전략 설정
    limit_dict = {1: 5000, 2: 10000, 3: 50000, 4: 100000}
    auto_target = limit_dict.get(dofs_count, 12000)
    target_points = args.max_points if args.max_points > 0 else auto_target
    
    print(f"    [Strategy] Target Samples: {target_points}, DOFs: {dofs_count}")
    
    # 이식된 get_sobol_points를 활용하여 초기 공간 배포
    if dofs_count == 1:
        print(f"{Colors.YELLOW}    [Info] 1 DOF detected. Using Linear Scan.{Colors.NC}")
        axis = np.linspace(-180, 180, target_points)
        initial_points = axis.reshape(-1, 1)
    elif dofs_count <= 8:
        corners = list(itertools.product([-180, 180], repeat=dofs_count))
        n_internal = 50 if dofs_count <= 6 else 100
        sobol = get_sobol_points(n_internal, dofs_count)
        initial_points = np.vstack([corners, sobol])
    else:
        n_points = min(2048, max(500, args.grid_points**3 * 10))
        initial_points = get_sobol_points(n_points, dofs_count)

    initial_values, initial_xyzs = evaluate_batch(initial_points, save_xyz=True)
    
    final_points = initial_points
    final_values = initial_values
    all_xyzs = list(initial_xyzs) if len(initial_xyzs) > 0 else []

    # Adaptive Sampling 메인 루프
    if dofs_count > 1:
        sampler = BoltzmannAdaptiveSampler(initial_points, initial_values)
        pbar = tqdm.tqdm(total=target_points, colour='green', desc='[Sampling]')
        pbar.update(len(initial_points))
        
        try:
            while len(sampler.points) < target_points:
                n_ask = min(100, target_points - len(sampler.points))
                if n_ask <= 0: break

                candidates = sampler.ask(n_points=n_ask)
                new_values, new_xyzs = evaluate_batch(candidates)

                if sampler.tell(candidates, new_values):
                    pbar.update(len(candidates))
                    if len(new_xyzs) > 0:
                        all_xyzs.extend(new_xyzs)
                else:
                    break
            
            final_points = sampler.points
            final_values = sampler.values
            pbar.close()

        except KeyboardInterrupt:
            # 강제 종료(Ctrl+C) 시 현재까지의 데이터를 안전하게 보존
            print(f"\n{Colors.YELLOW}[Interrupt] Stopping sampling manually...{Colors.NC}")
            pool.terminate()
            pbar.close()
            current_xyzs = np.array(all_xyzs) if len(all_xyzs) > 0 else None
            safety.handle_force_stop(name, sampler.points, sampler.values, project_root, xyzs=current_xyzs)
            return

    # 정상 종료 시 결과 HDF5 저장 및 시각화 파이프라인 호출
    try:
        save_path = os.path.join(output_dir, f"{name}.hdf5")
        final_xyzs_arr = np.array(all_xyzs) if len(all_xyzs) > 0 else None
        io.save_results_hdf5(save_path, final_points, final_values, xyzs=final_xyzs_arr, numbers=dummy_res[1])
        print(f"    [Done] Results saved to {save_path}")
        visual.analyze_and_save(save_path)
    finally:
        pool.close()
        pool.join()