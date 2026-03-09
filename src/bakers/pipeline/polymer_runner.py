"""
src/bakers/pipeline/polymer_runner.py

[설명]
샘플링 결과를 바탕으로 폴리머 블록을 조립하고, 하이브리드 최적화(Hybrid Optimization)를 
총괄하는 파이프라인의 핵심 모듈입니다. RDKit BFS 그래프 탐색 캐싱을 적용하여 조립 속도를 향상시켰습니다.
"""

import os
import math
import numpy as np
import tqdm
import torch
import contextlib
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
import argparse

from rdkit import Chem
from rdkit.Chem import SDMolSupplier

from bakers.chem import topology, align
from bakers.chem.transform import rdkit_to_ase, update_rdkit_coords
from bakers.chem.align import rotate_dihedral  # 이전 단계에서 최적화한 모듈 임포트
from bakers.sim.calculator import EnsembleAIMNet2
from bakers.sim.optimize import global_optimization
from bakers.utils import io, safety

class Colors:
    GREEN, RED, YELLOW, BLUE, NC = '\033[0;32m', '\033[0;31m', '\033[0;33m', '\033[0;34m', '\033[0m'

_WORKER_DATA: Dict[str, Any] = {}

def init_worker(residues: List[str], rotamers: List[int], rotamer_dir: str, params_path: str, model_files: List[str], device: str) -> None:
    """워커 메모리 초기화, 분자/파라미터/모델 적재 및 BFS 탐색 캐싱 수행"""
    global _WORKER_DATA
    _WORKER_DATA['molecules'] = {}
    _WORKER_DATA['params'] = {}
    _WORKER_DATA['moving_indices'] = {}
    
    full_params = topology.load_residue_params(params_path)
    unique_res = sorted(list(set(residues)))
    unique_rot = sorted(list(set(rotamers)))
    
    for res in unique_res:
        if res in full_params: 
            _WORKER_DATA['params'][res] = full_params[res]
            
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        if not os.path.exists(sdf_path):
            raise FileNotFoundError(f"[Error] SDF not found: {sdf_path}")
            
        suppl = SDMolSupplier(sdf_path, removeHs=False)
        for rot in unique_rot:
            mol = suppl[int(rot)]
            if mol is None:
                raise IndexError(f"[Error] Failed to load rotamer {rot} from {res}.sdf")
                
            key = f"{res}_{rot}"
            _WORKER_DATA['molecules'][key] = mol
            
            # [최적화] 각 자유도별 원자 그룹 1회 캐싱
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

    # AI 에너지 계산기 로드
    torch.set_num_threads(1)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        _WORKER_DATA['calc'] = EnsembleAIMNet2(model_files, device=device)

def build_polymer_task(args: Tuple[List[str], List[int], np.ndarray, Optional[np.ndarray], List[str], bool]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """모노머들을 연결하여 폴리머를 조립하고 필요 시 최적화하는 태스크"""
    full_residues, full_rotamers, full_dihedrals, base_xyz, unit_residues, use_opt = args

    accumulated_mol = None
    accumulated_params = None
    dih_idx = 0
    
    # 1. 100% Rigid Body 조립 (수학적 결합)
    for i, (res, rot) in enumerate(zip(full_residues, full_rotamers)):
        key = f"{res}_{rot}"
        mol_obj = _WORKER_DATA['molecules'].get(key)
        current_params = _WORKER_DATA['params'].get(res)
        moving_indices_list = _WORKER_DATA['moving_indices'].get(key, [])
        
        if mol_obj is None or current_params is None: 
            raise RuntimeError(f"Molecule or parameters missing for {key}")
            
        current_monomer = Chem.Mol(mol_obj)
        
        if len(unit_residues) == 1 and base_xyz is not None and len(base_xyz) == current_monomer.GetNumAtoms():
            coords = base_xyz.copy()
        else:
            coords = current_monomer.GetConformer().GetPositions()
        
        # 샘플링된 이면각 설정 (캐싱된 인덱스 활용하여 속도 향상, Fallback 제거)
        for dof_idx, dof in enumerate(current_params.get('dofs', [])):
            if dih_idx >= len(full_dihedrals):
                break
            angle = float(full_dihedrals[dih_idx])
            mov_idx = moving_indices_list[dof_idx] if dof_idx < len(moving_indices_list) else None
            
            coords = rotate_dihedral(
                coords, current_monomer, 
                int(dof[0]), int(dof[1]), int(dof[2]), int(dof[3]), 
                angle, moving_indices=mov_idx
            )
            dih_idx += 1
                
        update_rdkit_coords(current_monomer, coords)
        
        if i == 0:
            accumulated_mol = current_monomer
            accumulated_params = current_params.copy()
        else:
            p_pos = accumulated_mol.GetConformer().GetPositions()
            m_pos = current_monomer.GetConformer().GetPositions()
            
            # 이전 블록과 현재 블록 병합 (오류 발생 시 예외 던짐)
            merged_mol, merged_coords = align.merge_residues(
                accumulated_mol, p_pos, accumulated_params,
                current_monomer, m_pos, current_params
            )
            if merged_mol is None: 
                raise ValueError(f"Failed to merge structure at block index {i}")
                
            accumulated_mol = merged_mol
            accumulated_params = topology.analyze_residue_topology(accumulated_mol)

    # 2. 하이브리드 최적화 단계: 조립이 무사히 완료된 후에만 1회 최적화
    final_energy = 0.0
    if use_opt:
        calc = _WORKER_DATA.get('calc')
        # 최적화 모듈을 호출하여 전체 구조 이완 및 에너지 획득
        final_energy = global_optimization(accumulated_mol, calc)
    else:
        # 최적화 안 할 경우 Rigid 상태의 에너지를 평가하기 위해 1회 점검
        ase_atoms = rdkit_to_ase(accumulated_mol)
        calc = _WORKER_DATA.get('calc')
        if calc:
            ase_atoms.calc = calc
            final_energy = ase_atoms.get_potential_energy()

    # 최종 결과물 추출
    final_atoms = rdkit_to_ase(accumulated_mol)
    final_coords = final_atoms.get_positions()
    final_nums = final_atoms.get_atomic_numbers()
    
    return final_coords, final_nums, final_energy


def run_building_pipeline(args: argparse.Namespace, project_root: str) -> None:
    """폴리머 조립 파이프라인의 메인 진입점"""
    if len(args.residues) != len(args.rotamers):
        print(f"{Colors.RED}[Critical Error] Length mismatch! "
              f"--residues({len(args.residues)}) and --rotamers({len(args.rotamers)}) must match.{Colors.NC}")
        return

    unit_len = len(args.residues)
    if getattr(args, 'target_length', 0) > 0:
        target_len = args.target_length
        tile_count = math.ceil(target_len / unit_len)
        full_residues = (args.residues * tile_count)[:target_len]
        full_rotamers = (args.rotamers * tile_count)[:target_len]
        suffix = f"{target_len}mer"
    else:
        tile_count = getattr(args, 'repeats', 2)
        full_residues = args.residues * tile_count
        full_rotamers = args.rotamers * tile_count
        target_len = len(full_residues)
        suffix = f"poly_x{tile_count}"

    print(f"{Colors.BLUE}[Info] Sequence: {full_residues}{Colors.NC}")
    base_name = '-'.join(f'{r}_{i}' for r, i in zip(args.residues, args.rotamers))
    polymer_name = f"{base_name}_{suffix}"
    
    print(f"{Colors.GREEN}>>> [Step 3] Polymer Building: {polymer_name}{Colors.NC}")
    
    use_opt = getattr(args, 'optimize', False)
    if use_opt:
        print(f"{Colors.YELLOW}>>> [Notice] Optimize Mode: Rigid assembly + Global ASE optimization at the end.{Colors.NC}")
    else:
        print(f"{Colors.YELLOW}>>> [Notice] Assembly Mode: Generation of pure geometric structures (No optimization).{Colors.NC}")    
    
    output_dir = os.path.join(project_root, '1_data', 'polymers')
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = getattr(args, 'input_file', None) or os.path.join(project_root, '1_data', 'dimers', f"{base_name}.hdf5")
    if not os.path.exists(input_path):
        input_path = os.path.join(project_root, '1_data', 'monomers', f"{base_name}.hdf5")
    
    if not os.path.exists(input_path):
        print(f"{Colors.RED}[Error] Input file not found: {input_path}{Colors.NC}")
        return

    # 샘플링된 데이터 포인트 로드
    data = io.load_hdf5_data(input_path)
    points = data.get('points', [])
    base_xyzs = data.get('xyzs', [])
    
    if len(points) == 0: 
        print(f"{Colors.RED}[Error] No points found in HDF5.{Colors.NC}")
        return

    top_k = getattr(args, 'top_k', 100)
    top_n = min(len(points), top_k)
    tiled_points = np.tile(points[:top_n], (1, tile_count))
    
    if base_xyzs is not None and len(base_xyzs) >= top_n:
        tiled_xyzs = base_xyzs[:top_n]
    else:
        tiled_xyzs = [None] * top_n
    
    model_files = [os.path.join(project_root, '0_inputs', 'models', f) for f in os.listdir(os.path.join(project_root, '0_inputs', 'models')) if f.endswith('.jpt')]
    use_cuda = (getattr(args, 'use_gpu', 1) == 1)
    device = 'cuda' if use_cuda else 'cpu'

    # 워커 인자 세팅
    rotamer_dir = os.path.join(project_root, '0_inputs', 'rotamers')
    params_path = os.path.join(project_root, '0_inputs', 'residue_params.py')
    worker_args = (full_residues, full_rotamers, rotamer_dir, params_path, model_files, device)

    pool = None
    threads = getattr(args, 'threads', 20)
    if threads > 1:
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=threads, initializer=init_worker, initargs=worker_args)
    else:
        init_worker(*worker_args)

    results_vals, results_xyzs, results_points = [], [], []
    pbar = tqdm.tqdm(total=len(tiled_points), colour='cyan', desc='[Processing]')
    
    try:
        tasks = [(full_residues, full_rotamers, tiled_points[i], tiled_xyzs[i], args.residues, use_opt) for i in range(len(tiled_points))]
        
        # 에러 방어(Fallback)가 없으므로 터지면 바로 잡히도록 map 처리
        iterator = pool.imap(build_polymer_task, tasks) if pool else (build_polymer_task(t) for t in tasks)
        
        for i, res in enumerate(iterator):
            if res:
                coords, nums, energy = res
                results_xyzs.append(coords)
                results_vals.append(energy)
                results_points.append(tiled_points[i])
            pbar.update(1)

        save_path = os.path.join(output_dir, f"{polymer_name}.hdf5")
        if results_xyzs:
            io.save_results_hdf5(save_path, np.array(results_points), np.array(results_vals), np.array(results_xyzs), numbers=nums)
            print(f"\n{Colors.GREEN}    [Done] Saved {len(results_xyzs)} optimized structures.{Colors.NC}")
        else: 
            print(f"\n{Colors.RED}    [Error] All tasks failed. No valid structures generated.{Colors.NC}")

    except KeyboardInterrupt:
        if pool: pool.terminate() 
        safety.handle_force_stop(polymer_name, results_points, results_vals, project_root, xyzs=results_xyzs)
    except Exception as e:
        print(f"\n{Colors.RED}[Pipeline Exception] {e}{Colors.NC}")
        if pool: pool.terminate()
    finally:
        if pool: pool.close(); pool.join()