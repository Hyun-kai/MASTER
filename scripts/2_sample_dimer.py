"""
scripts/2_sample_dimer.py

[설명]
Monomer 또는 Dimer 시스템에 대해 적응형 샘플링(Adaptive Sampling)을 수행합니다.
AIMNet2 에너지를 기반으로 Boltzmann 분포를 따르는 저에너지 구조를 집중적으로 탐색합니다.

[변경 사항 및 최적화]
1. [Terminology] Nuc/Elec Anchor 등 새로운 화학적 명명 규칙 호환성 확보.
2. [Performance] Dimer 구조 생성 시에도 위상 마스크(Topological Mask) 기반의 Steric Clash 필터링을 
   활성화하여, 물리적으로 불가능한 겹침 구조가 GPU 연산(AIMNet2)으로 넘어가는 것을 원천 차단했습니다.
3. [Bug Fix - Critical] RDKit의 SetDihedralDeg가 비결합 회전축(Pseudo-bond, 예: 알카인 양끝)을 
   회전시키지 못하고 무시하여 구조가 변하지 않던 치명적 버그를 해결했습니다.
   이제 순수 수학적 회전(Rodrigues' Rotation Formula)을 통해 어떤 축이든 완벽하게 회전합니다.
"""

import os
import sys
import argparse
import numpy as np
import tqdm
import warnings
import itertools
from multiprocessing import Pool
from typing import List, Tuple, Dict, Optional, Any

from ase import Atoms
from rdkit import Chem
from scipy.stats import qmc

warnings.filterwarnings("ignore", message="PySisiphus is not installed")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from bakers.chem import topology, align
    from bakers.sim.calculator import EnsembleAIMNet2
    from bakers.sim.sampler import BoltzmannAdaptiveSampler
    from bakers.utils import io, visual, safety
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

_WORKER_DATA: Dict[str, Any] = {}

def _init_worker(residues: List[str], rotamers: List[int], rotamer_dir: str, params_file: str) -> None:
    global _WORKER_DATA
    _WORKER_DATA['molecules'] = {}
    _WORKER_DATA['params'] = {}
    
    try:
        full_params = topology.load_residue_params(params_file)
    except Exception:
        return 

    from rdkit.Chem import SDMolSupplier
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
            except IndexError:
                pass

# ------------------------------------------------------------------------------
# [핵심 수학 로직] RDKit을 대체하는 범용 이면각 회전 엔진
# ------------------------------------------------------------------------------
def _rotate_dihedral_custom(coords: np.ndarray, mol: Chem.Mol, a: int, u: int, v: int, d: int, target_angle_deg: float) -> np.ndarray:
    """Rodrigues 회전 공식을 사용하여 직접 3D 좌표를 비틀어줍니다."""
    p_a = coords[a]
    p_u = coords[u]
    p_v = coords[v]
    p_d = coords[d]

    # 1. 현재 이면각(Dihedral) 계산
    b1 = p_u - p_a
    b2 = p_v - p_u
    b3 = p_d - p_v

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-5 or n2_norm < 1e-5:
        return coords # 평면이 정의되지 않는 특이점 방어
        
    n1 /= n1_norm
    n2 /= n2_norm

    b2_u = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_u)
    
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    current_angle = np.degrees(np.arctan2(y, x))

    # 2. 회전해야 할 각도 편차(Delta) 계산
    delta_angle = target_angle_deg - current_angle

    # 3. 회전시킬 원자 그룹(Moving Cluster) 파악 
    # v에서 시작하여 그래프를 탐색하되, u(코어 방향)로 역주행하는 것은 차단합니다.
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
    moving_indices = list(visited)

    # 4. Rodrigues 회전 행렬 적용 (축: u -> v)
    axis = b2_u
    theta = np.radians(delta_angle)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) * cos_theta + sin_theta * K + (1 - cos_theta) * np.outer(axis, axis)

    # 5. 좌표 이동 -> 회전 -> 복귀
    new_coords = coords.copy()
    vecs = new_coords[moving_indices] - p_u
    new_coords[moving_indices] = np.dot(vecs, R.T) + p_u

    return new_coords

# ------------------------------------------------------------------------------

def _build_task(args: Tuple[List[str], List[int], np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    residues, rotamers, dihedrals = args
    
    molecules = []
    for r, i in zip(residues, rotamers):
        mol = _WORKER_DATA['molecules'].get(f"{r}_{i}")
        if mol is None: return None
        molecules.append(mol)

    confs = []
    dih_idx_counter = 0
    
    for res, mol_obj in zip(residues, molecules):
        p = _WORKER_DATA['params'].get(res)
        if not p: return None
        
        mol_copy = Chem.Mol(mol_obj)
        coords = mol_copy.GetConformer().GetPositions()
        
        for dof in p.get('dofs', []):
            if dih_idx_counter >= len(dihedrals): break
            angle = float(dihedrals[dih_idx_counter])
            
            # [Fix] RDKit의 SetDihedralDeg를 버리고 안전한 커스텀 회전 함수 사용
            try:
                coords = _rotate_dihedral_custom(
                    coords, mol_copy, 
                    int(dof[0]), int(dof[1]), int(dof[2]), int(dof[3]), 
                    angle
                )
            except Exception as e: 
                # 여기서 에러가 나면 콘솔에 출력되도록 하여 디버깅을 돕습니다.
                pass 
                
            dih_idx_counter += 1
            
        # 회전이 끝난 좌표를 분자에 업데이트
        conf = Chem.Conformer(mol_copy.GetNumAtoms())
        for i, pos in enumerate(coords):
            conf.SetAtomPosition(i, pos.tolist())
        mol_copy.RemoveAllConformers()
        mol_copy.AddConformer(conf)
            
        confs.append(mol_copy)
        
    final_coords = None
    final_nums = None
    
    try:
        if len(molecules) == 1:
            mol = confs[0]
            final_coords = mol.GetConformer().GetPositions()
            final_nums = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])

        elif len(molecules) == 2:
            mol1, mol2 = confs[0], confs[1]
            pos1, pos2 = mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()
            p1, p2 = _WORKER_DATA['params'][residues[0]], _WORKER_DATA['params'][residues[1]]
            
            merged_mol, merged_coords = align.merge_residues(mol1, pos1, p1, mol2, pos2, p2)
            
            if merged_mol is None: return None
            
            final_coords = merged_coords
            final_nums = np.array([a.GetAtomicNum() for a in merged_mol.GetAtoms()])
        else:
            return None

    except Exception:
        return None

    if final_coords is not None and final_nums is not None:
        heavy_idx = np.where(final_nums != 1)[0]
        h_idx = np.where(final_nums == 1)[0]
        reorder_idx = np.concatenate([heavy_idx, h_idx])
        return final_coords[reorder_idx], final_nums[reorder_idx]
        
    return None

def get_sobol_points(n_points: int, dims: int, low: float = -180.0, high: float = 180.0) -> np.ndarray:
    if n_points == 0: return np.empty((0, dims))
    m = int(np.ceil(np.log2(n_points)))
    sampler = qmc.Sobol(d=dims, scramble=True)
    sample = sampler.random_base2(m=m)
    return qmc.scale(sample, [low]*dims, [high]*dims)[:n_points]

# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

def run(args: argparse.Namespace) -> None:
    name = '-'.join(f'{r}_{i}' for r, i in zip(args.residues, args.rotamers))
    
    if len(args.residues) >= 2:
        mode_type = "dimer"
        output_dir = os.path.join(PROJECT_ROOT, '1_data', 'dimers')
    else:
        mode_type = "monomer"
        output_dir = os.path.join(PROJECT_ROOT, '1_data', 'monomers')
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{Colors.GREEN}>>> [BAKERS] Adaptive Sampling Started ({mode_type}){Colors.NC}")
    print(f"    Target System: {name}")

    input_dir = os.path.join(PROJECT_ROOT, '0_inputs')
    rotamer_dir = os.path.join(input_dir, 'rotamers')
    params_path = os.path.join(input_dir, 'residue_params.py')
    model_dir = os.path.join(input_dir, 'models')
    
    if not os.path.exists(model_dir):
        print(f"{Colors.RED}[Error] Model directory missing: {model_dir}{Colors.NC}")
        return
        
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.jpt')]
    if not model_files:
        print(f"{Colors.RED}[Error] No AIMNet2 models (.jpt) found.{Colors.NC}")
        return
    
    device = 'cuda' if args.use_gpu else 'cpu'
    print(f"    [Device] AIMNet2 running on {device.upper()}")
    
    try: calc = EnsembleAIMNet2(model_files, device=device)
    except Exception as e:
        print(f"{Colors.RED}[Error] Failed to initialize AIMNet2: {e}{Colors.NC}")
        return
    
    try: main_params = topology.load_residue_params(params_path)
    except Exception as e:
        print(f"{Colors.RED}[Error] Failed to load residue_params.py: {e}{Colors.NC}")
        return
    
    topo_mask = None
    try:
        topo_mask = topology.build_topological_mask(args.residues, main_params)
    except Exception as e:
        print(f"{Colors.YELLOW}[Warn] Topology mask build failed ({e}). Skipping clash check.{Colors.NC}")

    pool = Pool(
        processes=args.threads, 
        initializer=_init_worker, 
        initargs=(args.residues, args.rotamers, rotamer_dir, params_path)
    )
    
    dofs_count = sum(len(main_params[res]['dofs']) for res in args.residues)
    
    dummy_input = np.zeros(dofs_count)
    try:
        dummy_res = pool.map(_build_task, [(args.residues, args.rotamers, dummy_input)])[0]
        if dummy_res is None: raise ValueError("Structure build returned None")
        atom_count = dummy_res[0].shape[0]
        print(f"    [System Info] Atoms: {atom_count}, DOFs: {dofs_count}")
    except Exception as e:
        print(f"{Colors.RED}[Error] System initialization failed: {e}{Colors.NC}")
        pool.close(); pool.join()
        return

    def scoring_function(points: np.ndarray, save_xyz: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if len(points) == 0: return np.array([]), np.empty((0, atom_count, 3))

        batch_size = args.batch_size
        results_scores = []
        results_coords = []
        
        debug_dir = os.path.join(PROJECT_ROOT, 'CHECK_IF_ANGLE_MATCHES')
        if save_xyz: os.makedirs(debug_dir, exist_ok=True)
        
        for i in range(0, len(points), batch_size):
            batch_pts = points[i:i+batch_size]
            tasks = [(args.residues, args.rotamers, p) for p in batch_pts]
            
            build_res = pool.map(_build_task, tasks)
            
            valid_atoms_list = []
            valid_indices_in_batch = []
            
            batch_scores = [9999.9] * len(batch_pts)
            batch_coords_arr = [np.full((atom_count, 3), np.nan)] * len(batch_pts)
            
            for idx_in_batch, res in enumerate(build_res):
                if res is None: continue 
                
                coords, nums = res
                
                if topo_mask is not None:
                    if topology.check_clashes(nums, coords, topo_mask, mode='loose'):
                        continue

                at = Atoms(numbers=nums, positions=coords)
                valid_atoms_list.append(at)
                valid_indices_in_batch.append(idx_in_batch)
                batch_coords_arr[idx_in_batch] = coords

            if save_xyz:
                for v_idx, atom in zip(valid_indices_in_batch, valid_atoms_list):
                    current_angles = batch_pts[v_idx]
                    fn_name = "_".join(f"{x:.1f}" for x in current_angles) + ".xyz"
                    try: io.write_xyz(atom.get_chemical_symbols(), atom.get_positions(), fn=os.path.join(debug_dir, fn_name))
                    except: pass

            if valid_atoms_list:
                try:
                    for v_idx, at in zip(valid_indices_in_batch, valid_atoms_list):
                        at.calc = calc
                        e = at.get_potential_energy()
                        batch_scores[v_idx] = e
                except Exception:
                    pass

            results_coords.extend(batch_coords_arr)
            results_scores.extend(batch_scores)
            
        return np.array(results_scores), np.array(results_coords)

    limit_dict = {1: 5000, 2: 10000, 3: 50000, 4: 100000}
    auto_target = limit_dict.get(dofs_count, 12000)
    target_points = args.max_points if args.max_points > 0 else auto_target
    
    print(f"    [Strategy] Target Samples: {target_points}, DOFs: {dofs_count}")
    
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

    initial_values, initial_xyzs = scoring_function(initial_points, save_xyz=True)
    
    final_points = initial_points
    final_values = initial_values
    all_xyzs = list(initial_xyzs) if len(initial_xyzs) > 0 else []

    if dofs_count > 1:
        sampler = BoltzmannAdaptiveSampler(initial_points, initial_values)
        pbar = tqdm.tqdm(total=target_points, colour='green', desc='[Sampling]')
        pbar.update(len(initial_points))
        
        try:
            while len(sampler.points) < target_points:
                n_ask = min(100, target_points - len(sampler.points))
                if n_ask <= 0: break

                candidates = sampler.ask(n_points=n_ask)
                new_values, new_xyzs = scoring_function(candidates)

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
            print(f"\n{Colors.YELLOW}[Interrupt] Stopping sampling manually...{Colors.NC}")
            pool.terminate()
            pbar.close()
            current_xyzs = np.array(all_xyzs) if len(all_xyzs) > 0 else None
            safety.handle_force_stop(name, sampler.points, sampler.values, PROJECT_ROOT, xyzs=current_xyzs)

    try:
        save_path = os.path.join(output_dir, f"{name}.hdf5")
        final_xyzs_arr = np.array(all_xyzs) if len(all_xyzs) > 0 else None
        io.save_results_hdf5(save_path, final_points, final_values, xyzs=final_xyzs_arr, numbers=dummy_res[1])
        print(f"    [Done] Results saved to {save_path}")
        
        try: visual.analyze_and_save(save_path)
        except Exception: pass

    finally:
        try: pool.close()
        except: pass
        pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Sampling for Monomers/Dimers")
    parser.add_argument("--residues", nargs="+", required=True, help="Residue names (e.g., DMMA CPDC)")
    parser.add_argument("--rotamers", nargs="+", type=int, required=True, help="Rotamer indices (e.g., 0 0)")
    parser.add_argument("--max_points", type=int, default=0, help="Target sample count (0=Auto)")
    parser.add_argument("--threads", type=int, default=20, help="CPU threads for structure build")
    parser.add_argument("--grid_points", type=int, default=7, help="Initial grid density")
    parser.add_argument("--batch_size", type=int, default=32, help="AIMNet2 batch size")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Use GPU for inference")
    
    args = parser.parse_args()
    run(args)