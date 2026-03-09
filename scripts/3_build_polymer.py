"""
scripts/3_build_polymer.py

[설명]
Unit Block(Monomer/Dimer)의 샘플링 결과(HDF5)를 로드하여, 
지정된 길이의 폴리머(Polymer) 구조를 조립하고 필요시 전체 최적화를 수행합니다.

[최종 수정 내역 (Hybrid Optimization Mode)]
- [Feature] 하이브리드 최적화 도입: 조립은 100% 강체(Rigid Body)로 수학적 오차 없이 진행하며,
  조립이 모두 완료된 최종 산물에 대해서만 단 1회 전체 최적화(Global Optimization)를 수행하여
  구조 왜곡(Asymmetric Distortion)을 원천 차단했습니다.
- [Fix] 입력 파라미터 무결성 검증 로직 추가 (residues와 rotamers 길이 불일치 방지)
- [Fix] _init_worker 내 계산기(EnsembleAIMNet2) 로드 로직 복원 및 예외 처리 강화
"""

import os
import sys
import math
import argparse
import numpy as np
import tqdm
import warnings
import torch
import multiprocessing as mp
import contextlib

from ase import Atoms
from ase.optimize import BFGS # [수정] 최적화를 위한 옵티마이저 임포트
from rdkit import Chem
from rdkit.Chem import SDMolSupplier

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)

try:
    from bakers.chem import topology, align
    from bakers.sim.calculator import EnsembleAIMNet2 # [수정] 에너지 계산기 임포트 복구
    from bakers.utils import io, safety
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

GREEN, RED, YELLOW, BLUE, NC = '\033[0;32m', '\033[0;31m', '\033[0;33m', '\033[0;34m', '\033[0m'

# ==============================================================================
# 1. 헬퍼 함수 (Helper Functions) - 기존 기능 100% 유지
# ==============================================================================

def rdkit_to_ase(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    return Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], positions=conf.GetPositions())

def update_rdkit_coords(mol: Chem.Mol, new_positions: np.ndarray):
    conf = mol.GetConformer()
    for i, pos in enumerate(new_positions): 
        conf.SetAtomPosition(i, pos.tolist())

def _rotate_dihedral_custom(coords: np.ndarray, mol: Chem.Mol, a: int, u: int, v: int, d: int, target_angle_deg: float) -> np.ndarray:
    """Rodrigues 회전 공식을 사용하여 직접 3D 좌표를 비틀어줍니다."""
    p_a, p_u, p_v, p_d = coords[a], coords[u], coords[v], coords[d]

    b1 = p_u - p_a
    b2 = p_v - p_u
    b3 = p_d - p_v

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm, n2_norm = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1_norm < 1e-5 or n2_norm < 1e-5: return coords
        
    n1 /= n1_norm
    n2 /= n2_norm

    b2_u = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_u)
    
    x, y = np.dot(n1, n2), np.dot(m1, n2)
    current_angle = np.degrees(np.arctan2(y, x))

    delta_angle = target_angle_deg - current_angle

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

    axis = b2_u
    theta = np.radians(delta_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) * cos_theta + sin_theta * K + (1 - cos_theta) * np.outer(axis, axis)

    new_coords = coords.copy()
    vecs = new_coords[moving_indices] - p_u
    new_coords[moving_indices] = np.dot(vecs, R.T) + p_u

    return new_coords

# ==============================================================================
# 2. 전역 작업자 데이터 (Worker Data)
# ==============================================================================
_WORKER_DATA = {}

def _init_worker(residues, rotamers, rotamer_dir, params_path, model_files, device):
    global _WORKER_DATA
    _WORKER_DATA['molecules'] = {}
    _WORKER_DATA['params'] = {}
    
    try: 
        full_params = topology.load_residue_params(params_path)
    except Exception as e:
        print(f"{RED}[Worker Init Error] Params load failed: {e}{NC}")
        return

    unique_res = sorted(list(set(residues)))
    unique_rot = sorted(list(set(rotamers)))
    
    for res in unique_res:
        if res in full_params: _WORKER_DATA['params'][res] = full_params[res]
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        if os.path.exists(sdf_path):
            suppl = SDMolSupplier(sdf_path, removeHs=False)
            for rot in unique_rot:
                try:
                    mol = suppl[int(rot)]
                    if mol: _WORKER_DATA['molecules'][f"{res}_{rot}"] = mol
                except:
                    print(f"{YELLOW}[Worker Warning] Failed to load {res}_{rot} from SDF.{NC}")

    # [수정] AI 모델(계산기) 로드 로직 복구 (최적화를 위해 필수)
    try:
        torch.set_num_threads(1)
        # 불필요한 모델 로딩 출력 억제
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            _WORKER_DATA['calc'] = EnsembleAIMNet2(model_files, device=device)
    except Exception as e:
        print(f"{RED}[Worker Init Error] Calculator load failed: {e}{NC}")

# ==============================================================================
# 3. 폴리머 조립 코어 (Task - Hybrid Mode)
# ==============================================================================
def _build_polymer_task(args):
    # [수정] use_opt 인자를 추가로 받도록 언패킹 구조 변경
    full_residues, full_rotamers, full_dihedrals, base_xyz, unit_residues, use_opt = args

    accumulated_mol = None
    accumulated_params = None
    dih_idx = 0
    final_energy = 0.0

    # [수정] 하이브리드 최적화 로직으로 전면 재설계
    def perform_optimization(mol, is_final_step=False, optimize_flag=False):
        """
        조립이 100% 완료된 최종 구조(is_final_step=True)에 대해서만 
        전체 분자를 대상으로 구조 최적화(Global Relaxation)를 수행합니다.
        """
        # 스위치가 꺼져있거나, 아직 조립 중(중간 단계)이면 최적화 생략
        if not optimize_flag or not is_final_step:
            return 0.0
            
        calc = _WORKER_DATA.get('calc')
        if calc is None:
            return 0.0 # 계산기가 로드되지 않았다면 안전하게 스킵
            
        try:
            ase_atoms = rdkit_to_ase(mol)
            if hasattr(calc, 'reset'): calc.reset()
            ase_atoms.calc = calc
            
            # [핵심] 부분 고정(FixAtoms) 없이 전체 분자를 자유롭게 풀어주어 구조 왜곡 원천 차단
            opt = BFGS(ase_atoms, logfile=None)
            opt.run(fmax=0.05, steps=100) # 최대 100스텝으로 제한하여 무한 루프 방지
            
            # 최적화된 새로운 좌표를 RDKit 원본 객체에 동기화
            optimized_coords = ase_atoms.get_positions()
            update_rdkit_coords(mol, optimized_coords)
            
            # 안정화된 최종 잠재 에너지 반환
            return ase_atoms.get_potential_energy()
            
        except Exception:
            # 최적화 과정에서 양자 화학 계산 에러 발생 시, 원본 구조(Rigid 조립본)를 유지
            return 0.0

    # --------------------------------------------------------------------------
    # Main Build Loop (Strict Rigid Body Assembly)
    # --------------------------------------------------------------------------
    for i, (res, rot) in enumerate(zip(full_residues, full_rotamers)):
        
        mol_obj = _WORKER_DATA['molecules'].get(f"{res}_{rot}")
        if mol_obj is None: return None
        
        current_monomer = Chem.Mol(mol_obj)
        current_params = _WORKER_DATA['params'].get(res)
        
        # 100% 동일한 쌍둥이 구조 주입
        if len(unit_residues) == 1 and base_xyz is not None and len(base_xyz) == current_monomer.GetNumAtoms():
            coords = base_xyz.copy()
        else:
            coords = current_monomer.GetConformer().GetPositions()
        
        # 샘플링된 이면각 설정 (모든 모노머에 대해 정확히 똑같이 적용됨)
        for dof in current_params.get('dofs', []):
            if dih_idx < len(full_dihedrals):
                angle = float(full_dihedrals[dih_idx])
                try: 
                    coords = _rotate_dihedral_custom(
                        coords, current_monomer, 
                        int(dof[0]), int(dof[1]), int(dof[2]), int(dof[3]), 
                        angle
                    )
                except: pass
                dih_idx += 1
                
        update_rdkit_coords(current_monomer, coords)
        
        if i == 0:
            accumulated_mol = current_monomer
            accumulated_params = current_params.copy()
        else:
            p_pos = accumulated_mol.GetConformer().GetPositions()
            m_pos = current_monomer.GetConformer().GetPositions()
            
            # [기하학적 병합] 최적화 없이 수학적으로 오차 없이 이어 붙입니다.
            merged_mol, merged_coords = align.merge_residues(
                accumulated_mol, p_pos, accumulated_params,
                current_monomer, m_pos, current_params
            )
            if merged_mol is None: return None
            
            accumulated_mol = merged_mol
            try:
                accumulated_params = topology.analyze_residue_topology(accumulated_mol)
            except: return None

    # [수정] 조립 완료 후 단 1회 전체 최적화 수행 (스위치에 따라 작동)
    final_energy = perform_optimization(accumulated_mol, is_final_step=True, optimize_flag=use_opt)
    
    final_atoms = rdkit_to_ase(accumulated_mol)
    final_coords = final_atoms.get_positions()
    final_nums = final_atoms.get_atomic_numbers()
    
    return final_coords, final_nums, final_energy

# ==============================================================================
# 4. 메인 실행 로직 (Main Execution)
# ==============================================================================
def run(args):
    # [수정] 입력 파라미터 무결성 검증 (사일런트 버그 차단)
    if len(args.residues) != len(args.rotamers):
        print(f"{RED}[Critical Error] Length mismatch! "
              f"--residues({len(args.residues)}) and --rotamers({len(args.rotamers)}) must match.{NC}")
        sys.exit(1)

    unit_len = len(args.residues)
    if args.target_length > 0:
        target_len = args.target_length
        tile_count = math.ceil(target_len / unit_len)
        full_residues = (args.residues * tile_count)[:target_len]
        full_rotamers = (args.rotamers * tile_count)[:target_len]
        suffix = f"{target_len}mer"
    else:
        tile_count = args.repeats
        full_residues = args.residues * tile_count
        full_rotamers = args.rotamers * tile_count
        target_len = len(full_residues)
        suffix = f"poly_x{tile_count}"

    print(f"{BLUE}[Info] Sequence: {full_residues}{NC}")
    base_name = '-'.join(f'{r}_{i}' for r, i in zip(args.residues, args.rotamers))
    polymer_name = f"{base_name}_{suffix}"
    
    print(f"{GREEN}>>> [Step 3] Polymer Building: {polymer_name}{NC}")
    
    # args에 optimize가 없으면 에러를 내지 말고 False를 반환하도록 안전장치(getattr) 적용
    use_opt = getattr(args, 'optimize', False)
    
    if use_opt:
        print(f"{YELLOW}>>> [Notice] Optimize Mode: Strict rigid assembly + Global ASE optimization at the end.{NC}")
    else:
        print(f"{YELLOW}>>> [Notice] Assembly Mode: ASE Optimization is DISABLED. Generating pure geometric structures.{NC}")    
    
    output_dir = os.path.join(PROJECT_ROOT, '1_data', 'polymers')
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = args.input_file or os.path.join(PROJECT_ROOT, '1_data', 'dimers', f"{base_name}.hdf5")
    if not os.path.exists(input_path):
        input_path = os.path.join(PROJECT_ROOT, '1_data', 'monomers', f"{base_name}.hdf5")
    
    if not os.path.exists(input_path):
        print(f"{RED}[Error] Input file not found: {input_path}{NC}"); return

    data = io.load_hdf5_data(input_path)
    points = data.get('points', [])
    base_xyzs = data.get('xyzs', [])
    
    if len(points) == 0: print(f"{RED}[Error] No points.{NC}"); return

    top_n = min(len(points), args.top_k)
    tiled_points = np.tile(points[:top_n], (1, tile_count))
    
    if base_xyzs is not None and len(base_xyzs) >= top_n:
        tiled_xyzs = base_xyzs[:top_n]
    else:
        tiled_xyzs = [None] * top_n
    
    model_files = [os.path.join(PROJECT_ROOT, '0_inputs', 'models', f) for f in os.listdir(os.path.join(PROJECT_ROOT, '0_inputs', 'models')) if f.endswith('.jpt')]
    
    use_cuda = (args.use_gpu == 1)
    device = 'cuda' if use_cuda else 'cpu'

    pool = None
    if args.threads > 1:
        try:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(processes=args.threads, initializer=_init_worker, 
                            initargs=(full_residues, full_rotamers, os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers'), 
                                      os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py'), model_files, device))
        except:
            _init_worker(full_residues, full_rotamers, os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers'), 
                         os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py'), model_files, device)
    else:
        _init_worker(full_residues, full_rotamers, os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers'), 
                     os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py'), model_files, device)

    results_vals, results_xyzs, results_points = [], [], []
    pbar = tqdm.tqdm(total=len(tiled_points), colour='cyan', desc='[Processing]')
    
    try:
        # [수정됨] 안전장치가 적용된 use_opt 변수를 사용하여 Task를 생성
        use_opt = getattr(args, 'optimize', False)
        tasks = [(full_residues, full_rotamers, tiled_points[i], tiled_xyzs[i], args.residues, use_opt) for i in range(len(tiled_points))]
        iterator = pool.imap(_build_polymer_task, tasks) if pool else (_build_polymer_task(t) for t in tasks)
        
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
            print(f"\n{GREEN}    [Done] Saved {len(results_xyzs)} structures.{NC}")
        else: print(f"\n{RED}    [Error] All tasks failed.{NC}")

    except KeyboardInterrupt:
        # [수정] 강제 종료 시 프로세스풀 안전하게 종료 (데드락 방지)
        if pool: pool.terminate() 
        safety.handle_force_stop(polymer_name, results_points, results_vals, PROJECT_ROOT, xyzs=results_xyzs)
    finally:
        if pool: pool.close(); pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residues", nargs="+", required=True)
    parser.add_argument("--rotamers", nargs="+", type=int, required=True)
    parser.add_argument("--target_length", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--input_file", type=str, default=None)
    
    # [수정] 최적화 여부를 터미널에서 제어할 수 있는 스위치 추가 (기본값: False)
    parser.add_argument("--optimize", action="store_true", help="Enable global ASE optimization after rigid body assembly")
    
    args = parser.parse_args()
    run(args)