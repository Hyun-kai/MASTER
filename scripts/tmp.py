"""
scripts/3_build_polymer.py

[설명]
Unit Block(Monomer/Dimer)의 샘플링 결과(HDF5)를 로드하여, 
지정된 길이의 폴리머(Polymer) 구조를 조립합니다.

[수정 사항]
- [Critcal Fix] 최적화된 구조의 에너지를 반환받아 HDF5에 저장하도록 수정 (0.0 -> 실제 에너지).
- IndentationError 등 문법 오류 수정.
- 멀티프로세싱 로그 억제 및 데이터 진단 기능 포함.
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
from multiprocessing import Pool

# 화학/물리 라이브러리
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS 
from ase.constraints import FixAtoms
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg
from rdkit.Chem import SDMolSupplier

# 경고 억제
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# BAKERS 모듈 임포트
try:
    from bakers.chem import topology, align, capping
    from bakers.sim.calculator import EnsembleAIMNet2
    from bakers.utils import io, safety, visual
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

# 색상 코드
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

# ==============================================================================
# 2. 헬퍼 함수
# ==============================================================================

def rdkit_to_ase(mol: Chem.Mol) -> Atoms:
    """RDKit Mol 객체를 ASE Atoms 객체로 변환합니다."""
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return Atoms(numbers=numbers, positions=positions)

def update_rdkit_coords(mol: Chem.Mol, new_positions: np.ndarray):
    """ASE 최적화 후의 좌표를 RDKit Mol 객체에 반영합니다."""
    conf = mol.GetConformer()
    for i, pos in enumerate(new_positions):
        conf.SetAtomPosition(i, pos.tolist())

def get_alignment_frame(mol, cap_info, is_target_tail):
    """정렬을 위한 3점 프레임(좌표 인덱스)을 반환합니다."""
    anchor_idx = cap_info['anchor_index']
    cap_indices = cap_info['indices']
    
    def get_neighbors(idx, valid_set):
        atom = mol.GetAtomWithIdx(int(idx))
        nbrs = sorted([n for n in atom.GetNeighbors() if n.GetIdx() in valid_set],
                      key=lambda x: x.GetAtomicNum(), reverse=True)
        return [n.GetIdx() for n in nbrs]

    if is_target_tail:
        anchor_atom = mol.GetAtomWithIdx(int(anchor_idx))
        links = [n.GetIdx() for n in anchor_atom.GetNeighbors() if n.GetIdx() in cap_indices]
        if not links: return None
        link_idx = links[0]
        
        cap_nbrs = get_neighbors(link_idx, cap_indices)
        if not cap_nbrs: # Fallback
            core_nbrs = get_neighbors(anchor_idx, set(range(mol.GetNumAtoms())) - cap_indices)
            if core_nbrs: return [link_idx, anchor_idx, core_nbrs[0]]
            return None
        side_idx = cap_nbrs[0]
        return [link_idx, side_idx, anchor_idx]
    
    else:
        anchor_atom = mol.GetAtomWithIdx(int(anchor_idx))
        links = [n.GetIdx() for n in anchor_atom.GetNeighbors() if n.GetIdx() in cap_indices]
        if not links: return None
        link_idx = links[0]
        
        all_indices = set(range(mol.GetNumAtoms()))
        core_indices = all_indices - cap_indices
        core_nbrs = get_neighbors(anchor_idx, core_indices)
        if not core_nbrs: return None
        side_idx = core_nbrs[0]
        return [anchor_idx, side_idx, link_idx]

def calculate_transform_matrix(p_coords, m_coords):
    p_center = np.mean(p_coords, axis=0)
    m_center = np.mean(m_coords, axis=0)
    p_centered = p_coords - p_center
    m_centered = m_coords - m_center

    H = np.dot(m_centered.T, p_centered)
    U, S, Vt = np.linalg.svd(H)
    
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
        
    return R, p_center, m_center

def get_full_cap_indices(mol, cap_indices):
    """캡 인덱스뿐만 아니라, 캡에 붙어있는 수소(H)들까지 모두 찾아서 반환합니다."""
    full_indices = set(cap_indices)
    for idx in cap_indices:
        atom = mol.GetAtomWithIdx(int(idx))
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() == 1: # Hydrogen
                full_indices.add(nbr.GetIdx())
    return full_indices

# ==============================================================================
# 3. 전역 작업자 데이터
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
        if res in full_params:
            _WORKER_DATA['params'][res] = full_params[res]
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        if os.path.exists(sdf_path):
            suppl = SDMolSupplier(sdf_path, removeHs=False)
            for rot in unique_rot:
                try:
                    mol = suppl[int(rot)]
                    if mol: _WORKER_DATA['molecules'][f"{res}_{rot}"] = mol
                except: pass

    try:
        torch.set_num_threads(1)
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            _WORKER_DATA['calc'] = EnsembleAIMNet2(model_files, device=device)
    except Exception as e:
        print(f"{RED}[Worker Init Error] Calculator load failed: {e}{NC}")

def _build_polymer_task(args):
    full_residues, full_rotamers, full_dihedrals = args
    if 'calc' not in _WORKER_DATA: return None

    accumulated_mol = None
    dih_idx = 0
    final_energy = 0.0  # [Init] 에너지 초기화
    
    for i, (res, rot) in enumerate(zip(full_residues, full_rotamers)):
        
        # 1. Load Monomer & Apply Torsion
        mol_key = f"{res}_{rot}"
        mol_obj = _WORKER_DATA['molecules'].get(mol_key)
        if mol_obj is None: return None
        
        current_monomer = Chem.Mol(mol_obj)
        conf = current_monomer.GetConformer()
        current_params = _WORKER_DATA['params'].get(res)
        
        for dof in current_params.get('dofs', []):
            if dih_idx < len(full_dihedrals):
                try:
                    SetDihedralDeg(conf, int(dof[0]), int(dof[1]), int(dof[2]), int(dof[3]), float(full_dihedrals[dih_idx]))
                except: pass
                dih_idx += 1
        
        # 2. Main Logic
        if i == 0:
            accumulated_mol = current_monomer
        else:
            # Step A: Identify Caps
            poly_caps = capping.detect_caps(accumulated_mol)
            if not poly_caps: return None
            tail_cap = max(poly_caps, key=lambda x: max(x['indices'])) 
            
            mono_caps = capping.detect_caps(current_monomer)
            if not mono_caps: return None
            head_cap = min(mono_caps, key=lambda x: min(x['indices']))
            
            target_indices = get_alignment_frame(accumulated_mol, tail_cap, True)
            mobile_indices = get_alignment_frame(current_monomer, head_cap, False)
            
            if not target_indices or not mobile_indices: return None
            
            # Step B: Align
            p_pos = accumulated_mol.GetConformer().GetPositions()
            m_pos = current_monomer.GetConformer().GetPositions()
            
            target_pts = p_pos[target_indices]
            mobile_pts = m_pos[mobile_indices]
            
            R, p_cent, m_cent = calculate_transform_matrix(target_pts, mobile_pts)
            
            # Step C: Decap
            tail_remove_indices = get_full_cap_indices(accumulated_mol, tail_cap['indices'])
            head_remove_indices = get_full_cap_indices(current_monomer, head_cap['indices'])
            
            # Step D: Merge
            new_mol = Chem.RWMol()
            poly_map = {} 
            for atom in accumulated_mol.GetAtoms():
                old_idx = atom.GetIdx()
                if old_idx in tail_remove_indices: continue
                new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                new_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                poly_map[old_idx] = new_idx
                
            mono_map = {} 
            for atom in current_monomer.GetAtoms():
                old_idx = atom.GetIdx()
                if old_idx in head_remove_indices: continue
                new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                new_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                mono_map[old_idx] = new_idx
                
            for bond in accumulated_mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in poly_map and v in poly_map:
                    new_mol.AddBond(poly_map[u], poly_map[v], bond.GetBondType())
                    
            for bond in current_monomer.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in mono_map and v in mono_map:
                    new_mol.AddBond(mono_map[u], mono_map[v], bond.GetBondType())
            
            final_conf = Chem.Conformer(new_mol.GetNumAtoms())
            for old_idx, new_idx in poly_map.items():
                final_conf.SetAtomPosition(new_idx, p_pos[old_idx])
            
            m_pos_centered = m_pos - m_cent
            m_pos_transformed = np.dot(m_pos_centered, R.T) + p_cent
            for old_idx, new_idx in mono_map.items():
                final_conf.SetAtomPosition(new_idx, m_pos_transformed[old_idx])
            
            new_mol.AddConformer(final_conf)
            
            try:
                u_final = poly_map[tail_cap['anchor_index']]
                v_final = mono_map[head_cap['anchor_index']]
                new_mol.AddBond(u_final, v_final, Chem.BondType.SINGLE)
            except KeyError:
                return None
            
            try: Chem.SanitizeMol(new_mol)
            except: new_mol.UpdatePropertyCache(strict=False)
            accumulated_mol = new_mol.GetMol()

        # 4. Relax
        try:
            ase_atoms = rdkit_to_ase(accumulated_mol)
            calc = _WORKER_DATA['calc']
            if hasattr(calc, 'reset'): calc.reset()
            ase_atoms.calc = calc
            if i > 0:
                n_fixed = max(0, len(ase_atoms) - 40)
                if n_fixed > 0:
                    ase_atoms.set_constraint(FixAtoms(indices=range(n_fixed)))
            
            opt = BFGS(ase_atoms, logfile=None)
            opt.run(fmax=0.2, steps=30)
            update_rdkit_coords(accumulated_mol, ase_atoms.get_positions())
            
            # [Fix] 계산된 에너지 저장
            final_energy = ase_atoms.get_potential_energy()
            
        except Exception:
            pass

    final_atoms = rdkit_to_ase(accumulated_mol)
    final_coords = final_atoms.get_positions()
    final_nums = final_atoms.get_atomic_numbers()
    heavy_idx = np.where(final_nums != 1)[0]
    h_idx = np.where(final_nums == 1)[0]
    reorder_idx = np.concatenate([heavy_idx, h_idx])
    
    # [Return] 좌표, 원자번호, 그리고 **에너지** 반환
    return final_coords[reorder_idx], final_nums[reorder_idx], final_energy

# ==============================================================================
# 4. 메인 실행 로직
# ==============================================================================

def run(args):
    unit_len = len(args.residues)
    if args.target_length > 0:
        target_len = args.target_length
        tile_count = math.ceil(target_len / unit_len)
        raw_residues = args.residues * tile_count
        raw_rotamers = args.rotamers * tile_count
        full_residues = raw_residues[:target_len]
        full_rotamers = raw_rotamers[:target_len]
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
    
    output_dir = os.path.join(PROJECT_ROOT, '1_data', 'polymers')
    os.makedirs(output_dir, exist_ok=True)
    
    dimer_path = os.path.join(PROJECT_ROOT, '1_data', 'dimers', f"{base_name}.hdf5")
    monomer_path = os.path.join(PROJECT_ROOT, '1_data', 'monomers', f"{base_name}.hdf5")
    input_path = args.input_file if args.input_file else (dimer_path if os.path.exists(dimer_path) else monomer_path)
    
    if not input_path or not os.path.exists(input_path):
        print(f"{RED}[Error] Input file not found: {input_path}{NC}"); return

    data = io.load_hdf5_data(input_path)
    points = data.get('points', [])

    print(f"\n{YELLOW}[Debug Data Check]{NC}")
    print(f" - Input File: {input_path}")
    print(f" - Total Points found in file: {len(points)}")
    print(f" - Requested top_k: {args.top_k}")

    if len(points) == 0: 
        print(f"{RED}[Error] No points.{NC}"); return
    elif len(points) == 1:
        print(f"{RED}[Warning] Input file contains only 1 conformation!")
        print(f"          This is why only 1 polymer is being built.")
        print(f"          Check Step 2 (Dimer Sampling) results.{NC}")

    top_n = min(len(points), args.top_k)
    tiled_points = np.tile(points[:top_n], (1, tile_count))
    
    model_dir = os.path.join(PROJECT_ROOT, '0_inputs', 'models')
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.jpt')]
    if not model_files: print(f"{RED}[Error] No models found{NC}"); return

    print(f"    [Ensemble] Loading models for workers...")

    rotamer_dir = os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers')
    params_path = os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py')

    use_cuda = (args.use_gpu == 1)
    device = 'cuda' if use_cuda else 'cpu'
    
    pool = None
    if args.threads > 1:
        try:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(processes=args.threads, initializer=_init_worker, 
                            initargs=(full_residues, full_rotamers, rotamer_dir, params_path, model_files, device))
        except:
            _init_worker(full_residues, full_rotamers, rotamer_dir, params_path, model_files, device)
            pool = None
    else:
        _init_worker(full_residues, full_rotamers, rotamer_dir, params_path, model_files, device)

    results_vals = []
    results_xyzs = []
    results_points = []
    
    print(f"    -> Processing {len(tiled_points)} sequences...")
    pbar = tqdm.tqdm(total=len(tiled_points), colour='cyan', desc='[Processing]')
    
    try:
        tasks = [(full_residues, full_rotamers, dihedrals) for dihedrals in tiled_points]
        
        if pool:
            for i, res in enumerate(pool.imap(_build_polymer_task, tasks)):
                if res is not None:
                    # [Fix] Unpack Energy (3 values)
                    coords, nums, energy = res
                    results_xyzs.append(coords)
                    results_vals.append(energy)
                    results_points.append(tiled_points[i])
                pbar.update(1)
        else:
            for i, task_args in enumerate(tasks):
                res = _build_polymer_task(task_args)
                if res:
                    coords, nums, energy = res
                    results_xyzs.append(coords)
                    results_vals.append(energy)
                    results_points.append(tiled_points[i])
                pbar.update(1)

        save_path = os.path.join(output_dir, f"{polymer_name}.hdf5")
        if len(results_xyzs) > 0:
            print(f"\n{YELLOW}    [Note] Energies are computed.{NC}")
            io.save_results_hdf5(save_path, np.array(results_points), np.array(results_vals), np.array(results_xyzs))
            print(f"    [Done] Saved {len(results_xyzs)} structures.")
            try: visual.analyze_and_save(save_path)
            except: pass
        else:
            print(f"{RED}    [Error] All tasks failed.{NC}")

    except KeyboardInterrupt:
        pbar.close()
        if pool: pool.terminate()
        safety.handle_force_stop(polymer_name, results_points, results_vals, PROJECT_ROOT, xyzs=results_xyzs)
    finally:
        if pool: pool.close(); pool.join()

# ==============================================================================
# 5. 디버깅 및 단위 테스트
# ==============================================================================

def debug_check_environment():
    print(f"\n{BLUE}[Debug] Checking Environment Integrity...{NC}")
    if not os.path.exists(PROJECT_ROOT):
        print(f"{RED}[Fail] PROJECT_ROOT not found: {PROJECT_ROOT}{NC}")
        return False

    required_paths = {
        "Residue Params": os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py'),
        "Rotamer Dir": os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers'),
        "Model Dir": os.path.join(PROJECT_ROOT, '0_inputs', 'models'),
        "Test Molecule (CPDA.sdf)": os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers', 'CPDA.sdf')
    }

    all_pass = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"  - {name}: {GREEN}Found{NC}")
        else:
            print(f"  - {name}: {RED}Missing at {path}{NC}")
            all_pass = False
    return all_pass

def create_mock_hdf5(filepath, count=5):
    import h5py
    print(f"{BLUE}[Debug] Creating mock input data: {filepath}{NC}")
    mock_points = np.random.uniform(-180, 180, size=(count, 2))
    mock_values = np.zeros(count)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('points', data=mock_points)
        f.create_dataset('values', data=mock_values)
    return filepath

def run_debug_mode():
    print(f"\n{YELLOW}>>> Starting BAKERS Script Debugging Mode (Target: CPDA) <<<{NC}")
    if not debug_check_environment():
        print(f"\n{RED}[Critical] Environment check failed. Please fix missing files.{NC}")
        return

    test_input_file = os.path.join(CURRENT_DIR, "debug_temp_input.hdf5")
    try:
        create_mock_hdf5(test_input_file)
        class MockArgs:
            residues = ["CPDA", "CPDC"]
            rotamers = [0, 0]
            target_length = 0
            repeats = 1
            top_k = 2
            threads = 1
            use_gpu = 0
            input_file = test_input_file

        print(f"{BLUE}[Debug] Running build process with Mock Arguments (CPDA)...{NC}")
        run(MockArgs())
        print(f"\n{GREEN}>>> [Success] Debugging finished successfully! <<<")
    except Exception as e:
        import traceback
        print(f"\n{RED}>>> [Fail] An error occurred during debugging:{NC}")
        traceback.print_exc()
    finally:
        if os.path.exists(test_input_file):
            os.remove(test_input_file)
            print(f"{BLUE}[Debug] Removed temporary file: {test_input_file}{NC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residues", nargs="+", help="List of residue names")
    parser.add_argument("--rotamers", nargs="+", type=int, help="List of rotamer indices")
    parser.add_argument("--target_length", type=int, default=0, help="Target polymer length")
    parser.add_argument("--repeats", type=int, default=4, help="Number of repeats if target_length is 0")
    parser.add_argument("--top_k", type=int, default=100, help="Top K conformations to process")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU (1) or CPU (0)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input HDF5 file")
    
    parser.add_argument("--debug", action="store_true", help="Run in self-checking debug mode")

    args = parser.parse_args()

    if args.debug:
        run_debug_mode()
    else:
        if not args.residues or not args.rotamers:
            parser.error("the following arguments are required: --residues, --rotamers")
        if args.use_gpu and args.threads > 1:
            try: mp.set_start_method('spawn', force=True)
            except RuntimeError: pass 

        run(args)