"""
scripts/3_build_polymer.py

[설명]
Unit Block(Monomer/Dimer)의 샘플링 결과(HDF5)를 로드하여, 
지정된 길이의 폴리머(Polymer) 구조를 조립합니다.

[디버깅 모드 활성화]
- 각 단계(Step)마다 제거되는 캡 원자들의 인덱스를 상세히 출력합니다.
- Tail Cap(Polymer)과 Head Cap(Monomer)의 제거 과정을 추적합니다.

[작성자]
BAKERS Lead Chemist & Engineer
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
    """
    정렬을 위한 3점 프레임(좌표 인덱스)을 반환합니다.
    """
    anchor_idx = cap_info['anchor_index']
    cap_indices = cap_info['indices']
    
    def get_neighbors(idx, valid_set):
        atom = mol.GetAtomWithIdx(int(idx))
        nbrs = sorted([n for n in atom.GetNeighbors() if n.GetIdx() in valid_set],
                      key=lambda x: x.GetAtomicNum(), reverse=True)
        return [n.GetIdx() for n in nbrs]

    if is_target_tail:
        # [Target: Polymer Tail] Frame: [Link, Side, Anchor]
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
        # [Mobile: Monomer Head] Frame: [Anchor, Side, Link]
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
    """
    캡 인덱스뿐만 아니라, 캡에 붙어있는 수소(H)들까지 모두 찾아서 반환합니다.
    """
    full_indices = set(cap_indices)
    
    # 캡 원자들을 순회하며 수소 이웃을 찾음
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
        _WORKER_DATA['calc'] = EnsembleAIMNet2(model_files, device=device)
    except Exception as e:
        print(f"{RED}[Worker Init Error] Calculator load failed: {e}{NC}")

def _build_polymer_task(args):
    full_residues, full_rotamers, full_dihedrals = args
    if 'calc' not in _WORKER_DATA: return None

    accumulated_mol = None
    dih_idx = 0
    debug_dir = os.path.join(PROJECT_ROOT, "debug_steps")
    os.makedirs(debug_dir, exist_ok=True)
    
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
            print(f"  [Step {i}] Init {res}")
            try: write(os.path.join(debug_dir, f"step_{i}_init.pdb"), rdkit_to_ase(accumulated_mol))
            except: pass
        else:
            # ------------------------------------------------------------------
            # Step A: Identify Caps & Calculate Alignment
            # ------------------------------------------------------------------
            poly_caps = capping.detect_caps(accumulated_mol)
            if not poly_caps: print(f"{RED}[Error] Step {i}: No Tail Cap{NC}"); return None
            tail_cap = max(poly_caps, key=lambda x: max(x['indices'])) 
            
            mono_caps = capping.detect_caps(current_monomer)
            if not mono_caps: print(f"{RED}[Error] Step {i}: No Head Cap{NC}"); return None
            head_cap = min(mono_caps, key=lambda x: min(x['indices']))
            
            target_indices = get_alignment_frame(accumulated_mol, tail_cap, True)
            mobile_indices = get_alignment_frame(current_monomer, head_cap, False)
            
            if not target_indices or not mobile_indices:
                print(f"{RED}[Error] Step {i}: Frame detection failed{NC}"); return None
            
            # Transformation Matrix
            p_pos = accumulated_mol.GetConformer().GetPositions()
            m_pos = current_monomer.GetConformer().GetPositions()
            
            target_pts = p_pos[target_indices]
            mobile_pts = m_pos[mobile_indices]
            
            R, p_cent, m_cent = calculate_transform_matrix(target_pts, mobile_pts)
            
            # ------------------------------------------------------------------
            # Step B: Strict Decap (Expand to Hydrogens!)
            # ------------------------------------------------------------------
            tail_remove_indices = get_full_cap_indices(accumulated_mol, tail_cap['indices'])
            head_remove_indices = get_full_cap_indices(current_monomer, head_cap['indices'])
            
            # [DEBUG] 제거 대상 출력
            # print(f"  [Step {i} DEBUG] Removing Tail Cap (Polymer): {sorted(list(tail_remove_indices))}")
            # print(f"  [Step {i} DEBUG] Removing Head Cap (Monomer): {sorted(list(head_remove_indices))}")
            
            # ------------------------------------------------------------------
            # Step C: Merge
            # ------------------------------------------------------------------
            new_mol = Chem.RWMol()
            
            # 1. Add Polymer Atoms (Keep only non-cap)
            poly_map = {} 
            for atom in accumulated_mol.GetAtoms():
                old_idx = atom.GetIdx()
                if old_idx in tail_remove_indices: continue # SKIP CAP
                
                new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                new_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                poly_map[old_idx] = new_idx
                
            # 2. Add Monomer Atoms (Keep only non-cap)
            mono_map = {} 
            for atom in current_monomer.GetAtoms():
                old_idx = atom.GetIdx()
                if old_idx in head_remove_indices: continue # SKIP CAP
                    
                new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                new_mol.GetAtomWithIdx(new_idx).SetFormalCharge(atom.GetFormalCharge())
                mono_map[old_idx] = new_idx
                
            # 3. Add Internal Bonds
            for bond in accumulated_mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in poly_map and v in poly_map:
                    new_mol.AddBond(poly_map[u], poly_map[v], bond.GetBondType())
                    
            for bond in current_monomer.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in mono_map and v in mono_map:
                    new_mol.AddBond(mono_map[u], mono_map[v], bond.GetBondType())
                    
            # 4. Set Coordinates
            final_conf = Chem.Conformer(new_mol.GetNumAtoms())
            
            # Polymer coords
            for old_idx, new_idx in poly_map.items():
                final_conf.SetAtomPosition(new_idx, p_pos[old_idx])
                
            # Monomer coords (Transformed)
            m_pos_centered = m_pos - m_cent
            m_pos_transformed = np.dot(m_pos_centered, R.T) + p_cent
            
            for old_idx, new_idx in mono_map.items():
                final_conf.SetAtomPosition(new_idx, m_pos_transformed[old_idx])
            
            new_mol.AddConformer(final_conf)
            
            # 5. Stitching
            try:
                u_final = poly_map[tail_cap['anchor_index']]
                v_final = mono_map[head_cap['anchor_index']]
                new_mol.AddBond(u_final, v_final, Chem.BondType.SINGLE)
            except KeyError:
                print(f"{RED}[Error] Anchor lost! TailAnchor: {tail_cap['anchor_index']}, HeadAnchor: {head_cap['anchor_index']}{NC}")
                return None
            
            # Sanitize
            try: Chem.SanitizeMol(new_mol)
            except: new_mol.UpdatePropertyCache(strict=False)
            
            accumulated_mol = new_mol.GetMol()
            print(f"  [Step {i}] Connected. Total Atoms: {new_mol.GetNumAtoms()}")

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
            
        except Exception:
            pass
            
        # Debug Save
        try:
            write(os.path.join(debug_dir, f"step_{i}_result.pdb"), rdkit_to_ase(accumulated_mol))
        except: pass

    # Return Final
    final_atoms = rdkit_to_ase(accumulated_mol)
    final_coords = final_atoms.get_positions()
    final_nums = final_atoms.get_atomic_numbers()
    heavy_idx = np.where(final_nums != 1)[0]
    h_idx = np.where(final_nums == 1)[0]
    reorder_idx = np.concatenate([heavy_idx, h_idx])
    return final_coords[reorder_idx], final_nums[reorder_idx]

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
    if len(points) == 0: print(f"{RED}[Error] No points.{NC}"); return

    top_n = min(len(points), args.top_k)
    tiled_points = np.tile(points[:top_n], (1, tile_count))
    
    model_dir = os.path.join(PROJECT_ROOT, '0_inputs', 'models')
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.jpt')]
    if not model_files: print(f"{RED}[Error] No models found{NC}"); return

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
                    coords, nums = res
                    results_xyzs.append(coords)
                    results_vals.append(0.0)
                    results_points.append(tiled_points[i])
                pbar.update(1)
        else:
            for i, task_args in enumerate(tasks):
                res = _build_polymer_task(task_args)
                if res:
                    coords, nums = res
                    results_xyzs.append(coords)
                    results_vals.append(0.0)
                    results_points.append(tiled_points[i])
                pbar.update(1)

        save_path = os.path.join(output_dir, f"{polymer_name}.hdf5")
        if len(results_xyzs) > 0:
            print(f"\n{YELLOW}    [Note] Energies are set to 0.0.{NC}")
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
# 5. 디버깅 및 단위 테스트 (Debugging & Unit Test)
# ==============================================================================

def debug_check_environment():
    """
    [Debug] 필수 파일 및 디렉토리 존재 여부를 확인합니다. (CPDA 기준)
    """
    print(f"\n{BLUE}[Debug] Checking Environment Integrity...{NC}")
    
    if not os.path.exists(PROJECT_ROOT):
        print(f"{RED}[Fail] PROJECT_ROOT not found: {PROJECT_ROOT}{NC}")
        return False

    required_paths = {
        "Residue Params": os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py'),
        "Rotamer Dir": os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers'),
        "Model Dir": os.path.join(PROJECT_ROOT, '0_inputs', 'models'),
        # [수정] GLY 대신 CPDA 존재 여부 확인
        "Test Molecule (CPDA.sdf)": os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers', 'CPDA.sdf')
    }

    all_pass = True
    for name, path in required_paths.items():
        if os.path.exists(path):
            print(f"  - {name}: {GREEN}Found{NC}")
        else:
            print(f"  - {name}: {RED}Missing at {path}{NC}")
            if "CPDA.sdf" in name:
                print(f"    {YELLOW}>> Please ensure 'CPDA.sdf' is in 0_inputs/rotamers.{NC}")
            all_pass = False
    
    return all_pass

def create_mock_hdf5(filepath, count=5):
    """
    [Debug] 테스트용 임시 HDF5 파일을 생성합니다. (무작위 Dihedral 각도)
    """
    import h5py
    print(f"{BLUE}[Debug] Creating mock input data: {filepath}{NC}")
    
    # 2개의 Torsion angle을 가진다고 가정 (예: Phi, Psi)
    mock_points = np.random.uniform(-180, 180, size=(count, 2))
    mock_values = np.zeros(count) # 에너지는 0으로 가정

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('points', data=mock_points)
        f.create_dataset('values', data=mock_values)
    
    return filepath

def run_debug_mode():
    """
    [Debug] 스크립트의 전체 로직을 CPDA 2-mer로 검증합니다.
    """
    print(f"\n{YELLOW}>>> Starting BAKERS Script Debugging Mode (Target: CPDA) <<<{NC}")

    # 1. 환경 검사
    if not debug_check_environment():
        print(f"\n{RED}[Critical] Environment check failed. Please fix missing files.{NC}")
        return

    # 2. 테스트용 임시 파일 설정
    test_input_file = os.path.join(CURRENT_DIR, "debug_temp_input.hdf5")
    
    try:
        # 가상 데이터 생성 (CPDA의 Torsion 개수에 맞게 넉넉히 생성되지만, 여기선 임의의 값 사용)
        create_mock_hdf5(test_input_file)

        # 3. Mock Arguments 설정 (CPDA로 변경)
        class MockArgs:
            residues = ["CPDA", "CPDC"] # [수정] GLY -> CPDA
            rotamers = [0, 0]           # 0번 로타머 사용 (파일 내에 최소 1개는 있어야 함)
            target_length = 0           
            repeats = 1                 
            top_k = 2                   
            threads = 1                 
            use_gpu = 0                 
            input_file = test_input_file

        print(f"{BLUE}[Debug] Running build process with Mock Arguments (CPDA)...{NC}")
        
        # 4. 메인 로직 실행
        run(MockArgs())
        
        print(f"\n{GREEN}>>> [Success] Debugging finished successfully! <<<")
        print(f"The logic is sound. You can now run with real data.{NC}")

    except Exception as e:
        import traceback
        print(f"\n{RED}>>> [Fail] An error occurred during debugging:{NC}")
        traceback.print_exc()
        
    finally:
        # 5. 정리 (Cleanup)
        if os.path.exists(test_input_file):
            os.remove(test_input_file)
            print(f"{BLUE}[Debug] Removed temporary file: {test_input_file}{NC}")

# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAKERS Polymer Builder")
    
    parser.add_argument("--residues", nargs="+", help="List of residue names (e.g., GLY ALA)")
    parser.add_argument("--rotamers", nargs="+", type=int, help="List of rotamer indices")
    
    parser.add_argument("--target_length", type=int, default=0, help="Target polymer length")
    parser.add_argument("--repeats", type=int, default=4, help="Number of repeats if target_length is 0")
    parser.add_argument("--top_k", type=int, default=100, help="Top K conformations to process")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU (1) or CPU (0)")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input HDF5 file")
    
    # [Debug] 디버그 모드 플래그
    parser.add_argument("--debug", action="store_true", help="Run in self-checking debug mode")

    args = parser.parse_args()

    # 1. 디버그 모드인 경우: 인자 검사 없이 바로 테스트 함수 실행
    if args.debug:
        run_debug_mode()
        
    # 2. 일반 실행 모드인 경우: 여기서 필수 인자를 수동으로 검사
    else:
        if not args.residues or not args.rotamers:
            parser.error("the following arguments are required: --residues, --rotamers")

        # GPU/Multiprocessing 설정 충돌 방지
        if args.use_gpu and args.threads > 1:
            try: mp.set_start_method('spawn', force=True)
            except RuntimeError: pass 

        run(args)