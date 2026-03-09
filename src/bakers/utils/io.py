"""
src/bakers/utils/io.py

[목적 및 설명]
데이터 입출력(I/O) 및 파일 처리를 전담하는 코어 유틸리티 모듈입니다.
HDF5 궤적 데이터 관리, 분자 구조(PDB/SDF) 저장, 원자 정보 파싱 및 
RDKit을 활용한 화학적 구조 무결성(Sanitization) 검증 기능을 포괄합니다.

[설계 의도 및 구조 (Top-Down)]
1. Parsing & Guessing: 파일명 해석(정규식 기반) 및 원자 번호 할당
2. Chemical Validation: 3D 좌표 기반 결합 추론 및 화학적 타당성 검증
3. Basic I/O: 안전한 재귀적 직렬화(Recursive Serialization) 파라미터 저장
4. Structure I/O: 3D 분자 구조(PDB, SDF) 파일 쓰기 (오버로딩 적용)
5. HDF5 Operations: 대용량 시뮬레이션 데이터 로드 및 병합
6. High-Level Workflow: 전체 I/O 기능을 종합한 구조 추출 파이프라인
"""

import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union

# 외부 라이브러리 (필수 의존성)
try:
    from ase import Atoms
    from ase.io import write
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    print("[Warning] 'ase' or 'scipy' not found. PDB saving might fail.")

# [중요] 타입 힌트 평가 및 글로벌 참조를 위해 RDKit 모듈을 최상단에서 명시적으로 Import 합니다.
try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    print("[Warning] 'rdkit' not found. SDF saving and chemical validation will fail.")


# ==============================================================================
# [Section 1] 파싱 및 기본 추론 유틸리티 (Parsing & Guessing)
# ==============================================================================

def load_smiles_from_csv(csv_path: str) -> Dict[str, str]:
    """
    SMILES 데이터가 담긴 CSV 파일을 읽어 딕셔너리로 반환합니다.
    헤더의 존재 여부에 관계없이 문자열 데이터를 추출합니다.
    
    Args:
        csv_path (str): CSV 파일의 경로
    Returns:
        Dict[str, str]: {잔기명: SMILES} 형태의 딕셔너리
    """
    smiles_map = {}
    try:
        # 데이터 유실을 막기 위해 모든 데이터를 일단 string으로 읽음
        df = pd.read_csv(csv_path, header=None, dtype=str)
        for _, r in df.iterrows():
            if pd.notna(r[0]) and pd.notna(r[1]):
                res_name = str(r[0]).strip()
                smiles = str(r[1]).strip()
                
                # 유효한 문자열이 있는 경우에만 맵핑
                if res_name and smiles:
                    smiles_map[res_name] = smiles
    except Exception as e:
        print(f"[Warning] Failed to read SMILES data from {csv_path}: {e}")
    return smiles_map


def parse_filename_info(filename: str) -> Tuple[List[str], List[int], int]:
    """
    정규표현식을 사용하여 파일명에서 잔기 종류, 로타머 인덱스, 조립 반복 횟수를 견고하게 파싱합니다.
    
    Args:
        filename (str): 분석할 파일명 (예: AIB_0-DAL_1_hexamer.hdf5, AIB_0_x5.hdf5)
        
    Returns:
        Tuple: (잔기명 리스트, 로타머 인덱스 리스트, 반복 횟수)
    """
    base = os.path.basename(filename).replace('.hdf5', '')
    
    # 1. 영어 텍스트 기반 길이 맵핑 (확장성 고려)
    length_map = {
        'dimer': 2, 'trimer': 3, 'tetramer': 4, 'pentamer': 5,
        'hexamer': 6, 'heptamer': 7, 'octamer': 8, 'nonamer': 9, 'decamer': 10,
        'polymer': 0
    }
    
    target_length = 0
    clean_base = base.lower()
    
    # 텍스트 매칭 처리 및 확장자 제거
    for key, val in length_map.items():
        if key in clean_base:
            target_length = val
            base = re.sub(rf"_{key}$", "", base, flags=re.IGNORECASE)
            break
            
    # 숫자형 N-mer 파싱 (예: _12mer)
    nmer_match = re.search(r'_(\d+)mer$', base, re.IGNORECASE)
    if nmer_match:
        target_length = int(nmer_match.group(1))
        base = base[:nmer_match.start()]

    # 2. 명시적 배수 처리 (_xN)
    explicit_repeats = 0
    x_match = re.search(r'_x(\d+)$', base, re.IGNORECASE)
    if x_match:
        explicit_repeats = int(x_match.group(1))
        base = base[:x_match.start()]

    # 3. 잔기 및 로타머 쌍 파싱
    parts = base.split('-')
    residues, rotamers = [], []
    
    for part in parts:
        # AIB_0 패턴 매칭
        match = re.search(r'^([A-Za-z0-9]+)_(\d+)$', part)
        if match:
            residues.append(match.group(1))
            rotamers.append(int(match.group(2)))
        else:
            # 로타머 정보가 없을 경우 기본값 0 할당
            residues.append(part)
            rotamers.append(0)
            
    num_residues = len(residues)
    repeats = 1
    
    # 4. 반복 횟수(Repeats) 종합 계산
    if target_length > 0 and num_residues > 0:
        repeats = max(1, target_length // num_residues)
    elif explicit_repeats > 0:
        repeats = explicit_repeats
        
    return residues, rotamers, repeats


def get_atomic_numbers(residues: List[str], residue_params_dict: Dict, rotamers: Optional[List[int]] = None) -> np.ndarray:
    """
    조립될 펩타이드 시퀀스의 위상 정보를 바탕으로 최종 원자 번호 배열을 생성합니다.
    Heavy Atom 배열 후 H Atom 이 오도록 재정렬을 수행합니다.
    (Note: rotamers 인자는 하위 호환성을 위해 유지되나 로직에는 영향을 주지 않습니다.)
    """
    selections = []
    _numbers_list = []

    for res in residues:
        if res not in residue_params_dict:
            raise KeyError(f"Residue '{res}' not found in residue_params.")
        _numbers_list.append(np.array(residue_params_dict[res]['atoms']))

    for i, res in enumerate(residues):
        p = residue_params_dict[res]
        
        if len(residues) == 1:
            sel = list(range(len(p['atoms'])))
        elif i == 0:
            sel = p.get('n_term_indices', p['residue_indices'])
        elif i == len(residues) - 1:
            sel = p.get('c_term_indices', p['residue_indices'])
        else:
            sel = p['residue_indices']
            
        selections.append(sel)

    numbers = np.concatenate([nums[sel] for nums, sel in zip(_numbers_list, selections)])
    
    # Heavy 원자 우선 정렬 후 수소(H) 원자 부착
    mask_heavy = numbers != 1
    mask_h = numbers == 1
    return np.concatenate([numbers[mask_heavy], numbers[mask_h]])


def guess_elements_from_geometry(coords: np.ndarray) -> List[str]:
    """
    토폴로지 정보가 없을 경우, 3D 좌표 간의 거리(Geometry)만으로 원소 기호를 추론합니다.
    """
    n_atoms = len(coords)
    dists = squareform(pdist(coords))
    BOND_THRESHOLD = 1.7 
    
    adj = dists < BOND_THRESHOLD
    np.fill_diagonal(adj, False)
    
    elements = ['C'] * n_atoms
    carbonyl_carbons = []
    
    for i in range(n_atoms):
        neighbors = np.where(adj[i])[0]
        if len(neighbors) == 1:
            n_idx = neighbors[0]
            if dists[i, n_idx] < 1.2: 
                elements[i] = 'H'
            else: 
                elements[i] = 'O'
                carbonyl_carbons.append(n_idx) 
                
    for i in range(n_atoms):
        if elements[i] != 'C': continue
        deg = len(np.where(adj[i])[0])
        if deg == 3 and i not in carbonyl_carbons:
            neighbors = np.where(adj[i])[0]
            if any(n in carbonyl_carbons for n in neighbors):
                elements[i] = 'N'
                    
    return elements


# ==============================================================================
# [Section 2] 화학적 검증 및 분자 객체 생성 (Chemical Validation & Mol Creation)
# ==============================================================================

def _create_rdkit_mol_from_coords(numbers: np.ndarray, positions: np.ndarray, info: dict = None) -> Optional[Chem.Mol]:
    """
    RDKit의 공식 모듈(rdDetermineBonds)을 사용하여 3D 좌표로부터 결합선(Valence)과 차수를 완벽히 역추론합니다.
    """
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(numbers))
    
    for i, (z, pos) in enumerate(zip(numbers, positions)):
        atom = Chem.Atom(int(z))
        mol.AddAtom(atom)
        conf.SetAtomPosition(i, pos.tolist())
        
    mol.AddConformer(conf)
    
    try:
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
        rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        
        if info:
            for k, v in info.items():
                mol.SetProp(str(k), str(v))
                
        return mol.GetMol()
    except Exception:
        return None


def _is_chemically_valid(numbers: np.ndarray, positions: np.ndarray) -> bool:
    """
    추출된 3D 좌표가 화학적으로 합당한 분자인지 엄격하게 검증하는 필터입니다.
    산화수(Valence) 초과, 파괴된 고리 구조 등이 여기서 걸러집니다.
    """
    try:
        mol = _create_rdkit_mol_from_coords(numbers, positions)
        if mol is None: return False
            
        Chem.SanitizeMol(mol)
        
        mol_block = Chem.MolToMolBlock(mol)
        verified_mol = Chem.MolFromMolBlock(mol_block, sanitize=True)
        
        if verified_mol is None: return False
        return True
    except Exception:
        return False


# ==============================================================================
# [Section 3] 기본 파라미터 및 단순 텍스트 입출력 (Basic Text I/O)
# ==============================================================================

def _to_native(obj: Any) -> Any:
    """
    [안전성 강화] Numpy 객체 및 중첩된 컬렉션을 순수 Python 객체로 재귀적 변환합니다.
    파라미터 딕셔너리를 Python 코드로 저장할 때 구문 에러(NameError)를 원천 차단합니다.
    """
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def save_residue_params(params_dict: Dict[str, Any], filepath: str) -> None:
    """
    사전 계산된 파라미터 딕셔너리를 시뮬레이션 환경에서 'import' 하여 
    사용할 수 있도록 파이썬 코드(.py) 포맷으로 변환하여 저장합니다.
    I/O 병목을 피하기 위해 메모리 버퍼에서 문자열을 완성한 후 일괄 기록합니다.
    """
    lines = ["import numpy as np\n\nresidue_params = {\n"]
    
    for res_name, data in params_dict.items():
        lines.append(f"    '{res_name}': {{\n")
        safe_data = _to_native(data) # Numpy 의존성 완벽 제거
        
        for key, val in safe_data.items():
            lines.append(f"        '{key}': {repr(val)},\n")
        lines.append("    },\n")
        
    lines.append("}\n")
    
    with open(filepath, 'w') as f:
        f.writelines(lines)


def write_xyz(types: List[str], coords: np.ndarray, msg: str = "", fn: str = None, is_onehot: bool = False) -> str:
    """디버깅 및 빠른 구조 시각화를 위한 .xyz 텍스트 파일 생성 유틸리티입니다."""
    lines = [f"{coords.shape[0]}\n", f"{msg}\n"]
    for i in range(coords.shape[0]):
        lines.append(f"{types[i]}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n")
    
    xyz_str = "".join(lines)[:-1] # 마지막 줄바꿈 제거
    
    if fn is not None:
        with open(fn, "w") as w:
            w.write(xyz_str)
    return xyz_str


# ==============================================================================
# [Section 4] 3D 분자 구조 입출력 (Structure PDB/SDF I/O)
# ==============================================================================

def save_pdb(filepath: str, numbers: np.ndarray, positions: np.ndarray, info: dict = None) -> bool:
    """ASE 엔진을 사용하여 3D 좌표를 표준 PDB 포맷으로 저장합니다."""
    try:
        atoms = Atoms(numbers=numbers, positions=positions)
        if info:
            for k, v in info.items():
                atoms.info[k] = v
        write(filepath, atoms)
        return True
    except Exception as e:
        print(f"[IO Error] Failed to save PDB: {e}")
        return False


def save_sdf(arg1: Union[str, Chem.Mol], arg2: Any, arg3: Any, info: dict = None) -> bool:
    """
    SDF 포맷 파일 저장 유틸리티입니다. (다형성 지원 - Overloading)
    
    [Case 1: Monomer Prep]
    save_sdf(mol: Chem.Mol, valid_cids: List[int], filepath: str)
    -> 분자 객체 내의 다중 Conformer 중 유효한 것들만 선택하여 기록합니다.
    
    [Case 2: HDF5 Extraction]
    save_sdf(filepath: str, numbers: np.ndarray, positions: np.ndarray, info: dict = None)
    -> 원자 번호와 좌표만으로 결합을 새롭게 추론하여 단일 구조를 기록합니다.
    """
    
    # --- Case 1: 1_prep.py 파이프라인 대응 ---
    if isinstance(arg1, Chem.Mol):
        mol = arg1
        valid_cids = arg2
        filepath = arg3
        try:
            w = Chem.SDWriter(filepath)
            for i, cid in enumerate(valid_cids):
                mol.SetIntProp("Rotamer_Index", i)
                w.write(mol, confId=cid)
            w.close()
            return True
        except Exception as e:
            print(f"    [SDF Error] Failed to save conformers SDF: {e}")
            return False
            
    # --- Case 2: HDF5 추출 파이프라인 대응 ---
    elif isinstance(arg1, str):
        filepath = arg1
        numbers = arg2
        positions = arg3
        try:
            mol = _create_rdkit_mol_from_coords(numbers, positions, info)
            if mol is None: return False
                
            with Chem.SDWriter(filepath) as writer:
                writer.write(mol)
            return True
        except Exception as e:
            print(f"    [SDF Error] Failed to save structure SDF: {e}")
            return False
            
    else:
        print(f"    [SDF Error] Invalid argument types provided to save_sdf.")
        return False


# ==============================================================================
# [Section 5] 대용량 HDF5 입출력 파이프라인 (HDF5 Operations)
# ==============================================================================

def load_hdf5_data(filepath: str, sorted_by_energy: bool = True) -> Optional[Dict[str, np.ndarray]]:
    """HDF5 궤적 파일에서 3D 좌표(xyzs), 에너지(energies), 다이히드럴(points) 데이터를 로드합니다."""
    if not os.path.exists(filepath):
        print(f"[IO Error] File not found: {filepath}")
        return None

    try:
        with h5py.File(filepath, 'r') as f:
            xyzs = np.array(f['xyzs']) if 'xyzs' in f else np.array(f['positions']) if 'positions' in f else None
            
            energies = None
            if 'values' in f: energies = np.array(f['values'])
            elif 'energies' in f: energies = np.array(f['energies'])
            elif 'energy' in f: energies = np.array(f['energy'])

            points = np.array(f['points']) if 'points' in f else None
            numbers = np.array(f['numbers']) if 'numbers' in f else None

            if energies is None: return None

            if sorted_by_energy:
                idx = np.argsort(energies)
                energies = energies[idx]
                if xyzs is not None: xyzs = xyzs[idx]
                if points is not None: points = points[idx]

            return {'xyzs': xyzs, 'energies': energies, 'points': points, 'numbers': numbers}

    except Exception as e:
        print(f"[IO Error] Failed to read {filepath}: {e}")
        return None


def save_results_hdf5(filepath: str, points: np.ndarray, values: np.ndarray, xyzs: np.ndarray, numbers: np.ndarray = None) -> bool:
    """샘플링된 시뮬레이션 결과를 HDF5 데이터베이스 포맷으로 저장합니다."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('points', data=points, dtype='float32')
            f.create_dataset('values', data=values, dtype='float32')
            f.create_dataset('energies', data=values, dtype='float32') 
            
            if not isinstance(xyzs, np.ndarray): xyzs = np.array(xyzs)
            f.create_dataset('xyzs', data=xyzs, dtype='float32')
            
            if numbers is not None:
                if not isinstance(numbers, np.ndarray): numbers = np.array(numbers)
                f.create_dataset('numbers', data=numbers, dtype='int32')
            
        print(f"    [Save] HDF5 Saved: {filepath}")
        return True
    except Exception as e:
        print(f"    [Save Error] {e}")
        return False


def merge_hdf5_files(file_list: List[str], output_path: str, verbose: bool = True) -> bool:
    """여러 개의 HDF5 결과 파일을 단일 파일로 병합(Concatenate)합니다."""
    if not file_list:
        if verbose: print(" [Merge Error] No files to merge.")
        return False

    all_points, all_values, all_xyzs = [], [], []
    final_numbers = None 
    processed_count = 0

    if verbose: print(f" [Merge] Start merging {len(file_list)} files...")

    for fpath in file_list:
        if not os.path.exists(fpath):
            if verbose: print(f" [Skip] File not found: {fpath}")
            continue
            
        try:
            with h5py.File(fpath, 'r') as f:
                if 'points' in f and ('values' in f or 'energies' in f) and 'xyzs' in f:
                    all_points.append(f['points'][:])
                    all_values.append(f['values'][:] if 'values' in f else f['energies'][:])
                    all_xyzs.append(f['xyzs'][:])
                    
                    if 'numbers' in f and final_numbers is None:
                        final_numbers = f['numbers'][:]
                    processed_count += 1
                else:
                    if verbose: print(f" [Skip] Missing keys in {os.path.basename(fpath)}")
        except Exception as e:
            if verbose: print(f" [Error] Reading {os.path.basename(fpath)}: {e}")

    if processed_count == 0: return False

    try:
        final_points = np.concatenate(all_points, axis=0)
        final_values = np.concatenate(all_values, axis=0)
        final_xyzs = np.concatenate(all_xyzs, axis=0)
        
        save_results_hdf5(output_path, final_points, final_values, final_xyzs, numbers=final_numbers)
        if verbose: print(f" [Merge] Combined {processed_count} files. Shape: {final_xyzs.shape}")
        return True
    except Exception as e:
        if verbose: print(f" [Merge Error] {e}")
        return False


# ==============================================================================
# [Section 6] 상위 구조 추출 파이프라인 (High-Level Extraction Workflow)
# ==============================================================================

def extract_and_save_top_structures(target_file: str, output_dir: str, 
                                    top_n: int = 100, cluster_threshold: float = 45.0, 
                                    project_root: str = None, save_format: str = "both"):
    """
    [마스터 워크플로우] HDF5 궤적 파일에서 에너지가 가장 낮은 구조들을 불러와 
    화학적 검증(Sanitize)을 거친 후, 유효한(Valid) 구조만 PDB 및 SDF로 변환하여 추출합니다.
    """
    pdb_dir = os.path.join(output_dir, "pdb")
    sdf_dir = os.path.join(output_dir, "sdf")
    
    if save_format in ['pdb', 'both']: os.makedirs(pdb_dir, exist_ok=True)
    if save_format in ['sdf', 'both']: os.makedirs(sdf_dir, exist_ok=True)

    # 타겟 개수만큼 이미 추출되어 있다면 스킵
    existing_pdbs = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')] if os.path.exists(pdb_dir) else []
    existing_sdfs = [f for f in os.listdir(sdf_dir) if f.endswith('.sdf')] if os.path.exists(sdf_dir) else []
    
    if (save_format in ['pdb', 'both'] and len(existing_pdbs) >= top_n) or \
       (save_format in ['sdf', 'both'] and len(existing_sdfs) >= top_n):
        print(f"    [Skip] Target files already exist in {output_dir}. Skipping extraction.")
        return

    # 1. 에너지 순으로 데이터 로드
    data = load_hdf5_data(target_file, sorted_by_energy=True)
    if data is None or data['xyzs'] is None:
        print("    [Error] No coordinate data found for extraction.")
        return

    xyzs, energies, total_structures = data['xyzs'], data['energies'], len(data['xyzs'])
    
    # 2. 원자 번호(Topology) 복원 시도
    atomic_numbers = data.get('numbers')
    if atomic_numbers is None:
        residues, rotamers, repeats = parse_filename_info(target_file)
        params_path = os.path.join(project_root, '0_inputs', 'residue_params.py') if project_root else '0_inputs/residue_params.py'
            
        if os.path.exists(params_path):
            try:
                from bakers.chem import topology
                params = topology.load_residue_params(params_path)
                atomic_numbers = get_atomic_numbers(residues * repeats, params, rotamers * repeats)
            except Exception as e:
                print(f"    [Warn] Topology reconstruction failed: {e}")
        else:
            print("    [Warn] residue_params.py not found. Using geometry-based element guessing.")

    base_name = os.path.basename(target_file).replace('.hdf5', '')
    print(f"    [Extraction] Energy-sorted Top-{top_n} with Validation | Format: {save_format.upper()}")
    
    count_pdb, count_sdf, saved_count, current_rank = 0, 0, 0, 1
    
    # 3. 데이터 순회 및 화학적 필터링 적용
    for idx in range(total_structures):
        if saved_count >= top_n: break
            
        coords, energy = xyzs[idx], energies[idx]
        current_numbers = atomic_numbers
        
        # 원자 번호 정보가 전혀 없다면 3D 좌표 거리 기반으로 추론
        if current_numbers is None:
            current_numbers = np.array([Chem.GetPeriodicTable().GetAtomicNumber(e) for e in guess_elements_from_geometry(coords)])

        # [필터] 공간적으로 찌그러진 구조 제거
        if not _is_chemically_valid(current_numbers, coords):
            print(f"    [Sanitize Fail] Index {idx} has severe structural distortion. Skipping...")
            continue
            
        info = {'Energy': float(energy), 'Original_Index': int(idx)}

        # 4. 검증된 구조 파일 쓰기
        if save_format in ['pdb', 'both']:
            if save_pdb(os.path.join(pdb_dir, f"{base_name}_rank{current_rank}_idx{idx}.pdb"), current_numbers, coords, info=info):
                count_pdb += 1
                
        if save_format in ['sdf', 'both']:
            if save_sdf(os.path.join(sdf_dir, f"{base_name}_rank{current_rank}_idx{idx}.sdf"), current_numbers, coords, info=info):
                count_sdf += 1
                
        saved_count += 1
        current_rank += 1
                
    if save_format in ['pdb', 'both']: print(f"    [Saved] {count_pdb} Valid PDB files saved to {pdb_dir}/")
    if save_format in ['sdf', 'both']: print(f"    [Saved] {count_sdf} Valid SDF files saved to {sdf_dir}/")
    if saved_count < top_n:
        print(f"    [Warn] Only found {saved_count} valid structures out of {total_structures} generated.")