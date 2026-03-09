"""
scripts/6_extract_angles.py

[설명]
폴리머 생성 결과(HDF5)들을 일괄 분석하여 '단일 통합 파일'로 저장하는 스크립트입니다.

[수정 내역]
- [Logic] Cap Dihedral Inference: 캡(Cap)이 존재하는 경우, 단순히 NaN으로 두지 않고 
          분자 그래프(연결성)를 분석하여 캡 이면각(Angle 0, End)을 구성하는 원자 인덱스를 자동으로 추론합니다.
- [Feature] Full Comparison: Ref/Post-Opt의 캡 각도까지 모두 계산하여 Pre-Opt 데이터와 완벽하게 1:1 비교를 수행합니다.
"""

import os
import sys
import argparse
import math
import re
import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from typing import List, Tuple, Dict, Optional, Any
from collections import Counter

# ==============================================================================
# 1. 환경 설정 및 모듈 로드
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from bakers.chem import topology
    from bakers.utils import io
    from bakers.analytics import metrics 
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

# ==============================================================================
# 2. 토폴로지 및 캡 추론 (Advanced)
# ==============================================================================

def infer_cap_dof(res_name: str, residue_params: Dict[str, Any], term_type: str) -> Optional[Tuple[int, int, int, int]]:
    """
    잔기의 연결 정보를 바탕으로 Cap Dihedral을 구성하는 4개의 원자 인덱스를 추론합니다.
    
    알고리즘:
    1. Cap과 Core를 연결하는 'Bridge Bond'를 찾습니다.
    2. Bridge Bond를 중심축(axis)으로 하여, 양쪽에서 '가장 무거운(연결성이 많은)' 이웃 원자를 찾습니다.
    3. [Prev - Axis1 - Axis2 - Next] 형태의 Dihedral 인덱스를 반환합니다.
    """
    p = residue_params[res_name]
    bonds = p['bonds'] # List of (u, v)
    
    # Graph 구성 (Adjacency List)
    adj = {}
    for u, v in bonds:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    
    # Core/Cap 원자 집합 구분
    core_set = set(p['residue_indices'])
    
    if term_type == 'N':
        # N-term Cap 인덱스: n_term_indices 중 core가 아닌 것
        cap_set = set(p.get('n_term_indices', [])) - core_set
        if not cap_set: return None
        
        # 1. Find Bridge Bond (Cap <-> Core)
        bridge = None
        for u in cap_set:
            for v in adj.get(u, []):
                if v in core_set:
                    bridge = (u, v) # u=Cap(Axis1), v=Core(Axis2)
                    break
            if bridge: break
        
        if not bridge: return None
        a1, a2 = bridge
        
        # 2. Find Neighbors (Prefer heavy atoms / backbone)
        # a0 -> a1 -> a2 -> a3 (Dihedral)
        
        # a0: a1(Cap)의 이웃 중 a2가 아닌 것, 연결성이 높은 순
        candidates_0 = [n for n in adj[a1] if n != a2]
        if not candidates_0: return None
        # Sort by degree (number of connections), descending
        candidates_0.sort(key=lambda x: len(adj[x]), reverse=True)
        a0 = candidates_0[0]
        
        # a3: a2(Core)의 이웃 중 a1이 아닌 것, 연결성이 높은 순
        candidates_3 = [n for n in adj[a2] if n != a1]
        if not candidates_3: return None
        candidates_3.sort(key=lambda x: len(adj[x]), reverse=True)
        a3 = candidates_3[0]
        
        return (a0, a1, a2, a3)

    elif term_type == 'C':
        # C-term Cap 인덱스
        cap_set = set(p.get('c_term_indices', [])) - core_set
        if not cap_set: return None
        
        # 1. Find Bridge Bond (Core <-> Cap)
        bridge = None
        for u in core_set:
            for v in adj.get(u, []):
                if v in cap_set:
                    bridge = (u, v) # u=Core(Axis1), v=Cap(Axis2)
                    break
            if bridge: break
            
        if not bridge: return None
        a1, a2 = bridge
        
        # 2. Find Neighbors
        # a0 -> a1 -> a2 -> a3
        
        # a0: a1(Core)의 이웃 중 a2가 아닌 것 (High Degree)
        candidates_0 = [n for n in adj[a1] if n != a2]
        if not candidates_0: return None
        candidates_0.sort(key=lambda x: len(adj[x]), reverse=True)
        a0 = candidates_0[0]
        
        # a3: a2(Cap)의 이웃 중 a1이 아닌 것 (High Degree)
        candidates_3 = [n for n in adj[a2] if n != a1]
        if not candidates_3: return None
        candidates_3.sort(key=lambda x: len(adj[x]), reverse=True)
        a3 = candidates_3[0]
        
        return (a0, a1, a2, a3)

    return None

def build_global_index_map(residues: List[str], residue_params: Dict[str, Any]) -> Tuple[List[Dict[int, int]], List[int]]:
    """각 잔기의 로컬 인덱스를 글로벌 인덱스로 변환"""
    residue_maps = []
    atom_counts = []
    global_cursor = 0
    total_residues = len(residues)
    
    for i, res_name in enumerate(residues):
        if res_name not in residue_params:
            raise ValueError(f"Unknown residue: {res_name}")
        p = residue_params[res_name]
        
        if i == 0:
            selection = p.get('n_term_indices', p['residue_indices'])
        elif i == total_residues - 1:
            selection = p.get('c_term_indices', p['residue_indices'])
        else:
            selection = p['residue_indices']
            
        local_to_global = {}
        for j, local_idx in enumerate(selection):
            local_to_global[local_idx] = global_cursor + j
            
        residue_maps.append(local_to_global)
        count = len(selection)
        atom_counts.append(count)
        global_cursor += count
        
    return residue_maps, atom_counts

def get_mapped_topology(residues: List[str], residue_params: Dict[str, Any]) -> Tuple[List[Optional[Tuple]], List[Tuple], int]:
    """
    글로벌 DOF 리스트 추출.
    [수정] Cap이 있는 경우, 'infer_cap_dof'를 통해 실제 원자 인덱스를 찾아냅니다.
    """
    residue_maps, atom_counts = build_global_index_map(residues, residue_params)
    total_atoms = sum(atom_counts)
    
    global_dofs = []
    bond_pairs = []
    total_residues = len(residues)
    
    for i, res_name in enumerate(residues):
        p = residue_params[res_name]
        mapping = residue_maps[i]
        
        # [N-term Cap Inference]
        if i == 0 and p.get('is_capped', False):
            # 추론 시도
            cap_dof = infer_cap_dof(res_name, residue_params, 'N')
            if cap_dof and all(idx in mapping for idx in cap_dof):
                # 글로벌 인덱스로 변환
                global_cap = tuple(mapping[idx] for idx in cap_dof)
                global_dofs.append(global_cap)
            else:
                # 추론 실패 시 Placeholder (NaN)
                global_dofs.append(None)

        # Backbone DOFs
        for dof_tuple in p.get('dofs', []):
            if all(idx in mapping for idx in dof_tuple):
                global_tuple = tuple(mapping[idx] for idx in dof_tuple)
                global_dofs.append(global_tuple)
                
        # [C-term Cap Inference]
        if i == total_residues - 1 and p.get('is_capped', False):
            cap_dof = infer_cap_dof(res_name, residue_params, 'C')
            if cap_dof and all(idx in mapping for idx in cap_dof):
                global_cap = tuple(mapping[idx] for idx in cap_dof)
                global_dofs.append(global_cap)
            else:
                global_dofs.append(None)

    # Bond Pairs
    for i in range(len(residues) - 1):
        try:
            u_local = residue_params[residues[i]]['upper_connect_indices'][0]
            v_local = residue_params[residues[i+1]]['lower_connect_indices'][-1]
            map_i = residue_maps[i]
            map_next = residue_maps[i+1]
            if u_local in map_i and v_local in map_next:
                bond_pairs.append((map_i[u_local], map_next[v_local]))
        except: pass
            
    return global_dofs, bond_pairs, total_atoms

# ==============================================================================
# 3. 레퍼런스 로드 및 데이터 처리 유틸리티
# ==============================================================================

def load_references(ref_paths: List[str], global_dofs: List[Optional[Tuple]], expected_atoms: int) -> List[Dict]:
    """레퍼런스 PDB의 각도 데이터 추출"""
    refs = []
    print(f"    [Ref Check] Looking for {len(ref_paths)} reference files...")
    
    for path in ref_paths:
        if not os.path.exists(path):
            alt_path = os.path.join(PROJECT_ROOT, path.lstrip('/'))
            if os.path.exists(alt_path): path = alt_path
            else: continue
            
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
        if mol is None: continue
        
        pos = mol.GetConformer().GetPositions()
        
        if len(pos) != expected_atoms:
            print(f"    [Ref Warning] Atom mismatch in {os.path.basename(path)}: Exp {expected_atoms} vs Act {len(pos)}")
            continue
        
        angles = []
        try:
            for indices in global_dofs:
                if indices is None: 
                    angles.append(np.nan)
                else:
                    # 이제 Cap도 indices가 있으므로 계산 시도
                    pts = [pos[i] for i in indices]
                    angles.append(metrics.calculate_dihedral(*pts))
            
            refs.append({"name": os.path.basename(path), "angles": np.array(angles)})
            print(f"    [Ref Loaded] {os.path.basename(path)}")
        except Exception as e:
            print(f"    [Ref Error] {e}")
            pass
            
    return refs

def save_csv_with_header(df: pd.DataFrame, filepath: str, description: str):
    """상세 설명을 포함한 헤더와 함께 CSV 저장"""
    with open(filepath, 'w', encoding='utf-8-sig') as f:
        f.write(description.strip() + "\n")
        df.to_csv(f, index=False)
    print(f"    -> Saved: {os.path.basename(filepath)} ({len(df)} rows)")

# ==============================================================================
# 4. 파일 분석 (메모리 적재)
# ==============================================================================

def analyze_single_file_to_memory(filepath, args, residue_params, refs):
    """
    개별 파일을 분석하여 CSV로 저장하지 않고, 데이터 리스트를 반환합니다.
    """
    filename = os.path.basename(filepath)
    
    # 1. Load Data
    try: data = io.load_hdf5_data(filepath)
    except: return [], {}
    if not data: return [], {}

    pre_angles = data.get('points')
    post_xyzs = data.get('xyzs')
    energies = data.get('values')
    
    if pre_angles is None or post_xyzs is None: return [], {}
    
    n_samples = len(pre_angles)
    n_atoms_data = post_xyzs.shape[1]

    # 2. Sequence & Topology
    target_length = args.target_length
    match = re.search(r'[-_](\d+)mer', filename)
    if match: target_length = int(match.group(1))
    if target_length == 0: target_length = len(args.residues) * args.repeats
    
    unit_len = len(args.residues)
    if unit_len == 0: return [], {}
    
    tile_count = math.ceil(target_length / unit_len)
    full_residues = (args.residues * tile_count)[:target_length]
    
    try:
        # 여기서 Cap Inference가 포함된 Topology를 얻음
        global_dofs, check_bonds, expected_atoms = get_mapped_topology(full_residues, residue_params)
    except Exception as e:
        print(f"[Error] Topology {filename}: {e}"); return [], {}

    # DOF 매칭 (Data와의 길이 비교)
    n_dofs_topo = len(global_dofs)
    n_dofs_data = pre_angles.shape[1]
    
    # 길이가 다르면 조정 (보통 Inference가 성공하면 같아야 함)
    if n_dofs_topo != n_dofs_data:
        if n_dofs_data > n_dofs_topo:
            global_dofs = global_dofs + [None] * (n_dofs_data - n_dofs_topo)
        else:
            global_dofs = global_dofs[:n_dofs_data]

    # --------------------------------------------------------------------------
    # 4. 데이터 수집 및 유사도 평가
    # --------------------------------------------------------------------------
    
    rows = []
    sim_tracker = {r['name']: [] for r in refs}

    for idx in range(n_samples):
        coords = post_xyzs[idx]
        theta_pre = pre_angles[idx]
        energy = energies[idx] if energies is not None else 0.0
        
        # Bond Check
        max_bond = 0.0
        for u, v in check_bonds:
            if u < n_atoms_data and v < n_atoms_data:
                d = np.linalg.norm(coords[u] - coords[v])
                if d > max_bond: max_bond = d

        # -- Row 1: Pre-Opt --
        row_pre = {
            "ID": f"{filename}_idx{idx}",
            "Type": "Pre-Opt",
            "Energy": None,
            "Max_Bond": None,
            "Ref_Target": "-"
        }
        for i, ang in enumerate(theta_pre):
            row_pre[f"Angle_{i}"] = round(ang, 2)
        rows.append(row_pre)

        # -- Row 2: Post-Opt --
        theta_post_list = []
        for indices in global_dofs:
            if indices is None:
                theta_post_list.append(np.nan)
            else:
                try:
                    pts = [coords[i] for i in indices]
                    theta_post_list.append(metrics.calculate_dihedral(*pts))
                except: theta_post_list.append(np.nan)
        
        row_post = {
            "ID": f"{filename}_idx{idx}",
            "Type": "Post-Opt",
            "Energy": round(energy, 4),
            "Max_Bond": round(max_bond, 3),
            "Ref_Target": "-"
        }
        for i, ang in enumerate(theta_post_list):
            row_post[f"Angle_{i}"] = round(ang, 2) if not np.isnan(ang) else None
        
        rows.append(row_post)
        
        # (C) Similarity Check
        theta_post_arr = np.array(theta_post_list)
        for r in refs:
            ref_angles = r['angles']
            valid_mask = (~np.isnan(theta_post_arr)) & (~np.isnan(ref_angles))
            if np.sum(valid_mask) > 0:
                diffs = []
                for a, b in zip(theta_post_arr[valid_mask], ref_angles[valid_mask]):
                    diffs.append(metrics.calculate_angle_diff(a, b))
                mae = np.mean(diffs)
                
                sim_row = row_post.copy()
                sim_row['Ref_Target'] = r['name']
                sim_row['MAE_Score'] = round(mae, 2)
                sim_tracker[r['name']].append((mae, sim_row))
                
    return rows, sim_tracker

# ==============================================================================
# 5. Main (Batch Execution)
# ==============================================================================

def run_batch_analysis(args):
    params_path = os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py')
    try: residue_params = topology.load_residue_params(params_path)
    except Exception as e: print(f"[Error] Params load: {e}"); return

    input_path = os.path.abspath(os.path.expanduser(args.input_file))
    target_files = []
    if os.path.isdir(input_path):
        for f in sorted(os.listdir(input_path)):
            if f.endswith(".hdf5"): target_files.append(os.path.join(input_path, f))
    elif os.path.exists(input_path):
        target_files.append(input_path)
    
    if not target_files: print("No files."); return
    print(f">>> Found {len(target_files)} HDF5 files. Starting aggregation...")

    # [1] Auto-Detect Target Length for Reference Loading
    detected_len = 0
    if args.target_length > 0:
        detected_len = args.target_length
    else:
        lengths = []
        for f in target_files:
            match = re.search(r'[-_](\d+)mer', os.path.basename(f))
            if match: lengths.append(int(match.group(1)))
        
        if lengths:
            detected_len = Counter(lengths).most_common(1)[0][0]
            print(f"    [Auto-Detect] Detected Polymer Length: {detected_len}mer")
        else:
            detected_len = len(args.residues) * args.repeats
            print(f"    [Auto-Detect] Using default length: {detected_len} residues")

    # Reference 로드용 토폴로지 생성
    unit_len = len(args.residues)
    tile_cnt = math.ceil(detected_len / unit_len)
    full_res = (args.residues * tile_cnt)[:detected_len]
    
    # Cap Inference 적용된 Topology 얻기
    global_dofs_ref, _, expected_atoms_ref = get_mapped_topology(full_res, residue_params)
    
    # [2] Load References
    refs = []
    if args.ref_pdbs:
        print("\n>>> Loading References...")
        refs = load_references(args.ref_pdbs, global_dofs_ref, expected_atoms_ref)
        if not refs:
            print("[Warning] No references loaded.")
        else:
            print(f"    -> Successfully loaded {len(refs)} references.")

    # [3] Aggregation Containers
    master_rows = []
    
    for r in refs:
        row = {
            "ID": r['name'], "Type": "Reference", "Energy": None, "Max_Bond": None, "Ref_Target": "-"
        }
        for i, ang in enumerate(r['angles']):
            row[f"Angle_{i}"] = round(ang, 2) if not np.isnan(ang) else None
        master_rows.append(row)

    master_similarity = {r['name']: [] for r in refs}

    # [4] Iterate Files
    for f in tqdm.tqdm(target_files, desc="Aggregating Files"):
        try: 
            f_rows, f_sim = analyze_single_file_to_memory(f, args, residue_params, refs)
            master_rows.extend(f_rows)
            for rname, items in f_sim.items():
                master_similarity[rname].extend(items)
        except Exception as e: 
            print(f"[Error] Processing {os.path.basename(f)}: {e}")

    # [5] Save Merged Data
    if not master_rows:
        print("No data extracted."); return

    print(f"\n>>> Saving Merged Results...")
    
    df_all = pd.DataFrame(master_rows)
    meta_cols = ["ID", "Type", "Energy", "Max_Bond"]
    angle_cols = sorted([c for c in df_all.columns if c.startswith("Angle_")], key=lambda x: int(x.split('_')[1]))
    df_all = df_all[meta_cols + angle_cols]

    header_desc = (
        "# [Merged Dataset]\n"
        "# ID: File Identifier\n"
        "# Type: Reference / Pre-Opt / Post-Opt\n"
        "# Energy: AIMNet2 Potential Energy (kcal/mol)\n"
        f"# Angle_0~{len(angle_cols)-1}: Backbone Dihedral Angles (Degrees). Cap Angles are now CALCULATED and included.\n"
    )
    
    merged_csv_path = os.path.join(PROJECT_ROOT, '1_data', 'analysis', "merged_comparison.csv")
    os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)
    save_csv_with_header(df_all, merged_csv_path, header_desc)

    # [6] Save Global Top 50
    if not refs:
        print("\n[Info] No references loaded, skipping Top 50.")
    else:
        for ref_name, items in master_similarity.items():
            if not items: continue
            
            items.sort(key=lambda x: x[0])
            top_50 = [item[1] for item in items[:50]]
            
            df_top = pd.DataFrame(top_50)
            top_cols = ["ID", "Type", "MAE_Score", "Energy"] + angle_cols
            df_top = df_top[top_cols]
            
            top_desc = (
                f"# [Global Top 50 Similarity]\n"
                f"# Target: {ref_name}\n"
                f"# Metric: Mean Absolute Error (MAE) including Caps if inferred.\n"
            )
            
            top_csv_path = os.path.join(PROJECT_ROOT, '1_data', 'analysis', f"merged_top50_{ref_name}.csv")
            save_csv_with_header(df_top, top_csv_path, top_desc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--residues", nargs="+", required=True)
    parser.add_argument("--target_length", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--ref_pdbs", nargs='+', default=[])
    parser.add_argument("--angle_threshold", type=float, default=30.0)
    args = parser.parse_args()
    run_batch_analysis(args)

    '''
    python ./scripts/6_extract_angles.py \
  --input_file 1_data/polymers \
  --residues CPDA DMMA \
  --ref_pdbs 2_results/FILTERED_HBOND_PDB/CPDA_DMMA_1_converted.pdb \
             2_results/FILTERED_HBOND_PDB/CPDA_DMMA_2_converted.pdb
    '''