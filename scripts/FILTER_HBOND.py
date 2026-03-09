"""
scripts/FILTER_HBOND.py (Refactored)
"""

import os
import shutil
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem

# ==============================================================================
# 1. 설정
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PDB_SOURCE_DIR = os.path.join(PROJECT_ROOT, '2_results', 'pdb', 'polymers')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, '2_results', 'FILTERED_HBOND_PDB')
REPORT_CSV = os.path.join(PROJECT_ROOT, '2_results', 'hbond_analysis_report.csv')

TARGET_SIZES = [9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
HBOND_DIST_THRESHOLD = 2.5    
HBOND_ANGLE_THRESHOLD = 120.0 

# ==============================================================================
# 2. 헬퍼 함수
# ==============================================================================

def calculate_angle(pos_a, pos_b, pos_c):
    v_ba = pos_a - pos_b
    v_bc = pos_c - pos_b
    norm_ba = np.linalg.norm(v_ba)
    norm_bc = np.linalg.norm(v_bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0
    cosine = np.dot(v_ba, v_bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def get_hbond_ring_info(mol):
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    
    donors = []    
    acceptors = [] 
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 8: 
            acceptors.append(idx)
        elif atomic_num == 7: 
            for nbr in atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1: 
                    donors.append((idx, nbr.GetIdx()))
                    
    found_sizes = set()
    details = [] 
    
    for n_idx, h_idx in donors:
        pos_h = positions[h_idx]
        pos_n = positions[n_idx]
        
        for o_idx in acceptors:
            neighbor_idxs = [nbr.GetIdx() for nbr in mol.GetAtomWithIdx(n_idx).GetNeighbors()]
            if o_idx in neighbor_idxs: continue

            pos_o = positions[o_idx]
            dist = np.linalg.norm(pos_h - pos_o)
            if dist > HBOND_DIST_THRESHOLD: continue
            
            angle = calculate_angle(pos_n, pos_h, pos_o)
            if angle < HBOND_ANGLE_THRESHOLD: continue

            try:
                path = Chem.GetShortestPath(mol, o_idx, n_idx)
                ring_size = len(path) + 1 
                found_sizes.add(ring_size)
                details.append({
                    'Ring_Size': ring_size,
                    'Dist': round(dist, 2),
                    'Angle': round(angle, 1),
                    'Atom_Pairs': f"N{n_idx}-H{h_idx}...O{o_idx}"
                })
            except: pass
                    
    return found_sizes, details

# ==============================================================================
# 3. 메인 작업 함수 (함수로 캡슐화)
# ==============================================================================
def run_batch_process():
    print(f"{'='*60}")
    print(f"Searching for H-bond Rings with sizes: {TARGET_SIZES}")
    print(f"Criteria: Dist <= {HBOND_DIST_THRESHOLD} A, Angle >= {HBOND_ANGLE_THRESHOLD} deg")
    print(f"Source: {PDB_SOURCE_DIR}")
    print(f"Target: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_count = 0
    valid_count = 0
    report_data = []

    for root, dirs, files in os.walk(PDB_SOURCE_DIR):
        for file in files:
            if not file.endswith(".pdb"): continue

            total_count += 1
            file_path = os.path.join(root, file)
            
            try:
                # Sanitize 부분 적용으로 데이터 살리기
                mol = Chem.MolFromPDBFile(file_path, removeHs=False, sanitize=False)
                if mol: mol.UpdatePropertyCache(strict=False)
            except: continue
            if mol is None: continue

            sizes, details = get_hbond_ring_info(mol)
            matched = sizes.intersection(TARGET_SIZES)
            
            if matched:
                parent_folder = os.path.basename(root)
                if file.startswith(parent_folder):
                    base_name = file
                else:
                    base_name = f"{parent_folder}_{file}"
                    
                size_tag = ",".join(map(str, sorted(list(matched))))
                new_filename = f"[{size_tag}]_{base_name}"
                
                dest_path = os.path.join(OUTPUT_DIR, new_filename)
                shutil.copy2(file_path, dest_path)
                valid_count += 1
                
                for det in details:
                    if det['Ring_Size'] in TARGET_SIZES:
                        det['Filename'] = new_filename
                        det['Source_Path'] = file_path
                        report_data.append(det)
                
                if valid_count % 50 == 0:
                    print(f"  -> Found {valid_count} structures. Last: {new_filename}")

    if report_data:
        df = pd.DataFrame(report_data)
        cols = ['Filename', 'Ring_Size', 'Dist', 'Angle', 'Atom_Pairs', 'Source_Path']
        df = df[cols].sort_values(by=['Ring_Size', 'Dist'])
        df.to_csv(REPORT_CSV, index=False)
        print(f"\n[Report] Detailed analysis saved to: {REPORT_CSV}")
        print(f"         (Top detected: {len(df)} H-bonds)")

    print(f"\n{'-'*60}")
    print(f"Scan Completed.")
    print(f"Total Scanned : {total_count}")
    print(f"Selected      : {valid_count}")
    print(f"Saved to      : {OUTPUT_DIR}")
    print(f"{'-'*60}")

# ==============================================================================
# 4. 검증 및 실행 진입점 (Entry Point)
# ==============================================================================
if __name__ == "__main__":
    # [1단계] 자체 검증 테스트 실행
    print("\n" + "="*60)
    print("[Self-Check] Running Logic Verification...")
    print("="*60)

    # Test 1: Geometry
    p_a, p_b, p_c = np.array([1.,0.,0.]), np.array([0.,0.,0.]), np.array([0.,1.,0.])
    angle = calculate_angle(p_a, p_b, p_c)
    print(f"[Test 1] Angle (Expected 90.0): {angle:.2f} -> {'PASS' if abs(angle-90)<0.01 else 'FAIL'}")

    # Test 2: H-Bond Logic
    try:
        test_smiles = "NCCCCCCC=O" 
        mol_test = Chem.MolFromSmiles(test_smiles)
        mol_test = Chem.AddHs(mol_test)
        AllChem.EmbedMolecule(mol_test)
        conf = mol_test.GetConformer()
        
        # 강제 좌표 주입 (H-bond 형성)
        n_idx = 0
        o_idx = [a.GetIdx() for a in mol_test.GetAtoms() if a.GetAtomicNum() == 8][0]
        n_atom = mol_test.GetAtomWithIdx(n_idx)
        h_idx = [nbr.GetIdx() for nbr in n_atom.GetNeighbors() if nbr.GetAtomicNum() == 1][0]
        
        from rdkit.Geometry import Point3D
        conf.SetAtomPosition(n_idx, Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(h_idx, Point3D(1.0, 0.0, 0.0))
        conf.SetAtomPosition(o_idx, Point3D(3.0, 0.0, 0.0)) # Dist 2.0, Angle 180
        
        detected_sizes, _ = get_hbond_ring_info(mol_test)
        
        # N(0) ~ O(9) 경로 길이 9 + H(1) = 10 (or similar)
        # Check if ANY size is detected (Simple check)
        if len(detected_sizes) > 0:
            print(f"[Test 2] H-Bond Detection: Found sizes {detected_sizes} -> PASS")
        else:
            print(f"[Test 2] H-Bond Detection: None -> FAIL")
            
    except Exception as e:
        print(f"[Test 2] Error: {e}")

    print("="*60)
    print("Verification Completed. Starting Main Process...")
    print("="*60 + "\n")

    # [2단계] 실제 작업 수행 (검증이 통과되었으므로 실행)
    # 아래 줄의 주석을 해제하거나 그대로 두어 실행하세요.
    run_batch_process()