"""
scripts/check_angles.py

[설명]
지정된 경로 및 하위 폴더(예: '0 0' ~ '9 9')에 있는 모든 PDB 파일을 탐색하여 구조 품질을 진단합니다.
1. Amide Bond Angle (Omega): 180도(Trans) 고정 여부 확인
2. Min H-Bond Distance: 구조가 접혔는지(Folding) 확인
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# ==============================================================================
# 1. 경로 설정 (사용자 지정 경로)
# ==============================================================================
# 사용자가 지정한 절대 경로
PDB_SOURCE_DIR = '/home/khj2/CPDA_CPDC_180/2_results/pdb/polymers/CPDA_0-CPDC_0_9mer'

def check_structure_quality(pdb_dir):
    print(f"{'='*80}")
    print(f"[Diagnosis] Checking structures in: {pdb_dir}")
    print(f"Scanning subdirectories (0 0 ~ 9 9 etc)...")
    print(f"{'='*80}")
    
    if not os.path.exists(pdb_dir):
        print(f"[Error] Directory not found: {pdb_dir}")
        return

    results = []
    # Amide 결합 패턴: C(=O)N[H]
    amide_pattern = Chem.MolFromSmarts("C(=O)N[H]")
    
    # --------------------------------------------------------------------------
    # [수정] os.walk를 사용하여 하위 폴더의 모든 PDB 파일 수집
    # --------------------------------------------------------------------------
    pdb_files = []
    for root, dirs, files in os.walk(pdb_dir):
        for f in files:
            if f.endswith('.pdb'):
                # 전체 경로 저장
                full_path = os.path.join(root, f)
                pdb_files.append(full_path)

    total_files = len(pdb_files)
    
    if total_files == 0:
        print("[Warning] No PDB files found in the directory tree.")
        return

    print(f" -> Found {total_files} PDB files. Starting analysis...\n")

    for idx, path in enumerate(pdb_files):
        filename = os.path.basename(path)
        subdir = os.path.basename(os.path.dirname(path)) # 상위 폴더명 (예: "0 0")
        
        try:
            # PDB 로드 (수소 포함, Sanitize 유연하게)
            mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
            if mol: 
                try: mol.UpdatePropertyCache(strict=False)
                except: pass
            
            if mol is None: 
                continue
            
            # --------------------------------------------------------------
            # 1. Amide Angle (Omega) 측정
            # --------------------------------------------------------------
            matches = mol.GetSubstructMatches(amide_pattern)
            omegas = []
            conf = mol.GetConformer()
            
            for m in matches:
                # m indices: 0:C, 1:O, 2:N, 3:H -> Dihedral: O-C-N-H
                # Trans = 180 (or -180)
                ang = rdMolTransforms.GetDihedralDeg(conf, m[1], m[0], m[2], m[3])
                omegas.append(abs(ang)) 
            
            avg_omega = np.mean(omegas) if omegas else 0.0
            
            # --------------------------------------------------------------
            # 2. Min H-Bond Distance 측정 (O...H)
            # --------------------------------------------------------------
            min_dist = 999.0
            atoms = mol.GetAtoms()
            oxygens = [a.GetIdx() for a in atoms if a.GetAtomicNum() == 8]
            
            # Amide 질소에 붙은 수소만 추출
            hydrogens = []
            for a in atoms:
                if a.GetAtomicNum() == 1:
                    parent = a.GetParent()
                    if parent and parent.GetAtomicNum() == 7: # N에 붙은 H
                        hydrogens.append(a.GetIdx())

            positions = conf.GetPositions()
            
            if oxygens and hydrogens:
                o_pos = positions[oxygens]
                h_pos = positions[hydrogens]
                
                # 거리 계산 (Broadcasting)
                diff = o_pos[:, np.newaxis, :] - h_pos[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                
                # 1.2A 이하는 제외 (공유결합 등)
                valid_dists = dists[dists > 1.2] 
                
                if valid_dists.size > 0:
                    min_dist = np.min(valid_dists)

            results.append({
                'folder': subdir,       # "0 0" 등 폴더명 식별
                'file': filename,
                'avg_omega': avg_omega,
                'min_h_dist': min_dist
            })
            
        except Exception as e:
            continue
            
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{total_files} files...")

    # --------------------------------------------------------------------------
    # 결과 리포트 출력
    # --------------------------------------------------------------------------
    if not results:
        print("\n[Error] No valid structures analyzed.")
        return

    df = pd.DataFrame(results)
    
    # 전체 통계
    avg_omega_all = df['avg_omega'].mean()
    avg_hdist_all = df['min_h_dist'].mean()
    
    # 필터 통과 기준 (Trans: >170도, H-bond: <=2.5A)
    count_good_omega = len(df[df['avg_omega'] > 170.0])
    count_good_hbond = len(df[df['min_h_dist'] <= 2.5])
    
    print("\n" + "="*80)
    print(" [ANALYSIS REPORT] ")
    print("="*80)
    print(f" Target Directory : {PDB_SOURCE_DIR}")
    print(f" Total Analyzed   : {len(df)} structures")
    print("-" * 80)
    
    print(f"\n 1. Amide Bond Angle (Omega) [Ideal: ~180.0]")
    print(f"    - Average        : {avg_omega_all:.2f} deg")
    print(f"    - Valid (>170)   : {count_good_omega} / {len(df)} ({count_good_omega/len(df)*100:.1f}%)")
    
    print(f"\n 2. Shortest H-Bond Distance [Ideal: <= 2.5 A]")
    print(f"    - Average        : {avg_hdist_all:.2f} A")
    print(f"    - Valid (<=2.5)  : {count_good_hbond} / {len(df)} ({count_good_hbond/len(df)*100:.1f}%)")
    
    print("\n" + "-" * 80)
    
    # --------------------------------------------------------------------------
    # 폴더별(0 0 ~ 9 9) 통계 요약 (선택 사항)
    # --------------------------------------------------------------------------
    print(" [Folder-wise Summary (Top 5 Best H-Bond)]")
    # H-bond 거리가 가장 짧은(좋은) 순서로 폴더별 평균 정렬
    folder_stats = df.groupby('folder')['min_h_dist'].mean().sort_values()
    print(folder_stats.head(5))
    print("-" * 80)

    # 결론 및 제안
    if count_good_omega > 0.9 * len(df) and count_good_hbond < 0.1 * len(df):
        print(" [DIAGNOSIS] Amide bonds are FIXED (Trans), but structure is EXTENDED (Not folded).")
        print(" [ACTION]    Change input 'dihedral angles' (phi/psi) to helical values (e.g., -60, -45).")
    elif count_good_omega < 0.5 * len(df):
        print(" [DIAGNOSIS] Amide bonds are NOT fixed (Constraints failed).")
        print(" [ACTION]    Check ASE constraints and pre-conditioning logic.")
    else:
        print(" [DIAGNOSIS] Mixed results or partially successful.")
    print("="*80)

if __name__ == "__main__":
    check_structure_quality(PDB_SOURCE_DIR)