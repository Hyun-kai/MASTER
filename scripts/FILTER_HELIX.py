"""
scripts/FILTER_HELIX.py

[설명]
생성된 폴리머 PDB 파일들의 3D 기하학적 특성을 분석하여 
나선형(Helical) 구조 후보군을 선별(Filtering)하는 스크립트입니다.

[분석 지표]
1. Ree (End-to-End Distance): 폴리머 시작점과 끝점의 거리. (나선은 중간값을 가짐)
2. Rg (Radius of Gyration): 분자의 질량 중심으로부터의 퍼짐 정도.
3. Anisotropy (Relative Shape Anisotropy, k^2): 
   - 0에 가까울수록 완벽한 구형(뭉침)
   - 1에 가까울수록 완벽한 선형(막대기)
   - 나선형 구조는 고유의 중간 범위(보통 0.1 ~ 0.5 사이)를 가집니다.
"""

import os
import glob
import numpy as np
from ase.io import read
import pandas as pd

# 색상 코드
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
NC = '\033[0m'

def calculate_geometric_properties(atoms):
    """ASE Atoms 객체로부터 주요 기하학적 물성치를 계산합니다."""
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    
    # 1. 질량 중심 (Center of Mass)
    total_mass = np.sum(masses)
    com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass
    
    # 2. Radius of Gyration (Rg) 계산
    shifted_pos = positions - com
    rg_sq = np.sum(masses * np.sum(shifted_pos**2, axis=1)) / total_mass
    rg = np.sqrt(rg_sq)
    
    # 3. 관성 텐서(Gyration Tensor) 및 Shape Anisotropy 계산
    # 3x3 텐서 구성
    tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            tensor[i, j] = np.sum(masses * shifted_pos[:, i] * shifted_pos[:, j]) / total_mass
            
    # 고유값(Eigenvalues) 추출 및 정렬 (L1 <= L2 <= L3)
    eigenvalues, _ = np.linalg.eigh(tensor)
    eigenvalues = np.sort(eigenvalues)
    
    L1, L2, L3 = eigenvalues
    
    # Relative Shape Anisotropy (k^2)
    # k^2 = 1 - 3*(L1*L2 + L2*L3 + L3*L1) / (L1 + L2 + L3)^2
    # 0: 구형 (뭉침), 1: 선형 (막대)
    I1 = L1 + L2 + L3
    I2 = L1*L2 + L2*L3 + L3*L1
    if I1 == 0:
        anisotropy = 0
    else:
        anisotropy = 1.0 - 3.0 * (I2 / (I1**2))
        
    return rg, anisotropy

def analyze_polymers(pdb_dir):
    print(f"{BLUE}>>> [Analysis] Scanning PDB files in {pdb_dir}...{NC}")
    pdb_files = glob.glob(os.path.join(pdb_dir, "*.pdb"))
    
    if not pdb_files:
        print(f"{YELLOW}[Warn] No PDB files found.{NC}")
        return

    results = []
    
    for fpath in pdb_files:
        filename = os.path.basename(fpath)
        try:
            atoms = read(fpath)
            positions = atoms.get_positions()
            
            # End-to-End Distance (간단히 첫 원자와 마지막 원자 사이의 거리)
            # 정확도를 높이려면 양 끝단 Residue의 중심점(Center of Geometry) 거리를 구하는 것이 좋습니다.
            n_atoms = len(positions)
            chunk_size = n_atoms // 4 # 4-mer라고 가정
            
            head_com = np.mean(positions[:chunk_size], axis=0)
            tail_com = np.mean(positions[-chunk_size:], axis=0)
            ree = np.linalg.norm(head_com - tail_com)
            
            # 에너지 로드 (info 딕셔너리에 저장되어 있다고 가정)
            energy = atoms.info.get('Energy', 0.0)
            
            rg, anisotropy = calculate_geometric_properties(atoms)
            
            results.append({
                'Filename': filename,
                'Energy': energy,
                'Ree (Å)': round(ree, 2),
                'Rg (Å)': round(rg, 2),
                'Anisotropy': round(anisotropy, 3)
            })
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # DataFrame 생성 및 결과 정렬
    df = pd.DataFrame(results)
    
    # [정렬 기준] 에너지가 낮으면서(안정적), Anisotropy가 중간값(나선형)인 순서
    # 완전히 선형인 구조(Anisotropy > 0.8)와 완전히 뭉친 구조(Anisotropy < 0.1)를 배제하기 위해
    # Anisotropy 값이 0.2 ~ 0.6 사이인 것들을 필터링합니다. (이 수치는 폴리머 특성에 따라 조절 필요)
    
    print(f"\n{GREEN}--- [Top Candidates for Helical Structure] ---{NC}")
    print("Filter Condition: 0.15 < Anisotropy < 0.70")
    
    helical_candidates = df[(df['Anisotropy'] > 0.15) & (df['Anisotropy'] < 0.70)]
    helical_candidates = helical_candidates.sort_values(by='Energy')
    
    if helical_candidates.empty:
        print("No structures matched the strictly helical geometric criteria. Showing all sorted by Anisotropy:")
        print(df.sort_values(by='Anisotropy', ascending=False).to_string(index=False))
    else:
        print(helical_candidates.to_string(index=False))
        
    # 결과를 CSV로 저장하여 나중에 엑셀 등에서 열어볼 수 있게 합니다.
    csv_path = os.path.join(pdb_dir, "geometric_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n{BLUE}[Info] Full analysis saved to {csv_path}{NC}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default="1_data/polymers", help="Directory containing PDB files")
    args = parser.parse_args()
    
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.pdb_dir)
    analyze_polymers(target_dir)