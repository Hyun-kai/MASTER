"""
scripts/extract_dihedrals.py

[기능]
주어진 올리고머 PDB 파일에서 백본(Backbone) 경로를 탐색하고 이면각을 추출합니다.
PDB의 결합 정보 누락 문제를 해결하기 위해, 구조 로드 후 결합 차수 추론을 수행하여
캡(Cap) 인식을 돕습니다.

[개선 사항]
- [Visualization] 로드된 분자의 구조와 각 원자의 인덱스를 표시한 이미지를 저장하는 기능 추가.
- [Bond Order Inference] PDB 로드 후 기하 구조 기반으로 결합 차수를 추론하여 SMARTS 매칭률 향상.
- [Topology Fallback] 캡 인식이 여전히 실패할 경우, 그래프 기반 최장 경로 탐색으로 전환.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
# 시각화를 위한 모듈 임포트
from rdkit.Chem import Draw

# ==============================================================================
# 1. 환경 설정 및 경로 추가
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from bakers.chem import topology, capping
    from bakers.analytics import metrics
except ImportError as e:
    print(f"[Error] BAKERS 모듈을 불러오는데 실패했습니다: {e}")
    sys.exit(1)

# ==============================================================================
# 2. Helper Logic: 시각화 및 PDB 결합 차수 복구
# ==============================================================================

def visualize_mol_with_indices(mol, pdb_path):
    """
    분자의 구조와 각 원자의 인덱스를 이미지로 저장합니다.
    저장 경로는 입력 PDB 파일명 기반으로 생성됩니다 (예: op.pdb -> op_annotated.png).
    """
    try:
        # 원본 보존을 위해 복사
        mol_copy = Chem.Mol(mol)
        
        # 2D 좌표 생성을 위해 계산 (3D PDB를 보기 좋게 폄)
        AllChem.Compute2DCoords(mol_copy)
        
        # 각 원자에 'atomNote' 속성으로 인덱스 설정
        for atom in mol_copy.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
            
        # 이미지 생성 설정
        opts = Draw.MolDrawOptions()
        opts.addAtomIndices = False  # atomNote를 사용하므로 내장 인덱스 표시는 끔
        opts.bondLineWidth = 2
        
        # 이미지 저장 경로 생성
        base_name = os.path.splitext(os.path.basename(pdb_path))[0]
        dir_name = os.path.dirname(pdb_path)
        img_path = os.path.join(dir_name, f"{base_name}_annotated.png")
        
        # 이미지 그리기 및 저장
        img = Draw.MolToImage(mol_copy, size=(1000, 1000), options=opts)
        img.save(img_path)
        print(f"  [Info] 구조 이미지 저장 완료: {img_path}")
        
    except Exception as e:
        print(f"  [Warning] 시각화 실패: {e}")


def load_pdb_with_connect(pdb_path):
    """
    PDB를 로드하고 결합 정보(Bond Orders)를 최대한 복구합니다.
    """
    # 1. 일단 기본 로드 (수소 포함, Sanitize 안함)
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False, sanitize=False)
    if mol is None: return None

    try:
        # 2. 결합 차수 추론 (기하학적 구조 기반)
        mol.UpdatePropertyCache(strict=False)
        
        # 수동 보정: 카보닐(C=O) 패턴 찾기
        rw_mol = Chem.RWMol(mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.GetDegree() == 3: # SP2 Carbon Candidate
                for bond in atom.GetBonds():
                    other = bond.GetOtherAtom(atom)
                    # 산소이고, 산소의 이웃이 1개뿐이면(말단 산소) -> C=O
                    if other.GetAtomicNum() == 8 and other.GetDegree() == 1:
                        bond.SetBondType(Chem.BondType.DOUBLE)
        
        mol = rw_mol.GetMol()
        Chem.SanitizeMol(mol) # 이제 정식 분자로 변환
        
    except Exception as e:
        print(f"[Warn] 결합 차수 복구 중 경고: {e}")
        
    return mol

# ==============================================================================
# 3. Fallback Logic: 그래프 기반 백본 탐색
# ==============================================================================

def find_longest_path_backbone(mol):
    """[Fallback] 캡 인식 없이 분자 내에서 가장 긴 경로(Longest Path) 탐색"""
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
    
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        # 수소 제외 (Heavy Atom Backbone 탐색)
        if mol.GetAtomWithIdx(u).GetAtomicNum() > 1 and mol.GetAtomWithIdx(v).GetAtomicNum() > 1:
            G.add_edge(u, v)
    
    terminals = [n for n, d in G.degree() if d == 1]
    if len(terminals) < 2: terminals = list(G.nodes())

    longest_path = []
    max_len = -1

    for i in range(len(terminals)):
        for j in range(i + 1, len(terminals)):
            try:
                path = nx.shortest_path(G, source=terminals[i], target=terminals[j])
                if len(path) > max_len:
                    max_len = len(path)
                    longest_path = path
            except nx.NetworkXNoPath:
                continue
    return longest_path

def get_dofs_from_path(mol, path):
    """경로(Path) 상의 회전 가능한 결합(DOF) 추출"""
    dofs = []
    for i in range(len(path) - 1):
        u_idx = path[i]
        v_idx = path[i+1]
        
        bond = mol.GetBondBetweenAtoms(u_idx, v_idx)
        if bond is None: continue

        if bond.GetBondType() != Chem.BondType.SINGLE: continue
        if bond.IsInRing(): continue
        
        u_atom = mol.GetAtomWithIdx(u_idx)
        v_atom = mol.GetAtomWithIdx(v_idx)
        
        # 아미드 결합 체크 (C-N)
        is_amide = False
        if (u_atom.GetAtomicNum() == 6 and v_atom.GetAtomicNum() == 7) or \
           (u_atom.GetAtomicNum() == 7 and v_atom.GetAtomicNum() == 6):
            c_atom = u_atom if u_atom.GetAtomicNum() == 6 else v_atom
            # 이중결합 확인 (BondType=DOUBLE)
            for nbr in c_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 8:
                    b_type = mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondType()
                    if b_type == Chem.BondType.DOUBLE:
                        is_amide = True; break
        if is_amide: continue

        # Anchor 설정
        if i > 0: a_idx = path[i-1]
        else:
            neighbors = [n.GetIdx() for n in u_atom.GetNeighbors() if n.GetIdx() != v_idx]
            a_idx = neighbors[0] if neighbors else None

        if i < len(path) - 2: d_idx = path[i+2]
        else:
            neighbors = [n.GetIdx() for n in v_atom.GetNeighbors() if n.GetIdx() != u_idx]
            d_idx = neighbors[0] if neighbors else None
            
        if a_idx is not None and d_idx is not None:
            dofs.append((a_idx, u_idx, v_idx, d_idx))
            
    return dofs

# ==============================================================================
# 4. 메인 실행 함수
# ==============================================================================

def run(args):
    pdb_path = args.pdb
    output_path = args.out

    if not os.path.exists(pdb_path):
        print(f"[Error] 파일 없음: {pdb_path}")
        return

    print(f">>> Processing: {os.path.basename(pdb_path)}")
    
    # [Step 0] PDB 로드 및 결합 정보 복구 시도
    mol = load_pdb_with_connect(pdb_path)
    if mol is None:
        print("[Error] RDKit 로드 실패")
        return

    # [Visualization] 로드된 구조 시각화 (인덱스 포함)
    visualize_mol_with_indices(mol, pdb_path)

    dof_indices_list = []
    
    # [Step 1] 표준 위상 분석 (Capping 기반)
    try:
        print("  [Step 1] 캡(Cap) 기반 위상 분석 시도...")
        caps = capping.detect_caps(mol)
        if not caps:
            print("    -> 캡 인식 실패 (SMARTS 매칭 안됨). Fallback으로 이동.")
            raise ValueError("No caps found")
            
        topo_info = topology.analyze_residue_topology(mol)
        dof_indices_list = topology.get_dofs(mol, exclude_indices=set())
        print(f"    -> 성공! {len(dof_indices_list)}개 이면각 발견.")
        
    except Exception:
        dof_indices_list = [] # 초기화

    # [Step 2] Fallback (최장 경로 탐색)
    if not dof_indices_list:
        print("  [Step 2] 그래프 기반 최장 경로 탐색 시도...")
        try:
            longest_path = find_longest_path_backbone(mol)
            if not longest_path:
                print("    [Error] 경로 탐색 실패.")
                return
            
            dof_indices_list = get_dofs_from_path(mol, longest_path)
            print(f"    -> 성공! {len(dof_indices_list)}개 이면각 발견.")
            
        except Exception as e:
            print(f"    [Error] Fallback 실패: {e}")
            return

    # [Step 3] 계산 및 저장
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    results = []
    
    for i, atom_indices in enumerate(dof_indices_list):
        p1 = positions[atom_indices[0]]
        p2 = positions[atom_indices[1]]
        p3 = positions[atom_indices[2]]
        p4 = positions[atom_indices[3]]

        angle = metrics.calculate_dihedral(p1, p2, p3, p4)
        
        results.append({
            'Index': i + 1,
            'Atom_Indices': str(atom_indices),
            'Angle_Deg': angle
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"[Success] 이면각 데이터 저장 완료: {output_path}")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, help="입력 PDB 파일 경로")
    parser.add_argument("--out", default='./2_results/dihedrals.csv', help="출력 CSV 파일 경로")
    args = parser.parse_args()
    run(args)


'''
python ./scripts/extract_dihedrals.py --pdb ./2_results/pdb/CPDA_DMMA_1_converted.pdb
'''