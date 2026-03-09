
"""
scripts/BATCH_RMSD_TFD_FINAL_V23.py

[최종 해결책: Custom Geometric TFD Implementation]
1. [Problem] RDKit 내장 TFD는 화학적 무결성(Sanitization)을 요구하여, 깨진 PDB에서 999.9 에러 발생.
2. [Solution] 화학 속성을 무시하고, 순수 '기하학적(Geometric)'으로 비틀림 각도를 직접 계산하는 함수 구현.
3. [Method]
   - Reference 분자에서 회전 가능한 결합(Rotatable Bond)을 그래프 탐색으로 정의.
   - Reference와 Target(매핑된 원자)의 이면각(Dihedral Angle)을 각각 측정.
   - 두 각도의 차이를 정규화하여 0.0 ~ 1.0 사이의 값으로 반환.
"""

import os
import glob
import math
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# 1. 설정
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

REF_DIR = os.path.join(PROJECT_ROOT, "2_results", "FILTERED_HBOND_PDB")
REF1_PATH = os.path.join(REF_DIR, "[18]CPDA_CPDC_cap_converted.pdb")

TARGET_DIR = REF_DIR
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "2_results", "rmsd_tfd_results.csv")

# ==============================================================================
# 2. 유틸리티: 전처리 및 그래프 탐색
# ==============================================================================

def remove_hs_manually(mol):
    """수소 원자만 물리적으로 제거"""
    try:
        rw_mol = Chem.RWMol(mol)
        atoms = list(mol.GetAtoms())
        for atom in reversed(atoms):
            if atom.GetAtomicNum() == 1:
                rw_mol.RemoveAtom(atom.GetIdx())
        return rw_mol.GetMol()
    except:
        return mol

def get_longest_path(mol):
    """백본(Longest Path) 추출"""
    try:
        if mol.GetNumAtoms() == 0: return []
        adj = {}
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        if not adj: return []

        def bfs(start):
            visited = {start: 0}
            queue = [start]
            last = start
            while queue:
                curr = queue.pop(0)
                last = curr
                for n in adj.get(curr, []):
                    if n not in visited:
                        visited[n] = visited[curr] + 1
                        queue.append(n)
            return last

        node_a = bfs(0)
        node_b = bfs(node_a)
        
        parent = {node_a: None}
        queue = [node_a]
        while queue:
            curr = queue.pop(0)
            if curr == node_b: break
            for n in adj.get(curr, []):
                if n not in parent:
                    parent[n] = curr
                    queue.append(n)
        
        path = []
        curr = node_b
        while curr is not None:
            path.append(curr)
            curr = parent.get(curr)
        return path 
    except: return []

def rebuild_topology_temp(mol):
    """정렬용 임시 토폴로지 복구"""
    try:
        rw_mol = Chem.RWMol(mol)
        rw_mol.RemoveAllBonds()
        conf = rw_mol.GetConformer()
        n = rw_mol.GetNumAtoms()
        for i in range(n):
            pos_i = conf.GetAtomPosition(i)
            for j in range(i+1, n):
                if (pos_i - conf.GetAtomPosition(j)).Length() < 2.0:
                    rw_mol.AddBond(i, j, Chem.BondType.SINGLE)
        return rw_mol.GetMol()
    except: return mol

# ==============================================================================
# 3. [핵심] 커스텀 TFD 계산 로직 (기하학 기반)
# ==============================================================================

def get_torsion_definitions(mol):
    """
    [기능] 분자 내에서 비틀림 각도를 잴 수 있는 4개의 원자 조합(A-B-C-D)을 찾습니다.
    RDKit의 FindMolRotatableBonds 대신 그래프 구조를 직접 탐색합니다.
    """
    torsions = []
    try:
        # 고리 정보 계산 시도 (고리 내 결합은 보통 회전 불가로 간주)
        try: Chem.GetSymmSSSR(mol)
        except: pass
        ring_info = mol.GetRingInfo()

        for bond in mol.GetBonds():
            # 1. 고리에 포함된 결합은 제외 (단순화를 위해)
            if ring_info.NumRings() > 0 and ring_info.IsBondInRingOfSize(bond.GetIdx(), 0):
                continue
                
            atom_b = bond.GetBeginAtom()
            atom_c = bond.GetEndAtom()
            idx_b = atom_b.GetIdx()
            idx_c = atom_c.GetIdx()

            # 2. 말단 결합 제외 (양쪽 다 이웃이 있어야 각도 측정 가능)
            if atom_b.GetDegree() < 2 or atom_c.GetDegree() < 2:
                continue

            # 3. 이웃 원자 찾기 (A -> B -> C -> D)
            # B의 이웃 중 C가 아닌 놈 하나 (A)
            neighbors_b = [n for n in atom_b.GetNeighbors() if n.GetIdx() != idx_c]
            # C의 이웃 중 B가 아닌 놈 하나 (D)
            neighbors_c = [n for n in atom_c.GetNeighbors() if n.GetIdx() != idx_b]

            if neighbors_b and neighbors_c:
                # 첫 번째 이웃을 선택 (일관성 유지를 위해 정렬 후 선택)
                neighbors_b.sort(key=lambda x: x.GetIdx())
                neighbors_c.sort(key=lambda x: x.GetIdx())
                
                idx_a = neighbors_b[0].GetIdx()
                idx_d = neighbors_c[0].GetIdx()
                
                torsions.append((idx_a, idx_b, idx_c, idx_d))
                
    except Exception as e:
        print(f"Error defining torsions: {e}")
        
    return torsions

def calculate_custom_tfd(ref_mol, target_mol, atom_map, rmsd_val):
    """
    [기능] 기하학적 TFD 직접 계산
    1. Reference에서 정의된 비틀림 각도 리스트를 가져옵니다.
    2. 매핑된 Target 원자들의 비틀림 각도를 잽니다.
    3. 각도 차이의 평균을 0~1 사이로 정규화하여 반환합니다.
    """
    # Self-match (RMSD ~ 0)
    if rmsd_val < 1e-4: return 0.0

    # 매핑 딕셔너리 (Ref -> Tgt)
    ref_to_tgt = {r: t for t, r in atom_map}

    # Reference 기준 비틀림 정의 찾기
    ref_torsions = get_torsion_definitions(ref_mol)
    
    if not ref_torsions:
        return 999.9 # 회전 가능한 결합이 없거나 정의 실패

    diff_sum = 0.0
    valid_count = 0

    ref_conf = ref_mol.GetConformer()
    tgt_conf = target_mol.GetConformer()

    for idx_a, idx_b, idx_c, idx_d in ref_torsions:
        # 4개의 원자가 모두 매핑되었는지 확인
        if (idx_a in ref_to_tgt and idx_b in ref_to_tgt and 
            idx_c in ref_to_tgt and idx_d in ref_to_tgt):
            
            # 1. Reference 각도 (Radian)
            angle_ref = rdMolTransforms.GetDihedralRad(ref_conf, idx_a, idx_b, idx_c, idx_d)
            
            # 2. Target 각도 (Radian) - 매핑된 인덱스 사용
            t_a, t_b, t_c, t_d = ref_to_tgt[idx_a], ref_to_tgt[idx_b], ref_to_tgt[idx_c], ref_to_tgt[idx_d]
            angle_tgt = rdMolTransforms.GetDihedralRad(tgt_conf, t_a, t_b, t_c, t_d)
            
            # 3. 각도 차이 계산 (주기성 고려: -pi ~ pi)
            diff = abs(angle_ref - angle_tgt)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            
            # 정규화: 0(일치) ~ 1(180도 차이)
            # diff는 0 ~ pi 사이의 값임 -> pi로 나누면 0 ~ 1
            diff_sum += (diff / math.pi)
            valid_count += 1

    if valid_count == 0:
        return 999.9 # 매핑된 비틀림 결합이 하나도 없음

    # 평균 TFD 반환
    return diff_sum / valid_count

# ==============================================================================
# 4. 메인 실행 (정렬 및 계산)
# ==============================================================================

def match_and_align(ref_mol, target_mol):
    # (이전과 동일한 로직)
    ref_path = get_longest_path(ref_mol)
    ref_seq = [ref_mol.GetAtomWithIdx(i).GetSymbol() for i in ref_path]
    
    tgt_temp = rebuild_topology_temp(target_mol)
    tgt_path = get_longest_path(tgt_temp)
    tgt_seq = [target_mol.GetAtomWithIdx(i).GetSymbol() for i in tgt_path]
    
    backbone_map = []
    if ref_seq == tgt_seq:
        backbone_map = list(zip(tgt_path, ref_path))
    elif ref_seq == tgt_seq[::-1]:
        backbone_map = list(zip(tgt_path[::-1], ref_path))
    else:
        min_len = min(len(ref_path), len(tgt_path))
        if min_len >= 5:
            backbone_map = list(zip(tgt_path[:min_len], ref_path[:min_len]))
            
    if not backbone_map: return [], target_mol

    AllChem.AlignMol(target_mol, ref_mol, atomMap=backbone_map)
    
    full_map = list(backbone_map)
    mapped_ref = set([r for t, r in full_map])
    mapped_tgt = set([t for t, r in full_map])
    
    ref_conf = ref_mol.GetConformer()
    tgt_conf = target_mol.GetConformer()
    ref_syms = [a.GetSymbol() for a in ref_mol.GetAtoms()]
    tgt_syms = [a.GetSymbol() for a in target_mol.GetAtoms()]

    for r_idx in range(ref_mol.GetNumAtoms()):
        if r_idx in mapped_ref: continue
        r_pos = ref_conf.GetAtomPosition(r_idx)
        r_sym = ref_syms[r_idx]
        best_t = -1
        min_d = 2.0 
        for t_idx in range(target_mol.GetNumAtoms()):
            if t_idx in mapped_tgt: continue
            if tgt_syms[t_idx] != r_sym: continue
            if (tgt_conf.GetAtomPosition(t_idx) - r_pos).Length() < min_d:
                min_d = (tgt_conf.GetAtomPosition(t_idx) - r_pos).Length()
                best_t = t_idx
        if best_t != -1:
            full_map.append((best_t, r_idx))
            mapped_tgt.add(best_t)
            
    return full_map, target_mol

def main():
    print(f"{'='*60}")
    print(f"[Running: RMSD & Custom Geometric TFD]")
    
    # 1. Reference 로드
    ref1_mol = None
    if os.path.exists(REF1_PATH):
        try:
            m = Chem.MolFromPDBFile(REF1_PATH, removeHs=False, sanitize=False)
            if m:
                m = remove_hs_manually(m)
                try: m.UpdatePropertyCache(strict=False)
                except: pass
                # Reference에 한해서는 토폴로지 복구를 한 번 시도 (회전 결합 탐색용)
                ref1_mol = rebuild_topology_temp(m)
        except Exception as e:
            print(f" -> Ref Load Exception: {e}")
    
    if not ref1_mol: 
        print(f" -> [Critical Error] Ref not loaded: {REF1_PATH}")
        return
    
    # Reference의 회전 가능한 결합(Torsion) 미리 정의
    torsions = get_torsion_definitions(ref1_mol)
    print(f" Reference: {os.path.basename(REF1_PATH)}")
    print(f" -> Atoms: {ref1_mol.GetNumAtoms()}")
    print(f" -> Identified Torsions (Rotatable Bonds): {len(torsions)}")
    
    target_files = glob.glob(os.path.join(TARGET_DIR, "*.pdb"))
    results = []

    for idx, t_path in enumerate(target_files):
        t_name = os.path.basename(t_path)
        print(f"[{idx+1}/{len(target_files)}] {t_name}...", end='', flush=True)

        target = Chem.MolFromPDBFile(t_path, removeHs=False, sanitize=False)
        if not target: 
            print(" -> Load Fail")
            continue
        target = remove_hs_manually(target)
        try: target.UpdatePropertyCache(strict=False)
        except: pass

        row = {'Filename': t_name}
        
        try:
            full_map, target_aligned = match_and_align(ref1_mol, target)
            match_count = len(full_map)
            
            rmsd = 999.9
            tfd = 999.9
            
            if ref1_mol.GetNumAtoms() > 0 and match_count >= ref1_mol.GetNumAtoms() * 0.8:
                try: rmsd = AllChem.AlignMol(target_aligned, ref1_mol, atomMap=full_map)
                except: pass
                
                # [수정된 부분] 커스텀 TFD 계산 호출
                tfd = calculate_custom_tfd(ref1_mol, target_aligned, full_map, rmsd)
            
            row['RMSD'] = rmsd
            row['TFD'] = tfd
            row['Matched_Atoms'] = match_count
            
            status = "Fail (Low Match)"
            if tfd < 100:
                status = f"RMSD: {rmsd:.3f} / TFD: {tfd:.3f}"
            
            print(f" -> {status}")
            
        except Exception as e:
            print(f" -> Error: {e}")
            row['RMSD'] = 999.9
            row['TFD'] = 999.9

        results.append(row)

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by='RMSD')
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved Results to: {OUTPUT_CSV}")
        print(df[['Filename', 'Matched_Atoms', 'RMSD', 'TFD']].head().to_string(index=False))

if __name__ == "__main__":
    main()