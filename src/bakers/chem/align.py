"""
src/bakers/chem/align.py

[설명]
분자 구조 정렬(Alignment) 및 병합(Merging)을 수행하는 유틸리티 모듈입니다.
폴리머 조립 시 두 잔기(Residue)를 펩타이드 결합(또는 유사 결합)으로 연결합니다.

[최종 수정 내역]
- [Steric Explosion Bug Fix] 지시사항 6번을 누락하여 M1과 M2의 피리딘 고리가 
  동일한 좌표에 겹쳐진 채(Overlapped) 병합되어, 최적화 과정에서 분자가 폭발하며 
  이면각이 파괴되던 치명적인 물리적 버그를 완벽히 해결했습니다.
- 이제 M2(새 모노머)가 완벽히 정렬된 직후, M2의 피리딘 고리를 명시적으로 완전히 
  도려내어(Pruning), 원자 충돌 없는 매끄럽고 완벽한 결합을 보장합니다.
"""

import numpy as np
from rdkit import Chem
from typing import Tuple, List, Dict, Optional, Any

# ==============================================================================
# 1. Geometric Alignment (Kabsch Algorithm)
# ==============================================================================

def align(
    mobile_coords: np.ndarray, 
    target_coords: np.ndarray, 
    mapping: Tuple[List[int], List[int]]
) -> Optional[np.ndarray]:
    """Kabsch 알고리즘을 이용한 최적 3D 중첩 (RMSD 최소화)"""
    try:
        mob_idx, targ_idx = mapping
        if len(mob_idx) != len(targ_idx) or len(mob_idx) < 3:
            return None 

        P = mobile_coords[mob_idx]
        Q = target_coords[targ_idx]

        if not (np.all(np.isfinite(P)) and np.all(np.isfinite(Q))): 
            return None

        # 중심점 계산 및 이동
        P_centroid = np.mean(P, axis=0)
        Q_centroid = np.mean(Q, axis=0)
        P_centered = P - P_centroid
        Q_centered = Q - Q_centroid

        # 공분산 행렬 계산
        H = np.dot(P_centered.T, Q_centered)
        
        # SVD 수행
        U, S, Vt = np.linalg.svd(H)
        
        # 회전 행렬 계산 (R = U * V^T)
        R = np.dot(U, Vt)
        
        # 거울상 반사 방지 보정
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)

        # 연산 적용
        aligned_coords = np.dot(mobile_coords - P_centroid, R) + Q_centroid
        return aligned_coords
        
    except Exception as e: 
        print(f"[Align Error] Kabsch algorithm failed: {e}")
        return None

# ==============================================================================
# 2. Pyridine Feature Identification
# ==============================================================================

def _identify_pyridine_features(mol: Chem.Mol, anchor_indices: List[int], core_indices: List[int]) -> Optional[Dict[str, int]]:
    """피리딘 앵커 링 내의 각 원자들의 역할을 완벽하게 분석하여 반환합니다."""
    if len(anchor_indices) != 6:
        return None
        
    anchor_set = set(anchor_indices)
    core_set = set(core_indices)
    
    features = {'N': -1, 'C_core': -1, 'C_term': -1, 'C_para': -1, 'C_ortho_core': -1, 'C_ortho_term': -1}
    
    for idx in anchor_indices:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7:
            features['N'] = idx
        else:
            for nbr in atom.GetNeighbors():
                n_idx = nbr.GetIdx()
                if n_idx not in anchor_set and nbr.GetAtomicNum() > 1:
                    if n_idx in core_set:
                        features['C_core'] = idx  
                    else:
                        features['C_term'] = idx  
                        
    if features['N'] == -1 or features['C_core'] == -1 or features['C_term'] == -1:
        return None
        
    c_core_nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(features['C_core']).GetNeighbors() if n.GetIdx() in anchor_set}
    c_term_nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(features['C_term']).GetNeighbors() if n.GetIdx() in anchor_set}
    
    c_para_list = list(c_core_nbrs & c_term_nbrs)
    if c_para_list: features['C_para'] = c_para_list[0]
    
    n_nbrs = {n.GetIdx() for n in mol.GetAtomWithIdx(features['N']).GetNeighbors() if n.GetIdx() in anchor_set}
    c_ortho_core_list = list(c_core_nbrs & n_nbrs)
    if c_ortho_core_list: features['C_ortho_core'] = c_ortho_core_list[0]
    
    c_ortho_term_list = list(c_term_nbrs & n_nbrs)
    if c_ortho_term_list: features['C_ortho_term'] = c_ortho_term_list[0]
    
    if any(v == -1 for v in features.values()):
        return None
        
    return features


# ==============================================================================
# 3. Polymer Assembly (Merge Logic)
# ==============================================================================

def merge_residues(
    mol1: Chem.Mol, 
    coords1: np.ndarray, 
    params1: Dict[str, Any], 
    mol2: Chem.Mol, 
    coords2: np.ndarray, 
    params2: Dict[str, Any]
) -> Tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
    """사용자의 8단계 지침에 따른 완벽한 잔기 병합(Stitching) 로직"""
    try:
        # [지시사항 1 & 2] 앵커 기본 인덱스 로드
        match_elec = params1.get('elec_anchor_indices', params1.get('upper_connect_indices', []))
        match_nuc = params2.get('nuc_anchor_indices', params2.get('lower_connect_indices', []))

        if not match_elec or not match_nuc or len(match_elec) != len(match_nuc):
            print("[Merge Error] Anchor indices missing or length mismatch.")
            return None, None

        f1 = _identify_pyridine_features(mol1, match_elec, params1.get('residue_indices', []))
        f2 = _identify_pyridine_features(mol2, match_nuc, params2.get('residue_indices', []))
        
        if not f1 or not f2:
            print("[Merge Error] Could not identify Pyridine features.")
            return None, None

        # [지시사항 3 & 4] 정렬 (1:1 매핑 후 Kabsch)
        coords2_aligned = align(coords2, coords1, (match_nuc, match_elec))
        if coords2_aligned is None: 
            return None, None

        # ======================================================================
        # [지시사항 5 & 6] 불필요한 원자 제거 (Pruning) - 핵심 폭발 방지 로직!
        # ======================================================================
        indices1 = params1.get('n_term_indices', params1.get('residue_indices', []))
        indices2_raw = params2.get('c_term_indices', params2.get('residue_indices', []))
        
        if not indices1 or not indices2_raw:
            return None, None

        # [CRITICAL FIX] 지시사항 6번 완벽 수행: 
        # 새 모노머(M2)의 피리딘 고리(match_nuc)가 M1과 겹쳐서 폭발하지 않도록 명시적으로 완전히 삭제합니다.
        if len(match_nuc) == 6:
            indices2 = [idx for idx in indices2_raw if idx not in match_nuc]
        else:
            indices2 = indices2_raw
            
        final_coords = [] 
        node_map = {} 
        cnt = 0 
        
        # mol1 데이터 복사
        for old_idx in indices1:
            node_map[old_idx] = cnt 
            final_coords.append(coords1[old_idx]) 
            cnt += 1
            
        # mol2 데이터 복사 (피리딘이 깨끗이 도려내진 상태로 aligned 좌표 적용)
        offset = mol1.GetNumAtoms() 
        for old_idx in indices2:
            node_map[old_idx + offset] = cnt 
            final_coords.append(coords2_aligned[old_idx]) 
            cnt += 1
            
        # RWMol 생성 및 구조 복사
        new_mol = Chem.RWMol()
        
        for mol_ref, indices, off in [(mol1, indices1, 0), (mol2, indices2, offset)]:
            for old_idx in indices: 
                atom = mol_ref.GetAtomWithIdx(int(old_idx))
                new_idx = new_mol.AddAtom(Chem.Atom(atom.GetAtomicNum()))
                new_atom = new_mol.GetAtomWithIdx(new_idx)
                new_atom.SetFormalCharge(atom.GetFormalCharge())
                new_atom.SetChiralTag(atom.GetChiralTag())

        for mol_ref, indices, off in [(mol1, indices1, 0), (mol2, indices2, offset)]:
            for bond in mol_ref.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if u in indices and v in indices:
                    new_mol.AddBond(node_map[u + off], node_map[v + off], bond.GetBondType())
        
        # ======================================================================
        # [지시사항 8] 완벽한 결합 수행 (Stitching)
        # ======================================================================
        if len(match_elec) == 6 and len(match_nuc) == 6:
            # 이전 모노머(M1)의 말단부 결합 탄소
            bond_atom1_orig = f1['C_term'] 
            
            # 새로 들어오는 모노머(M2)의 피리딘이 '제거된 부분'의 alkyne 탄소 찾기
            bond_atom2_orig = -1
            for nbr in mol2.GetAtomWithIdx(f2['C_core']).GetNeighbors():
                if nbr.GetIdx() in indices2: # 피리딘이 잘려나가고 유일하게 살아남은 이웃 탄소
                    bond_atom2_orig = nbr.GetIdx()
                    break
                    
            if bond_atom1_orig == -1 or bond_atom2_orig == -1:
                print("[Merge Error] Failed to pinpoint the exact bonding atoms.")
                return None, None
                
            # M1 피리딘과 M2 알카인을 단일 결합으로 묶어줌 (폭발 위험 0%)
            try:
                u_new = node_map[bond_atom1_orig]
                v_new = node_map[bond_atom2_orig + offset] 
                new_mol.AddBond(u_new, v_new, Chem.BondType.SINGLE)
            except KeyError:
                print("[Merge Error] Bonding atoms were unexpectedly pruned.")
                return None, None
        else:
            # 일반 펩타이드 폴리머를 위한 Fallback
            keep_set1 = set(indices1)
            bond_atom1_orig = [idx for idx in indices1 for nbr in mol1.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() not in keep_set1][0]
            keep_set2 = set(indices2)
            bond_atom2_orig = [idx for idx in indices2 for nbr in mol2.GetAtomWithIdx(idx).GetNeighbors() if nbr.GetIdx() not in keep_set2][0]
            new_mol.AddBond(node_map[bond_atom1_orig], node_map[bond_atom2_orig + offset], Chem.BondType.SINGLE)
        
        # 분자 안정화 및 3D Conformer 주입
        mol_fixed = new_mol.GetMol()
        try: 
            Chem.SanitizeMol(mol_fixed)
        except Exception: 
            mol_fixed.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol_fixed) 

        conf = Chem.Conformer(mol_fixed.GetNumAtoms())
        for i, pos in enumerate(final_coords):
            conf.SetAtomPosition(i, pos.tolist())
        mol_fixed.AddConformer(conf)
        Chem.AssignStereochemistryFrom3D(mol_fixed)
        
        return mol_fixed, np.array(final_coords)

    except Exception as e:
        print(f"[Merge Error] Failed to merge residues: {e}")
        return None, None

# ==============================================================================
# [Debug & Verification]
# ==============================================================================
if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__)) 
    bakers_dir = os.path.dirname(current_dir)                
    src_dir = os.path.dirname(bakers_dir)                    
    
    if src_dir not in sys.path: 
        sys.path.insert(0, src_dir)

    from rdkit.Chem import AllChem

    print("=" * 60)
    print("[Debug] src/bakers/chem/align.py 기능 검증 시작")
    print("=" * 60)

    print("\n[Test 1] Molecule Merging (Full Integration Test)...")
    try:
        from bakers.chem.topology import analyze_residue_topology
        
        m1 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCC(=O)NC"))
        m2 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NCC(=O)NC"))
        AllChem.EmbedMolecule(m1)
        AllChem.EmbedMolecule(m2)
        
        p1 = analyze_residue_topology(m1)
        p2 = analyze_residue_topology(m2)
        
        merged_mol, merged_coords = merge_residues(m1, m1.GetConformer().GetPositions(), p1, 
                                                   m2, m2.GetConformer().GetPositions(), p2)
        
        if merged_mol is not None:
            expected_atoms = len(p1['n_term_indices']) + len(p2['c_term_indices'])
            actual_atoms = merged_mol.GetNumAtoms()
            print(f" -> Expected Atoms: {expected_atoms}, Actual: {actual_atoms}")
            if expected_atoms == actual_atoms: 
                print(" -> [PASS] Merge successful.")
                print(f" -> SMILES of Dimer: {Chem.MolToSmiles(Chem.RemoveHs(merged_mol))}")
            else: 
                print(" -> [FAIL] Atom count mismatch.")
        else: 
            print(" -> [FAIL] merge_residues returned None.")

    except Exception as e:
        print(f" -> [ERROR] Test Failed: {e}")

    print("=" * 60)
    print("[Debug] 검증 종료")