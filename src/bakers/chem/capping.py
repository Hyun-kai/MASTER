"""
src/bakers/chem/capping.py

[기능]
비표준 아미노산(NCAA) 및 대칭형 방향족 폴리머(Alkyne-Pyridine) 결합을 화학적으로 명확하게 분석합니다.
분자의 말단에 붙은 캡(Cap)을 식별하고, Core 영역을 분리하며, 모노머의 조립 토폴로지(Topology)를 판별합니다.
대칭형 모노머의 방향성(Symmetry Breaking) 부여 기능이 포함되어 있습니다.
"""

import operator
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from rdkit import Chem
from rdkit.Chem import rdMolAlign

class LinkerRole(Enum):
    NUCLEOPHILE = "nucleophile"   # 새로 들어오는 모노머의 결합부 (Head)
    ELECTROPHILE = "electrophile" # 기존 폴리머가 연장되는 말단부 (Tail)
    UNKNOWN = "unknown"

@dataclass
class CappingGroup:
    name: str
    bond_type: str
    elec_cap_smarts: str
    nuc_cap_smarts: str
    elec_cap_leave_maps: List[int]
    elec_cap_connect_map: int
    nuc_cap_leave_maps: List[int]
    nuc_cap_connect_map: int

# ==============================================================================
# 1. 캡(Cap) 정의 사전
# ==============================================================================
CAP_DEFINITIONS: Dict[str, CappingGroup] = {
    # ----------------------------------------------------------------------
    # [NEW] Alkyne-Pyridine Polymerization (사용자 정의 대칭형 폴리머)
    # 공통 SMARTS: [C:1]#[C:2] - [a:3]1:[a:4]:[a:5](-[C:6]#[C:7]-[C;H3:8]):[a:9]:[a:10]:[a:11]1
    # - 1, 2: 코어 쪽 내부 알카인
    # - 3, 4, 5, 9, 10, 11: 피리딘 고리 전체 (Alignment Template 역할)
    # - 6, 7, 8: 말단 메틸 알카인
    # ----------------------------------------------------------------------
    'ALKYNE_PYRIDINE': CappingGroup(
        name='Alkyne-Pyridine Polymer',
        bond_type='alkyne_aryl',
        
        # [Elec Cap (꼬리)] 폴리머 끝단. 피리딘은 보존하고 말단 메틸 알카인(-C#C-CH3)만 제거
        elec_cap_smarts='[C:1]#[C:2]-[a:3]1:[a:4]:[a:5](-[C:6]#[C:7]-[C;H3:8]):[a:9]:[a:10]:[a:11]1',
        elec_cap_leave_maps=[6, 7, 8],  # 떨어져 나가는 그룹
        elec_cap_connect_map=5,         # 피리딘 고리의 탄소 (새로운 결합이 형성될 앵커)
        
        # [Nuc Cap (머리)] 새 모노머. 피리딘 고리와 말단 메틸 알카인 전체 제거
        nuc_cap_smarts='[C:1]#[C:2]-[a:3]1:[a:4]:[a:5](-[C:6]#[C:7]-[C;H3:8]):[a:9]:[a:10]:[a:11]1',
        nuc_cap_leave_maps=[3, 4, 5, 6, 7, 8, 9, 10, 11], # 이탈기 (피리딘 + 말단알카인)
        nuc_cap_connect_map=2           # 내부 알카인 탄소 (새로운 결합이 형성될 앵커)
    ),
    'UREA': CappingGroup(
        name='Urea',
        bond_type='urea',
        # Core-NH - C(=O)-NH-CH3
        elec_cap_smarts='[!#1:1]-[N:2]-[C:3](=[O:4])-[N:5]-[C;H3:6]',
        elec_cap_leave_maps=[3, 4, 5, 6],
        elec_cap_connect_map=2,         # 앵커는 Core의 N
        
        # CH3-NH-C(=O) - NH-Core
        nuc_cap_smarts='[C;H3:1]-[N:2]-[C:3](=[O:4])-[N:5]-[!#1:6]',
        nuc_cap_leave_maps=[1, 2, 3, 4],
        nuc_cap_connect_map=5           # 앵커는 Core의 N
    ),
    'CARBAMATE': CappingGroup(
        name='Carbamate',
        bond_type='carbamate',
        # Core-O - C(=O)-NH-CH3
        elec_cap_smarts='[!#1:1]-[O:2]-[C:3](=[O:4])-[N:5]-[C;H3:6]',
        elec_cap_leave_maps=[3, 4, 5, 6],
        elec_cap_connect_map=2,         # 앵커는 Core의 O
        
        # CH3-O-C(=O) - NH-Core
        nuc_cap_smarts='[C;H3:1]-[O:2]-[C:3](=[O:4])-[N:5]-[!#1:6]',
        nuc_cap_leave_maps=[1, 2, 3, 4],
        nuc_cap_connect_map=5           # 앵커는 Core의 N
    ),
    'THIOESTER': CappingGroup(
        name='Thioester',
        bond_type='thioester',
        # Core-C(=O) - S-CH3
        elec_cap_smarts='[!#1:1]-[C:2](=[O:3])-[S:4]-[C;H3:5]',
        elec_cap_leave_maps=[4, 5],
        elec_cap_connect_map=2,         # 앵커는 Core의 C
        
        # CH3-C(=O) - S-Core
        nuc_cap_smarts='[C;H3:1]-[C:2](=[O:3])-[S:4]-[!#1:5]',
        nuc_cap_leave_maps=[1, 2, 3],
        nuc_cap_connect_map=4           # 앵커는 Core의 S
    ),
    'AMIDE': CappingGroup(
        name='Amide/Peptide',
        bond_type='peptide',
        elec_cap_smarts='[!#1:1]-[C:2](=[O:3])-[N:4]-[C;H3:5]',
        elec_cap_leave_maps=[4, 5], elec_cap_connect_map=2,
        nuc_cap_smarts='[C;H3:1]-[C:2](=[O:3])-[N:4]-[!#1:5]',
        nuc_cap_leave_maps=[1, 2, 3], nuc_cap_connect_map=4
    )
}

def _get_mapped_indices(mol: Chem.Mol, smarts: str) -> Dict[int, int]:
    query = Chem.MolFromSmarts(smarts)
    if not query: raise ValueError(f"Invalid SMARTS pattern: {smarts}")
    match = mol.GetSubstructMatch(query)
    if not match: raise ValueError(f"No substructure match found for SMARTS: {smarts}")
    
    return {query.GetAtomWithIdx(i).GetAtomMapNum(): mol_idx 
            for i, mol_idx in enumerate(match) if query.GetAtomWithIdx(i).GetAtomMapNum() > 0}

def splice_monomers(mol_elec: Chem.Mol, mol_nuc: Chem.Mol, cap_def: CappingGroup) -> Chem.Mol:
    if not mol_elec.GetNumConformers() or not mol_nuc.GetNumConformers():
        raise ValueError("Both molecules must have 3D conformers.")

    elec_map = _get_mapped_indices(mol_elec, cap_def.elec_cap_smarts)
    nuc_map = _get_mapped_indices(mol_nuc, cap_def.nuc_cap_smarts)

    common_maps = sorted(list(set(elec_map.keys()) & set(nuc_map.keys())))
    elec_match_idx = [elec_map[m] for m in common_maps]
    nuc_match_idx = [nuc_map[m] for m in common_maps]

    atom_map = list(zip(nuc_match_idx, elec_match_idx))
    rmsd, trans_matrix = rdMolAlign.GetAlignmentTransform(mol_nuc, mol_elec, atomMap=atom_map)
    Chem.TransformMol(mol_nuc, trans_matrix)

    offset = mol_elec.GetNumAtoms()
    combined = Chem.CombineMols(mol_elec, mol_nuc)
    rw_mol = Chem.RWMol(combined)

    idx_connect_elec = elec_map[cap_def.elec_cap_connect_map]
    idx_connect_nuc = nuc_map[cap_def.nuc_cap_connect_map] + offset
    rw_mol.AddBond(idx_connect_elec, idx_connect_nuc, Chem.BondType.SINGLE)

    atoms_to_remove = [elec_map[m] for m in cap_def.elec_cap_leave_maps] + \
                      [nuc_map[m] + offset for m in cap_def.nuc_cap_leave_maps]

    for idx in sorted(list(set(atoms_to_remove)), reverse=True):
        rw_mol.RemoveAtom(idx)

    final_mol = rw_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    Chem.AssignStereochemistryFrom3D(final_mol) 
    return final_mol

# ==============================================================================
# 2. 모노머 분석 및 대칭 붕괴(Symmetry Breaking) 로직
# ==============================================================================

def analyze_monomer(mol: Chem.Mol) -> Dict[str, Any]:
    nuc_caps_found = []
    elec_caps_found = []
    anchors = []
    cap_atom_indices = set()
    
    seen_nuc_anchors = set()
    seen_elec_anchors = set()
    
    for cap_id, cap_def in CAP_DEFINITIONS.items():
        # 1. Nucleophile (Head) Cap 탐색
        nuc_query = Chem.MolFromSmarts(cap_def.nuc_cap_smarts)
        for match in mol.GetSubstructMatches(nuc_query):
            map_to_idx = {nuc_query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if nuc_query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            anchor_idx = map_to_idx[cap_def.nuc_cap_connect_map]
            if anchor_idx in seen_nuc_anchors: continue
            seen_nuc_anchors.add(anchor_idx)
            
            leave_indices = [map_to_idx[m] for m in cap_def.nuc_cap_leave_maps]
            nuc_caps_found.append({'cap_type': cap_id, 'bond_type': cap_def.bond_type, 'leave_indices': leave_indices, 'anchor_idx': anchor_idx})
            anchors.append({'anchor_idx': anchor_idx, 'cap_type': cap_id, 'linker_role': LinkerRole.NUCLEOPHILE, 'atom_symbol': mol.GetAtomWithIdx(anchor_idx).GetSymbol()})
            cap_atom_indices.update(leave_indices)

        # 2. Electrophile (Tail) Cap 탐색
        elec_query = Chem.MolFromSmarts(cap_def.elec_cap_smarts)
        for match in mol.GetSubstructMatches(elec_query):
            map_to_idx = {elec_query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if elec_query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            anchor_idx = map_to_idx[cap_def.elec_cap_connect_map]
            if anchor_idx in seen_elec_anchors: continue
            seen_elec_anchors.add(anchor_idx)
            
            leave_indices = [map_to_idx[m] for m in cap_def.elec_cap_leave_maps]
            elec_caps_found.append({'cap_type': cap_id, 'bond_type': cap_def.bond_type, 'leave_indices': leave_indices, 'anchor_idx': anchor_idx})
            anchors.append({'anchor_idx': anchor_idx, 'cap_type': cap_id, 'linker_role': LinkerRole.ELECTROPHILE, 'atom_symbol': mol.GetAtomWithIdx(anchor_idx).GetSymbol()})
            cap_atom_indices.update(leave_indices)

    # --------------------------------------------------------------------------
    # [핵심] 대칭 붕괴 (Symmetry Breaking) 로직
    # 분자의 양 끝단이 동일하여 Nuc와 Elec가 각각 2개씩 매칭된 경우,
    # 물리적으로 서로 반대편(leave_indices가 겹치지 않는)에 있는 한 쌍만 골라내어
    # 하나의 유효한 방향성(HETEROBIFUNCTIONAL)을 강제합니다.
    # --------------------------------------------------------------------------
    if len(nuc_caps_found) >= 1 and len(elec_caps_found) >= 1:
        valid_nuc, valid_elec = None, None
        for n_cap in nuc_caps_found:
            for e_cap in elec_caps_found:
                # 두 이탈 그룹의 원자 인덱스 세트가 교집합이 없다면(서로 반대편 끝단이라면)
                if set(n_cap['leave_indices']).isdisjoint(set(e_cap['leave_indices'])):
                    valid_nuc = n_cap
                    valid_elec = e_cap
                    break
            if valid_nuc: break
            
        if valid_nuc and valid_elec:
            nuc_caps_found = [valid_nuc]
            elec_caps_found = [valid_elec]
            # 앵커 목록 재정의
            anchors = [
                {'anchor_idx': valid_nuc['anchor_idx'], 'cap_type': valid_nuc['cap_type'], 'linker_role': LinkerRole.NUCLEOPHILE, 'atom_symbol': mol.GetAtomWithIdx(valid_nuc['anchor_idx']).GetSymbol()},
                {'anchor_idx': valid_elec['anchor_idx'], 'cap_type': valid_elec['cap_type'], 'linker_role': LinkerRole.ELECTROPHILE, 'atom_symbol': mol.GetAtomWithIdx(valid_elec['anchor_idx']).GetSymbol()}
            ]
            # 사용되지 않고 탈락한 캡의 인덱스 복구
            cap_atom_indices = set(valid_nuc['leave_indices']) | set(valid_elec['leave_indices'])

    all_indices = set(range(mol.GetNumAtoms()))
    core_indices = sorted(list(all_indices - cap_atom_indices))

    num_nuc = len(nuc_caps_found)
    num_elec = len(elec_caps_found)
    
    monomer_type = 'UNKNOWN'
    if num_nuc == 1 and num_elec == 1: monomer_type = 'HETEROBIFUNCTIONAL'      
    elif num_nuc == 2 and num_elec == 0: monomer_type = 'DINUCLEOPHILE'     
    elif num_elec == 2 and num_nuc == 0: monomer_type = 'DIELECTROPHILE'     
    elif num_nuc == 1 and num_elec == 0: monomer_type = 'MONONUCLEOPHILE' 
    elif num_nuc == 0 and num_elec == 1: monomer_type = 'MONOELECTROPHILE' 
    elif num_nuc == 0 and num_elec == 0: monomer_type = 'CORE_ONLY'   

    return {
        'monomer_type': monomer_type,
        'nuc_caps': nuc_caps_found,
        'elec_caps': elec_caps_found,
        'anchors': anchors,
        'core_indices': core_indices,
        'caps_detected': [c['cap_type'] for c in nuc_caps_found + elec_caps_found],
        'is_valid': monomer_type != 'UNKNOWN'
    }

if __name__ == "__main__":
    print("=" * 60)
    print("[Debug] Monomer Analysis Verification (Chemical Nomenclature)")
    print("=" * 60)
    test_cases = [
        ("Case 1: Amino Acid-like", "CC(=O)NCC(=O)NC", "HETEROBIFUNCTIONAL", 2),
        ("Case 2: Diamine-like", "CC(=O)NCCNC(=O)C", "DINUCLEOPHILE", 2),
        ("Case 3: Diacid-like", "CNC(=O)CCC(=O)NC", "DIELECTROPHILE", 2),
    ]
    for name, smiles, exp_type, exp_cap_count in test_cases:
        mol = Chem.MolFromSmiles(smiles)
        result = analyze_monomer(mol)
        print(f"\nRunning {name}...")
        print(f"  -> Type Check: {'PASS' if result['monomer_type'] == exp_type else 'FAIL'}")
        print(f"  -> Anchors: {[a['linker_role'].value for a in result['anchors']]}")