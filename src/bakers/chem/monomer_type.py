"""
src/bakers/chem/monomer_type.py

[기능]
분자의 말단에 붙은 캡(Cap)을 식별하고, Core 영역을 분리하여 모노머 타입을 판별합니다.
"""

from rdkit import Chem
from enum import Enum

class LinkerType(Enum):
    DONOR = "donor"       # 예: Amine, Hydroxyl (Nucleophile) -> Head
    ACCEPTOR = "acceptor" # 예: Carboxyl, Ester (Electrophile) -> Tail
    UNKNOWN = "unknown"

# 1. 일반화된 Cap 정의 (SMARTS 패턴 수정됨)
# [Critical Fix] SMARTS 끝의 '-' 제거. RDKit은 '대상 없는 결합'을 문법 오류로 처리함.
CAP_DEFINITIONS = [
    {
        'name': 'acyl_cap', 
        # 수정전: '[CH3;X4][C;X3](=O)-'
        'smarts': '[CH3;X4][C;X3](=O)', # Acetyl group
        'type': LinkerType.ACCEPTOR 
    },
    {
        'name': 'methyl_ester_cap',
        # 수정전: '[CH3;X4][O;X2][C;X3](=O)-'
        'smarts': '[CH3;X4][O;X2][C;X3](=O)', # Methyl Ester group
        'type': LinkerType.ACCEPTOR 
    },
    {
        'name': 'n_methyl_cap',
        # 수정전: '[CH3;X4][NH;X3]-'
        'smarts': '[CH3;X4][NH;X3]', # N-Methyl amide end
        'type': LinkerType.ACCEPTOR 
    },
    {
        'name': 'boc_cap',
        # 수정전: 'CC(C)(C)OC(=O)[NH]-'
        'smarts': 'CC(C)(C)OC(=O)[NH]', # Boc protection group
        'type': LinkerType.DONOR 
    }
]

def analyze_monomer_generalized(mol, cap_defs=CAP_DEFINITIONS):
    """
    일반화된 캡 제거 및 앵커 탐지 알고리즘
    """
    mol_rw = Chem.RWMol(mol)
    atoms_to_remove = set()
    anchors = []
    
    detected_caps = []

    # 모든 원자의 인덱스 집합
    all_indices = set(range(mol.GetNumAtoms()))

    for cap_def in cap_defs:
        pattern = Chem.MolFromSmarts(cap_def['smarts'])
        if not pattern:
            print(f"[Warning] Invalid SMARTS for {cap_def['name']}")
            continue
            
        matches = mol.GetSubstructMatches(pattern)
        
        for match in matches:
            match_set = set(match)
            
            # 캡과 코어 사이의 연결 결합 찾기
            connection_found = False
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                for neighbor in atom.GetNeighbors():
                    n_idx = neighbor.GetIdx()
                    # 캡(Match)에 포함되지 않은 이웃 원자가 있다면, 그 원자가 바로 Anchor(Core의 연결점)
                    if n_idx not in match_set:
                        anchor_info = {
                            'anchor_idx': n_idx,
                            'cap_type': cap_def['name'],
                            'linker_role': cap_def['type'],
                            'atom_symbol': neighbor.GetSymbol()
                        }
                        anchors.append(anchor_info)
                        connection_found = True
            
            # 연결 부위가 확인된 매칭만 유효한 캡으로 인정
            if connection_found:
                atoms_to_remove.update(match_set)
                detected_caps.append(cap_def)

    # Core Indices 계산 (전체 - 캡)
    core_indices = list(all_indices - atoms_to_remove)
    core_indices.sort()
    
    # 모노머 타입 판별 로직
    n_donors = len([a for a in anchors if a['linker_role'] == LinkerType.DONOR])
    n_acceptors = len([a for a in anchors if a['linker_role'] == LinkerType.ACCEPTOR])
    
    monomer_type = 'unknown'
    # 로직 보정: Hetero는 Donor 1개 이상, Acceptor 1개 이상일 때
    if n_donors >= 1 and n_acceptors >= 1:
        monomer_type = 'hetero (amino_acid_like)'
    elif n_donors >= 2:
        monomer_type = 'head_head (diamine_like)'
    elif n_acceptors >= 2:
        monomer_type = 'tail_tail (diacid_like)'

    return {
        'monomer_type': monomer_type,
        'core_indices': core_indices,
        'anchors': anchors,
        'caps_detected': [c['name'] for c in detected_caps]
    }

# ==============================================================================
# [Debug & Verification]
# ==============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("[Debug] Monomer Analysis Verification (Fixed SMARTS)")
    print("=" * 60)

    test_cases = [
        (
            "Case 1: Amino Acid-like (Boc-Gly-OMe)", 
            "CC(C)(C)OC(=O)NCC(=O)OC",  
            "hetero (amino_acid_like)", 
            2
        ),
        (
            "Case 2: Diamine-like (Boc-NH-CH2-CH2-NH-Boc)",
            "CC(C)(C)OC(=O)NCCNC(=O)OC(C)(C)C",
            "head_head (diamine_like)",
            2
        ),
        (
            "Case 3: Diacid-like (MeO-C(=O)-CH2-C(=O)-OMe)",
            "COC(=O)CCC(=O)OC", 
            "tail_tail (diacid_like)",
            2
        ),
        (
            "Case 4: No Caps (Benzene)",
            "c1ccccc1",
            "unknown",
            0
        )
    ]

    for name, smiles, exp_type, exp_cap_count in test_cases:
        print(f"\nRunning {name}...")
        
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"  [ERROR] Invalid SMILES: {smiles}")
            continue
            
        # 함수 실행
        result = analyze_monomer_generalized(mol)
        
        detected_type = result['monomer_type']
        detected_caps = result['caps_detected']
        core_indices = result['core_indices']
        anchors = result['anchors']
        
        # 검증 1: 타입
        is_type_match = (detected_type == exp_type)
        status_type = "PASS" if is_type_match else f"FAIL (Expected '{exp_type}', Got '{detected_type}')"
        print(f"  -> Type Check: {status_type}")
        
        # 검증 2: 캡 개수
        is_count_match = (len(detected_caps) == exp_cap_count)
        status_count = "PASS" if is_count_match else f"FAIL (Expected {exp_cap_count}, Got {len(detected_caps)})"
        print(f"  -> Cap Count : {status_count} {detected_caps}")
        
        # 검증 3: 앵커 무결성 (앵커는 코어에 속해야 함)
        anchor_valid = True
        if anchors:
            for anc in anchors:
                if anc['anchor_idx'] not in core_indices:
                    anchor_valid = False
                    print(f"     [Error] Anchor atom {anc['anchor_idx']} is NOT in core indices!")
        
        status_anchor = "PASS" if anchor_valid else "FAIL"
        print(f"  -> Anchor Integrity: {status_anchor}")
        
        # 검증 4: 원자 보존성
        total_atoms = mol.GetNumAtoms()
        # 캡으로 분류된 원자 수 계산 (중복 제거된 합집합 크기 확인 필요하나, 여기선 간단히 검증)
        # 단순히 core size만 출력
        print(f"  -> Core Atoms Info: {len(core_indices)} atoms preserved out of {total_atoms}")
        
        if anchors:
            print("     [Anchor Details]")
            for i, anc in enumerate(anchors):
                print(f"       #{i+1}: AtomIdx {anc['anchor_idx']} ({anc['atom_symbol']}) - Role: {anc['linker_role'].value} (via {anc['cap_type']})")

    print("\n" + "=" * 60)
    print("[Debug] Verification Completed")