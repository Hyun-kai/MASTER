from rdkit import Chem

# ==============================================================================
# 1. 캡(Cap) 정의 및 설정 (Configuration)
# ==============================================================================
# 각 캡에 대해 다음 정보를 정의합니다:
# - smarts: 탐지할 패턴
# - type: 캡의 역할 (n_term_prot: 질소 보호, c_term_prot: 카보닐 보호)
# - cap_atom_indices: SMARTS 패턴 내에서 '제거될' 캡 원자들의 인덱스 리스트 (0부터 시작)
# - anchor_atom_idx: SMARTS 패턴 내에서 '남겨질' 코어 연결 원자의 인덱스
# ==============================================================================

CAP_DEFINITIONS = {
    # [Acyl Type] Acetyl 등. 
    # 패턴: C(0)-C(=O)(1,2) - [N_core](3)
    # 역할: 코어의 N을 보호함 (N-term 보호)
    'ACE': {
        'smarts': '[CH3:1][C:2](=[O:3])[#7:4]', 
        'type': 'n_term_prot', 
        'cap_atom_indices': [0, 1, 2], 
        'anchor_atom_idx': 3
    },

    # [Amine Type] N-Methyl amide 등.
    # 패턴: [C_core](0)(=O) - N(2)-C(3)
    # 역할: 코어의 C(=O)를 보호함 (C-term 보호)
    # 주의: SMARTS 작성 시 Core의 C(=O)가 0, 1번 인덱스에 오도록 작성
    'NME': {
        'smarts': '[C:1](=[O:2])[#7:3][CH3:4]', 
        'type': 'c_term_prot',
        'cap_atom_indices': [2, 3], 
        'anchor_atom_idx': 0
    },

    # [Urea Type] Methylurea 등.
    # 패턴: N(0)-C(=O)(1,2) - [N_core](3)
    # 역할: 코어의 N을 보호함 (N-term 보호)
    # 앞서 논의한 [CH3]NC(=O)N 패턴 활용
    'UREA': {
        'smarts': '[#7:1][C:2](=[O:3])[#7:4]', 
        'type': 'n_term_prot',
        'cap_atom_indices': [0, 1, 2], 
        'anchor_atom_idx': 3
    },

    # [Carbamate Type] Boc, Cbz 등.
    # 패턴: O(0)-C(=O)(1,2) - [N_core](3) (산소가 먼저 옴)
    # 역할: 코어의 N을 보호함 (N-term 보호)
    'CARBAMATE': {
        'smarts': '[#8:1][C:2](=[O:3])[#7:4]', 
        'type': 'n_term_prot',
        'cap_atom_indices': [0, 1, 2],
        'anchor_atom_idx': 3
    }
}

def detect_caps(mol):
    """
    분자에서 정의된 캡 패턴을 검색하여 캡 원자들의 인덱스와 앵커 정보를 반환합니다.
    
    Args:
        mol (rdkit.Chem.Mol): 분석할 분자 객체
        
    Returns:
        list[dict]: 발견된 캡 정보 리스트
    """
    caps_found = []
    
    # 이미 캡으로 식별된 원자들을 추적하여 중복 탐지 방지 (선택 사항)
    processed_atoms = set()

    for cap_name, rule in CAP_DEFINITIONS.items():
        pattern = Chem.MolFromSmarts(rule['smarts'])
        
        # SMARTS 패턴에 오류가 있으면 건너뜀 (안전 장치)
        if pattern is None:
            print(f"[Warning] Invalid SMARTS for {cap_name}")
            continue
            
        matches = mol.GetSubstructMatches(pattern)
        
        for match in matches:
            # match는 튜플로 반환됨 (예: (5, 6, 7, 8))
            # rule['cap_atom_indices']를 사용하여 실제 분자에서의 인덱스 매핑
            
            try:
                current_cap_indices = set()
                current_anchor_idx = -1
                
                # 1. 캡 원자 인덱스 추출
                for local_idx in rule['cap_atom_indices']:
                    atom_idx = match[local_idx]
                    current_cap_indices.add(atom_idx)
                
                # 2. 앵커(Core 연결부) 원자 인덱스 추출
                current_anchor_idx = match[rule['anchor_atom_idx']]
                
                # 중복 방지: 이미 처리된 원자가 포함되어 있다면 스킵할 수도 있음
                # 여기서는 단순히 추가하되, 나중에 Core 분석 시 집합 연산으로 처리
                
                caps_found.append({
                    'name': cap_name,
                    'type': rule['type'],
                    'indices': current_cap_indices,
                    'anchor_index': current_anchor_idx
                })
                
                processed_atoms.update(current_cap_indices)
                
            except IndexError:
                print(f"[Error] SMARTS match length mismatch for {cap_name}")
                continue
            
    return caps_found

def analyze_core_type(mol):
    """
    탐지된 캡 정보를 바탕으로 Core의 모노머 타입(AminoAcid, Diamine, Diacid)을 판별합니다.
    
    Logic:
        - N-term 보호 캡 (Acyl, Urea, Carbamate) 개수 확인
        - C-term 보호 캡 (Amine) 개수 확인
        - 조합에 따라 Core 타입 결정
    
    Returns:
        dict: 분석 결과 (타입, 코어 인덱스, 캡 정보 등)
    """
    caps = detect_caps(mol)
    
    all_indices = set(range(mol.GetNumAtoms()))
    all_cap_indices = set()
    
    # 캡의 유형별 개수 카운트
    n_term_prot_count = 0  # 질소 보호 그룹 개수
    c_term_prot_count = 0  # 카보닐 보호 그룹 개수
    
    for c in caps:
        all_cap_indices.update(c['indices'])
        if c['type'] == 'n_term_prot':
            n_term_prot_count += 1
        elif c['type'] == 'c_term_prot':
            c_term_prot_count += 1
            
    # Core 인덱스 계산 (전체 - 캡)
    core_indices = list(all_indices - all_cap_indices)
    core_indices.sort()
    
    # 모노머 타입 판별 로직
    monomer_type = 'unknown'
    
    if n_term_prot_count >= 1 and c_term_prot_count >= 1:
        # 양쪽 성질이 다름 -> Amino Acid (Hetero type)
        monomer_type = 'amino_acid'
    elif n_term_prot_count >= 2:
        # 양쪽 다 질소 보호 -> Core는 Diamine (Head-Head)
        monomer_type = 'diamine'
    elif c_term_prot_count >= 2:
        # 양쪽 다 카보닐 보호 -> Core는 Diacid (Tail-Tail)
        monomer_type = 'diacid'
    elif n_term_prot_count == 1 and c_term_prot_count == 0:
         # 한쪽만 막힘 (예외 케이스 혹은 말단 모노머)
         monomer_type = 'n_capped_fragment'
    elif n_term_prot_count == 0 and c_term_prot_count == 1:
         monomer_type = 'c_capped_fragment'

    return {
        'monomer_type': monomer_type,
        'core_indices': core_indices,
        'caps_summary': {
            'n_term_prot_count': n_term_prot_count,
            'c_term_prot_count': c_term_prot_count
        },
        'caps_details': caps
    }

# ==============================================================================
# 테스트 실행 코드 (Main)
# ==============================================================================
if __name__ == "__main__":
    print("--- Cap Detection Test ---")
    
    # 테스트 케이스 1: Amino Acid 형태 (Acyl + NME)
    # Acetyl-Glycine-N-Methylamide
    mol_aa = Chem.MolFromSmiles("CC(=O)NCC(=O)NC") 
    print(f"\n1. Amino Acid Check (CC(=O)NCC(=O)NC):")
    result_aa = analyze_core_type(mol_aa)
    print(f"   Type: {result_aa['monomer_type']}")
    print(f"   Counts: {result_aa['caps_summary']}")

    # 테스트 케이스 2: Diamine 형태 (Urea + Urea)
    # Methylurea - CH2CH2 - Methylurea (양끝이 Urea Cap)
    # 구조: CN-C(=O)-N(Core)-CC-N(Core)-C(=O)-NC
    mol_diamine = Chem.MolFromSmiles("CNC(=O)NCCNC(=O)NC")
    print(f"\n2. Diamine Check (CNC(=O)NCCNC(=O)NC):")
    result_da = analyze_core_type(mol_diamine)
    print(f"   Type: {result_da['monomer_type']}")
    print(f"   Counts: {result_da['caps_summary']}")
    
    # 테스트 케이스 3: Diamine 형태 (Carbamate + Urea 혼합)
    # Methoxycarbonyl(Carbamate) - N...N - Urea
    mol_mix = Chem.MolFromSmiles("COC(=O)NCCNC(=O)NC")
    print(f"\n3. Mixed Cap Check (COC(=O)NCCNC(=O)NC):")
    result_mix = analyze_core_type(mol_mix)
    print(f"   Type: {result_mix['monomer_type']}") 
    # Carbamate와 Urea 모두 'n_term_prot'이므로 diamine으로 인식되어야 함