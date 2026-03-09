"""
src/bakers/chem/topology.py

[목적 및 설명]
단일 분자(Residue)의 RDKit Mol 객체를 입력받아 위상(Topology) 구조를 분석합니다.
폴리머(Polymer) 조립 및 로타머 샘플링에 필수적인 
백본(Backbone) 경로, 캡(Caps), 연결 앵커(Connection Anchors), 그리고 자유도(DOFs)를 추출합니다.

[설계 의도 및 최적화 내역]
- Top-Down 접근 방식: 전체 위상 분석 -> 세부 앵커 추출 -> 자유도 계산 순서로 논리가 흐릅니다.
- Absolute Role-based Ordering: Nuc(Lower)/Elec(Upper) 앵커 배열 시, 방향성과 역할(Role)에 
  따른 절대적인 1:1 교차 맵핑을 영구적으로 보장합니다.
- Performance & Safety: SMARTS 쿼리 전역 캐싱, BFS 큐(deque) 도입 및 
  원본 딕셔너리 보호(Immutability 지향) 로직이 적용되었습니다.
"""

import os
import sys
import numpy as np
import networkx as nx
from collections import deque
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Dict, Any, Set, Tuple, Optional

# 프로젝트 루트 경로 확보 및 의존성 주입
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path: 
    sys.path.append(src_dir)

try:
    from bakers.chem import capping
except ImportError:
    pass

# ==============================================================================
# [Global Cache] SMARTS 쿼리 사전 컴파일 (성능 병목 해결 및 확장성 확보)
# ==============================================================================
_CAP_QUERIES = {
    'ALKYNE_PYRIDINE': Chem.MolFromSmarts('[C:1]#[C:2]-[c:3]1[c:4][c:5](-[C:6]#[C:7]-[C;H3:8])[c:9][n:10][c:11]1')
    # 향후 새로운 비표준 캡 추가 시 이곳에만 정의하면 됩니다.
}

# ==============================================================================
# [Section 1] 핵심 위상 분석 (Core Topology Analysis)
# ==============================================================================

def analyze_residue_topology(mol: Chem.Mol) -> Dict[str, Any]:
    """
    분자의 캡(Capping) 정보를 기반으로 N-말단, C-말단, 그리고 폴리머 연결 시 사용할
    핵심 앵커(Anchor) 인덱스들을 추출하여 딕셔너리로 반환합니다.
    """
    analysis = capping.analyze_monomer(mol)
    monomer_type = analysis.get('monomer_type', 'UNKNOWN')
    raw_nuc_caps = analysis.get('nuc_caps', [])
    raw_elec_caps = analysis.get('elec_caps', [])
    core_indices = analysis.get('core_indices', [])

    # [안전성] 원본 딕셔너리 훼손(Mutation) 방지를 위해 얕은 복사본을 생성하여 처리
    nuc_caps = []
    elec_caps = []
    
    for cap_list, target_list in [(raw_nuc_caps, nuc_caps), (raw_elec_caps, elec_caps)]:
        for cap in cap_list:
            safe_cap = cap.copy()
            heavy_atoms = list(safe_cap['leave_indices'])
            expanded = set(heavy_atoms)
            # 수소(H) 원자들까지 포괄하여 이탈 그룹 확장
            for idx in heavy_atoms:
                expanded.update([n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetAtomicNum() == 1])
            safe_cap['leave_indices'] = list(expanded)
            target_list.append(safe_cap)

    nuc_anchor, elec_anchor = [], []
    nuc_cap_indices, elec_cap_indices = set(), set()

    # 모노머 타입에 따라 하단(Nuc/Lower) 및 상단(Elec/Upper) 앵커 추출 방식 분기
    if monomer_type == 'HETEROBIFUNCTIONAL' and nuc_caps and elec_caps:
        nuc_cap_indices = set(nuc_caps[0]['leave_indices'])
        elec_cap_indices = set(elec_caps[0]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, nuc_caps[0]['anchor_idx'], nuc_caps[0]['leave_indices'], core_indices, nuc_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, elec_caps[0]['anchor_idx'], elec_caps[0]['leave_indices'], core_indices, elec_caps[0].get('cap_type', ''))
        
    elif monomer_type == 'DINUCLEOPHILE' and len(nuc_caps) >= 2:
        nuc_cap_indices, elec_cap_indices = set(nuc_caps[0]['leave_indices']), set(nuc_caps[1]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, nuc_caps[0]['anchor_idx'], nuc_caps[0]['leave_indices'], core_indices, nuc_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, nuc_caps[1]['anchor_idx'], nuc_caps[1]['leave_indices'], core_indices, nuc_caps[1].get('cap_type', ''))
        
    elif monomer_type == 'DIELECTROPHILE' and len(elec_caps) >= 2:
        nuc_cap_indices, elec_cap_indices = set(elec_caps[0]['leave_indices']), set(elec_caps[1]['leave_indices'])
        nuc_anchor = _extract_nuc_anchor(mol, elec_caps[0]['anchor_idx'], elec_caps[0]['leave_indices'], core_indices, elec_caps[0].get('cap_type', ''))
        elec_anchor = _extract_elec_anchor(mol, elec_caps[1]['anchor_idx'], elec_caps[1]['leave_indices'], core_indices, elec_caps[1].get('cap_type', ''))

    all_indices = set(range(mol.GetNumAtoms()))
    # 캡이 제거된 후 남게 될 실제 잔기의 말단 인덱스 추론
    n_term_indices = sorted(list(all_indices - elec_cap_indices)) 
    c_term_indices = sorted(list(all_indices - nuc_cap_indices))  

    return {
        'monomer_type': monomer_type,
        'residue_indices': sorted(list(all_indices - nuc_cap_indices - elec_cap_indices)),
        'n_term_indices': n_term_indices,
        'c_term_indices': c_term_indices,
        'nuc_anchor_indices': nuc_anchor,
        'elec_anchor_indices': elec_anchor,
        'lower_connect_indices': nuc_anchor,  # Nuc는 항상 Lower (이전 모노머와 연결)
        'upper_connect_indices': elec_anchor, # Elec는 항상 Upper (다음 모노머와 연결)
        'is_capped': bool(nuc_anchor and elec_anchor),
    }


# ==============================================================================
# [Section 2] 앵커 및 캡 추출 헬퍼 (Anchor & Cap Extraction)
# ==============================================================================

def _extract_nuc_anchor(mol: Chem.Mol, anchor_idx: int, leave_indices: List[int], core_indices: List[int], cap_type: str = "") -> List[int]:
    """하단(Lower/Nucleophile) 연결에 필요한 앵커 원자 배열을 추출합니다."""
    if cap_type == 'ALKYNE_PYRIDINE' and cap_type in _CAP_QUERIES:
        query = _CAP_QUERIES[cap_type]
        for match in mol.GetSubstructMatches(query):
            map_idx = {query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            if map_idx.get(2) == anchor_idx:
                ring_atoms = [map_idx[3], map_idx[4], map_idx[5], map_idx[9], map_idx[10], map_idx[11]]
                return _build_ordered_pyridine_anchor(mol, ring_atoms, core_indices, role='nuc')
                
    seq = _build_anchor_sequence(mol, anchor_idx, set(leave_indices), 5)
    return seq[:3][::-1] if len(seq) < 5 else seq[::-1]

def _extract_elec_anchor(mol: Chem.Mol, anchor_idx: int, leave_indices: List[int], core_indices: List[int], cap_type: str = "") -> List[int]:
    """상단(Upper/Electrophile) 연결에 필요한 앵커 원자 배열을 추출합니다."""
    if cap_type == 'ALKYNE_PYRIDINE' and cap_type in _CAP_QUERIES:
        query = _CAP_QUERIES[cap_type]
        for match in mol.GetSubstructMatches(query):
            map_idx = {query.GetAtomWithIdx(i).GetAtomMapNum(): idx for i, idx in enumerate(match) if query.GetAtomWithIdx(i).GetAtomMapNum() > 0}
            if map_idx.get(5) == anchor_idx:
                ring_atoms = [map_idx[3], map_idx[4], map_idx[5], map_idx[9], map_idx[10], map_idx[11]]
                return _build_ordered_pyridine_anchor(mol, ring_atoms, core_indices, role='elec')
                
    seq = _build_anchor_sequence(mol, anchor_idx, set(leave_indices) | set(core_indices), 5)
    return seq[:3] if len(seq) < 5 else seq

def _build_ordered_pyridine_anchor(mol: Chem.Mol, smarts_match: List[int], core_indices: List[int], role: str) -> List[int]:
    """
    [설계 의도] 피리딘 고리(Pyridine ring) 원자들의 역할을 파악하여 
    결합 대상(role)에 맞는 절대적인 1:1 교차 순서(Alignment) 배열을 반환합니다.
    - role='nuc' (Lower):  [C_core, C_para, C_term, C_ortho_term, N, C_ortho_core]
    - role='elec' (Upper): [C_term, C_para, C_core, C_ortho_core, N, C_ortho_term]
    """
    anchor_set = set(smarts_match)
    core_set = set(core_indices)
    
    n_idx, c_core, c_term = -1, -1, -1
    
    # 1. 핵심 원자(N, C_core, C_term) 식별
    for idx in smarts_match:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7:
            n_idx = idx
        else:
            for nbr in atom.GetNeighbors():
                n_idx_nbr = nbr.GetIdx()
                if n_idx_nbr not in anchor_set and nbr.GetAtomicNum() > 1:
                    if n_idx_nbr in core_set:
                        c_core = idx
                    else:
                        c_term = idx
                        
    if n_idx == -1 or c_core == -1 or c_term == -1:
        return list(smarts_match)
        
    c_core_atom = mol.GetAtomWithIdx(c_core)
    c_term_atom = mol.GetAtomWithIdx(c_term)
    
    # 2. C_para 식별
    c_core_nbrs = {n.GetIdx() for n in c_core_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    c_term_nbrs = {n.GetIdx() for n in c_term_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    
    c_para_list = list(c_core_nbrs & c_term_nbrs)
    if not c_para_list: return list(smarts_match)
    c_para = c_para_list[0]
    
    # 3. N과 인접한 Ortho 탄소들 식별
    n_atom = mol.GetAtomWithIdx(n_idx)
    n_nbrs = {n.GetIdx() for n in n_atom.GetNeighbors() if n.GetIdx() in anchor_set}
    
    c_ortho_core_list = list(c_core_nbrs & n_nbrs)
    if not c_ortho_core_list: return list(smarts_match)
    c_ortho_core = c_ortho_core_list[0]
    
    c_ortho_term_list = list(c_term_nbrs & n_nbrs)
    if not c_ortho_term_list: return list(smarts_match)
    c_ortho_term = c_ortho_term_list[0]
    
    # [핵심] 역할에 따른 맞춤형 배열 생성
    if role == 'nuc':
        return [c_core, c_para, c_term, c_ortho_term, n_idx, c_ortho_core]
    else:
        return [c_term, c_para, c_core, c_ortho_core, n_idx, c_ortho_term]

def _build_anchor_sequence(mol: Chem.Mol, start_idx: int, allowed_indices: Set[int], target_length: int = 5) -> List[int]:
    """너비 우선 탐색(BFS)을 사용하여 일반적인 선형/분기 앵커 배열을 생성합니다.
    (O(1) pop 처리를 위해 deque 자료구조 사용)
    """
    seq = [start_idx]
    queue = deque([start_idx])
    visited = {start_idx}
    
    while queue and len(seq) < target_length:
        curr = queue.popleft()
        nbrs = [(nbr.GetAtomicNum(), nbr.GetIdx()) for nbr in mol.GetAtomWithIdx(curr).GetNeighbors() 
                if nbr.GetIdx() in allowed_indices and nbr.GetIdx() not in visited]
        # 원자번호(Heavy atom) 우선, 그 다음 인덱스 순 정렬
        nbrs.sort(key=lambda x: (-x[0], x[1]))
        
        for _, idx in nbrs:
            if len(seq) < target_length:
                seq.append(idx)
                visited.add(idx)
                queue.append(idx)
    return seq


# ==============================================================================
# [Section 3] 백본 및 자유도(DOF) 추출 (Backbone & DOF)
# ==============================================================================

def _pick_end_neighbor(mol: Chem.Mol, atom_idx: int, exclude_indices: Set[int]) -> Optional[int]:
    """
    [DOF Helper] 이면각(Dihedral Angle) 계산을 위한 끝단 기준 원자를 선택합니다.
    화학적 방향성을 일관되게 유지하기 위해 무거운 원자(N > O > C)를 우선 선택합니다.
    """
    atom = mol.GetAtomWithIdx(atom_idx)
    cand = [a.GetIdx() for a in atom.GetNeighbors() if a.GetIdx() not in exclude_indices]
    if not cand: return None
    
    for target_z in [7, 8, 6]: 
        targets = [i for i in cand if mol.GetAtomWithIdx(i).GetAtomicNum() == target_z]
        if targets: return min(targets) 
    return min(cand)

def is_terminal_methyl(mol: Chem.Mol, atom_idx: int) -> bool:
    """[DOF Helper] 캡핑된 말단 메틸기(-CH3) 여부를 확인하여 불필요한 회전축 설정을 방지합니다."""
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 6: return False
    if len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]) != 1: return False
    if atom.GetTotalNumHs() == 3: return True
    return False

def get_backbone_path(mol: Chem.Mol, topo_info: Dict[str, Any]) -> List[int]:
    """
    위상 정보를 바탕으로 분자의 하단(Nuc) 앵커에서 상단(Elec) 앵커까지 
    이어지는 최단 백본 경로(Shortest Path)를 추출합니다.
    """
    nuc_conn = topo_info.get('nuc_anchor_indices', [])
    elec_conn = topo_info.get('elec_anchor_indices', [])
    if not nuc_conn or not elec_conn: return []
    try:
        # 새로 정의된 Absolute Ordering 룰에 따라 C_core 인덱스(0번 또는 2번)를 정확히 참조
        nuc_core = nuc_conn[0]
        elec_core = elec_conn[2] if len(elec_conn) == 6 else elec_conn[-1]
        return list(Chem.rdmolops.GetShortestPath(mol, nuc_core, elec_core))
    except Exception: 
        return []

def get_dofs(mol: Chem.Mol, exclude_indices: Set[int]) -> List[Tuple[int, int, int, int]]:
    """
    백본 경로를 따라 회전 가능한 이면각 자유도(Degrees of Freedom) 축을 추출합니다.
    단일 결합과 특별한 삼중 결합(Alkyne) 링커를 모두 스캔하여 4개의 원자(a, u, v, d) 쌍을 반환합니다.
    """
    dofs = []
    topo_info = analyze_residue_topology(mol) 
    bb_path = get_backbone_path(mol, topo_info)
    if not bb_path: return []

    core_indices = set(topo_info.get('residue_indices', []))
    bb_bonds_set = {tuple(sorted((bb_path[i], bb_path[i+1]))) for i in range(len(bb_path)-1)}
    
    # 1. 스캔할 결합 패턴 매칭 (단일 결합 및 Alkyne-Aryl 삼중결합)
    single_matches = list(mol.GetSubstructMatches(Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')))
    triple_query = Chem.MolFromSmarts('[*:1]-[C:2]#[C:3]-[*:4]')
    triple_matches = list(mol.GetSubstructMatches(triple_query))

    processed_bonds = set()

    # --- A. 단일 결합 (Single Bonds) 처리 ---
    for u, v in single_matches:
        if u in exclude_indices or v in exclude_indices: continue
        if is_terminal_methyl(mol, u) or is_terminal_methyl(mol, v): continue

        atom_u, atom_v = mol.GetAtomWithIdx(u), mol.GetAtomWithIdx(v)
        
        # 아마이드 결합(Amide bond, 펩타이드 결합)은 강직하므로 회전 자유도에서 제외
        is_amide = False
        if {atom_u.GetAtomicNum(), atom_v.GetAtomicNum()} == {6, 7}:
            c_atom = atom_u if atom_u.GetAtomicNum() == 6 else atom_v
            if any(nbr.GetAtomicNum() == 8 and mol.GetBondBetweenAtoms(c_atom.GetIdx(), nbr.GetIdx()).GetBondTypeAsDouble() == 2.0 for nbr in c_atom.GetNeighbors()):
                is_amide = True
        if is_amide: continue
        
        bond_tuple = tuple(sorted((u, v)))
        if bond_tuple not in bb_bonds_set or bond_tuple in processed_bonds: 
            continue
        processed_bonds.add(bond_tuple)

        idx_u, idx_v = bb_path.index(u), bb_path.index(v)
        if idx_u > idx_v: 
            u, v, idx_u, idx_v = v, u, idx_v, idx_u
        
        # 이면각을 정의할 양 끝단 참조 원자(a, d) 설정
        a = bb_path[idx_u - 1] if idx_u > 0 else _pick_end_neighbor(mol, u, {v})
        d = bb_path[idx_v + 1] if idx_v < len(bb_path) - 1 else _pick_end_neighbor(mol, v, {u})
        
        if a is not None and d is not None: 
            dofs.append((a, u, v, d))

    # --- B. 삼중 결합 (Alkyne Linker) 처리 ---
    for match in triple_matches:
        u, c1, c2, v = match[0], match[1], match[2], match[3]
        
        if u not in bb_path or v not in bb_path: 
            continue
            
        pseudo_bond = tuple(sorted((u, v)))
        if pseudo_bond in processed_bonds:
            continue
        processed_bonds.add(pseudo_bond)
        
        # 방향 정렬: u를 반드시 코어(Core) 원자로, v를 캡(Cap) 원자로 정렬
        u_core = u in core_indices
        v_core = v in core_indices
        if not u_core and v_core:
            u, v, c1, c2 = v, u, c2, c1
        elif not u_core and not v_core:
            if bb_path.index(u) > bb_path.index(v):
                u, v, c1, c2 = v, u, c2, c1
                
        # [핵심 1] Core 측 Reference (a) 선택
        a = None
        for nbr in get_neighbors(mol, u):
            if nbr in bb_path and nbr != c1:
                a = nbr
                break
        if a is None:
            a = _pick_end_neighbor(mol, u, exclude_indices={c1})
            
        # [핵심 2] Cap 측 Reference (d) 선택: 변경된 위상 앵커 배열에 맞춰 C_ortho 위치 참조
        d = None
        nuc_anchor = topo_info.get('nuc_anchor_indices', [])
        elec_anchor = topo_info.get('elec_anchor_indices', [])
        
        if nuc_anchor and v == nuc_anchor[0]:
            d = nuc_anchor[5]  # Nuc 배열: 5번 인덱스가 C_ortho_core
        elif elec_anchor and v == elec_anchor[2]:
            d = elec_anchor[3] # Elec 배열: 3번 인덱스가 C_ortho_core
        else:
            d = _pick_end_neighbor(mol, v, exclude_indices={c2})
                
        if a is not None and d is not None:
            dofs.append((a, u, v, d))
            
    return dofs

def identify_backbone_dofs(mol: Chem.Mol, dofs: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
    """추출된 회전 자유도를 'bb_1', 'bb_2' 식별자를 부여하여 딕셔너리로 반환합니다."""
    topo_info = analyze_residue_topology(mol)
    mapping = {'type': topo_info['monomer_type']}
    for i, dof in enumerate(dofs):
        mapping[f'bb_{i+1}'] = dof
    return mapping

def get_backbone_atoms(mol: Chem.Mol) -> Dict[str, Any]:
    """외부 모듈에서 백본 자유도를 즉시 요청할 때 사용하는 래퍼 함수입니다."""
    dofs = get_dofs(mol, set())
    return identify_backbone_dofs(mol, dofs)


# ==============================================================================
# [Section 4] 데이터 취합 (Data Aggregation)
# ==============================================================================

def build_parameter_dict(mol: Chem.Mol, full_smiles: str, topo_info: Dict[str, Any], dofs: List, dof_map: Dict) -> Dict[str, Any]:
    """
    최적화된 3D 분자 구조(mol)와 사전 분석된 모든 위상 정보를 하나로 병합하여,
    시뮬레이션 구동 및 직렬화(Serialization)에 사용될 최종 파라미터 구조체를 반환합니다.
    """
    try: 
        AllChem.ComputeGasteigerCharges(mol)
    except Exception as e: 
        # RDKit에서 특정 비표준 NCAA의 전하 계산을 실패할 수 있으므로 패스합니다.
        pass
    
    return {
        'residue_smiles': full_smiles,
        'monomer_type': topo_info.get('monomer_type', 'UNKNOWN'),
        'atoms': [a.GetAtomicNum() for a in mol.GetAtoms()],
        'bonds': [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()],
        # GasteigerCharge 정보가 없으면 0.0으로 기본값 할당
        'charges': [float(a.GetProp('_GasteigerCharge')) if a.HasProp('_GasteigerCharge') else 0.0 for a in mol.GetAtoms()],
        'radii': [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()],
        'dofs': dofs,
        'dof_map': dof_map,
        'residue_indices': topo_info.get('residue_indices', []),
        'n_term_indices': topo_info.get('n_term_indices', []),
        'c_term_indices': topo_info.get('c_term_indices', []),
        'lower_connect_indices': topo_info.get('lower_connect_indices', []),
        'upper_connect_indices': topo_info.get('upper_connect_indices', []),
        'is_capped': topo_info.get('is_capped', False)
    }


# ==============================================================================
# [Section 5] 일반 유틸리티 (General Utilities)
# ==============================================================================

def get_neighbors(mol: Chem.Mol, atom_idx: int) -> List[int]:
    """특정 원자에 직접 결합된 이웃 원자들의 인덱스 리스트를 반환합니다."""
    return [a.GetIdx() for a in mol.GetAtomWithIdx(atom_idx).GetNeighbors()]

def build_topological_mask(residues: List[str], residue_params_dict: Dict[str, Any]) -> np.ndarray:
    """폴리머 조립 시, 인접한 원자 간의 불필요한 Clash(충돌) 계산을 무시하기 위한 그래프 거리 기반 마스크를 생성합니다."""
    counts = [len(residue_params_dict[res]['atoms']) for res in residues]
    total_atoms = sum(counts)
    G = nx.Graph()
    G.add_nodes_from(range(total_atoms))
    offsets = [0] + list(np.cumsum(counts[:-1]))
    
    for i, res in enumerate(residues):
        for u, v in residue_params_dict[res]['bonds']: G.add_edge(u + offsets[i], v + offsets[i])
            
    for i in range(len(residues) - 1):
        u_idx = residue_params_dict[residues[i]]['upper_connect_indices'][0] + offsets[i]
        v_idx = residue_params_dict[residues[i+1]]['lower_connect_indices'][0] + offsets[i+1] 
        G.add_edge(u_idx, v_idx)

    path_lengths = dict(nx.all_pairs_shortest_path_length(G))
    mask = np.zeros((total_atoms, total_atoms), dtype=bool)
    
    # 그래프 최단 경로 길이가 4 이상인 원자 쌍만 
    # VdW 상호작용(충돌) 계산 대상에 포함하도록 마스킹 처리 (True = 무시)
    for i in range(total_atoms):
        for j in range(i + 1, total_atoms):
            if path_lengths.get(i, {}).get(j, 999) >= 4:
                mask[i, j] = mask[j, i] = True
    return mask

def check_clashes(numbers: np.ndarray, positions: np.ndarray, mask: np.ndarray, mode: str = 'strict') -> bool:
    """거리 기반 충돌(Steric Clash) 발생 여부를 확인합니다."""
    natoms = len(numbers)
    if mode == 'strict': 
        hh_lim, strict_lim, loose_lim = 1.0**2, 0.8**2, 0.5**2
    else: 
        hh_lim, strict_lim, loose_lim = 0.4**2, 0.4**2, 0.4**2
    
    for i in range(natoms):
        for j in range(i + 1, natoms):
            dist_sq = np.sum((positions[i] - positions[j]) ** 2)
            if numbers[i] == 1 and numbers[j] == 1:
                if dist_sq < hh_lim: return True
            else:
                limit = strict_lim if mask[i, j] else loose_lim
                if dist_sq < limit: return True
    return False

def load_residue_params(filepath: str) -> Dict[str, Any]:
    """동적으로 residue_params.py 파일을 임포트하여 딕셔너리를 로드합니다."""
    import importlib.util
    import os
    if not os.path.exists(filepath): 
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        spec = importlib.util.spec_from_file_location("residue_params_mod", filepath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.residue_params
    except Exception as e:
        raise RuntimeError(f"Failed to load residue params: {e}")
    

# ==============================================================================
# [Section 6] Debug & Verification (디버그용 실행 블록)
# ==============================================================================
if __name__ == "__main__":
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 사용자가 제공한 복잡한 비표준 아미노산 SMILES (삼중결합 및 피리딘 포함)
    test_smiles = "CC(C)(C)C1=CC(C#CC2=CC(C#CC)=CN=C2)=C(NC3=C4C=CC5=C3NC6=C5C=C(C(C)(C)C)C=C6C#CC7=CN=CC(C#CC)=C7)C4=C1"
    
    mol = Chem.MolFromSmiles(test_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    exclude_indices = set()

    print("=" * 60)
    print("[1] Topology Analysis (위상 구조 분석)")
    print("=" * 60)
    try:
        topo_info = analyze_residue_topology(mol)
        print(f"Monomer Type       : {topo_info.get('monomer_type', 'N/A')}")
        print(f"Nuc Anchor (Entry) : {topo_info.get('nuc_anchor_indices', 'N/A')}")
        print(f"Elec Anchor (Exit) : {topo_info.get('elec_anchor_indices', 'N/A')}")
    except Exception as e:
        print(f"❌ Topology 분석 중 오류 발생: {e}")

    print("\n" + "=" * 60)
    print("[2] Degrees of Freedom (자유도) 추출 검증")
    print("=" * 60)
    try:
        dofs = get_dofs(mol, exclude_indices)
        print(f"✅ 총 {len(dofs)}개의 회전 가능한 결합(자유도)이 추출되었습니다.\n")
        
        for i, (a, u, v, d) in enumerate(dofs):
            atom_a = mol.GetAtomWithIdx(a).GetSymbol()
            atom_u = mol.GetAtomWithIdx(u).GetSymbol()
            atom_v = mol.GetAtomWithIdx(v).GetSymbol()
            atom_d = mol.GetAtomWithIdx(d).GetSymbol()
            
            bond = mol.GetBondBetweenAtoms(u, v)
            bond_type = bond.GetBondTypeAsDouble()
            
            if bond_type == 3.0:
                bond_str = "삼중결합 (Alkyne Linker)"
                link_visual = "≡"
            else:
                bond_str = "단일결합 (Single Bond)"
                link_visual = "-"
                
            print(f"🔹 DOF #{i+1} [{bond_str}]")
            print(f"  - 원자 인덱스 : (a:{a:<3} | u:{u:<3} | v:{v:<3} | d:{d:<3})")
            print(f"  - 원자 기호   : ({atom_a}) - ({atom_u}) {link_visual} ({atom_v}) - ({atom_d})")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ DOF 추출 중 오류 발생: {e}")