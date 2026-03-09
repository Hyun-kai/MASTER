"""
src/bakers/analytics/metrics.py

[목적 및 설명]
3D 분자 구조의 기하학적 지표(RMSD, Dihedral, TFD)를 계산하고, 내부 좌표(Internal Coordinates)를
데카르트 좌표(Cartesian)로 변환하거나 유사한 구조들을 군집화(Clustering)하는 핵심 분석/수학 모듈입니다.

[아키텍처 (Top-Down 흐름)]
1. Pure Geometry: 배열 기반 각도 및 이면각 연산 (NumPy)
2. Coordinate Transformation: Kabsch 알고리즘(RMSD) 및 NeRF (좌표계 변환)
3. RDKit Molecular Metrics: RDKit Mol 객체를 활용한 화학적 구조 비교 (Heavy Atom RMSD, TFD)
4. Ensemble Clustering: 에너지 기반 Greedy 클러스터링 알고리즘

[최적화 및 수정 내역]
- DRY 원칙 적용: 중복되던 각도 차이 계산 함수 통합 (calculate_periodic_diff)
- 수학적 안전성(Mathematical Safety) 확보: 벡터 정규화 시 Divide-by-zero 방지 로직 추가
- 클러스터링 속도 최적화: 복잡한 인덱스 추적 대신 Numpy 순수 Boolean Masking 방식으로 리팩토링
"""

import math
import numpy as np
from typing import Union, List, Optional

# RDKit 의존성 임포트
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints


# ==============================================================================
# [Section 1] 순수 기하학 연산 (Pure Geometry & Angles)
# ==============================================================================

def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    4개의 3D 좌표점(p1, p2, p3, p4)이 이루는 이면각(Dihedral Angle)을 도(Degree) 단위로 계산합니다.
    
    Args:
        p1, p2, p3, p4 (array-like): 연속된 4개 원자의 3D 좌표
    Returns:
        float: 이면각 (-180.0 ~ 180.0)
    """
    # 나눗셈 오류 방지를 위해 입력값을 명시적 float로 캐스팅
    p1, p2, p3, p4 = map(lambda x: np.array(x, dtype=float), [p1, p2, p3, p4])
    
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # 중앙 회전축(b2) 정규화
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-9: return 0.0
    b2 /= norm_b2

    # 두 평면의 법선 벡터(Normal Vector) 계산
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    
    if norm_n1 < 1e-9 or norm_n2 < 1e-9: return 0.0
    
    n1 /= norm_n1
    n2 /= norm_n2

    # 투영된 벡터 간의 각도 계산 (부호 보정을 위해 atan2 사용)
    x = np.dot(n1, n2)
    m1 = np.cross(n1, b2)
    y = np.dot(m1, n2)
    
    return math.degrees(math.atan2(y, x))


def compute_dihedrals_vectorized(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """
    [Vectorized] 여러 개의 4점 세트에 대해 이면각을 NumPy로 초고속 일괄 계산합니다.
    대규모 폴리머 궤적(Trajectory) 분석 시 for 루프의 병목을 없애기 위해 사용됩니다.
    
    Args:
        p1, p2, p3, p4 (np.ndarray): (N, 3) 형태의 좌표 배열
    Returns:
        np.ndarray: (N,) 형태의 이면각 배열 (-180 ~ 180)
    """
    p1, p2, p3, p4 = [x.astype(float) for x in [p1, p2, p3, p4]]

    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # [Safety] b1 축 정규화 시 0 나누기 방지
    norm_b1 = np.linalg.norm(b1, axis=1, keepdims=True)
    norm_b1[norm_b1 == 0] = 1e-8
    b1 /= norm_b1

    # 법선 평면(b1에 수직인 평면)으로 벡터 투영
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1

    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    
    return np.degrees(np.arctan2(y, x))


def calculate_periodic_diff(a: Union[float, np.ndarray], b: Union[float, np.ndarray], period: float = 360.0) -> Union[float, np.ndarray]:
    """
    원형(Circular) 주기성을 고려하여 두 각도 간의 최소 거리(차이)를 계산합니다.
    (예: 350도와 10도의 차이는 340도가 아니라 20도로 반환)
    * 참고: 기존의 calculate_angle_diff와 get_periodic_diff를 하나로 통합한 함수입니다.
    """
    diff = np.abs(a - b)
    diff = np.minimum(diff, period - diff)
    return diff

# 하위 호환성을 위한 별칭(Alias) 설정
calculate_angle_diff = calculate_periodic_diff
get_periodic_diff = calculate_periodic_diff


# ==============================================================================
# [Section 2] 좌표계 변환 및 재구성 (Coordinate Transformation)
# ==============================================================================

def nerf(prev_atoms: np.ndarray, length: float, bond_angle: float, torsion: float) -> np.ndarray:
    """
    [NeRF Algorithm] (Natural Extension Reference Frame)
    내부 좌표(결합 길이, 결합 각도, 이면각) 정보를 바탕으로, 이전 3개 원자의 3D 좌표에서 
    다음 원자의 새로운 데카르트 3D 좌표를 기하학적으로 연역(Reconstruction)합니다.
    
    Args:
        prev_atoms (np.ndarray): 이전 3개 원자의 좌표 배열 (3, 3)
        length (float): M 원자와 새로 생성될 원자 간의 결합 길이
        bond_angle (float): M1 - M - New 간의 결합 각도 (도 단위)
        torsion (float): M2 - M1 - M - New 간의 이면각 (도 단위)
    Returns:
        np.ndarray: 새롭게 계산된 원자의 (3,) 좌표
    """
    prev_atoms = np.array(prev_atoms, dtype=float)
    m2, m1, m = prev_atoms[-3], prev_atoms[-2], prev_atoms[-1]
    
    # 1. 로컬 좌표계 축(X축 역할) 정의
    bc = m - m1
    bc /= np.linalg.norm(bc)
    
    # 2. 로컬 좌표계 평면(Z축 역할) 정의
    n = np.cross(m1 - m2, bc)
    n_norm = np.linalg.norm(n)
    
    # [안전 장치] 3점이 일직선(Collinear)에 가까울 경우 영점 나누기 방지
    if n_norm < 1e-6:
        n = np.array([0, 1, 0], dtype=float) if abs(bc[0]) > 0.9 else np.array([1, 0, 0], dtype=float)
    else:
        n /= n_norm
        
    # Y축 역할
    cross_n_bc = np.cross(n, bc)

    # 3. 구면 좌표계(Spherical) -> 데카르트 좌표계 전환
    # NeRF 정의에 따라 Bond Angle은 180도에 대한 보각을 사용합니다.
    angle_rad = np.radians(180.0 - bond_angle)
    torsion_rad = np.radians(torsion)

    x = length * np.cos(angle_rad)
    y = length * np.sin(angle_rad) * np.cos(torsion_rad)
    z = length * np.sin(angle_rad) * np.sin(torsion_rad)

    d = np.array([x, y, z])
    
    # 4. 로컬 좌표계를 글로벌 데카르트 좌표계로 회전 변환
    M = np.column_stack((bc, cross_n_bc, n))

    return m + np.dot(M, d)


def calculate_rmsd_array(P: np.ndarray, Q: np.ndarray) -> float:
    """
    [Kabsch Algorithm] NumPy 좌표 배열 기반 RMSD 계산 (Pure Math)
    이동시킬 좌표(P)를 기준 좌표(Q)에 최적으로 겹치도록(Translation & Rotation) 정렬한 뒤
    두 구조 간의 RMSD 값을 반환합니다.
    
    Args:
        P (np.ndarray): 이동시킬 좌표 (N, 3)
        Q (np.ndarray): 기준 좌표 (N, 3)
    Returns:
        float: 최소화된 RMSD 값
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape Mismatch: P{P.shape} != Q{Q.shape}")

    # 1. 질량 중심 이동 (Centering)
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # 2. 공분산 행렬 (Covariance Matrix) 계산
    H = np.dot(P_c.T, Q_c)

    # 3. 특이값 분해 (SVD, Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(H)

    # 4. 거울상 반사(Reflection) 보정 
    # 회전 행렬의 행렬식이 음수면 분자가 뒤집힌 것이므로 부호를 보정합니다.
    d = (np.linalg.det(np.dot(Vt.T, U.T)) < 0.0)
    if d:
        Vt[-1, :] *= -1
    
    R_mat = np.dot(Vt.T, U.T)

    # 5. 최적 회전 적용 및 차이(RMSD) 계산
    P_rotated = np.dot(P_c, R_mat)
    diff = P_rotated - Q_c
    
    return np.sqrt(np.sum(diff**2) / len(P))

# 하위 호환성을 위한 별칭(Alias) 설정
calculate_rmsd = calculate_rmsd_array


# ==============================================================================
# [Section 3] 화학적 구조 지표 (RDKit Molecular Metrics)
# ==============================================================================

def calculate_mol_rmsd(mol: Chem.Mol, ref_cid: int, prb_cid: int, heavy_only: bool = True) -> float:
    """
    [RDKit Wrapper] 동일한 분자의 두 컨포머 간의 RMSD를 계산합니다.
    (배열 기반이 아닌 RDKit 내부 정렬 알고리즘 AllChem.AlignMol 사용)
    
    Args:
        mol (Chem.Mol): 비교할 앙상블을 보유한 RDKit 분자 객체
        ref_cid (int): 기준 컨포머 ID
        prb_cid (int): 이동시킬 컨포머 ID
        heavy_only (bool): True일 경우 수소(H) 원자를 제외하고 무거운 원자(Heavy Atoms)만 비교
    Returns:
        float: RMSD 값 (계산 실패 시 999.9 반환으로 False Negative 방지)
    """
    try:
        atom_map = None
        if heavy_only:
            # 수소(H)를 제외하고 무거운 원자만 추출
            heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            # RDKit AlignMol은 원자 매핑을 [(probe_idx, ref_idx), ...] 형태의 튜플 리스트로 요구합니다.
            # 이 함수는 동일 분자(mol) 내의 컨포머 비교이므로 인덱스가 1:1 대응됩니다.
            atom_map = [(int(i), int(i)) for i in heavy_indices]
            
        return AllChem.AlignMol(mol, mol, prb_cid, ref_cid, atomMap=atom_map)
        
    except Exception as e:
        print(f"  [Metrics Error] Mol RMSD Failed: {e}")
        return 999.9


def calculate_mol_tfd(mol: Chem.Mol, ref_cid: int, prb_cid: int, use_weights: bool = True) -> float:
    """
    [RDKit Wrapper] Torsion Fingerprint Deviation (TFD) 계산
    RMSD가 전체적인 형상을 비교한다면, TFD는 내부 이면각의 차이를 중점적으로 비교합니다.
    비순환/순환 구조(Ring)에 독립적인 객관적 지표를 제공합니다.
    """
    try:
        # [안전 장치] TFD 계산 시 고리(Ring) 정보가 필수적입니다.
        # 분자 객체가 최적화 도중 고리 정보를 잃었을 경우를 대비하여 대칭 기반 고리 검색을 강제 주입합니다.
        try:
            Chem.GetSymmSSSR(mol)
        except Exception:
            pass

        tfd = TorsionFingerprints.GetTFDBetweenMolecules(
            mol, mol, 
            confId1=prb_cid, 
            confId2=ref_cid,
            useWeights=use_weights
        )
        return tfd
    except Exception as e:
        print(f"  [Metrics Error] TFD Failed: {e}")
        return 999.9


# ==============================================================================
# [Section 4] 앙상블 클러스터링 알고리즘 (Ensemble Clustering)
# ==============================================================================

def greedy_cluster_dihedrals(points: np.ndarray, values: np.ndarray, threshold: float = 45.0, metric: str = 'euclidean', top_k: Optional[int] = None) -> np.ndarray:
    """
    [Greedy Pruning Algorithm - 성능 최적화 버전]
    에너지(values)가 가장 낮은 구조부터 우선적으로 대표값(Centroid)으로 선정하고,
    선정된 구조와 임계값(threshold) 이내로 유사한 기하 구조(points)들을 일괄 제거(Pruning)합니다.
    복잡한 인덱스 추적을 제거하고, 전체 Boolean 마스크 업데이트 방식으로 속도를 향상시켰습니다.
    
    Args:
        points (np.ndarray): 컨포머들의 기하학적 특성 배열 (예: 이면각 배열 집합)
        values (np.ndarray): 컨포머들의 에너지 배열
        threshold (float): 군집으로 묶어 제거할 거리 임계값
        metric (str): 'euclidean' (L2 노름) 또는 'max' (최대 각도 차이)
        top_k (int, optional): 최종적으로 얻고 싶은 대표 군집의 최대 개수
    Returns:
        np.ndarray: 선택된 최적 컨포머들의 원본 인덱스 배열
    """
    # 1. 에너지가 낮은(가장 안정적인) 순서로 정렬 (오름차순)
    sorted_indices = np.argsort(values)
    sorted_points = points[sorted_indices]
    
    selected_orig_indices = []
    # 모든 데이터의 생존 여부를 나타내는 1D 불리언 마스크
    active_mask = np.ones(len(sorted_points), dtype=bool)
    
    for i in range(len(sorted_points)):
        if not active_mask[i]: 
            continue  # 이미 이전 대표값과 유사하여 군집화된 경우 스킵
            
        # 2. 새로운 대표값으로 등록
        selected_orig_indices.append(sorted_indices[i])
        
        # 목표 개수에 도달하면 조기 종료
        if top_k is not None and len(selected_orig_indices) >= top_k:
            break
            
        # 3. 벡터화된 거리 계산 (현재 대표값 vs 전체 배열)
        # 이미 죽은(False) 인덱스까지 계산하는 약간의 오버헤드가 있으나, 
        # Numpy의 브로드캐스팅(C레벨) 덕분에 슬라이싱/인덱스 매핑 로직보다 압도적으로 빠르고 코드가 깔끔해집니다.
        current_rep = sorted_points[i]
        diff_matrix = calculate_periodic_diff(sorted_points, current_rep)
        
        if metric == 'euclidean':
            dists = np.linalg.norm(diff_matrix, axis=1)
        else:
            dists = np.max(diff_matrix, axis=1)
            
        # 4. 임계값(Threshold) 이내의 유사 구조들을 마스크에서 제거(False)
        neighbors_mask = dists < threshold
        active_mask[neighbors_mask] = False
        
    return np.array(selected_orig_indices)


# ==============================================================================
# [Debug & Self-Check] 모듈 무결성 검증 파이프라인
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("[Debug] metrics.py 기능 검증 (Optimized Version)")
    print("="*60)

    # 1. Array RMSD 검증
    print("[1] NumPy Array RMSD Test")
    try:
        P = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
        Q = P + 1.0 # 1.0만큼 평행 이동
        rmsd = calculate_rmsd_array(P, Q)
        print(f" -> Result: {rmsd:.6f} (Expected: 0.000000)")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 2. 이면각(Dihedral) 계산 검증
    print("\n[2] Dihedral Calculation Test")
    try:
        p1 = [1, 0, 0]
        p2 = [0, 0, 0]
        p3 = [0, 1, 0]
        p4 = [0, 1, 1]
        
        angle = calculate_dihedral(p1, p2, p3, p4)
        print(f" -> Result: {angle:.2f} (Expected: 90.00)")
        if abs(angle - 90.0) < 1e-4: print(" -> [PASS]")
        else: print(" -> [FAIL]")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 3. RDKit 기반 메트릭 검증
    print("\n[3] RDKit Mol Metrics Test")
    try:
        mol = Chem.MolFromSmiles("C1CCCCC1")
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDG()
        ps.randomSeed = 0xF00D
        AllChem.EmbedMultipleConfs(mol, numConfs=2, params=ps)
        
        if mol.GetNumConformers() >= 2:
            val_rmsd = calculate_mol_rmsd(mol, 0, 1)
            print(f" -> Mol RMSD: {val_rmsd:.4f}")
            if val_rmsd != 999.9: print(" -> [PASS] RMSD OK")
            else: print(" -> [FAIL] RMSD Failed")

            val_tfd = calculate_mol_tfd(mol, 0, 1)
            print(f" -> Mol TFD:  {val_tfd:.4f}")
            if val_tfd != 999.9: print(" -> [PASS] TFD OK")
            else: print(" -> [FAIL] TFD Failed")
        else:
            print(" -> [SKIP] Conformer generation failed")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 4. NeRF 좌표 재구성 검증
    print("\n[4] NeRF Geometric Reconstruction Test")
    try:
        prev = [[-2,0,0], [-1,0,0], [0,0,0]] 
        # 직진 방향(180도 보각 기준)으로 길이 1.0만큼 확장
        next_pos = nerf(prev, 1.0, 180.0, 0.0)
        print(f" -> NeRF Result: {next_pos}")
        if np.allclose(next_pos, [1,0,0], atol=1e-4): print(" -> [PASS] NeRF OK")
        else: print(" -> [FAIL] NeRF Incorrect")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 5. Greedy 클러스터링 검증
    print("\n[5] Energy-Ordered Greedy Clustering Test")
    try:
        pts = np.array([[10], [12], [100], [105]], dtype=float)
        vals = np.array([0, 1, 2, 3]) # 에너지가 낮은 0, 2번 인덱스가 대표값으로 선택되어야 함
        idxs = greedy_cluster_dihedrals(pts, vals, threshold=10.0, top_k=2)
        print(f" -> Selected Indices: {idxs}")
        if len(idxs) == 2 and list(idxs) == [0, 2]: print(" -> [PASS] Clustering OK")
        else: print(" -> [FAIL]")
    except Exception as e: print(f" -> [ERROR] {e}")

    print("="*60)
    print("[Debug] 검증 종료")