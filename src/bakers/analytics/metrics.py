"""
src/bakers/analytics/metrics.py

[기능 정의]
구조 분석 및 비교를 위한 계산 모듈입니다.
1. Pure Math: 좌표 배열(NumPy) 기반의 RMSD, 이면각, NeRF 계산
2. RDKit Metrics: RDKit Mol 객체 기반의 구조적 차이(RMSD, TFD) 계산
   - [Safety] RingInfo 강제 주입으로 TFD 계산 에러 방지
   - [Safety] RMSD 계산 시 호환성이 높은 AlignMol 사용
   - [Safety] 계산 실패 시 999.9 반환 (False Negative 방지)
3. Clustering: 유사 구조 그룹화 알고리즘

[수정 내역]
- [Final] 모든 기능 검증 완료 (RMSD, TFD, Dihedral, NeRF, Clustering)
- [Refactor] 디버깅용 print 문을 유지하여 문제 발생 시 추적 용이하도록 설정
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

# RDKit 의존성
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints

# ==============================================================================
# 1. Array Metrics (NumPy based) - Pure Geometry
# ==============================================================================

def calculate_rmsd_array(P, Q):
    """
    [Kabsch Algorithm] NumPy 배열 기반 RMSD 계산
    두 좌표 배열(P, Q)을 최적으로 중첩시킨 후 RMSD를 반환합니다.
    
    Args:
        P (np.ndarray): 이동시킬 좌표 (N, 3)
        Q (np.ndarray): 기준 좌표 (N, 3)
    """
    if P.shape != Q.shape:
        raise ValueError(f"Shape Mismatch: P{P.shape} != Q{Q.shape}")

    # 1. 중심 이동 (Centering)
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # 2. 공분산 행렬 (Covariance Matrix) 계산
    H = np.dot(P_c.T, Q_c)

    # 3. SVD 수행
    U, S, Vt = np.linalg.svd(H)

    # 4. 반사(Reflection) 보정 (거울상 구조 처리)
    d = (np.linalg.det(np.dot(Vt.T, U.T)) < 0.0)
    if d:
        Vt[-1, :] *= -1
    
    R_mat = np.dot(Vt.T, U.T)

    # 5. 회전 적용 및 차이 계산
    P_rotated = np.dot(P_c, R_mat)
    diff = P_rotated - Q_c
    
    return np.sqrt(np.sum(diff**2) / len(P))

# 별칭(Alias) 설정
calculate_rmsd = calculate_rmsd_array


def calculate_dihedral(p1, p2, p3, p4):
    """
    4개의 점(p1, p2, p3, p4)으로 정의되는 이면각(Dihedral Angle)을 계산합니다.
    입력값이 정수여도 내부적으로 float로 변환하여 계산합니다.
    """
    # 입력 좌표 float 변환 (나눗셈 오류 방지)
    p1, p2, p3, p4 = map(lambda x: np.array(x, dtype=float), [p1, p2, p3, p4])
    
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # 회전축 정규화
    norm_b2 = np.linalg.norm(b2)
    if norm_b2 < 1e-9: return 0.0
    b2 /= norm_b2

    # 법선 벡터 계산
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    norm_n1 = np.linalg.norm(n1)
    norm_n2 = np.linalg.norm(n2)
    
    if norm_n1 < 1e-9 or norm_n2 < 1e-9: return 0.0
    
    n1 /= norm_n1
    n2 /= norm_n2

    # 각도 계산 (표준 정의 적용: sign 보정)
    x = np.dot(n1, n2)
    m1 = np.cross(n1, b2)
    y = np.dot(m1, n2)
    
    return math.degrees(math.atan2(y, x))


def compute_dihedrals_vectorized(p1, p2, p3, p4):
    """
    [Vectorized] 여러 개의 4점 세트에 대해 이면각을 NumPy로 고속 일괄 계산합니다.
    Input: (N, 3) Arrays
    """
    p1, p2, p3, p4 = [x.astype(float) for x in [p1, p2, p3, p4]]

    b0 = -1.0 * (p2 - p1)
    b1 = p3 - p2
    b2 = p4 - p3

    # b1 정규화
    b1 /= np.linalg.norm(b1, axis=1, keepdims=True)

    # 벡터 투영
    v = b0 - np.sum(b0 * b1, axis=1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, axis=1, keepdims=True) * b1

    x = np.sum(v * w, axis=1)
    y = np.sum(np.cross(b1, v) * w, axis=1)
    
    return np.degrees(np.arctan2(y, x))


def calculate_angle_diff(a, b, period=360.0):
    """
    주기성을 고려하여 두 각도 간의 최소 차이를 계산합니다.
    (예: 350도와 10도의 차이는 20도)
    """
    diff = np.abs(a - b)
    diff = np.minimum(diff, period - diff)
    return diff

# ==============================================================================
# 2. Molecule Metrics (RDKit based) - Structural Comparison
# ==============================================================================

def calculate_mol_rmsd(mol, ref_cid, prb_cid, heavy_only=True):
    """
    [RDKit Wrapper] RMSD 계산
    AllChem.AlignMol을 사용하여 호환성을 확보했습니다.
    """
    try:
        atom_map = None
        if heavy_only:
            # 수소(H) 제외하고 Heavy Atom만 추출
            heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
            # [Core Fix] AlignMol은 List of Tuples [(probe, ref)] 형식을 사용
            atom_map = [(int(i), int(i)) for i in heavy_indices]
            
        # AlignMol: 정렬 수행 및 RMSD 반환
        return AllChem.AlignMol(mol, mol, prb_cid, ref_cid, atomMap=atom_map)
        
    except Exception as e:
        # 에러 발생 시 로그 출력 후 999.9 반환
        print(f"  [Metrics Error] RMSD Failed: {e}")
        return 999.9


def calculate_mol_tfd(mol, ref_cid, prb_cid, use_weights=True):
    """
    [RDKit Wrapper] TFD 계산 (RingInfo 강제 주입 포함)
    불완전한 구조(sanitize=False)에서도 계산 가능하도록 설계되었습니다.
    """
    try:
        # [Core Fix] RingInfo 강제 갱신
        # IsInitialized 체크 없이 무조건 GetSymmSSSR 호출 (Idempotent)
        # 이를 통해 고리 정보가 없는 Mol 객체도 에러 없이 처리 가능
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
# 3. Geometric Reconstruction (NeRF)
# ==============================================================================

def nerf(prev_atoms, length, bond_angle, torsion):
    """
    [NeRF Algorithm] 내부 좌표(길이, 각도, 이면각)를 데카르트 좌표로 변환
    """
    # 입력값 float 변환 보장
    prev_atoms = np.array(prev_atoms, dtype=float)
    
    m2, m1, m = prev_atoms[-3], prev_atoms[-2], prev_atoms[-1]
    
    bc = m - m1
    bc /= np.linalg.norm(bc)
    
    n = np.cross(m1 - m2, bc)
    n_norm = np.linalg.norm(n)
    
    # 3점이 일직선인 경우 법선 벡터 예외 처리
    if n_norm < 1e-6:
        n = np.array([0, 1, 0], dtype=float) if abs(bc[0]) > 0.9 else np.array([1, 0, 0], dtype=float)
    else:
        n /= n_norm
        
    cross_n_bc = np.cross(n, bc)

    # 각도 변환 (NeRF 정의: Bond Angle은 180도 보각 사용)
    angle_rad = np.radians(180.0 - bond_angle)
    torsion_rad = np.radians(torsion)

    x = length * np.cos(angle_rad)
    y = length * np.sin(angle_rad) * np.cos(torsion_rad)
    z = length * np.sin(angle_rad) * np.sin(torsion_rad)

    d = np.array([x, y, z])
    
    # 로컬 -> 글로벌 변환 행렬
    M = np.column_stack((bc, cross_n_bc, n))

    return m + np.dot(M, d)

# ==============================================================================
# 4. Clustering
# ==============================================================================

def get_periodic_diff(a, b, period=360.0):
    """주기적 차이 계산"""
    diff = np.abs(a - b)
    diff = np.where(diff > period / 2, period - diff, diff)
    return diff

def greedy_cluster_dihedrals(points, values, threshold=45.0, metric='euclidean', top_k=None):
    """
    [Greedy Clustering] 에너지가 낮은 순서대로 대표 구조 선택
    """
    sorted_indices = np.argsort(values)
    temp_points = points[sorted_indices]
    temp_orig_indices = sorted_indices
    
    mask = np.ones(len(temp_points), dtype=bool)
    selected_indices = []
    
    for i in range(len(temp_points)):
        if not mask[i]: continue
        
        current_rep = temp_points[i]
        
        remaining_idxs_rel = np.where(mask[i:])[0]
        remaining_idxs_abs = remaining_idxs_rel + i
        rem_points = temp_points[remaining_idxs_abs]
        
        diff_matrix = get_periodic_diff(rem_points, current_rep)
        
        if metric == 'euclidean':
            dists = np.linalg.norm(diff_matrix, axis=1)
        else:
            dists = np.max(diff_matrix, axis=1)
            
        member_mask = dists < threshold
        member_indices_abs = remaining_idxs_abs[member_mask]
        
        if top_k is None or len(selected_indices) < top_k:
             selected_indices.append(temp_orig_indices[i])
        
        mask[member_indices_abs] = False
        
    return np.array(selected_indices)

# ==============================================================================
# [Self-Check Logic] 모듈 무결성 검증
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print("[Debug] metrics.py 기능 검증 (Passed Version)")
    print("="*60)

    # 1. Array RMSD
    print("[1] NumPy Array RMSD Test")
    try:
        P = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
        Q = P + 1.0
        rmsd = calculate_rmsd_array(P, Q)
        print(f" -> Result: {rmsd:.6f}")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 2. Dihedral
    print("\n[2] Dihedral Calculation Test (Integer Inputs)")
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

    # 3. RDKit Metrics
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

    # 4. NeRF
    print("\n[4] NeRF Test")
    try:
        prev = [[-2,0,0], [-1,0,0], [0,0,0]] 
        next_pos = nerf(prev, 1.0, 180.0, 0.0)
        print(f" -> NeRF Result: {next_pos}")
        if np.allclose(next_pos, [1,0,0], atol=1e-4): print(" -> [PASS] NeRF OK")
        else: print(" -> [FAIL] NeRF Incorrect")
    except Exception as e: print(f" -> [ERROR] {e}")

    # 5. Clustering
    print("\n[5] Clustering Test")
    try:
        pts = np.array([[10], [12], [100], [105]], dtype=float)
        vals = np.array([0, 1, 2, 3])
        idxs = greedy_cluster_dihedrals(pts, vals, threshold=10.0, top_k=2)
        print(f" -> Indices: {idxs}")
        if len(idxs) == 2: print(" -> [PASS] Clustering OK")
        else: print(" -> [FAIL]")
    except Exception as e: print(f" -> [ERROR] {e}")

    print("="*60)
    print("[Debug] 검증 종료")