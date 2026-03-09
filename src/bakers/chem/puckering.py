"""
src/bakers/chem/puckering.py

[목적 및 설명]
분자의 3D 컨포머(Conformer)를 생성, 최적화, 그리고 필터링(Clustering)하는 핵심 로직을 담당합니다.
Ring Puckering(고리 유연성)과 하이브리드 샘플링을 통해, 단일 분자가 가질 수 있는 
다양한 안정적 입체 구조(Rotamer/Conformer) 풀(Pool)을 확보합니다.

[주요 파이프라인 흐름]
1. 고리 분석 (Ring Analysis)
2. 초기 구조 임베딩 및 1차 다양성 확보 (Conformer Generation)
3. 역장 기반 구조 최적화 및 에너지 계산 (Optimization & Energy - Multi-threading 적용)
4. 최종 정예 구조 선별 (Clustering via metrics)
"""

import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import TorsionFingerprints
from rdkit.ML.Cluster import Butina

# ==============================================================================
# [Path Setup & Imports] 환경 설정 및 의존성 주입
# ==============================================================================
# 현재 스크립트: src/bakers/chem/puckering.py
current_dir = os.path.dirname(os.path.abspath(__file__))
bakers_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(bakers_dir)

# src 디렉토리를 시스템 경로에 추가
if src_dir not in sys.path:
    sys.path.append(src_dir)

# 거리 계산 로직(RMSD, TFD)을 위임받을 metrics 모듈 임포트
try:
    from bakers.analytics import metrics
except ImportError as e:
    print(f"[Critical Error] Failed to import 'bakers.analytics.metrics'. Check sys.path: {sys.path}")
    raise e


# ==============================================================================
# [Section 1] 고리 탐지 및 분석 (Ring Detection & Analysis)
# ==============================================================================

def detect_rings(mol: Chem.Mol) -> list:
    """
    분자 내에 존재하는 모든 고리(Ring)의 원자 인덱스 묶음을 반환합니다.
    
    Args:
        mol (Chem.Mol): RDKit 분자 객체
    Returns:
        list: 고리를 구성하는 원자 인덱스들의 튜플 리스트 (예: [(0, 1, 2, 3, 4, 5), ...])
    """
    if not mol: 
        return []
    return list(mol.GetRingInfo().AtomRings())


def has_flexible_rings(mol: Chem.Mol, max_ring_size: int = 7) -> bool:
    """
    분자에 입체적 형태 변화(Puckering)가 가능한 유연한 고리(보통 4~7원환)가 존재하는지 확인합니다.
    
    Args:
        mol (Chem.Mol): RDKit 분자 객체
        max_ring_size (int): 유연하다고 판단할 최대 고리 크기 (기본값: 7)
    Returns:
        bool: 유연한 고리가 하나라도 존재하면 True
    """
    rings = detect_rings(mol)
    for ring in rings:
        if 4 <= len(ring) <= max_ring_size:
            return True
    return False


# ==============================================================================
# [Section 2] 컨포머 앙상블 생성 (Conformer Generation)
# ==============================================================================

def embed_with_puckering(mol: Chem.Mol, n_confs: int, prune_thresh: float = 0.5, random_seed: int = -1) -> list:
    """
    (내부 유틸리티) RDKit의 최신 ETKDGv3 알고리즘을 사용하여 초기 3D 구조를 대량으로 임베딩합니다.
    매우 유사한 구조는 생성 단계에서 1차적으로 가지치기(Pruning)됩니다.
    
    Args:
        mol (Chem.Mol): RDKit 분자 객체 (수소가 추가되어 있어야 함)
        n_confs (int): 생성할 초기 구조의 개수
        prune_thresh (float): 가지치기를 위한 RMSD 임계값
        random_seed (int): 재현성을 위한 시드값 (-1은 랜덤)
    Returns:
        list: 생성에 성공한 컨포머 ID(cid) 리스트
    """
    ps = AllChem.ETKDGv3()
    ps.useRandomCoords = True
    ps.pruneRmsThresh = prune_thresh
    if random_seed != -1:
        ps.randomSeed = random_seed
    
    try:
        cids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
        return list(cids)
    except Exception:
        return []


def generate_conformers(mol: Chem.Mol, n_templates: int = 10, n_initial: int = 200, tfd_thresh: float = 0.2) -> Chem.Mol:
    """
    [핵심 생성 함수] 분자의 구조적 다양성을 확보하기 위한 초기 앙상블을 생성합니다.
    대량의 컨포머를 생성한 뒤, TFD(Torsion Fingerprint Deviation) 기반의 Butina 클러스터링을 적용하여 
    형태적으로 뚜렷하게 다른 대표 템플릿(Template)들만 엄선하여 반환합니다.
    
    Args:
        mol (Chem.Mol): RDKit 분자 객체
        n_templates (int): 최종적으로 확보할 대표 구조의 목표 개수
        n_initial (int): ETKDG로 최초에 생성할 대량의 컨포머 수
        tfd_thresh (float): Butina 클러스터링에 사용할 TFD 거리 임계값
    Returns:
        Chem.Mol: 엄선된 컨포머들만 포함된 새로운 분자 객체
    """
    # 1. 초기 컨포머 풀(Pool) 대량 생성
    cids = embed_with_puckering(mol, n_confs=n_initial, random_seed=42)
    
    # 생성 실패 시 Random Seed를 변경하여 재시도 (Fallback)
    if not cids:
        cids = embed_with_puckering(mol, n_confs=n_initial, random_seed=0xf00d)
        if not cids:
            # [버그 수정] 원본 mol을 그대로 리턴하면 외부에서 변경 시 원본이 훼손됨. 깊은 복사본 반환.
            return Chem.Mol(mol)
            
    cids = list(dict.fromkeys(cids))
    diverse_cids = []

    # 2. TFD & Butina 클러스터링을 통한 1차 다양성 추출
    if len(cids) > 1:
        try:
            # TFD(회전 결합 지문) 거리 매트릭스 계산
            tfd_matrix = TorsionFingerprints.GetTFDMatrix(mol)
            
            # Butina 군집화 알고리즘 실행 (isDistData=True: 거리 매트릭스 사용 명시)
            clusters = Butina.ClusterData(tfd_matrix, mol.GetNumConformers(), tfd_thresh, isDistData=True)
            
            # 각 클러스터의 중심(Centroid) 구조를 대표로 선택
            for cluster in clusters:
                centroid_idx = cluster[0]
                centroid_cid = mol.GetConformers()[centroid_idx].GetId()
                diverse_cids.append(centroid_cid)
                
                # 목표 템플릿 개수 도달 시 조기 종료
                if len(diverse_cids) >= n_templates:
                    break
        except Exception:
            # 회전 가능한 결합이 없어서 TFD 계산이 불가능한 경우, 순차적으로 선택
            diverse_cids = cids[:n_templates]
    else:
        diverse_cids = cids

    # 클러스터 수가 목표치보다 적을 경우, 선택되지 않은 나머지 구조들로 강제 보충
    if len(diverse_cids) < n_templates:
        for cid in cids:
            if cid not in diverse_cids:
                diverse_cids.append(cid)
            if len(diverse_cids) >= n_templates:
                break

    # 3. 새로운 Mol 객체를 생성하고 엄선된 구조만 복사하여 담기
    final_mol = Chem.Mol(mol)
    final_mol.RemoveAllConformers()
    
    for cid in diverse_cids:
        try:
            conf = mol.GetConformer(cid)
            final_mol.AddConformer(conf, assignId=True)
        except ValueError:
            continue
            
    return final_mol


# ==============================================================================
# [Section 3] 구조 최적화 및 에너지 계산 (Optimization & Energy calculation)
# ==============================================================================

def optimize_ensemble(mol: Chem.Mol, conf_ids: list, variant: str = 'MMFF94', max_iters: int = 200, num_threads: int = 0) -> None:
    """
    [성능 최적화] MMFF 역장을 사용하여 앙상블 구조들을 최적화(Relaxation)합니다.
    Python for 루프 대신 RDKit의 C++ 백엔드 멀티스레딩 알고리즘을 직접 호출하여 연산 속도를 수십 배 극대화했습니다.
    
    Args:
        mol (Chem.Mol): 최적화할 분자 객체
        conf_ids (list): 최적화를 수행할 컨포머 ID 리스트 (이 함수에서는 mol 내부의 모든 컨포머를 최적화함)
        variant (str): 사용할 역장 종류 ('MMFF94' 또는 'MMFF94s')
        max_iters (int): 최대 최적화 반복 횟수
        num_threads (int): 가용할 CPU 스레드 수 (0=모든 코어 사용)
    """
    try:
        # C++ 레벨에서 모든 컨포머를 병렬로 동시 최적화 (가장 빠르고 효율적인 방법)
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_threads, maxIters=max_iters, mmffVariant=variant)
    except Exception:
        # [Fallback] 멀티스레딩이 제한된 환경이거나 알 수 없는 에러 발생 시 기존 순차 처리 방식(Robustness 확보)
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
        if mp is None: 
            return

        for cid in conf_ids:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            if ff:
                try: 
                    ff.Minimize(maxIts=max_iters)
                except Exception: 
                    pass


def calculate_energies(mol: Chem.Mol, conf_ids: list, variant: str = 'MMFF94') -> list:
    """
    지정된 MMFF 역장을 기준으로 각 컨포머의 에너지를 계산합니다.
    (하드코딩을 제거하고 variant 매개변수를 수용하도록 확장성을 확보했습니다.)
    
    Args:
        mol (Chem.Mol): 분자 객체
        conf_ids (list): 에너지를 계산할 컨포머 ID 리스트
        variant (str): 사용할 역장 (기본값 'MMFF94')
    Returns:
        list: 각 컨포머의 ID와 에너지를 담은 딕셔너리 리스트 (예: [{'cid': 0, 'energy': 12.5}, ...])
    """
    props = []
    # [확장성] 외부에서 전달된 variant 인자를 존중함
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    
    for cid in conf_ids:
        e = 999.0  # 기본값 (에너지 계산 실패 시 페널티 높은 값 부여)
        if mp:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=cid)
            if ff: 
                e = ff.CalcEnergy()
        props.append({'cid': cid, 'energy': e})
    
    return props


# ==============================================================================
# [Section 4] 최종 정예 구조 선별 (Clustering)
# ==============================================================================

def cluster_ensemble(mol: Chem.Mol, props: list, method: str = 'rmsd', threshold: float = 2.0, max_confs: int = 10) -> list:
    """
    [최종 선별 함수] 에너지가 가장 낮은 구조부터 시작하여(Energy-Ordered Pruning), 
    기존에 선택된 구조들과 비교해 너무 유사한 구조를 걸러냅니다.
    거리 계산 로직은 외부 `metrics` 모듈에 전적으로 위임합니다.
    
    Args:
        mol (Chem.Mol): 분자 객체
        props (list): calculate_energies에서 반환된 에너지 정보 리스트
        method (str): 유사도 측정 방식 ('rmsd' 또는 'tfd')
        threshold (float): 중복으로 간주할 거리 임계값
        max_confs (int): 최종적으로 선택할 최대 컨포머 개수
    Returns:
        list: 선별이 완료된 최종 컨포머 ID 리스트
    """
    # 1. 에너지가 낮은(안정적인) 순서대로 정렬
    props.sort(key=lambda x: x['energy'])
    
    selected_cids = []
    
    for cand in props:
        if len(selected_cids) >= max_confs:
            break
            
        cid = cand['cid']
        
        # 첫 번째 구조(가장 안정한 구조)는 무조건 선택
        if not selected_cids:
            selected_cids.append(cid)
            continue
            
        is_distinct = True
        # 이미 선택된 정예 구조들과 하나씩 비교
        for existing_cid in selected_cids:
            dist = 999.0
            
            # 거리 계산을 외부 metrics 모듈에 위임 (결합도 하락, 응집도 상승)
            if method == 'tfd':
                dist = metrics.calculate_mol_tfd(mol, existing_cid, cid, use_weights=True)
            else:
                # RMSD 계산 (Heavy Atom Only 로직은 metrics 내부에서 처리됨)
                dist = metrics.calculate_mol_rmsd(mol, existing_cid, cid, heavy_only=True)
                
            # 임계값보다 거리가 가깝다면(유사하다면) 탈락 처리
            if dist < threshold:
                is_distinct = False 
                break
        
        # 기존의 어떤 구조와도 충분히 다르다면 선택 목록에 추가
        if is_distinct:
            selected_cids.append(cid)
            
    return selected_cids


# ==============================================================================
# [Section 5] 실행 및 테스트 블록 (Debug & Test)
# ==============================================================================
if __name__ == "__main__":
    print("="*60)
    print(" [DEBUG] Puckering Module Test Pipeline (Multi-threading Optimized)")
    print("="*60)
    
    # 1. 테스트용 분자 준비
    # 유연한 6원환 고리와 측쇄를 모두 가져 테스트에 적합한 Propylcyclohexane 사용
    smiles = "C1CCCCC1CCC"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # 3D 구조 생성을 위해서는 수소(H) 추가가 필수
    
    print(f"\n[1] 분자 준비 완료 (SMILES: {smiles})")
    print(f" - 전체 원자 수 (수소 포함): {mol.GetNumAtoms()}")
    
    # 2. 고리 유연성 테스트
    rings = detect_rings(mol)
    has_flex = has_flexible_rings(mol)
    print("\n[2] 고리 탐지 (Ring Detection)")
    print(f" - 발견된 고리 수: {len(rings)}")
    print(f" - 유연한 고리(4~7원환) 존재 여부: {has_flex}")
    
    # 3. 구조 대량 생성 및 TFD 대표 구조 추출
    print("\n[3] 초기 구조 대량 생성 및 TFD 기반 다양성 추출")
    print(" - ETKDGv3 임베딩 및 Butina 클러스터링 진행 중...")
    n_templates = 5
    n_initial = 50
    
    import time
    start_time = time.time()
    
    mol_3d = generate_conformers(mol, n_templates=n_templates, n_initial=n_initial, tfd_thresh=0.2)
    num_generated = mol_3d.GetNumConformers()
    
    print(f" - 생성 경과 시간: {time.time() - start_time:.3f} 초")
    print(f" - 내부 초기 생성 시도: {n_initial}개")
    print(f" - TFD 필터링 후 추출된 대표 템플릿 수: {num_generated}개 (목표: {n_templates}개)")
    if num_generated == 0:
        print(" [!] 컨포머 생성에 실패했습니다. 테스트를 종료합니다.")
        sys.exit()

    # 4. 역장 최적화 및 에너지 계산 (멀티스레딩 성능 체감)
    print("\n[4] 구조 최적화(MMFF94) 및 에너지 계산 (C++ 멀티스레딩)")
    cids = [conf.GetId() for conf in mol_3d.GetConformers()]
    
    pre_props = calculate_energies(mol_3d, cids)
    avg_pre_e = sum(p['energy'] for p in pre_props) / len(pre_props)
    print(f" - 최적화 전 평균 에너지: {avg_pre_e:.2f} kcal/mol")
    
    start_opt_time = time.time()
    # 최적화 수행 (모든 코어 사용)
    optimize_ensemble(mol_3d, cids, max_iters=200, num_threads=0)
    print(f" - 병렬 최적화 경과 시간: {time.time() - start_opt_time:.3f} 초")
    
    post_props = calculate_energies(mol_3d, cids)
    avg_post_e = sum(p['energy'] for p in post_props) / len(post_props)
    print(f" - 최적화 후 평균 에너지: {avg_post_e:.2f} kcal/mol")
    
    best_conf = min(post_props, key=lambda x: x['energy'])
    print(f" - 최저 에너지 컨포머 ID: {best_conf['cid']} ({best_conf['energy']:.2f} kcal/mol)")
    
    # 5. 최종 앙상블 클러스터링
    print("\n[5] 외부 모듈(metrics) 연동 클러스터링 테스트")
    try:
        selected_cids = cluster_ensemble(mol_3d, post_props, method='rmsd', threshold=0.5, max_confs=3)
        print(f" - RMSD 기반 클러스터링 성공!")
        print(f" - 최종 선택된 정예 컨포머 ID 목록 (최대 3개): {selected_cids}")
    except NameError:
        print(" - [경고] metrics 모듈을 찾을 수 없어 cluster_ensemble 테스트를 건너뜁니다.")
    except AttributeError as e:
        print(f" - [경고] metrics 모듈 내에 필요한 함수가 아직 구현되지 않았습니다: {e}")
    except Exception as e:
        print(f" - [에러] 알 수 없는 오류 발생: {e}")

    print("\n"+"="*60)
    print(" [DEBUG] 테스트 완료")
    print("="*60)