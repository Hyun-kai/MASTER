"""
src/bakers/analytics/criteria.py

[기능]
시뮬레이션 결과의 성공 여부를 판별하는 기준(Criteria) 검사 및
데이터셋 폴더의 무결성 통계(Stats)를 확인하는 모듈입니다.

[성공 판별 기준 (Scientific Basis)]
1. RMSD Threshold (Competitor Definition)
   - Standard: 2.0 A (General Cluster definition)
   - Strict:   1.5 A (Post-experimentation suggestion)
   - 의미: Global Minimum과 구조적으로 다르다고 판단하는 기준.
           이 기준을 넘는 구조들 중 가장 에너지가 낮은 것이 '경쟁 상태(Competitor)'가 됩니다.

2. Gap Criteria (Energy Threshold)
   - Standard: 0.66 kcal/mol per Residue (Based on thermal fluctuation ~kT)
   - Strict:   0.80 kcal/mol per Residue (Higher confidence filter)
   - 의미: (Energy_Competitor - Energy_GlobalMin) / Residues
           이 값이 임계값 이상이어야 실험적으로 유의미한 단일 구조를 형성한다고 봅니다.
"""

import os
import glob
import h5py
import numpy as np

# 순환 참조 방지 및 경로 호환성 처리
try:
    # 같은 패키지 내 import
    from .metrics import calculate_rmsd
except ImportError:
    # 외부 스크립트 실행 시 절대 경로 import
    from bakers.analytics.metrics import calculate_rmsd

# ==============================================================================
# Success Criteria Check (Energy Gap & Uniqueness)
# ==============================================================================

def check_energy_criteria(xyzs, energies, num_residues, rmsd_thresh=2.0, gap_criteria=0.66):
    """
    Global Minimum(S0) 구조가 구조적으로 다른 경쟁 상태(S_next)와 비교해
    충분한 에너지 격차(Energy Gap)를 가지는지 판별합니다.

    Args:
        xyzs (np.ndarray): 좌표 배열 (N, Atoms, 3). (에너지 오름차순 정렬 필수)
        energies (np.ndarray): 에너지 배열 (N,). (오름차순 정렬 필수)
        num_residues (int): 잔기 개수 (Gap 정규화용)
        rmsd_thresh (float): 구조적 차이를 인정하는 최소 RMSD (Angstrom).
        gap_criteria (float): 잔기당 필요한 최소 에너지 격차 (kcal/mol/res).

    Returns:
        dict: 판별 결과 {
            'pass': bool, 
            'global_min_e': float, 
            'gap_per_res': float, 
            'next_rmsd': float, 
            'note': str
        }
    """
    # 데이터 유효성 검사
    if len(xyzs) == 0 or len(energies) == 0 or len(xyzs) != len(energies):
        return {
            'pass': False, 
            'global_min_e': 0.0, 
            'gap_per_res': 0.0, 
            'next_rmsd': 0.0, 
            'note': 'No Data'
        }

    # 1. Global Minimum (S0) 설정
    s0_coords = xyzs[0]
    s0_energy = energies[0]

    s_next_idx = -1
    rmsd_val = 0.0
    
    # 2. Competitor 탐색 (Lowest energy structure with RMSD > threshold)
    # 이미 에너지가 오름차순 정렬되어 있으므로, 조건을 만족하는 첫 번째 구조가 가장 낮은 에너지의 경쟁자임.
    # 속도를 위해 최대 1000개까지만 검색 (충분히 많은 숫자)
    search_limit = min(len(xyzs), 1000)
    
    for i in range(1, search_limit):
        # metrics 모듈의 Kabsch RMSD 사용 (NumPy Array)
        curr_rmsd = calculate_rmsd(s0_coords, xyzs[i])
        
        if curr_rmsd > rmsd_thresh:
            s_next_idx = i
            rmsd_val = curr_rmsd
            break
    
    # 3. Gap 계산 및 판별
    
    # Case A: 경쟁자가 없는 경우 (모든 구조가 RMSD Threshold 안에 존재)
    # -> 샘플링된 모든 구조가 Global Min과 유사함 (단일 Funnel).
    # -> "갭"을 정의할 대안 상태가 없으므로 계산 불가하지만, 
    # -> "유일한 상태"라는 점에서는 긍정적일 수 있음. 
    # -> 하지만 여기서는 '경쟁 상태와의 갭'을 보는 것이므로 Fail 혹은 Undefined 처리.
    if s_next_idx == -1:
        return {
            'pass': False, 
            'global_min_e': s0_energy,
            'gap_per_res': 0.0, # 계산 불가 (Undefined)
            'next_rmsd': 0.0,
            'note': f'Fail: No competitor found > {rmsd_thresh}A (Sampling distinct states failed)'
        }

    # Case B: 경쟁자가 있는 경우 정상 계산
    delta_e = energies[s_next_idx] - s0_energy
    gap_per_res = delta_e / num_residues
    
    is_pass = gap_per_res >= gap_criteria
    
    return {
        'pass': is_pass,
        'global_min_e': s0_energy,
        'gap_per_res': gap_per_res,   
        'next_rmsd': rmsd_val,
        'note': 'Criteria Met' if is_pass else f'Fail: Low Gap ({gap_per_res:.4f} < {gap_criteria})'
    }

def print_criteria_report(res_crit, gap_target=0.66, rmsd_target=2.0):
    """
    판별 결과 딕셔너리를 받아 표준화된 리포트를 출력합니다.
    """
    # 색상 코드 (터미널 출력용)
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    status_icon = f"{GREEN}✅ PASS{RESET}" if res_crit['pass'] else f"{RED}❌ FAIL{RESET}"
    
    print(f"    [Criteria Report]")
    print(f"      Status:      {status_icon}")
    print(f"      Global Min:  {res_crit['global_min_e']:.4f} kcal/mol")
    
    gap_val = res_crit['gap_per_res']
    gap_str = f"{gap_val:.4f}" if gap_val > 0.0 else "Undefined"
    
    print(f"      Gap per Res: {gap_str} (Target: >{gap_target})")
    print(f"      Next RMSD:   {res_crit['next_rmsd']:.4f} A (Definition: >{rmsd_target})")
    print(f"      Note:        {res_crit['note']}")