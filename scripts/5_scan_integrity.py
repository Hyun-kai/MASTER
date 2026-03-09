"""
scripts/5_scan_integrity.py

[기능]
HDF5 파일들의 '데이터 무결성(Data Integrity)'을 진단하는 전용 도구입니다.
파일 손상 여부, 필수 데이터 키 존재 여부, 그리고 [결합각(Bond Angle)]을 계산하여
시뮬레이션 폭발(Explosion) 등 기하학적 이상 징후를 빠르게 포착합니다.

[점검 항목]
1. 파일 손상 여부 (Read Error)
2. 필수 키 존재 여부 ('xyzs', 'points', 'energy'/'values')
3. 기하학적 이상 (Bond Angle < 5도 or > 179.9도)

[수정 사항]
- [Smart Path] 프로젝트 루트 기준 및 1_data 폴더 자동 검색 추가
- [Compatibility] HDF5 Key 처리 로직을 io.py와 일치시킴
"""

import os
import sys
import argparse
import glob
import h5py
import numpy as np
import pandas as pd
import warnings

# RuntimeWarning 무시 (NaN 등)
warnings.filterwarnings("ignore")

# 색상 코드
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[0;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'

# ------------------------------------------------------------------------------
# 1. 환경 설정
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, '1_data')

# ------------------------------------------------------------------------------
# 2. 기하학적 계산 함수 (Vectorized)
# ------------------------------------------------------------------------------
def compute_bond_angles(positions):
    """
    [핵심 로직] 좌표 배열에서 연속된 3개 원자(i, i+1, i+2) 간의 결합각을 계산합니다.
    """
    # 차원 보정 (단일 프레임일 경우)
    if positions.ndim == 2:
        positions = positions[np.newaxis, ...]
        
    # 원자가 3개 미만이면 각도 계산 불가
    if positions.shape[1] < 3:
        return np.array([])

    # 벡터 계산: B->A (v1), B->C (v2)
    # A(i), B(i+1), C(i+2)
    v1 = positions[:, :-2, :] - positions[:, 1:-1, :] # Middle -> Prev
    v2 = positions[:, 2:, :] - positions[:, 1:-1, :]  # Middle -> Next
    
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    
    # Dot Product
    dot = np.sum(v1 * v2, axis=-1)
    
    # Cosine Rule (안정성을 위해 clip 적용)
    denominator = norm1 * norm2 + 1e-8
    cosine = dot / denominator
    cosine = np.clip(cosine, -1.0, 1.0)
    
    # 각도 변환 (Radian -> Degree)
    angles = np.degrees(np.arccos(cosine))
    
    return angles.flatten()

# ------------------------------------------------------------------------------
# 3. 상세 검사 함수
# ------------------------------------------------------------------------------
def inspect_hdf5_detailed(filepath):
    """
    HDF5 파일을 열어 필수 키 확인 및 에너지/각도 통계를 추출합니다.
    """
    fname = os.path.basename(filepath)
    result = {
        'filename': fname,
        'path': filepath,
        'valid': False,
        'n_frames': 0,
        'has_xyzs': False,
        'has_points': False,
        'has_energy': False,
        'min_energy': None,
        'min_angle': None,
        'max_angle': None,
        'avg_angle': None,
        'note': ''
    }

    try:
        with h5py.File(filepath, 'r') as f:
            # 1. Key Check & Data Loading
            
            # (A) XYZ / Positions
            if 'xyzs' in f:
                result['has_xyzs'] = True
                xyz_data = f['xyzs'][:]
            elif 'positions' in f:
                result['has_xyzs'] = True
                xyz_data = f['positions'][:]
            else:
                xyz_data = np.array([])

            if result['has_xyzs']:
                result['n_frames'] = xyz_data.shape[0]
                
                # [Calc] 각도 계산 (기하학적 붕괴 확인용)
                if xyz_data.size > 0:
                    angles = compute_bond_angles(xyz_data)
                    if angles.size > 0:
                        result['min_angle'] = float(np.min(angles))
                        result['max_angle'] = float(np.max(angles))
                        result['avg_angle'] = float(np.mean(angles))
                    else:
                        result['note'] += "[Info] Too few atoms (<3). "
            
            # (B) Points (Dihedrals)
            if 'points' in f:
                result['has_points'] = True
                # xyzs가 없어서 프레임 수가 0일 경우 points 기준으로 설정
                if result['n_frames'] == 0:
                    result['n_frames'] = f['points'].shape[0]

            # (C) Energy (Values or Energy)
            e_data = None
            if 'energies' in f: # io.py standard
                result['has_energy'] = True
                e_data = f['energies'][:]
            elif 'values' in f: # legacy
                result['has_energy'] = True
                e_data = f['values'][:]
            elif 'energy' in f: # potential legacy
                result['has_energy'] = True
                e_data = f['energy'][:]
            
            if e_data is not None and e_data.size > 0:
                result['min_energy'] = float(np.min(e_data))

            # 2. Validity Judgment
            # 기준: xyzs와 energy가 모두 있어야 '구조 분석'이 가능하다고 판단
            if result['has_xyzs'] and result['has_energy'] and result['n_frames'] > 0:
                result['valid'] = True
            else:
                missing = []
                if not result['has_xyzs']: missing.append('xyzs')
                if not result['has_energy']: missing.append('energy')
                if result['n_frames'] == 0: missing.append('empty')
                result['note'] = f"Missing: {', '.join(missing)} " + result['note']
                
            # 3. Integrity Warning (Angle Check)
            if result['min_angle'] is not None:
                if result['min_angle'] < 5.0 or result['max_angle'] > 179.9:
                    result['note'] += "[Warn] Geom Distortion. "

    except Exception as e:
        result['note'] = f"Read Error: {str(e)}"
        
    return result

# ------------------------------------------------------------------------------
# 4. 메인 실행 로직
# ------------------------------------------------------------------------------
def resolve_path(user_input):
    """
    [Smart Path] 입력된 경로가 없으면 1_data 폴더 하위를 검색합니다.
    """
    # 1. 입력된 경로가 그대로 존재하는지 확인
    if os.path.exists(user_input):
        return user_input
    
    # 2. 프로젝트 루트 기준 상대 경로 확인
    root_rel = os.path.join(PROJECT_ROOT, user_input)
    if os.path.exists(root_rel):
        return root_rel

    # 3. 파일명만 있는 경우 1_data 내부 검색 (Recursive)
    filename = os.path.basename(user_input)
    search_patterns = [
        os.path.join(DATA_DIR, user_input),
        os.path.join(DATA_DIR, '**', user_input),
        os.path.join(DATA_DIR, '**', filename)
    ]
    
    for pattern in search_patterns:
        found = glob.glob(pattern, recursive=True)
        if found:
            # 여러 개 발견 시 첫 번째 반환 (보통 가장 가까운 경로)
            return found[0]
            
    return None

def run(args):
    # 입력 소스 확인 및 경로 보정
    files_to_scan = []
    scan_source = None
    
    if args.file:
        resolved = resolve_path(args.file)
        if resolved and os.path.isfile(resolved):
            files_to_scan.append(resolved)
            scan_source = resolved
        else:
            print(f"{RED}[Error] File not found: {args.file}{NC}")
            print(f"        Searched in: {DATA_DIR} and subdirectories.")
            return
            
    elif args.dir:
        resolved = resolve_path(args.dir)
        if resolved and os.path.isdir(resolved):
            files_to_scan = glob.glob(os.path.join(resolved, "*.hdf5"))
            scan_source = resolved
        else:
            print(f"{RED}[Error] Directory not found: {args.dir}{NC}")
            print(f"        Searched in: {DATA_DIR} and subdirectories.")
            return
    else:
        print(f"{RED}[Error] Please specify --file or --dir{NC}")
        return

    print(f">>> [Integrity Scan] Source: {scan_source}")
    print(f"    Target Files: {len(files_to_scan)}")
    print("-" * 110)
    # 헤더 출력
    print(f"{'Filename':<35} | {'Frm':<5} | {'XYZ':<3} {'Pts':<3} {'Eng':<3} | {'Min E':<8} | {'Ang(Min/Max)':<14} | {'Status'}")
    print("-" * 110)

    results_list = []
    
    for fpath in files_to_scan:
        res = inspect_hdf5_detailed(fpath)
        results_list.append(res)
        
        # 포맷팅
        fname_short = res['filename']
        if len(fname_short) > 34:
            fname_short = fname_short[:31] + "..."
        
        status_icon = f"{GREEN}OK{NC}" if res['valid'] else f"{RED}FAIL{NC}"
        if res['valid'] and "Warn" in res['note']:
            status_icon = f"{YELLOW}WARN{NC}"

        min_e_str = f"{res['min_energy']:.1f}" if res['min_energy'] is not None else "-"
        
        if res['min_angle'] is not None:
            ang_str = f"{res['min_angle']:.0f}/{res['max_angle']:.0f}"
        else:
            ang_str = "-"

        mk_x = "O" if res['has_xyzs'] else "X"
        mk_p = "O" if res['has_points'] else "X"
        mk_e = "O" if res['has_energy'] else "X"

        print(f"{fname_short:<35} | {res['n_frames']:<5} | {mk_x:<3} {mk_p:<3} {mk_e:<3} | {min_e_str:<8} | {ang_str:<14} | {status_icon} {res['note']}")

    print("-" * 110)

    # CSV 저장
    if args.save_csv and results_list:
        # 저장 위치: 스캔한 대상 폴더 내부 (파일 하나면 그 파일 폴더)
        save_dir = os.path.dirname(scan_source) if os.path.isfile(scan_source) else scan_source
        out_csv = os.path.join(save_dir, "integrity_report.csv")
            
        try:
            df = pd.DataFrame(results_list)
            
            cols = ['filename', 'valid', 'n_frames', 'min_energy', 
                    'min_angle', 'max_angle', 'avg_angle', 
                    'has_xyzs', 'has_points', 'has_energy', 'note', 'path']
            
            exist_cols = [c for c in cols if c in df.columns]
            df = df[exist_cols]
            
            df.to_csv(out_csv, index=False)
            print(f"\n{GREEN}[Success] Report saved to: {out_csv}{NC}")
        except Exception as e:
            print(f"\n{RED}[Error] Failed to save CSV: {e}{NC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Integrity Scan for HDF5 Files")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single HDF5 file (or filename)")
    group.add_argument("--dir", help="Path to a directory containing HDF5 files")
    
    parser.add_argument("--save_csv", action="store_true", default=True, help="Save results to CSV (Default: True)")
    
    args = parser.parse_args()
    run(args)