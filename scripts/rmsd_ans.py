import math
import os
import glob
import csv
import numpy as np
import sys

# ------------------------------------------------------------------------------
# 1. 환경 설정 및 경로 지정
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../AIB_PRO_BAKERS/scripts
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                   # .../AIB_PRO_BAKERS
RESULTS_DIR = os.path.join(PROJECT_ROOT, '2_results')         # 결과 저장 폴더 (CSV)
TARGET_DIR = os.path.join(RESULTS_DIR, 'pdb')                 # 분석 대상 폴더 (PDB)
ANS_DIR = os.path.join(TARGET_DIR, 'ans')                     # 정답(Reference) 파일 폴더

# src 폴더를 시스템 경로에 추가하여 패키지 임포트 가능하게 설정
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

try:
    # metrics 모듈에서 calculate_rmsd 함수 임포트 (사용자 환경에 맞게 동작)
    from bakers.metrics import calculate_rmsd
except ImportError:
    # 혹시 경로 문제로 임포트가 안 될 경우를 대비한 더미 함수 (실제 환경에서는 위 경로가 맞아야 함)
    print("[Warning] 'bakers.metrics' module not found. Using placeholder function.")
    def calculate_rmsd(coords1, coords2):
        # 간단한 RMSD 계산 로직 (numpy 기반)
        diff = coords1 - coords2
        return np.sqrt((diff ** 2).sum() / len(coords1))

# ------------------------------------------------------------------------------
# 2. 헬퍼 함수: PDB 좌표 추출
# ------------------------------------------------------------------------------
def get_coordinates_from_pdb(pdb_path):
    """
    PDB 파일에서 ATOM 레코드의 XYZ 좌표를 추출하여 numpy array로 반환합니다.
    RMSD 계산을 위해 좌표 데이터가 필요하기 때문입니다.
    """
    coords = []
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    # PDB 포맷 표준에 따른 좌표 추출 (30~38: X, 38~46: Y, 46~54: Z)
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        coords.append([x, y, z])
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[Error] Failed to parse PDB: {pdb_path} ({e})")
        return None
    
    return np.array(coords)

# ------------------------------------------------------------------------------
# 3. 메인 실행
# ------------------------------------------------------------------------------
def main():
    # 1) 분석 대상 파일 검색 (이름에 'octamer'가 포함된 PDB)
    # search_pattern = os.path.join(TARGET_DIR, "*.pdb") # 기존 코드 수정
    search_pattern = os.path.join(TARGET_DIR, "*octamer*.pdb")
    pdb_files = glob.glob(search_pattern)
    pdb_files.sort()

    # 2) 정답(Reference) 파일 검색 (ans 폴더 내의 첫 번째 PDB 파일)
    ans_pattern = os.path.join(ANS_DIR, "*.pdb")
    ans_files = glob.glob(ans_pattern)
    
    if not ans_files:
        print(f"[Error] No reference PDB file found in {ANS_DIR}")
        return
    
    ref_file_path = ans_files[0] # 첫 번째 파일을 Reference로 사용
    ref_fname = os.path.basename(ref_file_path)
    
    print("=" * 100)
    print(f"Auto-Compare (RMSD) & CSV Export")
    print(f"Target Directory : {TARGET_DIR}")
    print(f"Reference File   : {ref_fname} (in {ANS_DIR})")
    print(f"CSV Save Dir     : {RESULTS_DIR}")
    print("=" * 100)

    # Reference 좌표 로드
    ref_coords = get_coordinates_from_pdb(ref_file_path)
    if ref_coords is None or len(ref_coords) == 0:
        print("[Error] Reference PDB has no coordinates.")
        return

    # 결과를 저장할 리스트 (파일명, RMSD, 좌표 수)
    result_list = []

    print(f"{'Filename':<50} | {'RMSD':<10} | {'Atom Count'}")
    print("-" * 100)

    for fpath in pdb_files:
        fname = os.path.basename(fpath)
        
        # Target 좌표 로드
        target_coords = get_coordinates_from_pdb(fpath)
        
        # 좌표 개수 검증 (RMSD 계산을 위해 원자 수가 같아야 함)
        rmsd_val = -1.0
        note = ""
        
        if target_coords is None:
            note = "Parse Error"
        elif len(target_coords) != len(ref_coords):
            # 원자 수가 다르면 RMSD 계산 불가 (혹은 정렬 필요)
            note = f"Atom mismatch ({len(target_coords)} vs {len(ref_coords)})"
            rmsd_val = 999.0 # 정렬 시 뒤로 보내기 위함
        else:
            # RMSD 계산 수행
            try:
                rmsd_val = calculate_rmsd(ref_coords, target_coords)
            except Exception as e:
                note = f"Calc Error"
                rmsd_val = 999.0

        # 결과 출력용 문자열 포맷팅
        if rmsd_val == 999.0 or rmsd_val == -1.0:
            display_rmsd = "N/A"
        else:
            display_rmsd = f"{rmsd_val:.4f}"

        # 1. 화면 출력
        print(f"{fname:<50} | {display_rmsd:<10} | {len(target_coords) if target_coords is not None else 0} {note}")
        
        # 2. 결과 리스트에 추가 (RMSD 값 기준으로 정렬하기 위해 숫자로 저장)
        result_list.append({
            'Filename': fname,
            'RMSD': rmsd_val,
            'Note': note
        })

    # 3. RMSD 기준 오름차순 정렬 (낮은 값이 상위)
    # 999.0 등 에러 값은 뒤로 가도록 정렬됨
    result_list.sort(key=lambda x: x['RMSD'])

    # 4. CSV 데이터 준비
    csv_data = []
    header = ['Filename', 'RMSD', 'Note']
    csv_data.append(header)
    
    for item in result_list:
        # CSV에는 RMSD가 유효한 경우 소수점 4자리, 아니면 N/A로 기록
        val_str = f"{item['RMSD']:.4f}" if item['RMSD'] < 900 else "N/A"
        csv_data.append([item['Filename'], val_str, item['Note']])

    # CSV 파일 저장
    csv_filename = "Compare_results_RMSD.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)

    try:
        # encoding='utf-8-sig'는 엑셀에서 한글 깨짐을 방지합니다.
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print("=" * 100)
        print(f" [Success] CSV file saved at: {csv_path}")
    except Exception as e:
        print("=" * 100)
        print(f" [Error] Failed to save CSV: {e}")

    # 상위 3개 결과 요약 출력
    print("=" * 100)
    print(f" [Top 3 Matches (Lowest RMSD)] ")
    top_n = min(3, len(result_list))
    for i in range(top_n):
        item = result_list[i]
        if item['RMSD'] < 900:
            print(f" {i+1}. {item['Filename']:<40} : RMSD {item['RMSD']:.4f}")
    print("=" * 100)

if __name__ == "__main__":
    main()