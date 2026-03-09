import math
import os
import glob
import csv

# ==========================================
# 1. 환경 설정 및 경로 지정
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../AIB_PRO_BAKERS/scripts
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                   # .../AIB_PRO_BAKERS
RESULTS_DIR = os.path.join(PROJECT_ROOT, '2_results')         # 결과 저장 폴더 (CSV)
TARGET_DIR = os.path.join(RESULTS_DIR, 'pdb')                 # 분석 대상 폴더 (PDB)

# ==========================================
# 2. 기하학적 분석 함수
# ==========================================

def get_centroid(atoms):
    """원자 좌표 리스트의 중심(Centroid) 계산"""
    if not atoms: return None
    n = len(atoms)
    return (sum(a[0] for a in atoms)/n, sum(a[1] for a in atoms)/n, sum(a[2] for a in atoms)/n)

def calc_dist(coord1, coord2):
    """유클리드 거리 계산"""
    if coord1 is None or coord2 is None: return 999.9
    return math.sqrt((coord1[0]-coord2[0])**2 + (coord1[1]-coord2[1])**2 + (coord1[2]-coord2[2])**2)

def parse_pdb_advanced(filename):
    """PDB 파싱: Backbone N 기준으로 잔기 그룹화 및 Sidechain 추출"""
    residues = []
    if not os.path.exists(filename): return []

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        current_res = {'N': None, 'O': None, 'Sidechain': []}
        
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                try:
                    coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                except ValueError: continue

                if atom_name == 'N':
                    if current_res['N'] is not None: residues.append(current_res)
                    current_res = {'N': coords, 'O': None, 'Sidechain': []}
                elif atom_name == 'O':
                    current_res['O'] = coords
                elif atom_name not in ['C', 'CA', 'H', 'HA']: 
                    if not atom_name.startswith('H'): # 수소 제외한 중원자만
                        current_res['Sidechain'].append(coords)
        
        if current_res['N'] is not None: residues.append(current_res)
    except Exception: return []
    return residues

def get_geometric_features(residues):
    """
    구조의 기하학적 특징(H-bond 수, Stacking 수)만 순수하게 반환
    """
    num_residues = len(residues)
    features = {
        "i3": 0, "i4": 0, "i5": 0, "stack": 0, 
        "total_hb": 0, "is_valid": True
    }
    
    if num_residues < 4:
        features["is_valid"] = False
        return features

    HBOND_MAX = 3.5
    STACK_MAX = 5.0

    # 1. H-Bond Check
    for i in range(num_residues):
        coord_O = residues[i]['O']
        if coord_O is None: continue

        # i -> i+3, i+4, i+5
        for offset in [3, 4, 5]:
            if i + offset < num_residues:
                coord_N = residues[i+offset]['N']
                if coord_N and calc_dist(coord_O, coord_N) <= HBOND_MAX:
                    features[f"i{offset}"] += 1

    features["total_hb"] = features["i3"] + features["i4"] + features["i5"]

    # 2. Stacking Check
    for i in range(num_residues - 1):
        sc_i = residues[i]['Sidechain']
        sc_next = residues[i+1]['Sidechain']
        # 고리를 형성할 만큼 원자가 충분한지(3개 이상) 체크
        if len(sc_i) >= 3 and len(sc_next) >= 3:
            if calc_dist(get_centroid(sc_i), get_centroid(sc_next)) <= STACK_MAX:
                features["stack"] += 1
                
    return features

# ==========================================
# 3. [핵심] 잔기 이름 + 기하학 통합 판별 함수
# ==========================================

def classify_structure_context_aware(filename):
    """
    파일명의 잔기 정보와 기하학적 분석 결과를 결합하여 최종 Type(A~G)을 판정합니다.
    """
    residues = parse_pdb_advanced(filename)
    feat = get_geometric_features(residues)
    fname_upper = os.path.basename(filename).upper()

    if not feat["is_valid"]:
        return "Too Short / Invalid"

    # ---------------------------------------------------------
    # 우선순위 1: Stacking 구조 (Herringbone / Aromatic)
    # ---------------------------------------------------------
    # Stacking이 존재하고 수소결합보다 우세하거나, 
    # 특정 잔기(QUIN, BEN2)가 있으면서 Stacking이 1개라도 감지된 경우
    is_stacking_dominant = (feat["stack"] > 0 and feat["stack"] >= feat["total_hb"])
    
    if "QUIN" in fname_upper or "PYRI" in fname_upper:
        if is_stacking_dominant:
            return "Type F: Herringbone (Quin/Pyri)"
        else:
            return "Unfolded Herringbone (Stacking missing)"

    if "BEN2" in fname_upper:
        if is_stacking_dominant or feat["stack"] > 0:
            return "Type G: Aromatic Helix (BEN2/Pro)"
        else:
            return "Unfolded Aromatic (Stacking missing)"

    # ---------------------------------------------------------
    # 우선순위 2: 18/16-Helix (Mixed / Wide)
    # ---------------------------------------------------------
    # Beta-2-Ala가 포함되어 있고 i->i+4 혹은 i->i+5 패턴이 보일 때
    if "B2ALA" in fname_upper or "B2" in fname_upper:
        # 18/16 헬릭스는 i->i+4와 i->i+5가 섞이거나 i->i+4가 넓게 형성됨
        if feat["i4"] > 0 or feat["i5"] > 0:
            return "Type E: 18/16-Helix (Ala/Beta2-Ala)"
        
    # ---------------------------------------------------------
    # 우선순위 3: i -> i+3 기반 구조들 (3_10, 11, 12, Ribbon)
    # ---------------------------------------------------------
    # 기하학적으로 i->i+3이 가장 우세해야 함
    if feat["i3"] >= feat["i4"] and feat["i3"] > 0:
        
        # Type B: Ribbon (Proline 포함)
        if "PRO" in fname_upper:
            return "Type B: Beta-Bend Ribbon (Aib/Pro)"
        
        # Type C: 11-Helix (ACPC 포함)
        elif "ACPC" in fname_upper:
            return "Type C: 11-Helix (Aib/ACPC)"
        
        # Type D: 12-Helix (Gamma residue 포함 - GALA, G-ALA 등)
        elif "GALA" in fname_upper or "G-ALA" in fname_upper or "GAMMA" in fname_upper:
            return "Type D: 12-Helix (Aib/Gamma-Ala)"
        
        # Type A: 3_10 Helix (Ala/Ala 혹은 Aib만 있을 때)
        # 보통 3_10은 i->i+3 이지만 Ala rich에서 나타남
        elif "ALA" in fname_upper or "AIB" in fname_upper:
             return "Type A: 3_10 Helix (Ala/Ala)"

    # ---------------------------------------------------------
    # 우선순위 4: Alpha-Helix (Standard)
    # ---------------------------------------------------------
    if feat["i4"] > feat["i3"] and feat["i4"] > 0:
        if "ALA" in fname_upper:
            return "Type A variant: Alpha-Helix (Ala/Ala - i+4 dominant)"
        return "Alpha-Helix (Type A like)"

    # ---------------------------------------------------------
    # 구조 없음
    # ---------------------------------------------------------
    return "No Structured Helix Found"

# ==========================================
# 4. 메인 실행
# ==========================================
def main():
    search_pattern = os.path.join(TARGET_DIR, "*.pdb")
    pdb_files = glob.glob(search_pattern)
    pdb_files.sort()
    
    print("=" * 100)
    print(f"Auto-Classification & CSV Export")
    print(f"PDB Directory : {TARGET_DIR}")
    print(f"CSV Save Dir  : {RESULTS_DIR}")
    print("=" * 100)

    # 결과를 저장할 리스트
    csv_data = []
    
    # 헤더 정의
    header = ['Filename', 'Classification Result']
    csv_data.append(header)

    summary_counts = {}

    print(f"{'Filename':<50} | {'Auto-Classified Type'}")
    print("-" * 100)

    for fpath in pdb_files:
        fname = os.path.basename(fpath)
        classification = classify_structure_context_aware(fpath)
        
        # 1. 화면 출력
        print(f"{fname:<50} | {classification}")
        
        # 2. CSV 데이터 수집
        csv_data.append([fname, classification])
        
        # 3. 통계 집계
        base_type = classification.split(":")[0]
        summary_counts[base_type] = summary_counts.get(base_type, 0) + 1

    # CSV 파일 저장
    csv_filename = "classification_results.csv"
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

    print("=" * 100)
    print(" [Classification Summary] ")
    for k, v in sorted(summary_counts.items()):
        print(f" - {k:<35} : {v} files")
    print("=" * 100)

if __name__ == "__main__":
    main()