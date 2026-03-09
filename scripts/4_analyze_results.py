"""
scripts/4_analyze_results.py

[설명]
BAKERS 프로젝트의 시뮬레이션 결과(HDF5)를 분석하는 파이프라인 스크립트입니다.
라이브러리(src/bakers)의 기능을 호출하여 다음 작업들을 순차적으로 수행합니다.

[주요 기능]
1. Criteria Check: 에너지 갭 및 구조적 안정성 기준 충족 여부 판별 (Pass/Fail)
2. Visualization: 에너지 분포(Hist/KDE), 랜드스케이프(PCA/t-SNE), RMSD 매트릭스 시각화
3. PDB Extraction: 클러스터링을 통해 구조적 다양성을 대표하는 Top-N 구조 추출 (.pdb)

[개선 사항]
- [Folder Structure] 결과 저장 시 monomers/dimers/polymers 하위 폴더로 자동 분류
- [Smart Path] 파일명만 입력해도 1_data 폴더 내의 파일을 자동으로 검색
- [Type Hinting] 코드 가독성 및 안정성을 위한 타입 힌트 추가
- [Optimization] 분석(Viz)과 PDB 추출 단계를 각각 독립적으로 체크하여 필요한 작업만 수행

[작성자]
BAKERS Lead Chemist & Engineer
"""

import os
import sys
import argparse
import glob
from typing import Optional, List, Dict, Any

# ==============================================================================
# 1. 환경 설정 및 라이브러리 로드
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, '1_data')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from bakers.utils import io, visual
    from bakers.analytics import criteria
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

# 터미널 색상 코드
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

# ==============================================================================
# 2. 헬퍼 함수: 스마트 경로 탐색
# ==============================================================================

def resolve_file_path(user_input: str) -> Optional[str]:
    """
    사용자가 입력한 파일 경로가 존재하지 않으면, 1_data 폴더 하위를 재귀적으로 검색합니다.
    
    Args:
        user_input (str): 사용자가 입력한 파일 경로 또는 파일명.
        
    Returns:
        Optional[str]: 발견된 절대 경로. 없으면 None.
    """
    if os.path.exists(user_input):
        return os.path.abspath(user_input)
    
    filename = os.path.basename(user_input)
    
    # 검색 우선순위 설정
    search_patterns = [
        os.path.join(DATA_DIR, filename),              # 1_data 루트
        os.path.join(DATA_DIR, 'dimers', filename),    # dimers 폴더
        os.path.join(DATA_DIR, 'monomers', filename),  # monomers 폴더
        os.path.join(DATA_DIR, 'polymers', filename),  # polymers 폴더
        os.path.join(DATA_DIR, '**', filename)         # 전체 검색 (느림)
    ]
    
    for pattern in search_patterns:
        # recursive=True는 '**' 패턴이 있을 때만 작동함
        found = glob.glob(pattern, recursive=True)
        if found:
            return os.path.abspath(found[0])
            
    return None

def determine_residue_count(filename: str) -> int:
    """파일명에서 잔기(Residue) 개수를 추론합니다."""
    fname_lower = filename.lower()
    if 'octamer' in fname_lower: return 8
    if 'hexamer' in fname_lower: return 6
    if 'tetramer' in fname_lower: return 4
    if 'trimer' in fname_lower: return 3
    if 'dimer' in fname_lower: return 2
    if '9mer' in fname_lower: return 9
    return 1 # Default (Monomer)

# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

def run(args: argparse.Namespace) -> None:
    raw_target = args.file if args.file else args.dir
    
    # --------------------------------------------------------------------------
    # [Mode A] 디렉토리 일괄 처리 (Batch Analysis)
    # --------------------------------------------------------------------------
    if raw_target and os.path.isdir(raw_target):
        print(f"{Colors.BLUE}>>> [Batch Analysis] Scanning Directory: {raw_target}{Colors.NC}")
        files = glob.glob(os.path.join(raw_target, "*.hdf5"))
        files.sort()
        
        if not files:
            print(f"{Colors.YELLOW}[Warn] No .hdf5 files found in {raw_target}{Colors.NC}")
            return

        for f in files:
            print("-" * 60)
            # 재귀 호출을 위해 args 복사 및 수정
            sub_args = argparse.Namespace(**vars(args))
            sub_args.file = f
            sub_args.dir = None
            run(sub_args)
        return

    # --------------------------------------------------------------------------
    # [Mode B] 단일 파일 처리 (Single File Analysis)
    # --------------------------------------------------------------------------
    target = resolve_file_path(raw_target)
    
    if not target:
        print(f"{Colors.RED}[Error] File not found: {raw_target}{Colors.NC}")
        print(f"        Searched in: {DATA_DIR} and subdirectories.")
        return

    # ==========================================================================
    # [최적화] 작업 필요 여부 사전 판별 (HDF5 로드 전)
    # ==========================================================================
    # 1. 파일 경로 및 이름 기반으로 저장 위치(Category/BaseName) 계산
    target_path_lower = target.lower()
    if 'monomers' in target_path_lower: category = 'monomers'
    elif 'dimers' in target_path_lower: category = 'dimers'
    elif 'polymers' in target_path_lower: category = 'polymers'
    else: category = 'others'

    fname = os.path.basename(target)
    base_name = os.path.splitext(fname)[0]

    # 2. 결과 저장 경로 설정
    analysis_dir = os.path.join(PROJECT_ROOT, '2_results', 'analysis', category, base_name)
    pdb_dir = os.path.join(PROJECT_ROOT, '2_results', 'pdb', category, base_name)
    
    # 3. 각 단계별 완료 여부 확인
    # (1) 분석(Viz): 폴더가 존재하고 비어있지 않아야 함
    is_analysis_done = os.path.exists(analysis_dir) and len(os.listdir(analysis_dir)) > 0
    
    # (2) PDB 추출: 폴더가 존재하고, .pdb 파일 개수가 요청한 top_n 이상이어야 함
    current_pdb_count = 0
    if os.path.exists(pdb_dir):
        # .pdb 확장자를 가진 파일만 카운트
        current_pdb_count = len([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])
    
    is_pdb_done = current_pdb_count >= args.top_n

    # 4. 전체 스킵 여부 결정
    if is_analysis_done and is_pdb_done:
        print(f"{Colors.GREEN}[Skip] All tasks completed for {base_name}.{Colors.NC}")
        print(f"       Analysis: OK, PDB: OK ({current_pdb_count}/{args.top_n})")
        return
    
    # 작업 상태 출력
    print(f"{Colors.GREEN}>>> [Processing] {base_name}{Colors.NC}")
    if is_analysis_done:
        print(f"    [Check] Analysis: DONE (Skipping Viz)")
    else:
        print(f"    [Check] Analysis: MISSING (Will run Viz)")
        
    if is_pdb_done:
        print(f"    [Check] PDB: DONE ({current_pdb_count} files)")
    else:
        # PDB가 부족한 경우 (예: 1개만 있음)
        print(f"    [Check] PDB: INCOMPLETE ({current_pdb_count}/{args.top_n}) (Will run Extraction)")

    # ==========================================================================
    # 데이터 로드 (필요한 경우에만 실행됨)
    # ==========================================================================
    try:
        data = io.load_hdf5_data(target, sorted_by_energy=True)
    except Exception as e:
        print(f"{Colors.RED}    [Error] Failed to load HDF5: {e}{Colors.NC}")
        return

    if data is None or data.get('xyzs') is None:
        print(f"{Colors.RED}    [Error] No coordinate data ('xyzs') found in HDF5.{Colors.NC}")
        return

    # 잔기 개수 추론 (파일명 기반)
    num_res = determine_residue_count(fname)

    # --------------------------------------------------------------------------
    # Step 1: Criteria Check (성공 기준 판별)
    # --------------------------------------------------------------------------
    # Criteria Check는 가벼운 작업이므로 항상 수행하여 로그에 상태를 남깁니다.
    try:
        res_crit = criteria.check_energy_criteria(
            data['xyzs'], data['energies'], num_residues=num_res
        )
        criteria.print_criteria_report(res_crit)
    except Exception as e:
        print(f"{Colors.YELLOW}    [Warn] Criteria check failed: {e}{Colors.NC}")

    # --------------------------------------------------------------------------
    # Step 2: Visualization (그래프 생성) - 조건부 실행
    # --------------------------------------------------------------------------
    if not is_analysis_done:
        os.makedirs(analysis_dir, exist_ok=True)
        try:
            # 1. 에너지 분포 및 Landscape
            visual.analyze_and_save(target, output_dir=analysis_dir)
            # 2. RMSD Matrix Heatmap
            visual.analyze_rmsd(target, output_dir=analysis_dir, num_residues=num_res)
            print(f"    [Visual] Plots saved to {analysis_dir}")
        except Exception as e:
            print(f"{Colors.YELLOW}    [Warn] Visualization failed: {e}{Colors.NC}")

    # --------------------------------------------------------------------------
    # Step 3: PDB Extraction (구조 추출) - 조건부 실행
    # --------------------------------------------------------------------------
    if not is_pdb_done:
        try:
            # Top-N 구조 추출
            # (io.py 내부에도 스킵 로직이 있을 수 있지만, 여기서는 강제로 실행 흐름을 제어합니다)
            io.extract_and_save_top_structures(
                target_file=target,
                output_dir=pdb_dir,
                top_n=args.top_n,
                cluster_threshold=args.cluster_threshold,
                project_root=PROJECT_ROOT 
            )
            print(f"    [Output] PDB structures saved to {pdb_dir}")
        except Exception as e:
            print(f"{Colors.RED}    [Error] PDB extraction failed: {e}{Colors.NC}")

# ------------------------------------------------------------------------------
# 4. CLI 진입점
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BAKERS Analysis & PDB Extraction Tool")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Target HDF5 file path or filename")
    group.add_argument("--dir", type=str, help="Target directory containing HDF5 files")
    
    parser.add_argument("--top_n", type=int, default=50, help="Number of representative structures to extract (default: 5)")
    parser.add_argument("--cluster_threshold", type=float, default=45.0, help="RMSD clustering threshold (default: 45.0)")
    
    args = parser.parse_args()
    run(args)