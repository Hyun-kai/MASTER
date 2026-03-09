"""
scripts/7_merge_and_analyze.py

[기능]
1. 지정된 폴더(input_dir) 내의 HDF5 파일들을 검색하여 하나로 병합합니다.
2. 병합된 파일을 기반으로 RMSD 분석 및 에너지 분포 시각화를 수행합니다.
3. 결과는 프로젝트 표준 구조인 '2_results/merged' 폴더에 저장됩니다.

[사용법]
python scripts/7_merge_and_analyze.py --pattern "*octamer.hdf5"

# Polymers 폴더 내의 모든 9mer 결과를 합쳐서 분석
python scripts/7_merge_and_analyze.py --pattern "*9mer.hdf5" --output_name merged_9mer.hdf5
"""


import os
import sys
import glob
import argparse
import shutil

# ------------------------------------------------------------------------------
# 1. 환경 설정 및 라이브러리 경로 추가
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, '1_data', 'polymers')
RESULT_DIR = os.path.join(PROJECT_ROOT, '2_results', 'merged')

# src 폴더를 시스템 경로에 추가
src_path = os.path.join(PROJECT_ROOT, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from bakers.utils import io, visual
except ImportError as e:
    print(f"[Error] Failed to import bakers modules: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. 메인 실행 함수
# ------------------------------------------------------------------------------
def run(args):
    # 입력 디렉토리 설정 (기본값: 1_data/polymers)
    input_dir = args.input_dir if args.input_dir else DATA_DIR
    input_dir = os.path.abspath(input_dir)
    
    if not os.path.exists(input_dir):
        print(f"[Error] Input directory not found: {input_dir}")
        return

    # 1. 병합할 파일 검색
    search_pattern = os.path.join(input_dir, args.pattern)
    files_to_merge = glob.glob(search_pattern)
    files_to_merge.sort()

    print("=" * 80)
    print(f">>> [Merge & Analyze Pipeline]")
    print(f"    Input Directory: {input_dir}")
    print(f"    Search Pattern:  {args.pattern}")
    print(f"    Found Files:     {len(files_to_merge)}")
    print("-" * 80)

    if not files_to_merge:
        print("[Error] No files found to merge. Please check the directory and pattern.")
        return

    # 2. 결과 폴더 준비
    os.makedirs(RESULT_DIR, exist_ok=True)
    merged_filepath = os.path.join(RESULT_DIR, args.output_name)

    # 3. 병합 수행 (Merge Logic)
    print(f">>> [Step 1] Processing HDF5 files...")
    
    if len(files_to_merge) == 1:
        # 파일이 하나뿐이면 병합 대신 복사 (또는 바로 사용)
        print("    [Info] Only 1 file found. Copying instead of merging.")
        shutil.copy2(files_to_merge[0], merged_filepath)
        success = True
    else:
        # 다수 파일 병합
        success = io.merge_hdf5_files(
            file_list=files_to_merge, 
            output_path=merged_filepath, 
            verbose=True
        )

    if not success:
        print("[Error] Merge failed. Aborting analysis.")
        return

    # 4. 분석 수행 (Visualization)
    print("-" * 80)
    print(f">>> [Step 2] Running Visualization Analysis...")
    
    # 분석 결과(그래프)가 저장될 하위 폴더 생성
    # 예: 2_results/merged/analysis/merged_octamer/
    base_name = args.output_name.replace('.hdf5', '')
    analysis_dir = os.path.join(RESULT_DIR, 'analysis', base_name)
    os.makedirs(analysis_dir, exist_ok=True)

    print(f"    Target File: {merged_filepath}")
    print(f"    Output Dir:  {analysis_dir}")

    try:
        # A. RMSD 분석 (구조적 다양성 및 Funnel 확인)
        # Polymer 길이에 따라 num_residues 추정 (파일명 기반)
        num_res = 1
        if 'octamer' in base_name: num_res = 8
        elif 'hexamer' in base_name: num_res = 6
        elif 'tetramer' in base_name: num_res = 4
        elif '9mer' in base_name: num_res = 9
        
        visual.analyze_rmsd(merged_filepath, output_dir=analysis_dir, num_residues=num_res)
        print("    [Success] RMSD Analysis done.")

        # B. 에너지 랜드스케이프 및 분포 분석
        visual.analyze_and_save(merged_filepath, output_dir=analysis_dir)
        print("    [Success] Energy Landscape Analysis done.")
        
        # C. PDB 추출 (Top 5 구조) - 선택적
        # 병합된 전체 데이터셋에서 최적 구조 추출
        pdb_dir = os.path.join(RESULT_DIR, 'pdb', base_name)
        io.extract_and_save_top_structures(
            target_file=merged_filepath,
            output_dir=pdb_dir,
            top_n=5,
            cluster_threshold=45.0, # RMSD 기반 클러스터링을 위해 무시될 수 있음
            project_root=PROJECT_ROOT
        )

    except Exception as e:
        print(f"    [Error] During analysis: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 80)
    print(f"[Done] Results saved to {RESULT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge HDF5 files and run analysis.")
    
    parser.add_argument("--input_dir", type=str, default=None, 
                        help="Directory containing source HDF5 files (Default: 1_data/polymers)")
    parser.add_argument("--pattern", type=str, default="*.hdf5", 
                        help="Glob pattern to filter files (e.g., *octamer.hdf5)")
    parser.add_argument("--output_name", type=str, default="merged_output.hdf5", 
                        help="Filename for the merged HDF5 file")

    args = parser.parse_args()
    run(args)