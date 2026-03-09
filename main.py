"""
BASE_ACID_2/main.py

[프로젝트 개요]
BAKERS 프로젝트의 통합 실행 컨트롤러(Unified Controller)입니다.
사용자의 CLI 명령을 해석하여 'scripts/' 폴더 내의 적절한 단계별 모듈을 동적으로 로드하고 실행합니다.

[수정 사항]
- [Fix] Argparse 호환성 수정: '--use_gpu' 옵션을 int(0/1)로 명시하여 전달
- [Refactor] Type Hinting 및 상세 주석 추가
- [Feature] 각 단계별 실행 로그 강화
- [Feature] Polymer 조립 단계에서 하이브리드 최적화를 제어할 수 있는 '--optimize' 플래그 추가
"""

import os
import sys
import argparse
import importlib
import multiprocessing
import warnings
from typing import Optional, List, Dict, Any

# [설정] 불필요한 경고 메시지 숨김
warnings.filterwarnings("ignore", message="PySisiphus is not installed")

# ==============================================================================
# 1. 환경 설정 및 경로 등록
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')

# 모듈 검색 경로 추가
for d in [SRC_DIR, SCRIPTS_DIR]:
    if d not in sys.path:
        sys.path.append(d)

# [Logger] 로깅 시스템 초기화
try:
    from bakers.utils import logger
    script_mode = "main"
    if len(sys.argv) > 1: 
        script_mode = sys.argv[1]
    logger.setup_logging(PROJECT_ROOT, script_name=script_mode)
except ImportError:
    pass

# 터미널 색상 코드
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

# ==============================================================================
# 2. 동적 스크립트 실행 함수 (Dispatcher)
# ==============================================================================
def run_script(module_name: str, args: argparse.Namespace) -> None:
    """
    scripts 폴더 내의 특정 모듈을 동적으로 임포트하고, 해당 모듈의 run(args) 함수를 실행합니다.
    
    Args:
        module_name (str): 실행할 모듈 이름 (예: '1_prep_rotamers')
        args (argparse.Namespace): 파싱된 인자 객체
    """
    try:
        print(f"{Colors.BLUE}>>> [Controller] Loading module: scripts.{module_name}{Colors.NC}")
        
        # importlib을 사용하여 문자열 이름으로 모듈 로드
        module = importlib.import_module(f"scripts.{module_name}")
        
        if hasattr(module, 'run'):
            module.run(args)
        else:
            print(f"{Colors.RED}[Main Error] Module 'scripts.{module_name}' has no 'run(args)' function.{Colors.NC}")
            
    except ImportError as e:
        print(f"{Colors.RED}[Main Error] Failed to import 'scripts.{module_name}': {e}{Colors.NC}")
        print(f"            Check if '{os.path.join(SCRIPTS_DIR, module_name + '.py')}' exists.")
    except Exception as e:
        print(f"{Colors.RED}[Execution Error] An unexpected error occurred:{Colors.NC}")
        import traceback
        traceback.print_exc()

# ==============================================================================
# 3. 메인 실행 로직 (CLI Parser)
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="BAKERS Unified Workflow Controller")
    subparsers = parser.add_subparsers(dest="mode", help="Workflow Step", required=True)

    # --------------------------------------------------------------------------
    # 1. Prep (전처리)
    # --------------------------------------------------------------------------
    p_prep = subparsers.add_parser("prep", help="Step 1: Prepare Rotamers & Topology")
    p_prep.add_argument("--residues", nargs="+", required=True, help="Residue names (e.g., DMMA CPDC)")

    # --------------------------------------------------------------------------
    # 2. Monomer (샘플링) - 2_sample_dimer.py 공유
    # --------------------------------------------------------------------------
    p_mono = subparsers.add_parser("monomer", help="Step 2-A: Monomer Sampling")
    p_mono.add_argument("--residues", nargs="+", required=True, help="Single residue name")
    p_mono.add_argument("--rotamers", nargs="+", type=int, required=True, help="Rotamer index")
    p_mono.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    p_mono.add_argument("--max_points", type=int, default=0, help="Max samples (0=Auto)")
    p_mono.add_argument("--grid_points", type=int, default=7, help="Initial grid density")
    p_mono.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    p_mono.add_argument("--use_gpu", type=int, default=1, help="1 for GPU, 0 for CPU")

    # --------------------------------------------------------------------------
    # 3. Residue/Dimer (샘플링) - 2_sample_dimer.py 공유
    # --------------------------------------------------------------------------
    p_res = subparsers.add_parser("residue", help="Step 2-B: Dimer Adaptive Sampling")
    p_res.add_argument("--residues", nargs="+", required=True, help="Two residue names")
    p_res.add_argument("--rotamers", nargs="+", type=int, required=True, help="Two rotamer indices")
    p_res.add_argument("--threads", type=int, default=4)
    p_res.add_argument("--max_points", type=int, default=0)
    p_res.add_argument("--grid_points", type=int, default=3)
    p_res.add_argument("--batch_size", type=int, default=32)
    p_res.add_argument("--use_gpu", type=int, default=1)

    # --------------------------------------------------------------------------
    # 4. Polymer (조립) - 3_build_polymer.py
    # --------------------------------------------------------------------------
    p_poly = subparsers.add_parser("polymer", help="Step 3: Build & Optimize Polymer")
    p_poly.add_argument("--residues", nargs="+", required=True, help="Sequence of residues")
    p_poly.add_argument("--rotamers", nargs="+", type=int, required=True, help="Sequence of rotamers")
    
    p_poly.add_argument("--target_length", type=int, default=0, help="Total length (e.g. 9 for 9mer)")
    p_poly.add_argument("--repeats", type=int, default=4, help="Multiples of unit block")
    
    p_poly.add_argument("--threads", type=int, default=4)
    p_poly.add_argument("--top_k", type=int, default=100, help="Number of candidates to keep")
    p_poly.add_argument("--use_gpu", type=int, default=1)
    
    # [Input] 명시적 입력 파일 (Unit Block HDF5) 지정
    p_poly.add_argument("--input_file", type=str, default=None, help="Explicit input HDF5 file path")

    # [수정] --optimize 플래그 정식 등록 (하위 스크립트로 안전하게 전달됨)
    p_poly.add_argument("--optimize", action="store_true", help="Enable global ASE optimization after assembly")

    # --------------------------------------------------------------------------
    # 5. Analyze (분석) - 4_analyze_results.py
    # --------------------------------------------------------------------------
    p_ana = subparsers.add_parser("analyze", help="Step 4: Analyze Results & Extract PDB")
    p_ana.add_argument("--file", type=str, help="Target HDF5 file (or filename)")
    p_ana.add_argument("--dir", type=str, help="Target directory (Batch mode)")
    p_ana.add_argument("--top_n", type=int, default=100, help="Num structures to extract")
    p_ana.add_argument("--cluster_threshold", type=float, default=45.0, help="Clustering threshold")

    # --------------------------------------------------------------------------
    # 6. Scan (검사) - 5_scan_integrity.py
    # --------------------------------------------------------------------------
    p_scan = subparsers.add_parser("scan", help="Check Data Integrity")
    group = p_scan.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to a single HDF5 file")
    group.add_argument("--dir", help="Path to a directory containing HDF5 files")
    p_scan.add_argument("--save_csv", action="store_true", default=True, help="Save stats to CSV")

    # --------------------------------------------------------------------------
    # 7. Merge (병합) - 7_merge_and_analyze.py
    # --------------------------------------------------------------------------
    p_merge = subparsers.add_parser("merge", help="Merge HDF5 files and Analyze")
    p_merge.add_argument("--pattern", type=str, required=True, help="Glob pattern (e.g., *9mer.hdf5)")
    p_merge.add_argument("--output_name", type=str, default="merged_output.hdf5", help="Output filename")
    p_merge.add_argument("--input_dir", type=str, default=None, help="Input directory (Optional)")

    # --------------------------------------------------------------------------
    # 실행 로직 (Dispatcher)
    # --------------------------------------------------------------------------
    args = parser.parse_args()

    # 모드별 스크립트 파일 매핑
    script_map = {
        "prep": "1_prep_rotamers",
        "monomer": "2_sample_dimer",
        "residue": "2_sample_dimer",
        "polymer": "3_build_polymer",
        "analyze": "4_analyze_results",
        "scan": "5_scan_integrity",
        "merge": "7_merge_and_analyze" 
    }

    if args.mode in script_map:
        run_script(script_map[args.mode], args)
    else:
        print(f"{Colors.RED}[Error] Unknown mode: {args.mode}{Colors.NC}")

if __name__ == "__main__":
    # Multiprocessing 설정 (Linux/Mac의 fork 방식 안전성 확보)
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()