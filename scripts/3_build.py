"""
scripts/3_build_polymer.py

[설명]
샘플링된 HDF5 데이터를 기반으로 지정된 길이의 폴리머(Polymer)를 조립합니다.
명령줄 인수를 파싱하고, bakers.pipeline.polymer_runner의 코어 로직을 호출하는 가벼운 래퍼 스크립트입니다.
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from bakers.pipeline.polymer_runner import run_building_pipeline
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Build and optimize polymer structures from sampled blocks")
    
    # 필수 인자
    parser.add_argument("--residues", nargs="+", required=True, help="Base unit residues (e.g., DMMA CPDC)")
    parser.add_argument("--rotamers", nargs="+", type=int, required=True, help="Base unit rotamers (e.g., 0 0)")
    
    # 조립 전략 인자
    parser.add_argument("--target_length", type=int, default=0, help="Absolute length of the target polymer")
    parser.add_argument("--repeats", type=int, default=2, help="Number of times to repeat the base unit")
    parser.add_argument("--top_k", type=int, default=100, help="Number of lowest energy structures to process")
    
    # 성능 및 I/O
    parser.add_argument("--threads", type=int, default=20, help="Number of multiprocessing threads")
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU (1) or CPU (0)")
    parser.add_argument("--input_file", type=str, default=None, help="Explicit HDF5 file path (Optional)")
    
    # 최적화 플래그
    parser.add_argument("--optimize", action="store_true", help="Enable global ASE optimization after rigid body assembly")
    
    args = parser.parse_args()
    
    # 파이프라인 트리거
    run_building_pipeline(args, project_root=PROJECT_ROOT)

if __name__ == "__main__":
    main()