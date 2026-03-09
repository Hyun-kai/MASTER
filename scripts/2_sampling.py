"""
scripts/2_adaptive_sampling.py

[설명]
Monomer 또는 Dimer 시스템에 대해 적응형 샘플링(Adaptive Sampling)을 수행합니다.
명령줄 인수(CLI Arguments)를 파싱하고, src.bakers.pipeline.sampling_runner의 
핵심 파이프라인을 트리거하는 가벼운 래퍼(Thin Wrapper) 스크립트입니다.
"""

import os
import sys
import argparse
import warnings

# PySisiphus 미설치 경고 무시
warnings.filterwarnings("ignore", message="PySisiphus is not installed")

# 시스템 경로에 프로젝트 src 폴더 등록
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    # 역할에 맞게 변경된 파이프라인 모듈 호출
    from bakers.pipeline.sampling_runner import run_sampling_pipeline
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found. Check environment: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Sampling for NCAA Monomers/Dimers")
    
    # 필수 인자
    parser.add_argument("--residues", nargs="+", required=True, 
                        help="Residue names (e.g., DMMA CPDC for Dimer, or just DMMA for Monomer)")
    parser.add_argument("--rotamers", nargs="+", type=int, required=True, 
                        help="Rotamer indices mapping to residues (e.g., 0 0)")
                        
    # 옵션 인자
    parser.add_argument("--max_points", type=int, default=0, 
                        help="Target sample count (0=Auto adjust by DOFs)")
    parser.add_argument("--threads", type=int, default=20, 
                        help="CPU threads for structural building and alignments")
    parser.add_argument("--grid_points", type=int, default=7, 
                        help="Initial grid density for multi-DOF sampling")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="AIMNet2 energy evaluation batch size")
    parser.add_argument("--use_gpu", action="store_true", default=True, 
                        help="Use GPU (CUDA) for AIMNet2 neural network inference")
    
    args = parser.parse_args()
    
    # 길이 검증
    if len(args.residues) != len(args.rotamers):
        print("[Error] Number of residues must match the number of rotamer indices.")
        sys.exit(1)
        
    # 파이프라인 실행
    run_sampling_pipeline(args, project_root=PROJECT_ROOT)

if __name__ == "__main__":
    main()