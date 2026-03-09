"""
run_PHI.py

[기능]
"직접 조립(Direct Assembly) 방식: 모노머 샘플링 및 분석 -> 모노머 반복 연결(Tiling) -> 폴리머 분석"
하나의 모노머(Monomer)를 타겟 길이(예: 4-mer)만큼 직접 연결하여 폴리머를 생성합니다.

[수정 사항]
- 다이머(Dimer) 생성 단계를 완전히 제거하고, 모노머를 직접 폴리머로 조립하도록 파이프라인을 단축했습니다.
- 모노머 샘플링 직후 해당 모노머를 평가하는 분석(Analysis) 로직을 추가했습니다.
- 하위 스크립트(main.py) 호출 시 GPU 옵션 전달 방식을 정수형(0 or 1)으로 고정하여 안정성을 높였습니다.
- [추가] 하이브리드 최적화 제어를 위한 --optimize 플래그 연동 로직을 추가했습니다.
"""

import os
import sys
import argparse
import subprocess
import itertools
import time

# ==============================================================================
# [UI/UX] 콘솔 출력을 위한 색상 코드
# ==============================================================================
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

# ==============================================================================
# 1. 유틸리티 함수 (Helper Functions)
# ==============================================================================

def count_rotamers_in_sdf(sdf_path: str) -> int:
    """SDF 파일 내의 분자($$$$) 개수를 셉니다."""
    count = 0
    try:
        with open(sdf_path, 'r') as f:
            for line in f:
                if line.strip() == '$$$$': count += 1
    except Exception: 
        sys.exit(1)
    return count

def run_command(cmd: str, desc: str, allow_fail: bool = False):
    """쉘 명령어를 실행하고 소요 시간과 결과를 출력합니다."""
    print(f"{YELLOW}>>> [{desc}] Running...{NC}")
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        if allow_fail:
            print(f"{RED}    [WARN] {desc} Failed - Continuing...{NC}")
        else:
            print(f"{RED}    [FAILED] {desc} ({elapsed:.2f}s) - STOPPING.{NC}")
            sys.exit(1)
    else:
        print(f"{GREEN}    [DONE] {desc} ({elapsed:.2f}s){NC}\n")

def get_paths(main_script_path: str) -> dict:
    """프로젝트의 주요 디렉토리 경로를 생성하여 반환합니다."""
    root = os.path.dirname(main_script_path)
    return {
        'root': root,
        'monomers': os.path.join(root, '1_data', 'monomers'),
        'polymers': os.path.join(root, '1_data', 'polymers')
    }

# ==============================================================================
# 2. 메인 파이프라인 (Main Pipeline)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scientific Pipeline: Monomer -> Polymer Assembly")
    parser.add_argument("--residues", nargs="+", default=["PHI"])
    parser.add_argument("--target_length", type=int, default=4)
    parser.add_argument("--grid_points", type=int, default=7)
    parser.add_argument("--threads", type=int, default=20)
    
    # [설정] GPU 사용 여부 (0: CPU, 1: GPU)
    parser.add_argument("--use_gpu", type=int, default=1, help="1 for GPU, 0 for CPU")
    
    # [수정] 최적화 제어 스위치 추가
    parser.add_argument("--optimize", action="store_true", help="Enable global ASE optimization after assembly")
    
    args = parser.parse_args()

    # main.py 경로 동적 탐색
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = None
    for p in [os.path.join(current_dir, "main.py"), os.path.join(os.path.dirname(current_dir), "main.py")]:
        if os.path.exists(p): 
            main_script = p
            break
            
    if not main_script: 
        print(f"{RED}[Error] 'main.py' not found.{NC}")
        sys.exit(1)

    paths = get_paths(main_script)
    rotamer_dir = os.path.join(paths['root'], "0_inputs", "rotamers")
    
    # --------------------------------------------------------------------------
    # [Phase 1] Prep Rotamers (초기 형태학적 로타머 준비)
    # --------------------------------------------------------------------------
    res_str = " ".join(args.residues)
    run_command(f"python {main_script} prep --residues {res_str}", "Phase 1: Prep Rotamers")

    # 로타머 개수 파악을 통한 조합(Combinations) 생성
    rotamer_ranges = []
    for res in args.residues:
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        rotamer_ranges.append(range(count_rotamers_in_sdf(sdf_path)))

    combinations = list(itertools.product(*rotamer_ranges))
    
    # 하위 스크립트로 전달할 GPU 옵션 문자열
    gpu_opt = f"--use_gpu {args.use_gpu}"
    
    print(f"{BLUE}======================================================{NC}")
    print(f"{BLUE}   Pipeline: Monomer -> {args.target_length}-mer Assembly            {NC}")
    print(f"{BLUE}   * Method: Direct Monomer Tiling & Analysis         {NC}")
    print(f"{BLUE}======================================================{NC}\n")

    for idx, combo in enumerate(combinations):
        rot_list = [str(c) for c in combo]
        rot_str = " ".join(rot_list)
        job_id = f"Job {idx+1}/{len(combinations)}"
        
        # 단위 블록의 기본 파일명 생성 (예: PHI_0)
        base_parts = [f"{r}_{c}" for r, c in zip(args.residues, rot_list)]
        unit_base_name = "-".join(base_parts)

        print(f"{BLUE}######################################################{NC}")
        print(f"{BLUE}   [{job_id}] Processing Unit: {unit_base_name} {NC}")
        print(f"{BLUE}######################################################{NC}")

        # ----------------------------------------------------------------------
        # Phase 2: Monomer Sampling & Analysis
        # ----------------------------------------------------------------------
        print(f"\n{BLUE}--- [Phase 2] Monomer Sampling & Analysis ---{NC}")
        primary_mono_file = ""
        
        for i, (r, rot) in enumerate(zip(args.residues, rot_list)):
            mono_name = f"{r}_{rot}"
            mono_file = os.path.join(paths['monomers'], f"{mono_name}.hdf5")
            
            # 폴리머 조립의 뼈대가 될 첫 번째 모노머 파일을 지정
            if i == 0: 
                primary_mono_file = mono_file 
                
            # 모노머 샘플링 진행
            if not os.path.exists(mono_file):
                cmd_mono = (f"python {main_script} monomer --residues {r} --rotamers {rot} "
                            f"--grid_points {args.grid_points} --threads {args.threads} {gpu_opt}")
                run_command(cmd_mono, f"Sampling Monomer {mono_name}")
            else:
                print(f"{YELLOW}[Info] Monomer {mono_name} already exists.{NC}")

            # 샘플링 완료 직후 모노머 구조 분석(Analysis) 수행
            if os.path.exists(mono_file):
                run_command(f"python {main_script} analyze --file {mono_file}", f"Analyzing Monomer {mono_name}")

        # 모노머 파일이 정상적으로 준비되지 않았다면 안전하게 건너뛰기
        if not primary_mono_file or not os.path.exists(primary_mono_file):
            print(f"{RED}[Error] Monomer generation failed. Skipping assembly for {unit_base_name}.{NC}")
            continue

        # ----------------------------------------------------------------------
        # Phase 3: Polymer Assembly (Direct Tiling)
        # ----------------------------------------------------------------------
        print(f"\n{BLUE}--- [Phase 3] {args.target_length}-mer Assembly (Direct Tiling) ---{NC}")
        
        # [수정] 최적화 옵션 문자열 생성
        opt_str = " --optimize" if args.optimize else ""
        
        # main.py의 polymer 모드 호출 
        # (방금 생성 및 분석을 마친 primary_mono_file을 입력 파일로 강제 주입하여 이어붙임)
        cmd_poly = (f"python {main_script} polymer --residues {res_str} --rotamers {rot_str} "
                    f"--target_length {args.target_length} --threads {args.threads} {gpu_opt} "
                    f"--input_file {primary_mono_file}{opt_str}") 
                    
        run_command(cmd_poly, f"Assembling {args.target_length}-mer from Monomer {unit_base_name}")

        # ----------------------------------------------------------------------
        # Phase 4: Final Polymer Analysis
        # ----------------------------------------------------------------------
        print(f"\n{BLUE}--- [Phase 4] Final Polymer Analysis ---{NC}")
        target_suffix = f"{args.target_length}mer"
        target_file = os.path.join(paths['polymers'], f"{unit_base_name}_{target_suffix}.hdf5")
        
        if os.path.exists(target_file):
            run_command(f"python {main_script} analyze --file {target_file}", f"Final Analysis of {target_suffix}")
        else:
            print(f"{RED}[Error] Polymer assembly failed for {unit_base_name}.{NC}")

    print(f"\n{BLUE}======================================================{NC}")
    print(f"{BLUE}      Pipeline Completed Successfully!                {NC}")
    print(f"{BLUE}======================================================{NC}")

if __name__ == "__main__":
    main()