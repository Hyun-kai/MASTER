"""
run_CPDC-CPDA_assemble.py

[기능]
"과학적 방식: 모노머 -> 다이머(Unit) 최적화 -> Unit 반복 및 Slicing으로 9-mer 생성"
이종 중합체인 CPDC-CPDA의 9-mer를 가장 안정적인 Dimer 구조를 기반으로 조립합니다.

[수정 사항 (v4.1 - Fix)]
- [Critical Fix] 명령어 실행 코드가 주석 처리되어 있던 문제 해결.
- [Smart Retry] OOM 충돌 방지 및 자동 복구 메커니즘.
"""

import os
import sys
import argparse
import subprocess
import itertools
import time

# 색상 코드
GREEN = '\033[0;32m'
BLUE = '\033[0;34m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

# ==============================================================================
# GPU Safety Functions
# ==============================================================================

def get_free_gpu_memory(gpu_id=0):
    """
    nvidia-smi를 호출하여 특정 GPU의 여유 메모리(MiB)를 반환합니다.
    """
    try:
        # nvidia-smi 쿼리 실행
        result = subprocess.check_output(
            f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits --id={gpu_id}",
            shell=True
        )
        return int(result.strip())
    except Exception as e:
        print(f"{RED}[GPU Check Error] Failed to check GPU memory: {e}{NC}")
        return 0

def wait_for_safe_gpu(gpu_id=0, required_mb=7200, check_interval=5):
    """
    GPU 여유 메모리가 required_mb 이상이 될 때까지 대기합니다.
    """
    first_wait = True
    while True:
        free_mem = get_free_gpu_memory(gpu_id)
        
        if free_mem >= required_mb:
            if not first_wait:
                print(f"\n{GREEN}[GPU Safety] Memory recovered! Starting job...{NC}")
            return
        else:
            # 부족하면 대기 메시지 출력 (덮어쓰기 위해 \r 사용)
            if first_wait:
                print(f"{YELLOW}[GPU Safety] Not enough memory. Waiting until > {required_mb}MiB free...{NC}")
                first_wait = False
            print(f"    [Waiting] Current: {free_mem}MiB < Required: {required_mb}MiB   ", end='\r')
            time.sleep(check_interval)

# ==============================================================================
# Main Helper Functions
# ==============================================================================

def count_rotamers_in_sdf(sdf_path):
    """SDF 파일 내의 분자($$$$) 개수를 셉니다."""
    count = 0
    try:
        with open(sdf_path, 'r') as f:
            for line in f:
                if line.strip() == '$$$$': count += 1
    except: sys.exit(1)
    return count

def run_command_with_retry(cmd, desc, gpu_check=True, required_mb=7200, gpu_id=0):
    """
    명령어를 실행하되, 실패 시(GPU OOM 등) 대기 후 무한 재시도합니다.
    """
    attempt = 1
    while True:
        # 1. GPU 메모리 확보 대기 (실행 직전)
        if gpu_check:
            wait_for_safe_gpu(gpu_id, required_mb)

        print(f"{YELLOW}>>> [{desc}] Running (Attempt {attempt})...{NC}")
        print(f"    Command: {cmd}") # 디버깅용 명령어 출력
        start_time = time.time()
        
        # 2. 명령어 실행
        result = subprocess.run(cmd, shell=True)
        elapsed = time.time() - start_time
        
        # 3. 결과 확인
        if result.returncode == 0:
            print(f"{GREEN}    [DONE] {desc} ({elapsed:.2f}s){NC}\n")
            return True
        else:
            # 실패 시 (OOM 가능성 높음)
            print(f"\n{RED}    [FAILED] {desc} crashed! (Likely OOM or Logic Error){NC}")
            print(f"{RED}    [Retry] Waiting 60 seconds for other processes to finish...{NC}\n")
            time.sleep(60) # 1분 쿨타임 (다른 프로세스가 지나가길 기다림)
            attempt += 1

def get_paths(main_script_path):
    root = os.path.dirname(main_script_path)
    return {
        'root': root,
        'monomers': os.path.join(root, '1_data', 'monomers'),
        'dimers': os.path.join(root, '1_data', 'dimers'),
        'polymers': os.path.join(root, '1_data', 'polymers')
    }

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scientific Pipeline: Assembly Only Mode (Smart Retry)")
    parser.add_argument("--residues", nargs="+", default=["CPDC", "CPDA"])
    parser.add_argument("--target_length", type=int, default=9)
    parser.add_argument("--grid_points", type=int, default=7)
    parser.add_argument("--threads", type=int, default=20)
    
    # [설정] GPU 사용 여부 (0: CPU, 1: GPU)
    parser.add_argument("--use_gpu", type=int, default=1, help="1 for GPU, 0 for CPU")
    
    args = parser.parse_args()

    # main.py 경로 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_script = None
    for p in [os.path.join(current_dir, "main.py"), os.path.join(os.path.dirname(current_dir), "main.py")]:
        if os.path.exists(p): main_script = p; break
    if not main_script: print(f"{RED}[Error] 'main.py' not found.{NC}"); sys.exit(1)

    paths = get_paths(main_script)
    rotamer_dir = os.path.join(paths['root'], "0_inputs", "rotamers")
    
    # --------------------------------------------------------------------------
    # [Step 1] Prep Rotamers
    # --------------------------------------------------------------------------
    res_str = " ".join(args.residues)
    missing_sdfs = False
    for r in args.residues:
        if not os.path.exists(os.path.join(rotamer_dir, f"{r}.sdf")):
            missing_sdfs = True
            break
            
    if missing_sdfs:
        run_command_with_retry(f"python {main_script} prep --residues {res_str}", "Step 1: Prep Rotamers", gpu_check=False)
    else:
        print(f"{GREEN}[Info] All Rotamer SDF files found. Skipping Prep step.{NC}")

    # 로타머 개수 파악
    rotamer_ranges = []
    for res in args.residues:
        sdf_path = os.path.join(rotamer_dir, f"{res}.sdf")
        rotamer_ranges.append(range(count_rotamers_in_sdf(sdf_path)))

    combinations = list(itertools.product(*rotamer_ranges))
    
    # GPU 옵션 문자열 생성
    gpu_opt = f"--use_gpu {args.use_gpu}"
    
    print(f"{BLUE}======================================================{NC}")
    print(f"{BLUE}   Pipeline: [Assembly Only Mode + Smart Retry]       {NC}")
    print(f"{BLUE}   * Skipping Monomer/Dimer Generation                {NC}")
    print(f"{BLUE}   * Auto-Retry on OOM Crash                          {NC}")
    print(f"{BLUE}======================================================{NC}\n")

    for idx, combo in enumerate(combinations):
        rot_list = [str(c) for c in combo]
        rot_str = " ".join(rot_list)
        job_id = f"Job {idx+1}/{len(combinations)}"
        
        # Dimer Base Name (e.g., CPDC_0-CPDA_0)
        base_parts = [f"{r}_{c}" for r, c in zip(args.residues, rot_list)]
        dimer_base_name = "-".join(base_parts)

        print(f"{BLUE}######################################################{NC}")
        print(f"{BLUE}   [{job_id}] Processing Unit: {dimer_base_name} {NC}")
        print(f"{BLUE}######################################################{NC}")

        # ----------------------------------------------------------------------
        # Check Dimer Existence
        # ----------------------------------------------------------------------
        dimer_file = os.path.join(paths['dimers'], f"{dimer_base_name}.hdf5")
        
        if not os.path.exists(dimer_file):
            print(f"{RED}[Skip] Dimer file not found: {dimer_file}. Skipping assembly.{NC}")
            continue
        else:
            print(f"{GREEN}[Info] Using existing Dimer: {dimer_base_name}{NC}")

        # ----------------------------------------------------------------------
        # Phase 4: Polymer Assembly (Tiling & Slicing)
        # ----------------------------------------------------------------------
        print(f"\n{BLUE}--- [Phase 4] {args.target_length}-mer Assembly (Tiling Dimer) ---{NC}")
        
        # 명령어 생성
        cmd_poly = (f"python {main_script} polymer --residues {res_str} --rotamers {rot_str} "
                    f"--target_length {args.target_length} --threads {args.threads} {gpu_opt} "
                    f"--input_file {dimer_file}")
        
        # [수정됨] 주석 해제 및 실행
        success = run_command_with_retry(
            cmd_poly, 
            f"Assembling {args.target_length}-mer from Dimer", 
            gpu_check=(args.use_gpu == 1), 
            required_mb=7200
        )

        if success:
            # 결과 확인 및 분석
            target_suffix = f"{args.target_length}mer"
            target_file = os.path.join(paths['polymers'], f"{dimer_base_name}_{target_suffix}.hdf5")
            
            if os.path.exists(target_file):
                # 분석 실행 (CPU 위주 작업이라 GPU 체크 끔)
                run_command_with_retry(
                    f"python {main_script} analyze --file {target_file}", 
                    "Final Analysis", 
                    gpu_check=False
                )
            else:
                print(f"{RED}[Error] Polymer assembly finished but output not found: {target_file}{NC}")
        else:
             print(f"{RED}[Error] Polymer assembly command failed.{NC}")

    print(f"\n{BLUE}======================================================{NC}")
    print(f"{BLUE}       Assembly Pipeline Completed Successfully!       {NC}")
    print(f"{BLUE}======================================================{NC}")

if __name__ == "__main__":
    main()