"""
src/bakers/utils/safety.py

[기능]
시뮬레이션 실행 중 예기치 않은 종료(KeyboardInterrupt 등)가 발생했을 때,
현재까지 계산된 데이터를 안전하게 백업하고 시각화하는 안전장치 기능을 제공합니다.
"""

import os
import sys
from . import io, visual

# 터미널 출력용 색상 코드
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
NC = '\033[0m'

def handle_force_stop(name, points, values, project_root, xyzs=None):
    """
    강제 종료(Ctrl+C) 신호가 감지되었을 때 호출되는 핸들러입니다.
    
    데이터를 '2_results/ForceStop/<name>' 폴더에 저장하고,
    즉시 분석(Landscape 시각화)을 수행하여 사용자가 상태를 파악할 수 있게 합니다.

    Args:
        name (str): 실험 이름 (예: ALA_0-B2ALA_0)
        points (np.ndarray): 현재까지 샘플링된 좌표 데이터
        values (np.ndarray): 현재까지 계산된 에너지 데이터
        project_root (str): 프로젝트 루트 디렉토리 경로
    """
    print(f"\n{YELLOW}>>> [Interrupt] 강제 종료(Ctrl+C) 감지! 데이터 백업 절차를 시작합니다...{NC}")

    # 1. 저장할 폴더 경로 생성 (/BAKERS/2_results/ForceStop/<name>)
    force_stop_root = os.path.join(project_root, '2_results', 'ForceStop')
    specific_dir = os.path.join(force_stop_root, name)
    
    try:
        os.makedirs(specific_dir, exist_ok=True)
    except OSError as e:
        print(f"{RED}    [Error] 백업 폴더 생성 실패: {e}{NC}")
        return

    # 2. 데이터 저장 (.hdf5)
    save_path = os.path.join(specific_dir, f"{name}.hdf5")
    
    if len(points) > 0:
        try:
            io.save_results_hdf5(save_path, points, values, xyzs=xyzs)
            print(f"{YELLOW}    [Saved] 긴급 백업 완료: {save_path}{NC}")
            print(f"    [Info] 백업된 데이터 포인트: {len(points)}개")
            if xyzs is not None:
                print(f"    [Info] 좌표 데이터 포함됨 ({len(xyzs)}개)")
        except Exception as e:
            print(f"{RED}    [Error] 데이터 저장 실패: {e}{NC}")
            return
            
        # 3. 자동 분석 및 시각화 (Landscape 그리기)
        #    사용자가 중단된 시점의 탐색 상태를 눈으로 확인할 수 있도록 함
        if len(points) > 10:
            print(f"    [Analysis] 긴급 시각화를 수행합니다...")
            try:
                visual.analyze_and_save(save_path)
                print(f"    [Visual] 시각화 파일 생성 완료")
            except Exception as e:
                print(f"{RED}    [Warning] 시각화 실패: {e}{NC}")
        else:
            print(f"    [Info] 데이터가 너무 적어 시각화는 생략합니다.")
    else:
        print(f"{RED}    [Empty] 저장할 유효 데이터가 없습니다.{NC}")