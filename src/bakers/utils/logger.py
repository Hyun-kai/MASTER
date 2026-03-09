"""
src/bakers/utils/logger.py

터미널 출력을 파일로 동시에 저장(Tee)하는 로깅 유틸리티입니다.
환경 변수를 사용하여 하나의 세션(부모-자식 프로세스 포함)이 단일 로그 파일을 공유하도록 합니다.
"""

import sys
import os
import datetime
import time
from contextlib import contextmanager

@contextmanager
def check_time(name="작업"):
    # 1. 시작 시간 기록 (perf_counter가 미세 시간 측정에 더 정확함)
    start_time = time.perf_counter()
    print(f"▶️ [{name}] 시작...")
    
    try:
        # 2. 본론 실행 (여기로 코드가 들어감)
        yield
    finally:
        # 3. 종료 시간 기록 (finally를 써야 에러가 나도 시간은 찍고 죽음)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        
        # 보기 좋게 소수점 4자리까지만 출력
        print(f"⏹️ [{name}] 완료! (소요 시간: {elapsed:.4f}초)")

class DualLogger(object):
    """
    터미널(stdout/stderr)과 파일 양쪽에 동시에 출력하는 클래스
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.filename = filename
        # 'a' 모드로 열어서 기존 내용 뒤에 이어서 씀
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        # 터미널에 출력 (이미 DualLogger라면 중복 방지)
        if not isinstance(self.terminal, DualLogger):
            self.terminal.write(message)
        
        # 파일에 출력
        self.log.write(message)
        self.log.flush()  

    def flush(self):
        # 파이썬 3 호환성을 위해 필요
        if not isinstance(self.terminal, DualLogger):
            self.terminal.flush()
        self.log.flush()

def setup_logging(project_root, script_name="run"):
    """
    로그 폴더를 생성하고 stdout/stderr를 리다이렉트합니다.
    환경 변수 'BAKERS_SESSION_LOG'를 확인하여, 이미 설정된 로그 파일이 있다면 
    새 파일을 만들지 않고 해당 파일에 이어서 기록합니다.
    
    Args:
        project_root (str): 프로젝트 루트 경로
        script_name (str): 로그 파일명 접두사 (새 파일 생성 시에만 사용됨)
        
    Returns:
        str: 로그 파일 경로
    """
    # 1. 이미 로깅이 설정되어 있는지 확인 (한 프로세스 내 중복 호출 방지)
    if isinstance(sys.stdout, DualLogger):
        return sys.stdout.filename

    # 2. 환경 변수에서 기존 로그 경로 확인 (부모-자식 프로세스 간 공유)
    log_path = os.environ.get("BAKERS_SESSION_LOG")

    if log_path and os.path.exists(log_path):
        # [Case A] 이미 로그 파일이 환경 변수에 지정되어 있음 -> 해당 파일 사용
        pass 
    else:
        # [Case B] 첫 실행임 (환경 변수 없음) -> 새 파일 생성
        log_dir = os.path.join(project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)

        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{script_name}_{now}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # 환경 변수에 등록하여 이후 실행되는 자식 프로세스들이 이 경로를 알게 함
        os.environ["BAKERS_SESSION_LOG"] = log_path

    # 3. 리다이렉션 설정
    # stdout(일반 출력)과 stderr(에러)를 가로채서 파일과 터미널에 동시 출력
    try:
        dual_logger = DualLogger(log_path)
        sys.stdout = dual_logger
        sys.stderr = dual_logger # 에러도 같은 파일에 기록
    except Exception as e:
        # 혹시 파일 접근 권한 등으로 실패할 경우 기존 stdout 유지
        print(f"[Logger Error] Failed to setup logging: {e}")

    return log_path