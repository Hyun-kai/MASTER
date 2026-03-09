"""
scripts/1_prep.py

[목적 및 설명]
1. 입력된 잔기명(Residues)을 바탕으로 'SMILES_Data.csv' 파일에서 SMILES 구조를 참조합니다.
2. 각 잔기의 3D 로타머(Rotamer) 구조를 생성하여 *.sdf 파일로 저장합니다.
3. 생성된 잔기의 위상 정보와 물리화학적 특성을 파싱하여 'residue_params.py' 파일로 저장합니다.

* 본 스크립트는 실행 로직(Flow)만을 제어하며, 핵심 알고리즘은 src/bakers/pipeline 에 위치합니다.
* 하드코딩된 Fallback(기본 아미노산)을 제거하여 사용자가 제공한 CSV 데이터만 엄격하게 신뢰합니다.

[사용 모듈]
io.py, monomer_runner.py, check_topology_grid.py

[실행 방법]
$ python scripts/1_prep.py --residues AIB DAL NME --max_rotamers 10
"""

import os
import sys
import argparse

# ==============================================================================
# 1. 환경 설정 및 라이브러리 연동
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# src 디렉토리를 시스템 경로에 추가하여 내부 bakers 모듈을 임포트 가능하게 함
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# 내부 모듈 임포트
try:
    from bakers.utils import io
    from bakers.pipeline import monomer_runner
    
    # 시각화 모듈 로드 가능 여부 검사 (의존성 문제 방지)
    from bakers.analytics import check_topology_grid
    HAS_VISUALIZATION = True
except ImportError as e:
    print(f"[Warning] Visualization module or dependencies missing: {e}")
    HAS_VISUALIZATION = False


# ==============================================================================
# 2. 메인 파이프라인 로직
# ==============================================================================
def run(args: argparse.Namespace) -> None:
    """
    모노머 준비 과정을 일괄 통제하는 메인 함수입니다.
    
    Args:
        args: argparse를 통해 입력된 커맨드라인 인자 객체
    """
    print(f">>> [PREP] Rotamer Preparation Pipeline Started")
    print(f"    - Target Residues : {args.residues}")
    print(f"    - Max Rotamers    : {args.max_rotamers}")
    print(f"    - RMSD/TFD Thresh : {args.rmsd_thresh} / {args.tfd_thresh}")
    
    # [1] I/O 경로 설정 및 디렉토리 생성
    input_dir = os.path.join(PROJECT_ROOT, '0_inputs')
    rotamer_dir = os.path.join(input_dir, 'rotamers')
    vis_dir = os.path.join(input_dir, 'analysis', 'topology_checks')
    csv_path = os.path.join(input_dir, 'SMILES_Data.csv')
    
    os.makedirs(rotamer_dir, exist_ok=True)
    if HAS_VISUALIZATION:
        os.makedirs(vis_dir, exist_ok=True)
    
    # [2] SMILES 데이터 로드 (Fallback 삭제)
    if not os.path.exists(csv_path):
        print(f"[Error] Required file 'SMILES_Data.csv' not found at {csv_path}. Aborting.")
        sys.exit(1)
        
    # io 모듈을 통해 데이터 파싱
    smiles_map = io.load_smiles_from_csv(csv_path)
    if not smiles_map:
        print("[Error] Failed to load any valid SMILES data from the CSV. Aborting.")
        sys.exit(1)
    
    print(f"    [Info] Loaded {len(smiles_map)} SMILES entries from CSV.")

    # 각 잔기별 추출된 파라미터를 담을 저장소
    params_dict = {}

    # [3] 요청된 개별 잔기 순회 처리
    for res_name in args.residues:
        if res_name not in smiles_map:
            print(f"    [Skip] '{res_name}' not found in SMILES list. Please check the CSV.")
            continue
            
        full_smiles = smiles_map[res_name]
        
        # 내부 pipeline 모듈로 복잡한 비즈니스 로직(생성, 최적화, 위상분석 등) 위임
        entry = monomer_runner.prepare_monomer(
            res_name=res_name,
            full_smiles=full_smiles,
            rotamer_dir=rotamer_dir,
            vis_dir=vis_dir,
            has_visualization=HAS_VISUALIZATION,
            max_rotamers=args.max_rotamers,
            threshold_rmsd=args.rmsd_thresh,
            threshold_tfd=args.tfd_thresh
        )
        
        # 파이프라인 처리가 성공하여 데이터를 반환한 경우에만 딕셔너리에 병합
        if entry is not None:
            params_dict[res_name] = entry

    # [4] 전체 결과 데이터 일괄 저장
    if params_dict:
        params_path = os.path.join(input_dir, 'residue_params.py')
        # io 모듈을 통해 파이썬 딕셔너리를 파일 형태로 저장
        io.save_residue_params(params_dict, params_path)
        print(f"    [Done] Saved generated parameters to {params_path}")
        
        if HAS_VISUALIZATION:
            print(f"    [Done] Visual topology reports saved to {vis_dir}")
    else:
        print("    [Warning] No residues were successfully processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monomer Pipeline: Prep Rotamers & Topology")
    
    # 필수 입력: 처리할 잔기 이름 리스트
    parser.add_argument("--residues", nargs="+", required=True, 
                        help="List of residue names to process (must exist in SMILES_Data.csv)")
    
    # 선택 입력: 로타머 샘플링 세부 설정 (하드코딩 제거 목적)
    parser.add_argument("--max_rotamers", type=int, default=10, 
                        help="Maximum rotamers to generate and save per residue (default: 10)")
    parser.add_argument("--rmsd_thresh", type=float, default=2.0, 
                        help="RMSD clustering threshold for acyclic molecules in Angstroms (default: 2.0)")
    parser.add_argument("--tfd_thresh", type=float, default=0.1, 
                        help="TFD clustering threshold for cyclic molecules (default: 0.1)")
    
    run(parser.parse_args())