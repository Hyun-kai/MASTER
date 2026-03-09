"""
scripts/make_mock_input_manual.py

[기능]
사용자가 직접 측정한 각도(Values)를 입력받아,
파이프라인이 즉시 사용할 수 있는 HDF5 형식으로 포장합니다.
불안정한 PDB 파싱 과정을 건너뛰고 '정답 데이터'를 강제 주입할 때 사용합니다.
"""

import os
import argparse
import h5py
import numpy as np

def run(args):
    # 입력받은 각도 리스트
    input_angles = args.values
    
    # 데이터 형변환 (list -> numpy array)
    # Shape: (1, N_degrees_of_freedom)
    points_data = np.array([input_angles], dtype=np.float32)
    
    # 에너지는 검증 단계에서 무시되거나 최우선 순위가 되도록 설정
    # 매우 낮은 값(-9999)을 주어 샘플러가 이 포인트를 "절대적인 정답"으로 취급하게 유도
    energies_data = np.array([-9999.0], dtype=np.float32)

    # 저장 경로 설정
    save_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    
    # HDF5 파일 생성
    with h5py.File(save_path, 'w') as f:
        # 'points': 각도 데이터 (degree)
        f.create_dataset('points', data=points_data)
        # 'energies' & 'values': 에너지 데이터 (호환성 유지)
        f.create_dataset('energies', data=energies_data)
        f.create_dataset('values', data=energies_data)
        
    print(f"\n[Success] Manual Input Saved: {save_path}")
    print(f"    -> Injected Angles: {input_angles}")
    print(f"    -> You can now run 'scripts/3_build_polymer.py' with this file.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mock HDF5 from manual angle values")
    parser.add_argument("--values", nargs='+', type=float, required=True, 
                        help="List of dihedral angles in order (e.g. -96.87 -96.85 -30.07 -70.00)")
    parser.add_argument("--out", required=True, 
                        help="Output path (e.g. 1_data/dimers/CPDA_0-DMMA_0.hdf5)")
    
    args = parser.parse_args()
    run(args)