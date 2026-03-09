"""
src/bakers/sim/optimize.py

[설명]
완성된 폴리머 구조의 기하학적 불안정성 및 구조 왜곡(Distortion)을 해소하기 위해 
ASE 프레임워크와 제공된 계산기(예: AIMNet2)를 사용하여 전역 최적화를 수행합니다.
"""

import numpy as np
from ase.optimize import BFGS
from rdkit import Chem

# 방금 분리한 변환 유틸리티 임포트
from bakers.chem.transform import rdkit_to_ase, update_rdkit_coords

def global_optimization(mol: Chem.Mol, calc, fmax: float = 0.05, steps: int = 100) -> float:
    """
    강체(Rigid Body) 조립이 완료된 최종 분자에 대해 전체 구조 최적화(Relaxation)를 수행합니다.

    Args:
        mol (Chem.Mol): 최적화를 수행할 RDKit 분자 객체
        calc: 에너지와 힘(Force)을 계산할 ASE 호환 계산기 (예: EnsembleAIMNet2)
        fmax (float): 수렴 기준이 되는 최대 힘 (Force) 임계값. 기본값 0.05
        steps (int): 최대 최적화 스텝 제한 (무한 루프 방지). 기본값 100

    Returns:
        float: 최적화가 완료된 분자의 최종 잠재 에너지 (Potential Energy)
    """
    if calc is None:
        raise ValueError("[Error] Calculator is not provided for optimization.")

    # 1. 계산을 위해 ASE 객체로 형 변환
    ase_atoms = rdkit_to_ase(mol)
    
    # 2. 계산기 초기화 및 할당
    if hasattr(calc, 'reset'):
        calc.reset()
    ase_atoms.calc = calc
    
    # 3. 전체 구조 이완 (BFGS 옵티마이저)
    # logfile=None 설정으로 불필요한 콘솔 출력을 억제합니다.
    opt = BFGS(ase_atoms, logfile=None)
    
    # 예외 발생 시 에러를 숨기지 않고 파이프라인으로 전파하여 불량 데이터를 추적합니다. (Fallback 제거)
    opt.run(fmax=fmax, steps=steps)
    
    # 4. 안정화된 좌표를 원본 RDKit 객체에 동기화
    optimized_coords = ase_atoms.get_positions()
    update_rdkit_coords(mol, optimized_coords)
    
    # 5. 최종 에너지 반환
    return ase_atoms.get_potential_energy()