"""
src/bakers/chem/transform.py

[설명]
RDKit의 화학 정보 객체(Chem.Mol)와 양자 화학 계산 및 최적화에 사용되는 
ASE(Atomic Simulation Environment) 객체(Atoms) 간의 데이터 형 변환을 담당하는 유틸리티입니다.
"""

import numpy as np
from ase import Atoms
from rdkit import Chem

def rdkit_to_ase(mol: Chem.Mol) -> Atoms:
    """
    RDKit 분자 객체로부터 ASE Atoms 객체를 생성합니다.
    
    Args:
        mol (Chem.Mol): RDKit 분자 객체 (반드시 3D 좌표를 포함하는 Conformer가 있어야 함)
        
    Returns:
        Atoms: 원자 번호와 3D 좌표가 매핑된 ASE Atoms 객체
    """
    if mol.GetNumConformers() == 0:
        raise ValueError("[Error] RDKit molecule does not have 3D conformers.")
        
    conf = mol.GetConformer()
    atomic_numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    positions = conf.GetPositions()
    
    return Atoms(numbers=atomic_numbers, positions=positions)

def update_rdkit_coords(mol: Chem.Mol, new_positions: np.ndarray) -> None:
    """
    외부(예: ASE 최적화기)에서 갱신된 3D 좌표를 원본 RDKit 객체에 덮어씁니다.
    
    Args:
        mol (Chem.Mol): 좌표를 업데이트할 대상 RDKit 분자 객체
        new_positions (np.ndarray): 새로운 3D 좌표 배열 (N x 3)
    """
    if mol.GetNumAtoms() != len(new_positions):
        raise ValueError("[Error] Number of atoms and length of new positions do not match.")
        
    conf = mol.GetConformer()
    for i, pos in enumerate(new_positions): 
        conf.SetAtomPosition(i, pos.tolist())