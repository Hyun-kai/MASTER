"""
scripts/3a_modify_terminal.py

[설명]
이전 단계에서 생성된 폴리머 구조(SDF 파일)를 읽어들여, 
양 끝단에 남아있는 미반응 Methyl Alkyne (-C#C-CH3) 그룹을 
Methyl (-CH3) 그룹으로 치환한 뒤, 최종 기하 구조 최적화를 수행합니다.

[최종 수정 내역]
- [Type Hint Fix] 파이썬 내장 타입 힌트인 소문자 'tuple'을 사용하여 
  NameError('Tuple' is not defined)를 해결했습니다.
"""

import os
import sys
import glob
import argparse
import numpy as np
import tqdm
import warnings
import torch

from ase import Atoms
from ase.optimize import BFGS
from rdkit import Chem

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

if SRC_DIR not in sys.path: sys.path.append(SRC_DIR)

try:
    from bakers.sim.calculator import EnsembleAIMNet2
    from bakers.utils import io
except ImportError as e:
    print(f"[Critical Error] BAKERS modules not found: {e}")
    sys.exit(1)

GREEN, RED, YELLOW, BLUE, NC = '\033[0;32m', '\033[0;31m', '\033[0;33m', '\033[0;34m', '\033[0m'

# ==============================================================================
# 1. 분자 구조 편집 (Terminal Modification)
# ==============================================================================

def replace_terminal_alkynes(mol: Chem.Mol) -> Chem.Mol:
    """
    분자 내의 '-C#C-CH3' 그룹을 찾아 '-CH3'로 변환합니다.
    """
    # SMARTS: [어떤원자:1] - [탄소:2] # [탄소:3] - [메틸탄소:4]
    smarts = '[*:1]-[C:2]#[C:3]-[C;H3:4]'
    query = Chem.MolFromSmarts(smarts)
    
    matches = mol.GetSubstructMatches(query)
    if not matches:
        return mol # 변경할 말단이 없으면 원본 반환
        
    rwmol = Chem.RWMol(mol)
    conf = rwmol.GetConformer()
    
    atoms_to_remove = []
    
    for match in matches:
        idx_core, idx_a1, idx_a2, idx_me = match
        
        pos_core = np.array(conf.GetAtomPosition(idx_core))
        pos_a1 = np.array(conf.GetAtomPosition(idx_a1))
        pos_me = np.array(conf.GetAtomPosition(idx_me))
        
        # 1. 3D 좌표 보정 (메틸 그룹을 본체 쪽으로 당겨옵니다)
        # 방향 벡터: 본체(core) -> 알카인 첫번째 탄소(a1)
        direction_vec = pos_a1 - pos_core
        norm = np.linalg.norm(direction_vec)
        
        if norm > 0:
            direction_vec = direction_vec / norm
            # 이상적인 C-C 단일 결합 거리 (1.50 Å) 위치 계산
            target_pos_me = pos_core + direction_vec * 1.50
            
            # 이동해야 할 변위(Translation Vector) 계산
            translation = target_pos_me - pos_me
            
            # 메틸 탄소 및 결합된 수소 원자들 탐색
            moving_indices = [idx_me]
            me_atom = rwmol.GetAtomWithIdx(idx_me)
            for nbr in me_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 1: # 수소인 경우 함께 이동
                    moving_indices.append(nbr.GetIdx())
                    
            # 좌표 일괄 이동 적용
            for idx in moving_indices:
                old_pos = np.array(conf.GetAtomPosition(idx))
                conf.SetAtomPosition(idx, (old_pos + translation).tolist())
                
        # 2. 새로운 결합 생성 (본체와 메틸 탄소 직접 연결)
        rwmol.AddBond(idx_core, idx_me, Chem.BondType.SINGLE)
        
        # 3. 삭제할 알카인 탄소 인덱스 수집
        atoms_to_remove.extend([idx_a1, idx_a2])
        
    # 4. 원자 삭제 (인덱스 변동을 막기 위해 반드시 역순으로 삭제해야 함)
    # 중복 제거 후 내림차순 정렬
    atoms_to_remove = sorted(list(set(atoms_to_remove)), reverse=True)
    for idx in atoms_to_remove:
        rwmol.RemoveAtom(idx)
        
    # 구조 검증 및 안정화
    modified_mol = rwmol.GetMol()
    Chem.SanitizeMol(modified_mol)
    
    return modified_mol

# ==============================================================================
# 2. ASE 최적화 (Optimization)
# ==============================================================================

# [핵심 수정] Tuple -> tuple 로 변경하여 내장 타입 힌트 사용
def optimize_molecule(mol: Chem.Mol, calc: EnsembleAIMNet2) -> tuple[Chem.Mol, float]:
    """ASE를 이용해 분자의 구조를 최적화합니다."""
    conf = mol.GetConformer()
    ase_atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()], 
                      positions=conf.GetPositions())
                      
    if hasattr(calc, 'reset'): calc.reset()
    ase_atoms.calc = calc
    
    # 뼈대를 자유롭게 풀어 분자 전체의 장력을 완벽히 이완(Relaxation)시킵니다.
    try:
        opt = BFGS(ase_atoms, logfile=None)
        opt.run(fmax=0.05, steps=500) # 최대 500스텝으로 충분히 최적화
        
        final_coords = ase_atoms.get_positions()
        final_energy = ase_atoms.get_potential_energy()
        
        # 최적화된 좌표를 RDKit Mol에 업데이트
        for i, pos in enumerate(final_coords): 
            conf.SetAtomPosition(i, pos.tolist())
            
        return mol, final_energy
    except Exception as e:
        print(f" {RED}[Opt Error] Optimization failed: {e}{NC}")
        return mol, 0.0

# ==============================================================================
# 3. 메인 실행 로직 (Main)
# ==============================================================================

def run(args):
    print(f"{GREEN}>>> [Step 4] Terminal Modification & Final Optimization{NC}")
    
    # 입력 폴더 설정 (기본값: 2_results/pdb/polymers/*/sdf/)
    input_dir = args.input_dir
    if not input_dir:
        print(f"{RED}[Error] Please provide an input directory containing SDF files using --input_dir{NC}")
        return
        
    sdf_files = glob.glob(os.path.join(input_dir, "*.sdf"))
    if not sdf_files:
        print(f"{YELLOW}[Warn] No SDF files found in {input_dir}{NC}")
        return
        
    print(f"{BLUE}[Info] Found {len(sdf_files)} SDF files to process.{NC}")
    
    # 출력 폴더 설정
    output_dir = args.output_dir
    if not output_dir:
        parent_dir = os.path.dirname(input_dir.rstrip('/'))
        output_dir = os.path.join(parent_dir, "modified_sdfs")
        
    os.makedirs(output_dir, exist_ok=True)
    print(f"{BLUE}[Info] Optimized structures will be saved to: {output_dir}{NC}")
    
    # AIMNet2 계산기 초기화
    model_files = [os.path.join(PROJECT_ROOT, '0_inputs', 'models', f) for f in os.listdir(os.path.join(PROJECT_ROOT, '0_inputs', 'models')) if f.endswith('.jpt')]
    use_cuda = (args.use_gpu == 1) and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    try:
        calc = EnsembleAIMNet2(model_files, device=device)
    except Exception as e:
        print(f"{RED}[Error] Failed to load AIMNet2 calculator: {e}{NC}")
        return

    success_count = 0
    
    # 처리 루프
    pbar = tqdm.tqdm(sdf_files, desc="[Modifying & Optimizing]", colour='cyan')
    for sdf_path in pbar:
        filename = os.path.basename(sdf_path)
        
        try:
            # 1. 분자 로드
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
            mol = suppl[0]
            if mol is None: continue
            
            # 2. 말단 치환 (CCCH3 -> CH3)
            mod_mol = replace_terminal_alkynes(mol)
            
            # 3. 구조 최적화
            opt_mol, final_energy = optimize_molecule(mod_mol, calc)
            
            # 4. 정보 업데이트 및 저장
            opt_mol.SetProp("Energy", str(final_energy))
            opt_mol.SetProp("Modified", "True")
            
            save_path = os.path.join(output_dir, f"mod_{filename}")
            
            # 기존 io.py의 save_sdf를 사용하여 결합 정보를 완벽하게 유지하며 저장
            positions = opt_mol.GetConformer().GetPositions()
            numbers = np.array([atom.GetAtomicNum() for atom in opt_mol.GetAtoms()])
            info = {"Energy": final_energy, "Modification": "Terminal_Alkyne_Removed"}
            
            if io.save_sdf(save_path, numbers, positions, info):
                success_count += 1
                
        except Exception as e:
            print(f"\n{RED}[Error] Processing {filename} failed: {e}{NC}")
            
    print(f"\n{GREEN}>>> [Done] Successfully modified and optimized {success_count}/{len(sdf_files)} structures.{NC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input SDF files generated from Step 3.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save modified SDFs. Defaults to 'modified_sdfs' next to input.")
    parser.add_argument("--use_gpu", type=int, default=1, help="Use GPU for AIMNet2 optimization (1=True, 0=False).")
    
    args = parser.parse_args()
    run(args)