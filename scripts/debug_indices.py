"""
scripts/debug_indices.py

[기능]
1. Residue Params와 Rotamer SDF를 로드하여, 원자 인덱스가 표시된 2D 이미지를 생성합니다.
   (params가 올바른 원자를 가리키는지 시각적으로 확인용)
2. 사용자가 입력한 각도를 적용하여 단일 모노머 PDB를 저장합니다.
   (각도 적용 로직이 정상인지 확인용)
"""

import os
import sys
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# 프로젝트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from bakers.chem import topology
from rdkit.Chem.rdMolTransforms import SetDihedralDeg

def draw_atom_indices(mol, name, out_dir):
    """원자 인덱스를 포함한 분자 이미지를 그립니다."""
    d = Draw.MolDraw2DCairo(600, 600)
    
    # 원자 인덱스 라벨링
    for atom in mol.GetAtoms():
        atom.SetProp('atomLabel', str(atom.GetIdx()))
    
    d.DrawMolecule(mol)
    d.FinishDrawing()
    
    png_path = os.path.join(out_dir, f"DEBUG_MAP_{name}.png")
    with open(png_path, 'wb') as f:
        f.write(d.GetDrawingText())
    print(f"[Map] Saved Atom Index Map: {png_path}")

def run(args):
    # 1. 로드
    params_path = os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py')
    full_params = topology.load_residue_params(params_path)
    
    rotamer_dir = os.path.join(PROJECT_ROOT, '1_data', 'rotamers')
    if not os.path.exists(rotamer_dir):
        rotamer_dir = os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers')

    out_dir = os.path.join(PROJECT_ROOT, '2_results', 'debug')
    os.makedirs(out_dir, exist_ok=True)

    target_res = args.residue
    if target_res not in full_params:
        print(f"[Error] Residue {target_res} not in params.")
        return

    # 2. SDF 로드
    sdf_path = os.path.join(rotamer_dir, f"{target_res}.sdf")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0] # Rotamer 0
    
    # 3. 인덱스 맵 그리기
    draw_atom_indices(mol, target_res, out_dir)
    
    # 4. Params 정보 출력
    p = full_params[target_res]
    print(f"\n[Analysis: {target_res}]")
    print(f"  - DOFs (Angles): {p['dofs']}")
    print(f"  - Upper Connect (Head): {p['upper_connect_indices']}")
    print(f"  - Lower Connect (Tail): {p['lower_connect_indices']}")
    print(f"  - Residue Indices (Core): {p['residue_indices']}")

    # 5. 각도 적용 테스트
    if args.angle is not None:
        conf = mol.GetConformer()
        dofs = p['dofs']
        
        # 첫 번째 DOF에만 테스트 각도 적용 (또는 순차 적용)
        # 여기서는 입력된 값을 순서대로 적용
        angle_vals = args.angle
        
        print(f"\n[Test] Applying Angles: {angle_vals}")
        for i, val in enumerate(angle_vals):
            if i >= len(dofs): break
            indices = dofs[i] # (a1, a2, a3, a4)
            print(f"  -> Setting DOF {i+1} (Atoms {indices}) to {val} deg")
            SetDihedralDeg(conf, indices[0], indices[1], indices[2], indices[3], float(val))
        
        pdb_path = os.path.join(out_dir, f"DEBUG_TEST_{target_res}.pdb")
        Chem.MolToPDBFile(mol, pdb_path)
        print(f"[Test] Saved Modified Structure: {pdb_path}")
        print("  -> Open this PDB in PyMOL to verify if the angle matches your expectation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residue", required=True, help="Residue Name (e.g. CPDA)")
    parser.add_argument("--angle", nargs='+', type=float, help="Test angles (e.g. -30.0 180.0)")
    args = parser.parse_args()
    run(args)