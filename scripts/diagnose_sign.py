"""
scripts/diagnose_sign.py

[기능]
1. 모노머에 강제로 '음수 각도(-30.0)'를 적용합니다.
2. 결과 구조의 각도를 측정하여 출력합니다.
   - 결과가 -30.0도 근처면: RDKit과 파라미터 정의는 정상 -> 모노머 카이랄성 문제 의심.
   - 결과가 +30.0도 근처면: 파라미터 정의(dofs) 순서가 반대 -> 순서 뒤집기 필요.
3. (옵션) 카이랄 중심(Chiral Centers)의 R/S 설정을 출력하여 모노머의 입체 이성질체 종류를 확인합니다.
"""

import os
import sys
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

# 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from bakers.chem import topology

def run(args):
    # 1. 로드
    params_path = os.path.join(PROJECT_ROOT, '0_inputs', 'residue_params.py')
    full_params = topology.load_residue_params(params_path)
    
    rotamer_dir = os.path.join(PROJECT_ROOT, '1_data', 'rotamers')
    if not os.path.exists(rotamer_dir):
        rotamer_dir = os.path.join(PROJECT_ROOT, '0_inputs', 'rotamers')
        
    res_name = args.residue
    if res_name not in full_params:
        print(f"[Error] {res_name} not found in params.")
        return

    sdf_path = os.path.join(rotamer_dir, f"{res_name}.sdf")
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = suppl[0]
    conf = mol.GetConformer()
    
    p = full_params[res_name]
    dofs = p['dofs']
    
    print(f"========================================")
    print(f" DIAGNOSIS REPORT: {res_name}")
    print(f"========================================")

    # 2. 카이랄성 확인 (Chirality Check)
    # SDF 파일이 가진 본연의 카이랄 정보를 출력
    Chem.AssignAtomChiralTagsFromStructure(mol)
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    print(f"[1] Chiral Centers (SDF Origin):")
    if chiral_centers:
        for idx, chiral_type in chiral_centers:
            print(f"    - Atom {idx}: {chiral_type}")
    else:
        print(f"    - None (Achiral or Planar)")

    # 3. 부호 적용 테스트
    test_angle = -30.0
    print(f"\n[2] Sign Inversion Test:")
    print(f"    -> Applying angle: {test_angle} deg")
    
    for i, dof in enumerate(dofs):
        # (1) 적용
        rdMolTransforms.SetDihedralDeg(conf, dof[0], dof[1], dof[2], dof[3], test_angle)
        
        # (2) 재측정 (RDKit 내부 값)
        read_angle = rdMolTransforms.GetDihedralDeg(conf, dof[0], dof[1], dof[2], dof[3])
        
        print(f"    [DOF {i+1} Indices: {dof}]")
        print(f"      - Input: {test_angle}")
        print(f"      - Read : {read_angle:.4f}")
        
        if abs(read_angle - test_angle) < 1.0:
            print(f"      -> [PASS] Sign matches. (Definition order implies Input direction)")
        else:
            # 보통 360도 주기로 인해 330도가 나올 수 있으나, 부호가 반대(+30)라면 문제
            if abs(read_angle - (-test_angle)) < 1.0:
                 print(f"      -> [FAIL] Sign FLIPPED! (+30 detected)")
            else:
                 print(f"      -> [WARN] Value mismatch (Check periodicity)")

    # 4. 저장
    out_path = os.path.join(PROJECT_ROOT, '2_results', 'debug', f"SIGN_TEST_{res_name}.pdb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Chem.MolToPDBFile(mol, out_path)
    print(f"\n[3] Saved Test Structure: {out_path}")
    print(f"    -> Open in PyMOL. Measure angle {dofs[0]}.")
    print(f"       If PyMOL says +30 but here it says -30, your PyMOL clicking order is reverse.")
    print(f"       If PyMOL says +30 AND here it says +30, the definition is reversed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--residue", required=True)
    args = parser.parse_args()
    run(args)