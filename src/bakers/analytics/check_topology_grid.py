"""
src/bakers/analytics/check_topology_grid.py

[기능]
residue_params.py의 정보를 받아 2x3 Grid 이미지로 시각화합니다.
1_prep_rotamers.py에서 직접 호출되어 실행됩니다.
"""

import os
import io
import sys
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# 색상 정의 (R, G, B)
COLOR_C = (0.2, 0.2, 0.2)
COLOR_N = (0.2, 0.2, 1.0)
COLOR_O = (1.0, 0.2, 0.2)
COLOR_H = (0.8, 0.8, 0.8)
COLOR_S = (0.8, 0.8, 0.2)
COLOR_DEFAULT = (0.5, 0.8, 0.5)
COLOR_DOF_BOND = (1.0, 0.6, 0.0) # Highlight Orange

# 영역별 색상
COLOR_CORE  = (0.6, 0.8, 1.0) # Light Blue
COLOR_NTERM = (1.0, 1.0, 0.6) # Light Yellow
COLOR_CTERM = (1.0, 0.6, 0.6) # Light Red
COLOR_LOWER = (0.2, 0.8, 0.2) # Green
COLOR_UPPER = (0.6, 0.2, 0.8) # Purple

def get_atom_color(atomic_num):
    if atomic_num == 6: return COLOR_C
    if atomic_num == 7: return COLOR_N
    if atomic_num == 8: return COLOR_O
    if atomic_num == 1: return COLOR_H
    if atomic_num == 16: return COLOR_S
    return COLOR_DEFAULT

def draw_panel(mol, highlight_atoms, highlight_bonds, atom_colors, bond_colors, title, use_ball_style=False):
    width, height = 400, 400 # 그리드에 맞게 사이즈 조정
    d = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = d.drawOptions()
    opts.legendFontSize = 24
    opts.annotationFontScale = 0.8
    opts.bondLineWidth = 3
    
    # 인덱스 표시
    for atom in mol.GetAtoms():
        opts.atomLabels[atom.GetIdx()] = str(atom.GetIdx())
    
    if use_ball_style:
        all_atoms = list(range(mol.GetNumAtoms()))
        master_atom_cols = {}
        for idx in all_atoms:
            at_num = mol.GetAtomWithIdx(idx).GetAtomicNum()
            master_atom_cols[idx] = get_atom_color(at_num)
            
        d.DrawMolecule(
            mol, highlightAtoms=all_atoms, highlightAtomColors=master_atom_cols,
            highlightBonds=highlight_bonds, highlightBondColors=bond_colors, legend=title
        )
    else:
        d.DrawMolecule(
            mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors,
            highlightBonds=[], highlightBondColors={}, legend=title
        )

    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))

def create_grid_report(name, data, output_dir):
    """
    메인 로직: 데이터를 받아 이미지를 생성하고 저장합니다.
    Args:
        name (str): 잔기 이름 (예: DMMA)
        data (dict): residue_params의 entry 딕셔너리
        output_dir (str): 저장할 폴더 경로
    """
    smiles = data.get('residue_smiles', '')
    if not smiles: return

    mol = Chem.MolFromSmiles(smiles)
    if not mol: return
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    # 데이터 추출
    res_idx = data.get('residue_indices', [])
    n_idx = data.get('n_term_indices', [])
    c_idx = data.get('c_term_indices', [])
    lower = data.get('lower_connect_indices', [])
    upper = data.get('upper_connect_indices', [])
    dofs = data.get('dofs', [])

    # 1. Master View
    dof_bonds = []
    dof_bond_colors = {}
    for dof in dofs:
        u, v = dof[1], dof[2]
        bond = mol.GetBondBetweenAtoms(u, v)
        if bond:
            b_idx = bond.GetIdx()
            dof_bonds.append(b_idx)
            dof_bond_colors[b_idx] = COLOR_DOF_BOND

    img1 = draw_panel(mol, [], dof_bonds, {}, dof_bond_colors, f"1. Master: {name} (Org=DOF)", use_ball_style=True)

    # 2~6 Panels
    cols = {i: COLOR_CORE for i in res_idx}
    img2 = draw_panel(mol, res_idx, [], cols, {}, f"2. Core ({len(res_idx)})")

    cols = {i: COLOR_NTERM for i in n_idx}
    img3 = draw_panel(mol, n_idx, [], cols, {}, f"3. N-Term ({len(n_idx)})")

    cols = {i: COLOR_CTERM for i in c_idx}
    img4 = draw_panel(mol, c_idx, [], cols, {}, f"4. C-Term ({len(c_idx)})")

    cols = {i: COLOR_LOWER for i in lower}
    img5 = draw_panel(mol, lower, [], cols, {}, f"5. Lower Conn (N-side)")

    cols = {i: COLOR_UPPER for i in upper}
    img6 = draw_panel(mol, upper, [], cols, {}, f"6. Upper Conn (C-side)")

    # Stitching
    w, h = img1.size
    grid_img = Image.new('RGB', (w * 3, h * 2), (255, 255, 255))
    grid_img.paste(img1, (0, 0)); grid_img.paste(img2, (w, 0)); grid_img.paste(img3, (w * 2, 0))
    grid_img.paste(img4, (0, h)); grid_img.paste(img5, (w, h)); grid_img.paste(img6, (w * 2, h))

    save_path = os.path.join(output_dir, f"{name}_topology_check.png")
    grid_img.save(save_path)
    return save_path


if __name__ == "__main__":
    import os

    # 1. 테스트할 분자의 SMILES 및 저장 경로 설정
    test_smiles = "CC(C)(C)C1=CC(C#CC2=CC(C#CC)=CN=C2)=C(NC3=C4C=CC5=C3NC6=C5C=C(C(C)(C)C)C=C6C#CC7=CN=CC(C#CC)=C7)C4=C1"
    output_dir = "." # 현재 폴더에 저장

    # 2. create_grid_report 함수가 요구하는 데이터 구조 생성
    # 참고: 현재는 예시 인덱스이며, 실제 시각화를 원하는 원자 인덱스로 수정해야 합니다.
    test_data = {
        'residue_smiles': test_smiles,
        'residue_indices': list(range(15)),      # 예: 0번~14번 원소를 Core(Panel 2)로 색칠
        'n_term_indices': [15, 16],              # 예: N-Term 영역(Panel 3)
        'c_term_indices': [17, 18],              # 예: C-Term 영역(Panel 4)
        'lower_connect_indices': [],             # 예: Lower Connect (Panel 5)
        'upper_connect_indices': [],             # 예: Upper Connect (Panel 6)
        'dofs': [
            # [type, atom1_idx, atom2_idx] 형태
            # 예: 1번과 2번 원자 사이의 결합을 하이라이트(Panel 1)
            ['bond', 1, 2],
            ['bond', 3, 4]
        ]
    }

    # 3. 메인 함수 실행 및 결과 확인
    try:
        saved_file = create_grid_report("Custom_Molecule", test_data, output_dir)
        if saved_file:
            print(f"✅ 토폴로지 분석 이미지가 성공적으로 저장되었습니다: {saved_file}")
        else:
            print("❌ 이미지 생성 실패: SMILES가 유효하지 않거나 데이터가 누락되었습니다.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

