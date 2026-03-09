import numpy as np
import trimesh
import trimesh.creation
from rdkit import Chem
from rdkit.Chem import AllChem

def create_molecule_glb(smiles, output_filename="molecule.glb", dummy_alpha=0):
    """
    SMILES를 GLB로 변환하며 * (Dummy Atom)을 투명하게 처리합니다.
    dummy_alpha=0 이면 해당 부분의 메쉬를 아예 생성하지 않습니다.
    """
    
    # 1. RDKit으로 분자 로드
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    # [핵심] Dummy Atom(*)을 찾아 태그(Property)를 심고, 탄소(C)로 변경
    # 리스트(dummy_indices) 대신 분자 자체에 태그를 달아야 AddHs 후에도 추적 가능합니다.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIntProp("is_dummy", 1) # 태그 설정
            atom.SetAtomicNum(6)           # 탄소로 변경 (Geometry 계산용)

    # 2. 수소 추가 (이 과정에서 원자 인덱스가 변할 수 있음)
    mol = Chem.AddHs(mol, addCoords=True)
    
    # 3. 3D 좌표 생성
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    try:
        AllChem.EmbedMolecule(mol, params)
    except Exception as e:
        print(f"Embedding failed for {smiles}: {e}")
        return
    
    conf = mol.GetConformer()
    scene = trimesh.Scene()
    
    # 색상표 (RGBA)
    atom_colors = {
        1: [255, 255, 255, 255],   # H
        6: [50, 50, 50, 255],      # C
        7: [0, 0, 255, 255],       # N
        8: [255, 0, 0, 255],       # O
        16: [255, 255, 0, 255],    # S
        0: [255, 0, 255, dummy_alpha] # Dummy (Pink)
    }
    atom_radius = {0: 0.3, 1: 0.2, 6: 0.4, 7: 0.4, 8: 0.4}
    default_radius = 0.4

    # 4. 원자(Atom) 메쉬 생성
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atomic_num = atom.GetAtomicNum()
        
        # [판별 로직: 태그 기반]
        is_dummy_atom = atom.HasProp("is_dummy")
        
        # 더미에 붙은 수소인지 확인
        is_dummy_hydrogen = False
        if atomic_num == 1: # 수소인 경우 이웃 확인
            for neighbor in atom.GetNeighbors():
                if neighbor.HasProp("is_dummy"):
                    is_dummy_hydrogen = True
                    break
        
        # [중요] 완전 투명(0)이면 아예 생성하지 않음 (건너뛰기)
        if (is_dummy_atom or is_dummy_hydrogen) and dummy_alpha == 0:
            continue
            
        # 색상 결정
        rad = atom_radius.get(atomic_num, default_radius)
        # 더미 본체라면 색상 키를 0으로, 아니면 원자번호 그대로
        color_key = 0 if is_dummy_atom else atomic_num
        color = atom_colors.get(color_key, [0, 255, 0, 255]).copy() # copy 필수
        
        # 반투명 설정 (alpha > 0 인 경우)
        if is_dummy_atom or is_dummy_hydrogen:
            color[3] = dummy_alpha

        # 구체 생성
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=rad)
        sphere.apply_translation([pos.x, pos.y, pos.z])
        sphere.visual.face_colors = color
        scene.add_geometry(sphere)

    # 5. 결합(Bond) 메쉬 생성
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        
        idx1 = begin_atom.GetIdx()
        idx2 = end_atom.GetIdx()
        
        pos1 = conf.GetAtomPosition(idx1)
        pos2 = conf.GetAtomPosition(idx2)
        p1 = np.array([pos1.x, pos1.y, pos1.z])
        p2 = np.array([pos2.x, pos2.y, pos2.z])
        
        # 결합된 두 원자 중 하나라도 '더미 계열(본체 or 수소)'인지 확인
        
        # 1) 더미 본체인가?
        is_dummy_bond = (begin_atom.HasProp("is_dummy") or end_atom.HasProp("is_dummy"))
        
        # 2) 더미에 붙은 수소인가?
        # (begin이 수소이고 이웃이 더미인지, end가 수소이고 이웃이 더미인지 확인)
        is_dummy_h_bond = False
        
        def is_atom_dummy_h(atom):
            if atom.GetAtomicNum() != 1: return False
            for n in atom.GetNeighbors():
                if n.HasProp("is_dummy"): return True
            return False

        if is_atom_dummy_h(begin_atom) or is_atom_dummy_h(end_atom):
            is_dummy_h_bond = True

        # [중요] 더미 관련 결합이고 alpha가 0이면 생성 생략
        if (is_dummy_bond or is_dummy_h_bond) and dummy_alpha == 0:
            continue

        # 색상 및 투명도 설정
        bond_alpha = 255
        if is_dummy_bond or is_dummy_h_bond:
            bond_alpha = dummy_alpha
            
        bond_color = [100, 100, 100, bond_alpha]

        # 실린더 생성
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length < 1e-6: continue
        
        cylinder = trimesh.creation.cylinder(radius=0.1, height=length, sections=12)
        cylinder.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], vec))
        cylinder.apply_translation((p1 + p2) / 2)
        cylinder.visual.face_colors = bond_color
        scene.add_geometry(cylinder)

    # 6. GLB 내보내기
    scene.export(output_filename)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    target = ["CC(=O)N[C@@H](C)CCC(=O)NC","CC(=O)N1[C@@H](CCC1)C(=O)NC","CC(=O)NCc1nc(ccc1)C(=O)NC","CC(=O)Nc1c2nc(ccc2ccc1)C(=O)NC"]
    output = ["G4ALA.glb","PRO.glb","HUCP.glb","HUCQ.glb"]
    
    for t, o in zip(target, output):
        create_molecule_glb(t, o, dummy_alpha=0)