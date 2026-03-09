import os
import glob
import sys
import argparse
import numpy as np
import trimesh
from rdkit import Chem

# ==============================================================================
# 1. 환경 설정 및 경로 지정
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

PDB_DIR = os.path.join(PROJECT_ROOT, '2_results', 'pdb')
GLB_DIR = os.path.join(PROJECT_ROOT, '2_results', 'glb')

os.makedirs(GLB_DIR, exist_ok=True)

# ==============================================================================
# 2. 색상 설정 (흰색 배경용 고대비)
# ==============================================================================
ATOM_COLORS = {
    6: [30, 30, 30, 255],      # C (탄소): 진한 검정
    7: [20, 50, 255, 255],     # N (질소): 선명한 파랑
    8: [255, 10, 10, 255],     # O (산소): 선명한 빨강
    16: [255, 230, 0, 255],    # S (황): 선명한 노랑
    1: [200, 200, 200, 255],   # H (수소): 은색
    9: [0, 200, 0, 255],       # F: 녹색
    17: [0, 200, 0, 255],      # Cl: 녹색
}
DEFAULT_COLOR = [150, 0, 255, 255] # 기타: 보라색

ATOM_RADIUS_MAP = {1: 0.25, 6: 0.4, 7: 0.4, 8: 0.4, 16: 0.45}
DEFAULT_RADIUS = 0.4
BOND_RADIUS = 0.15
BOND_COLOR = [180, 180, 180, 255]

# ==============================================================================
# 3. 변환 함수 (오류 원인 제거됨)
# ==============================================================================
def pdb_to_glb(pdb_path, output_path):
    # PDB 로드
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=False)
    if mol is None:
        print(f"   [Skip] Load Failed: {os.path.basename(pdb_path)}")
        return False

    conf = mol.GetConformer()
    scene_objects = []

    # 1) 원자(Sphere) 생성
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx)
        atomic_num = atom.GetAtomicNum()
        
        color = ATOM_COLORS.get(atomic_num, DEFAULT_COLOR)
        radius = ATOM_RADIUS_MAP.get(atomic_num, DEFAULT_RADIUS)
        
        # 구체 생성
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        sphere.apply_translation([pos.x, pos.y, pos.z])
        
        # [핵심 수정] material 속성 접근 코드 완전 삭제
        # 단순히 색상만 지정합니다.
        sphere.visual.face_colors = color
        
        scene_objects.append(sphere)

    # 2) 결합(Cylinder) 생성
    for bond in mol.GetBonds():
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        p1 = np.array(conf.GetAtomPosition(start_idx))
        p2 = np.array(conf.GetAtomPosition(end_idx))
        
        # 원기둥 생성
        cylinder = trimesh.creation.cylinder(radius=BOND_RADIUS, segment=[p1, p2])
        
        # [핵심 수정] material 속성 접근 코드 완전 삭제
        cylinder.visual.face_colors = BOND_COLOR
        
        scene_objects.append(cylinder)

    # 3) 저장
    if scene_objects:
        # scene으로 합쳐서 저장
        scene = trimesh.Scene(scene_objects)
        scene.export(output_path)
        print(f"   [Done] -> {os.path.basename(output_path)}")
        return True
    return False

# ==============================================================================
# 4. 파일 선택 로직
# ==============================================================================
def select_files(all_files, filter_keyword=None):
    if filter_keyword:
        filtered = [f for f in all_files if filter_keyword in os.path.basename(f)]
        print(f"Running in Filter Mode: '{filter_keyword}'")
        return filtered

    print("\n" + "="*60)
    print(f" [File Selection Menu] Found {len(all_files)} PDB files")
    print("="*60)
    
    for idx, fpath in enumerate(all_files):
        fname = os.path.basename(fpath)
        print(f" {idx+1:3d}. {fname}")
    
    print("-" * 60)
    print(" Enter numbers to process (e.g., '1' or '1,3' or '1-5' or 'all')")
    user_input = input(" > Selection: ").strip()

    if user_input.lower() == 'all':
        return all_files
    
    selected_files = []
    try:
        parts = user_input.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_files.extend(all_files[start-1:end])
            else:
                idx = int(part) - 1
                if 0 <= idx < len(all_files):
                    selected_files.append(all_files[idx])
    except Exception as e:
        print(f" [Error] Invalid input: {e}")
        return []

    return sorted(list(set(selected_files)))

# ==============================================================================
# 5. 메인 실행
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="PDB to GLB Converter")
    parser.add_argument("--filter", type=str, help="Auto-select files containing this keyword")
    args = parser.parse_args()

    pdb_files = glob.glob(os.path.join(PDB_DIR, "*.pdb"))
    pdb_files.sort()

    if not pdb_files:
        print(f" [Error] No PDB files found in {PDB_DIR}")
        return

    targets = select_files(pdb_files, args.filter)

    if not targets:
        print(" [Info] No files selected. Exiting.")
        return

    print("\n" + "="*60)
    print(f" Starting Conversion for {len(targets)} files...")
    print("="*60)

    success_count = 0
    for pdb_path in targets:
        base_name = os.path.basename(pdb_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        glb_name = f"{file_name_no_ext}.glb"
        output_path = os.path.join(GLB_DIR, glb_name)
        
        try:
            if pdb_to_glb(pdb_path, output_path):
                success_count += 1
        except Exception as e:
            # 혹시 모를 다른 에러에 대비해 에러 메시지를 더 자세히 출력
            import traceback
            print(f"   [Fail] Error converting {base_name}:")
            traceback.print_exc()

    print("="*60)
    print(f" Completed: {success_count}/{len(targets)} files saved to {GLB_DIR}")

if __name__ == "__main__":
    main()