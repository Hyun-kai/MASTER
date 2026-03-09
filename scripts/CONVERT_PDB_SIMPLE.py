import os
import sys
from rdkit import Chem

# ==============================================================================
# 1. 설정: 변환할 파일 목록 및 경로
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# 원본 파일이 있는 폴더
SOURCE_DIR = os.path.join(PROJECT_ROOT, "2_results", "ANSWER") 

# 변환된 파일이 저장될 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "2_results", "FILTERED_HBOND_PDB")

# [Input List] 변환 대상 파일 목록
TARGET_FILES = [
    "[18]CPDA_CPDC_cap.pdb","[12]CPDA_CPDC_cap.pdb","[12]CPDA_CPDC_cap_2.pdb"
]

# ==============================================================================
# 2. 변환 로직 (RDKit 기반)
# ==============================================================================
def fix_and_convert_pdb(input_path, output_path):
    file_name = os.path.basename(input_path)
    print(f"[Processing] {file_name}...", end='')
    
    # -------------------------------------------------------------------------
    # [수정된 부분] sanitize=False로 로드하여 Valence 에러 우회
    # -------------------------------------------------------------------------
    try:
        # 1. 로드 (sanitize=False: 화학적 오류가 있어도 일단 객체 생성 시도)
        mol = Chem.MolFromPDBFile(input_path, removeHs=False, sanitize=False)
        
        if mol is None:
            print(" -> [Error] Load Failed (Parse Error).")
            return

        # 2. 속성 갱신 (strict=False: Valence 오류가 있어도 무시하고 계산)
        # 이를 수행하지 않으면 GetAtomicNum() 등에서 에러가 날 수 있음
        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            pass # 치명적이지 않으면 무시
            
    except Exception as e:
        print(f" -> [Error] Initial Load Failed: {e}")
        return

    try:
        # 3. 원자 재배열 (Heavy Atoms -> Hydrogens)
        heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        h_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
        
        new_order = heavy_indices + h_indices
        mol_ordered = Chem.RenumberAtoms(mol, new_order)
        
        # 4. 메타데이터 단순화 (Residue Name -> MOL, Atom Name -> Element)
        for atom in mol_ordered.GetAtoms():
            info = atom.GetPDBResidueInfo()
            if info:
                # 잔기 정보 변경
                info.SetResidueName("MOL")
                info.SetResidueNumber(1)
                info.SetChainId("A")
                
                # 원자 이름 변경
                symbol = atom.GetSymbol()
                target_name = f" {symbol:<3}" if len(symbol) == 1 else f" {symbol:<2}"
                info.SetName(target_name)

        # 5. 저장 (RDKit이 CONECT 정보를 자동으로 작성함)
        Chem.MolToPDBFile(mol_ordered, output_path)
        
        out_name = os.path.basename(output_path)
        print(f" -> Done. Saved to: {out_name}")
        
    except Exception as e:
        print(f" -> [Error] Processing Failed: {e}")

# ==============================================================================
# 3. 메인 실행 (Batch)
# ==============================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"PDB Format Fixer (Topology & Reorder)")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {OUTPUT_DIR}")
    print(f"{'='*60}")

    count = 0
    for filename in TARGET_FILES:
        input_path = os.path.join(SOURCE_DIR, filename)
        new_name = filename.replace(".pdb", "_converted.pdb")
        output_path = os.path.join(OUTPUT_DIR, new_name)
        
        if os.path.exists(input_path):
            fix_and_convert_pdb(input_path, output_path)
            count += 1
        else:
            print(f"[Skip] File not found: {filename}")

    print(f"{'='*60}")
    print(f"Completed. {count}/{len(TARGET_FILES)} files processed.")