from rdkit import Chem
from rdkit.Chem import TorsionFingerprints
import os, sys
import pandas as pd



smiles_dict = {
"DMMA":"CNC(=O)C(C)(C)C(=O)NC",
"CPDC":"CNC(=O)[C@@H]1[C@H](CCC1)C(=O)NC",
"CPDA":"CC(=O)N[C@@H]1[C@H](CCC1)NC(=O)C",
"EDA":"CC(=O)NCCNC(=O)C"
}

mol = Chem.MolFromSmiles(smiles_dict["CPDC"])
symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
print(symbols)
print(TorsionFingerprints.CalculateTorsionLists(mol, maxDev='equal', symmRadius=2, ignoreColinearBonds=True))
print("==============================================================================")

# 기본 표준 아미노산 SMILES (Fallback용)
DEFAULT_SMILES = {
    'ALA': 'CC(=O)N[C@@H](C)C(=O)NC',
    'GLY': 'CC(=O)NCC(=O)NC',
}

# ==============================================================================
# 2. 메인 실행 로직
# ==============================================================================

if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

    # 소스 코드 경로 추가
    if SRC_DIR not in sys.path:
        sys.path.append(SRC_DIR)    
    # 경로 설정
    input_dir = os.path.join(PROJECT_ROOT, '0_inputs')

    # --------------------------------------------------------------------------
    # [Step 1] SMILES 데이터 로드
    # --------------------------------------------------------------------------
    smiles_map = DEFAULT_SMILES.copy()
    csv_path = os.path.join(input_dir, 'SMILES_Data.csv')
    
    if os.path.exists(csv_path):
        try:
            # CSV 포맷: [이름, SMILES] (헤더 없음)
            df = pd.read_csv(csv_path, header=None, names=['Name','Smiles'])
            for _, r in df.iterrows():
                #print(_)
                print(r)
                if pd.notna(r['Name']) and pd.notna(r['Smiles']):
                    smiles_map[str(r['Name']).strip()] = str(r['Smiles']).strip()
            print(f"    [Info] Loaded {len(smiles_map)} SMILES entries from CSV.")
        except Exception as e:
            print(f"[Warning] Failed to read SMILES_Data.csv: {e}")
    else:
        print(f"[Warning] SMILES_Data.csv not found at {csv_path}. Using defaults.")

    params_dict = {}