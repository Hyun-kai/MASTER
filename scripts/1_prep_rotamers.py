"""
scripts/1_prep_rotamers.py

[설명]
pass
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# 소스 코드 경로 추가
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# BAKERS 모듈 임포트
try:
    from bakers.chem import puckering, topology
    from bakers.analytics import check_topology_grid
    HAS_VISUALIZATION = True
except ImportError as e:
    print(f"[Warning] Visualization module or dependencies (PIL, matplotlib) missing: {e}")
    HAS_VISUALIZATION = False

# 기본 표준 아미노산 SMILES (Fallback용)
DEFAULT_SMILES = {
    'ALA': 'CC(=O)N[C@@H](C)C(=O)NC',
    'GLY': 'CC(=O)NCC(=O)NC',
}

# 로타머 생성 설정
MAX_ROTAMERS = 10         # 최대 저장할 로타머 개수
THRESHOLD_RMSD = 2.0      # 비순환(Acyclic) 구조 클러스터링 임계값 (Angstrom)
THRESHOLD_TFD = 0.1       # 순환(Cyclic) 구조 Torsion Fingerprint 임계값

# ==============================================================================
# 2. 메인 실행 로직
# ==============================================================================

def run(args):
    print(f">>> [BAKERS] Rotamer Preparation Started for: {args.residues}")
    
    # 경로 설정
    input_dir = os.path.join(PROJECT_ROOT, '0_inputs')
    rotamer_dir = os.path.join(input_dir, 'rotamers')
    # 시각화 결과 저장 경로 (검증용 이미지)
    vis_dir = os.path.join(input_dir, 'analysis', 'topology_checks')
    
    os.makedirs(rotamer_dir, exist_ok=True)
    if HAS_VISUALIZATION:
        os.makedirs(vis_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # [Step 1] SMILES 데이터 로드
    # --------------------------------------------------------------------------
    smiles_map = DEFAULT_SMILES.copy()
    csv_path = os.path.join(input_dir, 'SMILES_Data.csv')
    
    if os.path.exists(csv_path):
        try:
            # CSV 포맷: [이름, SMILES] (헤더 없음)
            df = pd.read_csv(csv_path, header=None, names=['N','S'])
            for _, r in df.iterrows():
                if pd.notna(r['N']) and pd.notna(r['S']):
                    smiles_map[str(r['N']).strip()] = str(r['S']).strip()
            print(f"    [Info] Loaded {len(smiles_map)} SMILES entries from CSV.")
        except Exception as e:
            print(f"[Warning] Failed to read SMILES_Data.csv: {e}")
    else:
        print(f"[Warning] SMILES_Data.csv not found at {csv_path}. Using defaults.")

    params_dict = {}

    # 각 잔기에 대해 반복 처리
    for res_name in args.residues:
        if res_name not in smiles_map:
            print(f"    [Skip] '{res_name}' not found in SMILES list.")
            continue
            
        print(f"    -> Processing {res_name} ... ", end="")
        sys.stdout.flush()
        
        # ----------------------------------------------------------------------
        # [Step 2] 분자 객체 생성 및 초기화
        # ----------------------------------------------------------------------
        full_smiles = smiles_map[res_name]
        mol = Chem.MolFromSmiles(full_smiles)
        
        if mol is None:
            print("Invalid SMILES.")
            continue
            
        # 수소 추가 (3D 구조 생성 필수)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms() # 몰객체의 전체 원자 수 반환
        
        # ----------------------------------------------------------------------
        # [Step 3] 위상 분석 (Topology Analysis)
        # ----------------------------------------------------------------------
        # Backbone 경로, N/C 터미널, 연결 부위 등을 자동으로 탐지합니다.
        try:
            topo_info = topology.analyze_residue_topology(mol)
            m_type = topo_info.get('monomer_type', 'unknown')
        except Exception as e:
            print(f"Topology Analysis Error ({e}).")
            continue

        # ----------------------------------------------------------------------
        # [Step 4] 자유도(DOF) 분석
        # ----------------------------------------------------------------------
        # Backbone 원자를 제외하고 회전 가능한 결합(Rotatable Bonds)을 찾습니다.
        all_indices = set(range(num_atoms))
        # 토폴로지 분석에서 Backbone으로 지정된 원자들은 제외 (보존)
        exclude_indices = all_indices - set(topo_info['residue_indices'])
        
        # Sidechain DOF 추출
        dofs = topology.get_dofs(mol, exclude_indices)
        
        # Backbone DOF (Phi/Psi 등) 매핑
        dof_map = topology.identify_backbone_dofs(mol, dofs)
        
        print(f"[{m_type.upper()} | Atoms: {num_atoms} | DOFs: {len(dofs)}] ... ", end="", flush=True)

        # ----------------------------------------------------------------------
        # [Step 5] 구조 생성 (Conformer Generation)
        # ----------------------------------------------------------------------
        # Hybrid Sampling: ETKDG (Distance Geometry) + Systematic Torsion Scan
        final_mol = puckering.generate_conformers(mol)
        
        if final_mol.GetNumConformers() == 0:
            print("Embedding Failed.")
            continue
            
        # ----------------------------------------------------------------------
        # [Step 6] 구조 최적화 (Optimization)
        # ----------------------------------------------------------------------
        # MMFF Force Field를 사용하여 기하학적 구조를 이완(Relaxation)시킵니다.
        cids = [c.GetId() for c in final_mol.GetConformers()]
        puckering.optimize_ensemble(final_mol, cids)
        
        # ----------------------------------------------------------------------
        # [Step 7] 클러스터링 및 필터링 (Clustering)
        # ----------------------------------------------------------------------
        # 유사한 구조들을 그룹화하여 대표 로타머만 선별합니다.
        props = puckering.calculate_energies(final_mol, cids)
        is_cyclic = (final_mol.GetRingInfo().NumRings() > 0)
        
        if is_cyclic:
            # 순환 구조는 Torsion Fingerprint Deviation (TFD) 사용
            valid_cids = puckering.cluster_ensemble(
                final_mol, props, method='tfd', threshold=THRESHOLD_TFD, max_confs=MAX_ROTAMERS
            )
            print(f"Selected {len(valid_cids)} (Cyclic, TFD).")
        else:
            # 비순환 구조는 RMSD 사용
            valid_cids = puckering.cluster_ensemble(
                final_mol, props, method='rmsd', threshold=THRESHOLD_RMSD, max_confs=MAX_ROTAMERS
            )
            print(f"Selected {len(valid_cids)} (Acyclic, RMSD).")

        # ----------------------------------------------------------------------
        # [Step 8] SDF 파일 저장
        # ----------------------------------------------------------------------
        sdf_path = os.path.join(rotamer_dir, f"{res_name}.sdf")
        w = Chem.SDWriter(sdf_path)
        
        for i, cid in enumerate(valid_cids):
            # 로타머 인덱스 속성 추가 (중요: 추후 식별용)
            final_mol.SetIntProp("Rotamer_Index", i)
            # Conformer ID를 기반으로 구조 저장
            w.write(final_mol, confId=cid)
        w.close()

        # ----------------------------------------------------------------------
        # [Step 9] 파라미터 딕셔너리 구축
        # ----------------------------------------------------------------------
        # 시뮬레이션 및 분석에 필요한 모든 물리화학적/위상 정보를 딕셔너리로 구성
        try: 
            AllChem.ComputeGasteigerCharges(final_mol)
        except: 
            pass
        
        entry = {
            'residue_smiles': full_smiles,
            'monomer_type': m_type,
            'atoms': [a.GetAtomicNum() for a in final_mol.GetAtoms()],
            'bonds': [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in final_mol.GetBonds()],
            # 전하 정보가 없으면 0.0으로 처리
            'charges': [float(a.GetProp('_GasteigerCharge')) if a.HasProp('_GasteigerCharge') else 0.0 for a in final_mol.GetAtoms()],
            'radii': [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in final_mol.GetAtoms()],
            'dofs': dofs,
            'dof_map': dof_map,
            'residue_indices': topo_info['residue_indices'],
            'n_term_indices': topo_info['n_term_indices'],
            'c_term_indices': topo_info['c_term_indices'],
            # Optional 연결 부위 인덱스 (기본값 빈 리스트)
            'lower_connect_indices': topo_info.get('lower_connect_indices', []),
            'upper_connect_indices': topo_info.get('upper_connect_indices', []),
            'is_capped': topo_info['is_capped']
        }
        params_dict[res_name] = entry
        
        # ----------------------------------------------------------------------
        # [Step 10] 즉시 시각화 수행 (Visual Check)
        # ----------------------------------------------------------------------
        # 토폴로지 분석이 올바르게 되었는지 눈으로 확인하기 위한 리포트 생성
        if HAS_VISUALIZATION:
            try:
                # 딕셔너리 데이터를 바로 넘겨서 이미지를 생성합니다.
                check_topology_grid.create_grid_report(res_name, entry, vis_dir)
            except Exception as e:
                print(f"        [Visual Error] Failed to generate report: {e}")

    # --------------------------------------------------------------------------
    # [Step 11] residue_params.py 파일 저장
    # --------------------------------------------------------------------------
    params_path = os.path.join(input_dir, 'residue_params.py')
    with open(params_path, 'w') as f:
        f.write("import numpy as np\n\nresidue_params = {\n")
        for k, v in params_dict.items():
            f.write(f"    '{k}': {{\n")
            for sk, sv in v.items():
                # Numpy 배열은 리스트 형태로 변환하여 저장
                val = repr(sv.tolist()) if isinstance(sv, np.ndarray) else repr(sv)
                f.write(f"        '{sk}': {val},\n")
            f.write("    },\n")
        f.write("}\n")
        
    print(f"    [Done] Saved parameters to {params_path}")
    if HAS_VISUALIZATION:
        print(f"    [Done] Visual reports saved to {vis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prep Rotamers & Topology")
    parser.add_argument("--residues", nargs="+", required=True, help="List of residue names to process")
    run(parser.parse_args())