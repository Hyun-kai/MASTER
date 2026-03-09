'''
src/bakers/pipeline/monomer_runner.py
 
[사용 모듈]
pucking.py, topology.py, io.py, check_topology_grid.py

'''


import os
import sys
import traceback
from typing import Dict, Any, Optional
from rdkit import Chem

from bakers.chem import puckering, topology
from bakers.utils import io

def prepare_monomer(
    res_name: str, 
    full_smiles: str, 
    rotamer_dir: str, 
    vis_dir: Optional[str] = None, 
    has_visualization: bool = False,
    max_rotamers: int = 10,
    threshold_rmsd: float = 2.0,
    threshold_tfd: float = 0.1
) -> Optional[Dict[str, Any]]:
    """
    단일 잔기(Monomer) 파이프라인. 예외 처리를 강화하여 파이프라인의 붕괴를 막습니다.
    """
    print(f"    -> Processing {res_name} ... ", end="")
    sys.stdout.flush()

    try:
        # 1. 초기화 및 수소 추가
        mol = Chem.MolFromSmiles(full_smiles)
        mol = Chem.AddHs(mol)
        num_atoms = mol.GetNumAtoms()
        
        # 2. 위상(Topology) 분석
        topo_info = topology.analyze_residue_topology(mol)
        m_type = topo_info.get('monomer_type', 'unknown')

        # 3. 자유도(DOF) 분석
        all_indices = set(range(num_atoms))
        exclude_indices = all_indices - set(topo_info['residue_indices'])
        
        dofs = topology.get_dofs(mol, exclude_indices)
        dof_map = topology.identify_backbone_dofs(mol, dofs)
        
        print(f"[{m_type.upper()} | Atoms: {num_atoms} | DOFs: {len(dofs)}] ... ", end="", flush=True)

        # 4. Conformer 생성
        final_mol = puckering.generate_conformers(mol)
        if final_mol.GetNumConformers() == 0:
            print("Embedding Failed.")
            return None

        # 5. 최적화 및 클러스터링
        cids = [c.GetId() for c in final_mol.GetConformers()]
        puckering.optimize_ensemble(final_mol, cids)
        
        props = puckering.calculate_energies(final_mol, cids)
        is_cyclic = (final_mol.GetRingInfo().NumRings() > 0)
        
        if is_cyclic:
            valid_cids = puckering.cluster_ensemble(
                final_mol, props, method='tfd', threshold=threshold_tfd, max_confs=max_rotamers
            )
            print(f"Selected {len(valid_cids)} (Cyclic).")
        else:
            valid_cids = puckering.cluster_ensemble(
                final_mol, props, method='rmsd', threshold=threshold_rmsd, max_confs=max_rotamers
            )
            print(f"Selected {len(valid_cids)} (Acyclic).")

        # 6. 파일 저장
        sdf_path = os.path.join(rotamer_dir, f"{res_name}.sdf")
        io.save_sdf(final_mol, valid_cids, sdf_path)

        # 7. 파라미터 딕셔너리 구축
        entry = topology.build_parameter_dict(final_mol, full_smiles, topo_info, dofs, dof_map)

        # 8. 시각화 (Optional)
        if has_visualization and vis_dir is not None:
            from bakers.analytics import check_topology_grid
            check_topology_grid.create_grid_report(res_name, entry, vis_dir)

        return entry

    except Exception as e:
        # 복잡한 분자 구조에서 RDKit 내부 에러 발생 시, 스택 트레이스 대신 깔끔한 로그를 남기고 스킵
        print(f"Failed. Error: {str(e)}")
        # 디버깅이 필요할 땐 아래 주석 해제
        # traceback.print_exc()
        return None