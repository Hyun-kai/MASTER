"""
src/bakers/sim/calculator.py

AIMNet2 모델을 사용하여 에너지 및 힘을 계산하고, 
OpenMM 시뮬레이션 환경에서 중원자 구속(Heavy Atom Restraint)을 포함한 
구조 최적화를 수행하는 모듈입니다.
"""

import os
import sys
import time
import uuid
import numpy as np
import torch
from typing import Optional, List, Tuple
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

# OpenMM 및 관련 라이브러리 임포트 처리
try:
    from openmm import app, unit, CustomExternalForce
    from openmm.app import Simulation
except ImportError:
    # OpenMM이 없는 환경에서도 ASE 계산기는 동작할 수 있도록 예외 처리
    pass

# aimnet2calc 모듈 임포트 시도
try:
    import aimnet2calc
except ImportError:
    pass

# ==============================================================================
# 1. Core AIMNet2 Calculator (OpenMM & Optimization with Restraints)
# ==============================================================================

class AIMNet2Calculator:
    """
    AIMNet2 Neural Network Potential을 이용하여 OpenMM 시뮬레이션 내에서 
    구조 최적화 및 에너지 계산을 수행하는 핵심 엔진입니다.
    중원자 구속(Heavy Atom Restraint) 기능을 통해 구조 폭발을 방지합니다.
    """
    def __init__(self, model_paths: List[str], device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model_paths (list): 로드할 AIMNet2 모델 파일(.jpt) 경로 리스트
            device (str): 계산 장치 ('cuda' or 'cpu')
        """
        self.device = torch.device(device)
        self.models = [torch.jit.load(p, map_location=self.device) for p in model_paths]
        for m in self.models:
            m.eval()

    def _apply_restraints(self, system, positions, heavy_atom_indices, k=1000.0):
        """
        OpenMM System에 중원자(Heavy Atoms)를 원래 위치에 잡아두는 구속 조건을 추가합니다.
        
        Args:
            system: OpenMM System 객체
            positions: 구속의 기준이 될 원자 좌표
            heavy_atom_indices: 구속을 적용할 원자 인덱스 리스트
            k (float): 구속 상수 (kJ/mol/nm^2)
        """
        # 조화 구속(Harmonic Restraint) 식 정의: 0.5 * k * delta^2
        restraint_force = CustomExternalForce('0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
        restraint_force.addGlobalParameter('k', k * unit.kilojoule_per_mole/unit.nanometer**2)
        restraint_force.addPerParticleParameter('x0')
        restraint_force.addPerParticleParameter('y0')
        restraint_force.addPerParticleParameter('z0')

        for idx in heavy_atom_indices:
            pos = positions[idx]
            # 각 원자의 초기 위치(x0, y0, z0)를 파라미터로 저장
            restraint_force.addParticle(idx, pos.value_in_unit(unit.nanometer))
        
        system.addForce(restraint_force)
        return restraint_force

    def optimize(self, 
                 simulation: 'Simulation', 
                 heavy_atom_indices: Optional[List[int]] = None,
                 use_restraints: bool = True,
                 max_iterations: int = 0) -> float:
        """
        중원자 구속을 활용한 단계별 구조 최적화를 수행합니다.
        
        Args:
            simulation: OpenMM Simulation 객체
            heavy_atom_indices: 중원자 인덱스 리스트 (None일 경우 구속 미적용)
            use_restraints: 구속 조건 사용 여부
            max_iterations: 최적화 최대 반복 횟수 (0은 수렴 시까지)
            
        Returns:
            float: 최적화 후의 최종 에너지 (kcal/mol)
        """
        system = simulation.system
        context = simulation.context
        # 현재 좌표를 구속의 기준점으로 획득
        initial_positions = context.getState(getPositions=True).getPositions()

        if use_restraints and heavy_atom_indices:
            print("    [Optimizer] Applying heavy atom restraints for initial minimization...", flush=True)
            # 1단계: 중원자를 고정한 상태에서 수소 및 위치가 어긋난 결합 최적화
            restraint_force = self._apply_restraints(system, initial_positions, heavy_atom_indices)
            context.reinitialize(preserveState=True) # 시스템 변경 사항 적용
            simulation.minimizeEnergy(maxIterations=max_iterations)
            
            # 2단계: 구속 상수를 0으로 설정하여 구속을 풀고 전체 구조 최적화
            context.setParameter('k', 0.0)
            print("    [Optimizer] Restraints released. Finalizing optimization...", flush=True)
            simulation.minimizeEnergy(maxIterations=max_iterations)
        else:
            # 구속 없이 일반 최적화 수행
            print("    [Optimizer] Starting standard minimization (no restraints)...", flush=True)
            simulation.minimizeEnergy(maxIterations=max_iterations)

        final_state = context.getState(getEnergy=True)
        return final_state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    def calculate_energy(self, coordinates: np.ndarray, species: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        AIMNet2 모델을 사용하여 단일 포인트 에너지 및 힘(Forces)을 계산합니다.
        """
        coord_t = torch.tensor(coordinates, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_(True)
        spec_t = torch.tensor(species, dtype=torch.long, device=self.device).unsqueeze(0)
        
        energies = []
        for model in self.models:
            out = model({'coord': coord_t, 'numbers': spec_t})
            energies.append(out['energy'])
        
        # 앙상블 평균 계산
        avg_energy = torch.mean(torch.stack(energies))
        avg_energy.backward()
        
        forces = -coord_t.grad.squeeze(0).cpu().numpy()
        energy_val = avg_energy.item()
        
        return energy_val, forces


# ==============================================================================
# 2. Ensemble Calculator (for High Accuracy - ASE Compatibility)
# ==============================================================================

class EnsembleAIMNet2(Calculator):
    """
    여러 개의 AIMNet2 모델을 로드하여 앙상블 평균(Ensemble Mean)을 계산하는 ASE Calculator입니다.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, model_files, device='cuda'):
        super().__init__()
        self.calculators = []
        self.device = device
        
        if not model_files:
            raise ValueError("[Ensemble] No model files provided.")

        print(f"    [Ensemble] Loading {len(model_files)} models on {device}...", flush=True)
        
        for model_path in model_files:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            try:
                if hasattr(aimnet2calc, 'AIMNet2ASE'):
                    calc = aimnet2calc.AIMNet2ASE(model_path)
                else:
                    calc = aimnet2calc.AIMNet2Calculator(model_path)
                
                # 장치 설정 및 평가 모드 활성화
                if hasattr(calc, 'model'):
                    calc.model.to(device)
                    calc.model.eval()
                elif hasattr(calc, 'models'):
                    for m in calc.models:
                        m.to(device)
                        m.eval()
                        
                self.calculators.append(calc)
            except Exception as e:
                raise RuntimeError(f"Failed to load model {model_path}: {e}")

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        energies = []
        forces_list = []
        
        for calc in self.calculators:
            calc.calculate(atoms, properties, system_changes)
            energies.append(calc.results['energy'])
            
            if 'forces' in properties or 'forces' in calc.results:
                forces_list.append(calc.results['forces'])
        
        self.results['energy'] = np.mean(energies)
        if forces_list:
            self.results['forces'] = np.mean(forces_list, axis=0)


# ==============================================================================
# 3. Remote Calculator (for Multiprocessing Efficiency)
# ==============================================================================

def _gpu_server_process(input_queue, result_dict, device='cuda'):
    """
    백그라운드에서 실행되는 GPU 서버 프로세스입니다.
    """
    print(f"[GPU Server] Initializing AIMNet2 on {device}...", flush=True)
    
    try:
        from aimnet2 import AIMNet2Calculator as Aim2Calc
        calc = Aim2Calc(model='aimnet2_wb97m_ens', device=device)
    except ImportError:
        try:
            from aimnet.calculators import AIMNet2ASE as AIMNetCalculator
            calc = AIMNetCalculator('aimnet2')
        except Exception as e:
            print(f"[GPU Server] Critical Error: Failed to import AIMNet2 ({e})")
            return

    print("[GPU Server] Ready to process requests.", flush=True)
    
    while True:
        try:
            req = input_queue.get()
            
            if req == "STOP":
                print("[GPU Server] Stopping...", flush=True)
                break
            
            req_id = req['id']
            numbers = req['numbers']
            positions = req['positions']
            
            atoms = Atoms(numbers=numbers, positions=positions)
            atoms.calc = calc
            
            try:
                e = atoms.get_potential_energy()
                f = atoms.get_forces()
                result_dict[req_id] = {'energy': e, 'forces': f, 'error': None}
            except Exception as e:
                result_dict[req_id] = {'energy': 0.0, 'forces': None, 'error': str(e)}
                
        except Exception as e:
            print(f"[GPU Server] Unexpected error loop: {e}", flush=True)
            pass


class LocalRemoteCalculator(Calculator):
    """
    GPU 서버 프로세스와 통신하는 '가벼운' ASE Calculator입니다.
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, input_queue, result_dict):
        super().__init__()
        self.input_queue = input_queue
        self.result_dict = result_dict 

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        
        req_id = str(uuid.uuid4())
        payload = {
            'id': req_id, 
            'numbers': atoms.get_atomic_numbers(), 
            'positions': atoms.get_positions()
        }
        
        self.input_queue.put(payload)
        
        while True:
            if req_id in self.result_dict:
                resp = self.result_dict.pop(req_id)
                if resp['error']:
                    raise RuntimeError(f"GPU Calculation Failed: {resp['error']}")
                self.results['energy'] = resp['energy']
                self.results['forces'] = resp['forces']
                break
            time.sleep(0.0001)