"""
src/bakers/sim/sampler.py

[설명]
Boltzmann 분포를 기반으로 에너지 랜드스케이프를 효율적으로 탐색하는 
적응형 샘플링(Adaptive Sampling) 클래스를 정의합니다.

Delaunay Triangulation을 사용하여 공간을 분할하고, 
각 분할된 영역(Simplex)의 부피와 Boltzmann 확률(에너지)을 결합한 
Loss 함수를 통해 탐색할 가치가 높은 영역을 선정합니다.

[주요 기능]
1. 초기 포인트에 대한 Delaunay Mesh 구축 및 캐싱 (Pickle)
2. Boltzmann 가중치를 적용한 유망 탐색 영역(Candidate) 추천 (Ask)
3. 새로운 샘플링 결과 업데이트 및 메쉬 재구성 (Tell)
4. 샘플링 속도 조절을 위한 동적 Controller

[작성자]
BAKERS Lead Chemist & Engineer

[업데이트]
- 캐싱 로직 통합 및 안정성 강화
- Type Hinting 및 상세 주석 추가
"""

import os
import math
import pickle
import numpy as np
from scipy.spatial import Delaunay
from timeit import default_timer
from typing import Optional, Tuple, Callable, Union, List

# Numpy math patch (Legacy support: 일부 구형 코드 호환성 유지)
np.math = math

class BoltzmannAdaptiveSampler(object):
    """
    Delaunay Triangulation과 Boltzmann 가중치를 사용하여
    에너지가 낮은 영역을 집중적으로 탐색하는 Sampler입니다.
    """
    def __init__(
        self, 
        points: np.ndarray, 
        values: np.ndarray, 
        kt: float = 3.0, 
        clip: float = 12.0, 
        cache_path: Optional[str] = None
    ):
        """
        Sampler 초기화 및 초기 메쉬 구축.

        Args:
            points (np.ndarray): 초기 데이터 포인트 좌표 배열 (N, D).
            values (np.ndarray): 초기 포인트의 값 (에너지) 배열 (N,).
            kt (float): Boltzmann 온도 계수 (탐색/활용 균형 조절). 기본값 3.0.
            clip (float): 에너지 값의 상한 클리핑 (수치 안정성). 기본값 12.0.
            cache_path (str, optional): Delaunay 메쉬를 저장/로드할 경로.
        """
        assert len(points) == len(values), "Points and values must have the same length."
        
        self.kt = kt
        self.clip = clip
        self.dims = points.shape[-1]
        
        # 데이터 복사 방지 및 형변환
        self.values = np.array(values, dtype=float)
        
        # 샘플링 속도 조절 변수 (Controller)
        # 초기값: 차원에 따라 보수적으로 시작 (음수 값)
        # 값이 높을수록 한 번에 더 많은 포인트를 탐색하려고 시도함
        self.controller = float(-self.dims)
        
        # Delaunay Mesh 구축 (캐싱 지원)
        self.delaunay: Optional[Delaunay] = None
        self._build_and_save_delaunay(points, cache_path)

    def _build_and_save_delaunay(self, points: np.ndarray, cache_path: Optional[str]) -> None:
        """
        Delaunay Triangulation을 수행합니다. 
        계산 비용이 높으므로 cache_path가 제공되면 로드/저장을 시도합니다.

        Args:
            points (np.ndarray): 메쉬를 구축할 포인트들.
            cache_path (str, optional): 캐시 파일 경로.
        """
        self.delaunay = None
        
        # 1. 캐시 로드 시도
        if cache_path and os.path.exists(cache_path):
            print(f"      [Sampler] Found cache! Loading Delaunay mesh from {cache_path}...", flush=True)
            try:
                with open(cache_path, 'rb') as f:
                    self.delaunay = pickle.load(f)
                print(f"      [Sampler] Cache loaded successfully.", flush=True)
                
                # 캐시된 포인트와 현재 입력 포인트가 일치하는지 검증 (개수 비교)
                if len(self.delaunay.points) != len(points):
                    print(f"      [Warning] Point count mismatch (Cache: {len(self.delaunay.points)}, Input: {len(points)}). Rebuilding...", flush=True)
                    self.delaunay = None
            except Exception as e:
                print(f"      [Warning] Cache load failed ({e}). Rebuilding...", flush=True)
                self.delaunay = None

        # 2. 메쉬 구축 (캐시가 없거나 로드 실패 시)
        if self.delaunay is None:
            print(f"      [Sampler] Building Delaunay mesh for {len(points)} points... (CPU Heavy)", flush=True)
            t_start = default_timer()
            
            # Qhull 옵션 설명:
            # Q0: 초기 옵션
            # Q12: 동일평면(Coplanar) 면 병합 허용
            # Qc: Coplanar 포인트 유지
            # QJ: Joggled input (좌표를 미세하게 흔들어 수치적 에러 방지 및 삼각화 보장)
            try:
                self.delaunay = Delaunay(points, incremental=True, qhull_options='Q0 Q12 Qc QJ')
                elapsed = default_timer() - t_start
                print(f"      [Sampler] Mesh built in {elapsed:.2f}s.", flush=True)
            except Exception as e:
                 print(f"      [Critical Error] Delaunay triangulation failed: {e}")
                 raise e
            
            # 3. 캐시 저장
            if cache_path:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(self.delaunay, f)
                    print(f"      [Sampler] Saved mesh to {cache_path}.", flush=True)
                except Exception as e:
                    print(f"      [Warning] Failed to save cache: {e}", flush=True)

    def ask(self, n_points: Optional[int] = None) -> np.ndarray:
        """
        다음으로 샘플링할 후보 포인트들을 추천합니다.
        Loss(부피 * Boltzmann 확률)가 높은 영역의 가중 중심을 반환합니다.
        
        Args:
            n_points (int, optional): 요청할 포인트 개수. 
                                      None이면 Controller 상태에 따라 자동 결정.
        
        Returns:
            np.ndarray: 추천된 포인트 좌표 배열 (K, D)
        """
        if n_points is None:
            # Controller 기반 동적 배치 크기 결정
            # simplices 수에 비례하되, controller 값으로 지수적 조절 (10^controller)
            # controller가 낮으면 매우 적게, 높으면 많이 탐색
            estimated_size = int((10**self.controller) * len(self.simplices) + 1)
            run_size = min(len(self.simplices), estimated_size)
            run_size = max(1, run_size)
        else:
            run_size = n_points

        # Loss가 높은 영역 선택
        # np.argpartition을 사용하여 상위 run_size개의 인덱스를 빠르게 추출 (정렬보다 효율적)
        if len(self.losses) < run_size:
            selection = np.arange(len(self.losses))
        else:
            # -run_size: 뒤에서부터 run_size개 (상위 값들)
            selection = np.argpartition(self.losses, -run_size)[-run_size:]
            
        return self.weighted_centers[selection]

    def tell(self, points: np.ndarray, values: np.ndarray) -> bool:
        """
        새로 계산된 포인트와 그 값을 Sampler에 업데이트합니다.
        Delaunay 메쉬를 증분 업데이트(Incremental Update)합니다.
        
        Args:
            points (np.ndarray): 새로 계산된 포인트 좌표 (K, D)
            values (np.ndarray): 해당 포인트의 값 (에너지) (K,)
            
        Returns:
            bool: 업데이트 성공 여부 (Qhull 에러 발생 시 False 반환)
        """
        if len(points) == 0:
            return True
            
        try:
            # Delaunay 메쉬에 점 추가 (증분)
            self.delaunay.add_points(points)
            self.values = np.concatenate([self.values, values])
            
            # 성공 시 Controller 값을 약간 증가시켜 다음 배치 크기를 늘림 (Aggressive Strategy)
            # 최대 0까지 증가 (너무 커지는 것 방지)
            self.controller = min(0.0, self.controller + 0.1)
            return True
            
        except Exception as e:
            print(f"[Sampler Error] Qhull update failed: {e}")
            # 실패 시 Controller 값을 대폭 감소시켜 다음 배치를 보수적으로 설정 (Conservative Strategy)
            self.controller -= 0.5
            return False

    def run(self, scoring_function: Callable[[np.ndarray], np.ndarray]) -> int:
        """
        (편의 메서드) ask -> score -> tell 과정을 한 번 수행합니다.
        
        Args:
            scoring_function (callable): 좌표 배열(np.ndarray)을 받아 값 배열(np.ndarray)을 반환하는 함수.
            
        Returns:
            int: 새로 추가된(성공적으로 업데이트된) 포인트의 개수.
        """
        # 1. Ask (후보 생성)
        candidates = self.ask()
        
        # 2. Evaluate (계산/평가)
        new_values = scoring_function(candidates)
        
        # 3. Tell (업데이트)
        success = self.tell(candidates, new_values)
        
        return len(candidates) if success else 0

    # ==========================================================================
    # Properties (Calculation Logic)
    # ==========================================================================
    
    @property
    def points(self) -> np.ndarray:
        """현재 메쉬의 모든 포인트 좌표"""
        return self.delaunay.points

    @property
    def simplices(self) -> np.ndarray:
        """현재 메쉬를 구성하는 Simplex(삼각형/사면체)의 정점 인덱스 배열"""
        return self.delaunay.simplices

    @property
    def centers(self) -> np.ndarray:
        """
        각 Simplex의 기하학적 중심 (Centroid).
        Returns: (N_simplices, D)
        """
        return np.average(self.points[self.simplices], axis=1)

    @property
    def weighted_centers(self) -> np.ndarray:
        """
        각 Simplex의 가중 중심. 
        단순 중심(Centroid)보다 중심에서 먼 쪽으로 약간 치우치게 하여 
        이미 탐색된 영역보다는 경계면이나 넓은 영역을 선호하도록 유도합니다.
        
        Returns: (N_simplices, D)
        """
        # (N_simplices, Dim+1, Dim) - (N_simplices, 1, Dim) broadcasting
        # 각 정점과 중심 사이의 거리 벡터 계산
        diff = self.points[self.simplices] - self.centers[:, np.newaxis, :]
        
        # 유클리드 거리 (L2 norm)
        dists = np.linalg.norm(diff, axis=2)[:, :, np.newaxis]
        
        # 거리 기반 가중치 (멀수록 가중치 높음 -> 중심으로의 쏠림 방지)
        # 1e-9는 0으로 나누기 방지용 안전상수
        weights = np.tile(dists, (1, 1, self.dims)) + 1e-9
        
        return np.average(self.points[self.simplices], axis=1, weights=weights)

    @property
    def delta_values(self) -> np.ndarray:
        """최솟값 기준 상대 에너지 (Delta E). 클리핑 적용."""
        return np.clip(self.values - np.min(self.values), 0.0, self.clip)

    @property
    def boltzmann_values(self) -> np.ndarray:
        """
        Boltzmann 확률 인자 (exp(-dE/kT)).
        에너지가 낮을수록 값이 커집니다.
        """
        return np.exp(self.delta_values / -self.kt)

    @property
    def volumes(self) -> np.ndarray:
        """
        각 Simplex의 부피(Volume) 계산.
        다차원 공간에서의 부피는 행렬식(Determinant)을 이용해 계산됩니다.
        """
        # Simplex의 첫 번째 점을 기준으로 나머지 점들과의 벡터 행렬 생성
        # matrix shape: (N_simplices, Dim, Dim)
        matrix = self.points[self.simplices][:, 1:, :] - self.points[self.simplices][:, :1, :]
        
        # 행렬식의 절댓값 계산
        det = np.linalg.det(matrix)
        
        # Simplex 부피 공식: |det| / Dim!
        return np.abs(det / np.math.factorial(self.dims))

    @property
    def losses(self) -> np.ndarray:
        """
        각 Simplex의 탐색 중요도(Loss).
        정의: 부피(Volume) * 평균 Boltzmann 확률(Probability)
        
        의미:
        1. 부피가 클수록 (탐색이 덜 된 영역) -> Loss 높음
        2. 에너지가 낮을수록 (Boltzmann 확률 높음) -> Loss 높음
        결과적으로 '넓으면서도 에너지가 낮은' 영역을 우선적으로 탐색합니다.
        """
        # Simplex 내부 포인트들의 평균 Boltzmann 값 계산
        avg_boltzmann = np.squeeze(np.mean(self.boltzmann_values[self.simplices], axis=1))
        
        losses = self.volumes * avg_boltzmann
        
        # NaN 처리 및 정규화
        losses = np.nan_to_num(losses, nan=0.0)
        sum_losses = np.sum(losses)
        
        # 모든 Loss가 0인 경우 균등 확률 반환
        if sum_losses == 0:
            return np.ones_like(losses) / len(losses)
            
        return losses / sum_losses