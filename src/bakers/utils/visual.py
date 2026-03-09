"""
src/bakers/utils/visual.py

[기능 정의]
시뮬레이션 결과(HDF5)를 로드하여 시각화(Landscape, Energy Dist)하고,
bakers.analytics.criteria 모듈을 활용하여 성공 여부를 과학적으로 판별하는 통합 모듈입니다.

[수정 내역]
- [Visual Fix] 'Dataset has 0 variance' 경고 해결
  : 데이터의 분산이 0일 경우(모든 에너지가 같을 때), kdeplot 대신 histogram을 그리도록 분기 처리.
  : 불필요한 경고 메시지 억제 (warnings.filterwarnings).
- [Restore] Landscape Plot, Combinatorial Plot 등 고급 시각화 로직 복구.
- [Fix] HDF5 데이터 키 수정 ('values' -> 'energies').
- [Integration] bakers.analytics.criteria 및 metrics 모듈 연동.
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import combinations
import warnings

# [설정] 불필요한 경고 메시지 억제 (특히 분산 0일 때의 KDE 경고)
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# [Dependency] Criteria Module Integration (성공 여부 판별)
try:
    from bakers.analytics.criteria import check_energy_criteria, print_criteria_report
except ImportError:
    check_energy_criteria = None
    def print_criteria_report(res): pass

# [Dependency] Metrics Helper (RMSD 계산)
try:
    from bakers.analytics.metrics import calculate_rmsd
except ImportError:
    # 모듈이 없을 경우 더미 함수 제공 (의존성 최소화)
    def calculate_rmsd(P, Q): return 999.9

# ==============================================================================
# 1. Plotting Style & Helpers
# ==============================================================================

def setup_plotting_style():
    """논문 게재 가능한 수준의 고품질 플롯 스타일을 설정합니다."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update({
        'font.family': 'sans-serif', 
        'axes.linewidth': 1.5,
        'figure.dpi': 300, 
        'savefig.bbox': 'tight',
        'xtick.direction': 'in', 
        'ytick.direction': 'in',
    })

def get_smart_bounds(x_data, y_data, padding=20.0):
    """
    데이터 분포에 따라 축 범위를 스마트하게 결정합니다.
    주기성(Periodic Boundary)이 감지되면 (-180, 180) 전체를 보여줍니다.
    """
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    # 데이터가 양 끝단(-180 근처와 +180 근처)에 걸쳐 있는지 확인 (주기성 체크)
    is_wrapped_x = (x_min < -170 and x_max > 170)
    is_wrapped_y = (y_min < -170 and y_max > 170)
    
    # 범위가 넓거나 주기적이면 전체 범위, 아니면 데이터 주변만 표시
    xlim = (-180, 180) if (x_max - x_min > 240 or is_wrapped_x) else (max(-180, x_min - padding), min(180, x_max + padding))
    ylim = (-180, 180) if (y_max - y_min > 240 or is_wrapped_y) else (max(-180, y_min - padding), min(180, y_max + padding))
    return xlim, ylim

def calculate_periodic_distance(p1, p2):
    """주기적 경계 조건(-180 ~ 180)을 고려한 각도 거리 계산"""
    diff = np.abs(p1 - p2)
    diff = np.minimum(diff, 360.0 - diff)
    return np.linalg.norm(diff)

def get_distinct_candidates(df, angle_cols, threshold=30.0, top_n=5):
    """
    에너지가 낮은 순서대로 구조를 훑으며, 
    기존 후보들과 구조적으로(각도 거리) 유사하지 않은 'Distinct'한 구조만 추출합니다.
    """
    sorted_df = df.sort_values('Energy').reset_index(drop=True)
    candidates = []
    
    for _, row in sorted_df.iterrows():
        current_angles = row[angle_cols].values.astype(float)
        is_distinct = True
        
        # 기존 후보들과 비교
        for cand in candidates:
            cand_angles = cand[angle_cols].values.astype(float)
            dist = calculate_periodic_distance(current_angles, cand_angles)
            if dist < threshold:
                is_distinct = False
                break
        
        if is_distinct:
            candidates.append(row)
            if len(candidates) >= top_n: break
            
    return pd.DataFrame(candidates)

# ==============================================================================
# 2. Plotting Functions (Visualization Core)
# ==============================================================================

def plot_energy_distribution(df, output_path):
    """
    에너지 분포 히스토그램을 그립니다.
    분산이 0인 경우(단일 값) KDE Plot 대신 일반 히스토그램을 사용합니다.
    """
    setup_plotting_style()
    plt.figure(figsize=(8, 5))
    
    # [Visual Fix] 분산 0 체크 -> KDE 대신 Hist만 그림
    if df['Energy'].std() < 1e-6:
        sns.histplot(df['Energy'], bins=30, kde=False, color='#4c72b0', alpha=0.6)
        plt.title("Energy Distribution (Uniform)", fontweight='bold')
    else:
        try:
            sns.histplot(df['Energy'], bins=50, kde=True, color='#4c72b0', alpha=0.6)
            plt.title("Energy Distribution", fontweight='bold')
        except:
            # KDE 계산 실패 시(데이터 부족 등) Fallback
            sns.histplot(df['Energy'], bins=50, kde=False, color='#4c72b0', alpha=0.6)
            
    plt.xlabel("Energy (kcal/mol)", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path)
    plt.close()

def plot_rmsd_vs_energy(df, output_path):
    """
    에너지 vs RMSD 산점도(Funnel Plot)를 그립니다.
    구조 예측이 성공적이라면 깔때기(Funnel) 모양이 나타나야 합니다.
    """
    setup_plotting_style()
    plt.figure(figsize=(7, 6))
    
    x, y = df['RMSD'], df['Rel_Energy']
    plt.scatter(x, y, s=15, c='#0077b6', alpha=0.6, edgecolors='none', rasterized=True)
    
    plt.xlabel("Backbone RMSD ($\AA$)", fontweight='bold', fontsize=14)
    plt.ylabel("$\Delta$Energy (kcal/mol)", fontweight='bold', fontsize=14)
    
    # Y축 범위 제한 (너무 높은 에너지는 잘라냄)
    plt.ylim(-0.2, min(y.max(), 10.0))
    plt.xlim(left=-0.1)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_monomer_landscape(df, x_col, y_col, output_path, top_candidates=None):
    """
    2차원 에너지 랜드스케이프(Ramachandran Plot 등)를 그립니다.
    - Contourf: 에너지 등고선
    - Scatter: 샘플링 포인트
    - Marginal: 각 축의 분포 (KDE/Hist)
    """
    setup_plotting_style()
    
    global_min_e = df['Energy'].min()
    if 'Rel_Energy' not in df.columns:
        df['Rel_Energy'] = df['Energy'] - global_min_e
        
    x_limits, y_limits = get_smart_bounds(df[x_col], df[y_col])
    viz_cutoff = 6.0 # 시각화할 에너지 상한선 (kcal/mol)
    
    # 레이아웃 설정 (메인 플롯 + 상단/우측 분포 플롯)
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[15, 1], height_ratios=[1, 15], wspace=0.02, hspace=0.02)
    
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    
    plt.setp(ax_top.get_xticklabels(), visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_main.set_facecolor('#f0f0f0')
    
    # 컬러맵 안전 설정
    try:
        current_cmap = plt.colormaps['turbo_r'].copy()
    except:
        current_cmap = plt.cm.get_cmap('turbo_r').copy()
    current_cmap.set_bad(color='#f0f0f0')
    
    # 1. Triangulation Mesh (배경 그리드)
    try:
        triang_all = tri.Triangulation(df[x_col], df[y_col])
        ax_main.triplot(triang_all, color='#999999', alpha=0.2, linewidth=0.5, zorder=1)
    except: pass 

    # 2. Contour Plot (에너지 등고선)
    mask = df['Rel_Energy'] <= viz_cutoff
    # [Visual Fix] 분산이 0이거나 데이터가 너무 적으면 Contour 생략
    if mask.sum() > 3 and df.loc[mask, 'Rel_Energy'].std() > 1e-6:
        try:
            triang_masked = tri.Triangulation(df.loc[mask, x_col], df.loc[mask, y_col])
            levels = np.linspace(0, viz_cutoff, 41)
            cntr = ax_main.tricontourf(triang_masked, df.loc[mask, 'Rel_Energy'], levels=levels, cmap=current_cmap, extend='max', alpha=0.9, zorder=2)
            ax_main.triplot(triang_masked, color='black', alpha=0.2, linewidth=0.5, zorder=3)
            
            # Colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6])
            cbar = plt.colorbar(cntr, cax=cbar_ax)
            cbar.set_label(r'$\Delta$ Energy (kcal/mol)', rotation=270, labelpad=20)
        except: pass

    # 3. Scatter Plot (저에너지 포인트)
    df_low = df[mask]
    ax_main.scatter(df_low[x_col], df_low[y_col], c='black', s=5, alpha=0.1, edgecolors='none', zorder=5, rasterized=True)
    
    # 4. Top Candidates Highlight (Global Min 등 표시)
    if top_candidates is not None and not top_candidates.empty:
        rank1 = top_candidates.iloc[0]
        # Global Min (Red Star)
        ax_main.scatter(rank1[x_col], rank1[y_col], c='#d62728', s=300, marker='*', edgecolor='white', linewidth=1.5, zorder=10, label='Global Min')
        
        # Other Top Candidates (White Diamond)
        if len(top_candidates) > 1:
            others = top_candidates.iloc[1:]
            others_in = others[(others[x_col] >= x_limits[0]) & (others[x_col] <= x_limits[1]) & (others[y_col] >= y_limits[0]) & (others[y_col] <= y_limits[1])]
            if not others_in.empty:
                ax_main.scatter(others_in[x_col], others_in[y_col], c='white', s=80, marker='D', edgecolor='black', zorder=9)

    # 5. Marginal Plots (Side Histograms)
    if not df_low.empty:
        # [Visual Fix] 분산 0 체크 -> KDE 대신 Hist 사용
        has_variance = (df_low[x_col].std() > 1e-6 and df_low[y_col].std() > 1e-6)
        try:
            if has_variance:
                sns.kdeplot(x=df_low[x_col], ax=ax_top, fill=True, color='#303030', alpha=0.3, linewidth=0, bw_adjust=0.15, warn_singular=False)
                sns.kdeplot(y=df_low[y_col], ax=ax_right, fill=True, color='#303030', alpha=0.3, linewidth=0, bw_adjust=0.15, warn_singular=False)
            else:
                # 분산 없으면 히스토그램으로 대체
                sns.histplot(x=df_low[x_col], ax=ax_top, color='#303030', alpha=0.3, kde=False)
                sns.histplot(y=df_low[y_col], ax=ax_right, color='#303030', alpha=0.3, kde=False, orientation='horizontal')
        except: pass
    
    # 축 정리
    ax_top.axis('off'); ax_right.axis('off')
    ax_main.set_xlabel(x_col, fontweight='bold'); ax_main.set_ylabel(y_col, fontweight='bold')
    ax_main.set_xlim(x_limits); ax_main.set_ylim(y_limits)
    
    plt.savefig(output_path)
    plt.close()

# ==============================================================================
# 3. Main Analysis Wrappers (External Interface)
# ==============================================================================

def analyze_and_save(file_path, output_dir=None):
    """
    [일반 분석] HDF5 파일을 로드하여 에너지 분포 및 Landscape를 시각화하고 저장합니다.
    """
    if not os.path.exists(file_path): return
    try:
        with h5py.File(file_path, 'r') as f:
            if 'points' not in f: return
            
            # [Fix] HDF5 Key Compatibility ('energies' vs 'values')
            if 'energies' in f: values = f['energies'][:]
            elif 'values' in f: values = f['values'][:]
            else: return
                
            points = f['points'][:]
    except Exception as e:
        print(f"[Visual Error] Failed to load {file_path}: {e}")
        return

    # 컬럼 이름 자동 생성 (phi, psi, theta ...)
    num_angles = points.shape[1]
    if num_angles == 2: angle_cols = ['phi', 'psi']
    elif num_angles == 3: angle_cols = ['phi', 'theta', 'psi']
    else: angle_cols = [f'Angle_{i+1}' for i in range(num_angles)]
        
    df = pd.DataFrame(points, columns=angle_cols)
    df['Energy'] = values

    if output_dir is None:
        file_name = os.path.basename(file_path).replace('.hdf5', '')
        output_dir = os.path.join('2_results', 'analysis', file_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"    [Analysis] Saving basic results to: {output_dir}")

    # 저에너지 구조 필터링 및 후보 추출
    min_e = df['Energy'].min()
    df_vis = df[df['Energy'] <= min_e + 500.0].copy() # 시각화용 데이터 (Outlier 제거)
    
    top_candidates = get_distinct_candidates(df_vis, angle_cols[:num_angles], threshold=45.0, top_n=5)
    
    # CSV 저장
    df.to_csv(os.path.join(output_dir, 'structural_data.csv'), index=False)
    top_candidates.to_csv(os.path.join(output_dir, 'top_candidates.csv'), index_label='Rank')

    # 1. Energy Distribution Plot
    plot_energy_distribution(df_vis, os.path.join(output_dir, 'energy_dist.png'))
    
    # 2. Landscape Plots (조합별 생성)
    if num_angles >= 2:
        for (x_col, y_col) in combinations(angle_cols, 2):
            plot_name = "landscape.png" if num_angles == 2 else f"landscape_{x_col}_{y_col}.png"
            plot_monomer_landscape(df_vis, x_col, y_col, 
                                   os.path.join(output_dir, plot_name), 
                                   top_candidates)

def analyze_rmsd(file_path, output_dir=None, num_residues=1):
    """
    [RMSD 분석] 에너지 vs RMSD 플롯을 생성하고, 성공 기준(Criteria)을 판별합니다.
    """
    if not os.path.exists(file_path): return
    try:
        with h5py.File(file_path, 'r') as f:
            # [Fix] HDF5 Key Compatibility
            if 'energies' in f: values = f['energies'][:]
            elif 'values' in f: values = f['values'][:]
            else: return
                
            if 'xyzs' not in f: return
            xyzs = f['xyzs'][:]
    except: return

    df = pd.DataFrame({'Energy': values})
    df['Rel_Energy'] = df['Energy'] - df['Energy'].min()

    if output_dir is None:
        file_name = os.path.basename(file_path).replace('.hdf5', '')
        output_dir = os.path.join('2_results', 'analysis', file_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"    [Analysis] Calculating RMSD and Checking Criteria for: {output_dir}")

    try:
        # RMSD 계산 (Global Min 기준)
        min_idx = df['Energy'].idxmin()
        ref_coords = xyzs[min_idx]
        
        rmsd_list = [calculate_rmsd(coord, ref_coords) for coord in xyzs]
        df['RMSD'] = rmsd_list
        
        # Criteria Check (성공 여부 판별)
        if check_energy_criteria:
            sorted_indices = np.argsort(values)
            sorted_xyzs = xyzs[sorted_indices]
            sorted_energies = values[sorted_indices]
            
            criteria_result = check_energy_criteria(sorted_xyzs, sorted_energies, num_residues=num_residues)
            print_criteria_report(criteria_result)
            
            # 리포트 저장
            with open(os.path.join(output_dir, 'criteria_report.txt'), 'w') as f:
                f.write(str(criteria_result))
        
        # Plotting
        df_rmsd_vis = df[df['Rel_Energy'] <= 10.0].copy()
        plot_rmsd_vs_energy(df_rmsd_vis, os.path.join(output_dir, 'rmsd_vs_energy.png'))
        
    except Exception as e:
        print(f"    [Warn] RMSD calculation/Analysis failed: {e}")