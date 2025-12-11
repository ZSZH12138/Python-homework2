from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
import matplotlib as mpl

# 数据加载、预处理、模型训练
iris = load_iris()
x = iris.data[:, 1:]  # 使用第1-3列（Sepal Width, Petal Length, Petal Width）
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
clf_gp = GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0, 1.0]), random_state=42)
clf_gp.fit(x_train, y_train)

# 本处风格与preview有所不同 因为实验发现preview颜色浅在这里看得不是很清楚
BG_COLOR = '#EDE9E3'
TEXT_COLOR = '#1F1F1F'
SPINE_COLOR = '#303030'
WIREFRAME_COLOR = '#555555'

# 深蓝 → 米白 → 深红
custom_Logit_cmap = LinearSegmentedColormap.from_list(
    "custom_Logit_deep",
    ["#003F7F", "#F2EFEA", "#7F0000"],
    N=256
)

LEGEND_PATCHES = [
    Patch(facecolor="#7F0000", edgecolor=SPINE_COLOR, label='Logit > 0 (P > 0.5)'),
    Patch(facecolor="#F2EFEA", edgecolor=SPINE_COLOR, label='Logit = 0 (P = 0.5)'),
    Patch(facecolor="#003F7F", edgecolor=SPINE_COLOR, label='Logit < 0 (P < 0.5)')
]

AXE_NAMES = ["Sepal Width", "Petal Length", "Petal Width"]
CLASS_NAMES = ["Setosa Probability", "Versicolor Probability", "Virginica Probability"]
TITLES_COL = ["Low Slice", "Mean Slice", "High Slice"]
CLASS_INDICES = [0, 1, 2]

X_MIN = [x[:, i].min() - 0.5 for i in range(3)]
X_MAX = [x[:, i].max() + 0.5 for i in range(3)]
X3_VALS = [[np.percentile(x[:, i], 25), np.mean(x[:, i]), np.percentile(x[:, i], 75)] for i in range(3)]


# 绘制单组 3×3 图
def plot_logit_surfaces(fixed_z_idx, clf, norm_logit, levels, cmap_to_use):
    global Z_proj, Z_lim_logit

    other_indices = [i for i in range(3) if i != fixed_z_idx]
    x_idx, y_idx = other_indices[0], other_indices[1]

    h = 0.05
    x_range = np.arange(X_MIN[x_idx], X_MAX[x_idx], h)
    y_range = np.arange(X_MIN[y_idx], X_MAX[y_idx], h)
    XX, YY = np.meshgrid(x_range, y_range)
    z_fixed_values = X3_VALS[fixed_z_idx]

    plt.rcParams['figure.facecolor'] = BG_COLOR
    fig = plt.figure(figsize=(18, 16))

    fixed_feature_name = AXE_NAMES[fixed_z_idx]
    fig.suptitle(
        f"Logit (Log-Odds) Surfaces Sliced by: {fixed_feature_name}",
        fontsize=20, fontweight='bold', color=TEXT_COLOR, y=0.985
    )

    idx = 0
    for row_class in CLASS_INDICES:
        for col_slice in range(3):

            ax = fig.add_subplot(3, 3, idx + 1, projection='3d')
            z_fixed = z_fixed_values[col_slice]

            X_grid_ordered = np.zeros((XX.size, 3))
            X_grid_ordered[:, x_idx] = XX.ravel()
            X_grid_ordered[:, y_idx] = YY.ravel()
            X_grid_ordered[:, fixed_z_idx] = z_fixed

            P_all = clf.predict_proba(X_grid_ordered)
            P_k = P_all[:, row_class]
            P_k_clipped = np.clip(P_k, 1e-15, 1 - 1e-15)
            Z = np.log(P_k_clipped / (1 - P_k_clipped)).reshape(XX.shape)

            ax.set_facecolor(BG_COLOR)
            ax.plot_wireframe(XX, YY, Z, color=WIREFRAME_COLOR, linewidth=0.35)

            # 只保留 Z 方向投影
            ax.contourf(
                XX, YY, Z,
                zdir='z', offset=Z_proj,
                levels=levels, norm=norm_logit,
                cmap=cmap_to_use, alpha=0.78
            )

            ax.set_xlim(X_MIN[x_idx], X_MAX[x_idx])
            ax.set_ylim(X_MIN[y_idx], X_MAX[y_idx])
            ax.set_zlim(Z_lim_logit)

            ax.set_xlabel(AXE_NAMES[x_idx], fontsize=9, color=TEXT_COLOR)
            ax.set_ylabel(AXE_NAMES[y_idx], fontsize=9, color=TEXT_COLOR)
            ax.set_zlabel(f"Logit L({CLASS_NAMES[row_class]})", fontsize=9, color=TEXT_COLOR)
            if row_class==0:
                ax.set_title(TITLES_COL[col_slice], fontsize=11, fontweight='bold',
                             color=TEXT_COLOR, pad=0)

            ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=7)
            ax.grid(axis='both', linestyle='--', alpha=0.25, color='#777777')
            ax.view_init(elev=20, azim=-120)

            idx += 1

    # 颜色条
    sm = mpl.cm.ScalarMappable(norm=norm_logit, cmap=cmap_to_use)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Logit Value (Log-Odds)', color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    cbar.ax.set_facecolor(BG_COLOR)
    cbar.outline.set_edgecolor(SPINE_COLOR)

    ticks = np.linspace(Z_lim_logit[0], Z_lim_logit[1], 9)
    if 0 not in ticks:
        ticks = np.sort(np.append(ticks, 0.0))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    fig.legend(handles=LEGEND_PATCHES, loc='upper center',
               bbox_to_anchor=(0.5, 0.96), ncol=3,
               frameon=True, fontsize=10, facecolor=BG_COLOR, edgecolor=SPINE_COLOR)

    plt.tight_layout(rect=[0.08, 0.0, 0.90, 0.94])
    plt.show()

if __name__ == '__main__':
    Z_all = []

    for fixed_z_idx in range(3):
        other = [i for i in range(3) if i != fixed_z_idx]
        x_idx, y_idx = other[0], other[1]

        h = 0.05
        x_range = np.arange(X_MIN[x_idx], X_MAX[x_idx], h)
        y_range = np.arange(X_MIN[y_idx], X_MAX[y_idx], h)
        XX, YY = np.meshgrid(x_range, y_range)
        z_fixed_values = X3_VALS[fixed_z_idx]

        for cls in CLASS_INDICES:
            for slice_i in range(3):

                z_fixed = z_fixed_values[slice_i]
                X_grid = np.zeros((XX.size, 3))
                X_grid[:, x_idx] = XX.ravel()
                X_grid[:, y_idx] = YY.ravel()
                X_grid[:, fixed_z_idx] = z_fixed

                P_k = clf_gp.predict_proba(X_grid)[:, cls]
                P_k_clipped = np.clip(P_k, 1e-15, 1 - 1e-15)
                Z = np.log(P_k_clipped / (1 - P_k_clipped)).reshape(XX.shape)

                Z_all.append(Z)

    # 全局 Logit 范围
    Z_min = min(z.min() for z in Z_all)
    Z_max = max(z.max() for z in Z_all)
    max_abs_Z = max(abs(Z_min), abs(Z_max))

    Z_lim_logit = (-max_abs_Z, max_abs_Z)
    Z_proj = Z_lim_logit[0]

    norm_logit = TwoSlopeNorm(vmin=Z_lim_logit[0], vcenter=0.0, vmax=Z_lim_logit[1])
    levels = np.linspace(Z_lim_logit[0], Z_lim_logit[1], 21)

    # 依次绘制三组 3×3 图
    for fixed_z_idx in range(3):
        plot_logit_surfaces(fixed_z_idx, clf_gp, norm_logit, levels, custom_Logit_cmap)
