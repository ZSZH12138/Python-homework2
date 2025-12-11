import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mcolors
from sklearn.datasets import load_iris
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline

# 载入数据
iris = load_iris()
X = iris.data[:, 2:]  # 作业要求使用两个特征进行回归
y = iris.target
# 获取标签
# 关键修正：将 target_names 转换为 Python 列表，避免 Matplotlib 内部报错
target_names = iris.target_names.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 这里选择ppt内展示的七个模型
classifiers = {
    "Logistic regression\n(C=0.01)": LogisticRegression(C=0.1),
    "Logistic regression\n(C=1)": LogisticRegression(C=100),
    "Gaussian Process": GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0])),
    "Logistic regression\n(RBF features)": make_pipeline(
        Nystroem(kernel="rbf", gamma=5e-1, n_components=50, random_state=1),
        LogisticRegression(C=10),
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(),
    "Logistic regression\n(binned features)": make_pipeline(
        KBinsDiscretizer(n_bins=5, quantile_method="averaged_inverted_cdf"),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
    "Logistic regression\n(spline features)": make_pipeline(
        SplineTransformer(n_knots=5),
        PolynomialFeatures(interaction_only=True),
        LogisticRegression(C=10),
    ),
}

# 设置绘图风格（与preview保持一致）
bg_color = "#F5F3EE"
edge_color = "#444444"
grid_color = "#BBBBBB"
title_color = "#333333"
class_colors_hex = ["#98C1D9", "#E8D8C4", "#D0A5C0"]
markers = ["o", "s", "D"]
contour_color = "#666666"
contour_linestyle = "--"
plt.style.use("seaborn-v0_8-white")

# 生成网格数据
pad = 0.8
# 通过实际数据的最值以及pad 找到我们网格所要覆盖的面积
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
res = 300 # 300*300的网格
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, res),
    np.linspace(y_min, y_max, res)
)
# 将一个个点坐标转换为一个个列表
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 颜色映射
# 生成色阶 0表示浅 12表示深
levels = np.linspace(0, 1, 12)
class_cmaps_discrete = []
class_cmaps_continuous = []
for hex_color in class_colors_hex:
    new_cmap_discrete = ListedColormap([mcolors.to_rgba(hex_color, alpha=a)
                                        for a in np.linspace(0.1, 1.0, len(levels) - 1)])
    class_cmaps_discrete.append(new_cmap_discrete)
    rgba_start = mcolors.to_rgba(hex_color, alpha=0.1)
    # 使颜色平滑过渡 共256阶
    new_cmap_continuous = mcolors.LinearSegmentedColormap.from_list(
        'custom_gradient',
        [rgba_start, hex_color],
        N=256
    )
    class_cmaps_continuous.append(new_cmap_continuous)
# 12个等级映射到0 1区间 方便后面直接取 用于离散
norm = BoundaryNorm(levels, len(levels) - 1)
# 用于连续
cbar_norm = mcolors.Normalize(vmin=0, vmax=1)

def style_ax(ax):
    """设置画布 与preview里面做法类似"""
    ax.set_facecolor(bg_color)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color=grid_color)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(edge_color)
    ax.spines["bottom"].set_color(edge_color)
    ax.tick_params(colors=title_color)

# 动态创建大画布和子图
n_classifiers = len(classifiers)
fig, axes = plt.subplots(n_classifiers, 4, figsize=(18, 4.5 * n_classifiers), squeeze=False) # 保证axes是二维数组
fig.patch.set_facecolor(bg_color)

# 调整顶部和底部边距：增加底部空间用于放置图例
fig.subplots_adjust(left=0.18, top=0.92, bottom=0.10, hspace=0.6, wspace=0.3)

fig.suptitle(
    "Task 1: Visualize the results of different classifiers",
    fontsize=18,
    fontweight='bold',
    color=title_color,
    y=0.98
)

# 用于收集一次性图例元素的列表
legend_elements = []
print("Starting classifier training and plotting...")
for row_idx, (name, classifier) in enumerate(classifiers.items()):
    print(f"Training and plotting: {name.replace('\n', ' ')}")
    classifier.fit(X_train, y_train)
    # 获取每个点属于三类的三个概率
    proba_grid = classifier.predict_proba(grid_points)
    # 把概率分配到每个点头上
    proba_grid_reshaped = proba_grid.reshape(xx.shape[0], xx.shape[1], -1)
    # 行标题
    fig.text(
        0.005,
        axes[row_idx, 0].get_position().y0 + axes[row_idx, 0].get_position().height / 2,
        name,
        fontsize=14,
        fontweight='bold',
        color=title_color,
        ha='left',
        va='center',
        rotation=0
    )
    # 绘制当前分类器的 4 个子图
    for col_idx in range(4):
        ax = axes[row_idx, col_idx]
        # 对于前三个子图
        if proba_grid_reshaped.shape[2] >= 3 > col_idx:
            prob_i = proba_grid_reshaped[:, :, col_idx]
            # 概率热力图
            im = ax.imshow(
                prob_i,
                origin="lower",
                extent=(x_min, x_max, y_min, y_max),
                cmap=class_cmaps_discrete[col_idx],
                norm=norm,
                alpha=0.92,
                aspect="auto"
            )
            # 绘制决策边界
            ax.contour(
                xx, yy,
                prob_i,
                levels=[0.5],
                colors=[contour_color],
                linestyles=contour_linestyle,
                linewidths=1.1,
                alpha=0.8
            )
            # 散点
            for ci in range(len(target_names)):
                mask = (y == ci)
                if row_idx == 0 and col_idx == 0:
                    label = target_names[ci]
                else:
                    label = "_nolegend_" # 图例统一绘制
                ax.scatter(
                    X[mask, 0], X[mask, 1],
                    c=class_colors_hex[ci],
                    marker=markers[ci],
                    s=35,
                    edgecolor=edge_color,
                    linewidth=0.5,
                    alpha=0.95,
                    label=label
                )

                # 在第一次迭代中收集图例元素 (确保只收集一次)
                if row_idx == 0 and col_idx == 0:
                    legend_elements.append(
                        plt.Line2D([0], [0],
                                   marker=markers[ci],
                                   color='w',
                                   label=target_names[ci],
                                   markerfacecolor=class_colors_hex[ci],
                                   markeredgecolor=edge_color,
                                   markersize=8)
                    )
            if row_idx==0:
                ax.set_title(target_names[col_idx], fontsize=12, fontweight="bold", color=title_color)
            if col_idx == 0:
                ax.set_ylabel("Petal Width", fontsize=10, color=title_color)

            # 颜色映射器以及颜色条
            sm = cm.ScalarMappable(norm=cbar_norm, cmap=class_cmaps_continuous[col_idx])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=title_color)

        elif col_idx == 3:  # 第四个图：整体预测区域（平滑渐变）
            pred_class_idx = np.argmax(proba_grid_reshaped, axis=2)
            max_proba_values = np.max(proba_grid_reshaped, axis=2)

            final_image_data = np.zeros((res, res, 4))
            for r in range(res):
                for c in range(res):
                    class_idx = pred_class_idx[r, c]
                    proba_val = max_proba_values[r, c]

                    rgba_color_gradient = [mcolors.to_rgba(class_colors_hex[class_idx], alpha=a)
                                           for a in np.linspace(0.1, 1.0, 100)]
                    gradient_idx = int(proba_val * (len(rgba_color_gradient) - 1))
                    final_image_data[r, c] = rgba_color_gradient[gradient_idx]

            ax.imshow(
                final_image_data,
                origin="lower",
                extent=(x_min, x_max, y_min, y_max),
                alpha=0.9,
                aspect="auto"
            )
            if row_idx==0:
                ax.set_title("Overall Decision Boundaries", fontsize=12, fontweight="bold", color=title_color)

            # 叠加真实点
            for ci, mark in enumerate(markers):
                mask = (y == ci)
                ax.scatter(
                    X[mask, 0], X[mask, 1],
                    marker=mark,
                    s=50,
                    edgecolor=edge_color,
                    linewidth=0.6,
                    facecolor=class_colors_hex[ci],
                    alpha=0.95
                )

        # 每个子图通用的样式和 X 轴标签
        style_ax(ax)
        ax.set_xlabel("Petal Length", fontsize=10, color=title_color)

# 在 Figure 级别绘制单个图例
print("Plotting complete. Displaying figure...")
fig.subplots_adjust(hspace=0.5)
if legend_elements:
    fig.legend(handles=legend_elements,
               labels=target_names,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.02),
               ncol=len(target_names),
               frameon=False,
               fontsize=12,
               labelcolor=title_color)

# 显示
plt.show()