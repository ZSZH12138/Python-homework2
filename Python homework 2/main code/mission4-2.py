from sklearn.datasets import load_iris
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 风格保持与preview一致
BG_COLOR = '#F5F3EE'
TEXT_COLOR = '#333333'
SPINE_COLOR = '#444444'
GRID_COLOR = '#BBBBBB'
BOUNDARIES_ALPHA = 0.15

iris = load_iris()
x_raw = iris.data
y = iris.target
x = x_raw[:, 1:]  # 使用后三个特征 (Sepal Width, Petal Length, Petal Width)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
kernel = 1.0 * RBF([1.0, 1.0, 1.0])
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
gpc.fit(x_train, y_train)
print(f"Train accuracy: {gpc.score(x_train, y_train):.2f}")
print(f"Test accuracy: {gpc.score(x_test, y_test):.2f}")

# 构建 3D 空间网格
resolution = 40
x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
x3_min, x3_max = x[:, 2].min() - 0.5, x[:, 2].max() + 0.5
xx1, xx2, xx3 = np.meshgrid(
    np.linspace(x1_min, x1_max, resolution),
    np.linspace(x2_min, x2_max, resolution),
    np.linspace(x3_min, x3_max, resolution),
    indexing='ij'
)
grid_points = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]

# 寻找边界
probs = gpc.predict_proba(grid_points)
probs_sorted = np.sort(probs, axis=1)
margin = probs_sorted[:, -1] - probs_sorted[:, -2]

# 设定阈值：只有由于度极小(比如小于0.05)的点，才被认为是"边界"
threshold = 0.05
boundary_mask = margin < threshold
# 提取边界点
boundary_points = grid_points[boundary_mask]
boundary_predictions = gpc.predict(boundary_points)

# 绘图
# 设置全局背景色
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.style.use('default')

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')


ax.set_facecolor(BG_COLOR)
ax.xaxis.pane.set_facecolor(BG_COLOR)
ax.yaxis.pane.set_facecolor(BG_COLOR)
ax.zaxis.pane.set_facecolor(BG_COLOR)

ax.xaxis.pane.set_edgecolor(GRID_COLOR)
ax.yaxis.pane.set_edgecolor(GRID_COLOR)
ax.zaxis.pane.set_edgecolor(GRID_COLOR)

colors_data = ['#98C1D9', '#E8D8C4', '#D0A5C0']
colors_boundary = ['blue', 'yellow', 'red']
labels = iris.target_names

markers = ['o', 's', 'D']

# 绘制原始数据点
for i, c in enumerate(colors_data):
    ax.scatter(x[y == i, 0], x[y == i, 1], x[y == i, 2],
               c=c,
               label=f'Data: {labels[i]}',
               s=50,
               edgecolor=SPINE_COLOR,
               linewidth=0.5,
               alpha=0.9,
               marker=markers[i])

# 绘制决策边界
if len(boundary_points) > 0:
    # 创建边界点的颜色列表，使用 colors_boundary
    boundary_colors_list = [colors_boundary[p] for p in boundary_predictions]

    # 实际边界点云的绘制
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
               c=boundary_colors_list,
               alpha=BOUNDARIES_ALPHA,
               s=5,
               marker='.') # 边界点继续使用小点

    # 绘制一个额外的点作为边界线的图例占位符
    ax.scatter([], [], [],
               c='gray',
               alpha=0.5,
               s=50,
               marker='o',
               label=f'Uncertainty Region ($\Delta P < {threshold}$)')

# 设置轴标签和标题风格
ax.set_xlabel('Sepal Width', color=TEXT_COLOR, fontsize=12)
ax.set_ylabel('Petal Length', color=TEXT_COLOR, fontsize=12)
ax.set_zlabel('Petal Width', color=TEXT_COLOR, fontsize=12)

# 调整刻度颜色
ax.tick_params(axis='x', colors=TEXT_COLOR)
ax.tick_params(axis='y', colors=TEXT_COLOR)
ax.tick_params(axis='z', colors=TEXT_COLOR)

# 设置标题风格
ax.set_title(
    '3D GPC Decision Uncertainty Regions',
    color=TEXT_COLOR,
    fontsize=16,
    fontweight='bold',
    pad=15
)

# 调整图例风格
ax.legend(
    loc='upper right',
    fontsize=10,
    facecolor=BG_COLOR,  # 图例背景色
    edgecolor=SPINE_COLOR,  # 图例边框色
    labelcolor=TEXT_COLOR
)

ax.view_init(elev=20, azim=130)  # 设置初始视角

# 整体背景色
fig.patch.set_facecolor(BG_COLOR)
plt.tight_layout()
plt.show()