from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.datasets import load_iris
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage import measure
from sklearn.metrics import confusion_matrix

# 风格与preview保持一致
PLOT_COLORS = ['#98C1D9', '#E8D8C4', '#D0A5C0']
PLOT_LABELS = ['setosa', 'versicolor', 'virginica']
MARKERS = ['o', 's', 'D'] # 为散点图添加标记样式

def plot_3d_decision_surface(gpc, x, y, labels, colors, resolution=60, threshold=0.4):
    """
    使用 Marching Cubes 算法绘制的 3D 决策曲面
    """
    plt.style.use("seaborn-v0_8-white") 

    x1_min, x1_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    x2_min, x2_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    x3_min, x3_max = x[:, 2].min() - 0.5, x[:, 2].max() + 0.5

    # 构建网格
    xx1, xx2, xx3 = np.meshgrid(
        np.linspace(x1_min, x1_max, resolution),
        np.linspace(x2_min, x2_max, resolution),
        np.linspace(x3_min, x3_max, resolution),
        indexing='ij'
    )
    grid_points = np.c_[xx1.ravel(), xx2.ravel(), xx3.ravel()]
    probs = gpc.predict_proba(grid_points)
    probs_vol = probs.reshape(xx1.shape + (3,))

    fig = plt.figure(figsize=(12, 10))
    # 设置背景色
    fig.patch.set_facecolor('#F5F3EE') 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F5F3EE') 

    # 绘制每个类别的等值面
    for i in range(3):
        vol_data = probs_vol[:, :, :, i]
        verts, faces, _, _ = measure.marching_cubes(vol_data, level=threshold)

        # 坐标映射回实际特征空间
        verts_mapped = np.zeros_like(verts)
        verts_mapped[:, 0] = x1_min + verts[:, 0] * (x1_max - x1_min) / (resolution - 1)
        verts_mapped[:, 1] = x2_min + verts[:, 1] * (x2_max - x2_min) / (resolution - 1)
        verts_mapped[:, 2] = x3_min + verts[:, 2] * (x3_max - x3_min) / (resolution - 1)
        
        # 决策面使用更柔和的颜色和透明度
        mesh = Poly3DCollection(verts_mapped[faces], alpha=0.15, label=f'{labels[i]} Boundary')
        mesh.set_facecolor(colors[i])
        mesh.set_edgecolor('none')
        ax.add_collection3d(mesh)

    # 绘制原始数据点
    for i, c in enumerate(colors):
        ax.scatter(x[y==i, 0], x[y==i, 1], x[y==i, 2],
                   c=c, 
                   label=f'Data {labels[i]}', 
                   s=30, # 更小的点
                   edgecolor="#444444", # 深灰色边缘
                   linewidth=0.4,
                   alpha=0.8, # 略微透明
                   marker=MARKERS[i], # 不同标记
                   zorder=10)

    # 轴标签和标题
    ax.set_xlabel('Sepal Width', color='#333333'); ax.set_ylabel('Petal Length', color='#333333'); ax.set_zlabel('Petal Width', color='#333333')
    ax.set_title('3D Boundary for mission4', fontsize=14, fontweight='bold', color='#333333', pad=10)
    ax.set_xlim(x1_min, x1_max); ax.set_ylim(x2_min, x2_max); ax.set_zlim(x3_min, x3_max)
    ax.view_init(elev=25, azim=135)
    
    # 隐藏网格线和边框
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#BBBBBB')
    ax.yaxis.pane.set_edgecolor('#BBBBBB')
    ax.zaxis.pane.set_edgecolor('#BBBBBB')


    handles, lbs = ax.get_legend_handles_labels()
    by_label = dict(zip(lbs, handles)); ax.legend(by_label.values(), by_label.keys(), loc='best', facecolor='#F5F3EE', edgecolor='#BBBBBB')
    plt.show()

def plot_2d_slice(model, x_data, y_data, slice_value, colors, labels, tolerance=0.25):
    """
    绘制3D模型的2D切片图
    """
    # 应用新风格
    plt.style.use("seaborn-v0_8-white") 

    feature_index_to_slice = 2 # Petal Width
    x_min, x_max = x_data[:, 0].min() - 0.5, x_data[:, 0].max() + 0.5
    y_min, y_max = x_data[:, 1].min() - 0.5, x_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_2d = np.c_[xx.ravel(), yy.ravel(), np.full(xx.ravel().shape, slice_value)]

    # 预测模式逻辑 (修正Setosa支配性)
    if slice_value < 0.8:
        Z = model.predict(grid_2d)
        target_classes_for_plot = [0, 1, 2]
    else:
        probs = model.predict_proba(grid_2d)
        Z_arg = np.argmax(probs[:, 1:3], axis=1) # 1 vs 2
        Z = Z_arg + 1
        target_classes_for_plot = [1, 2]
    Z = Z.reshape(xx.shape)

    # 筛选数据点 (用于 scatter 和 CM)
    mask = np.abs(x_data[:, feature_index_to_slice] - slice_value) < tolerance
    x_slice, y_slice = x_data[mask], y_data[mask]

    # 混淆矩阵和准确率 (用于验证)
    y_pred_slice = model.predict(x_slice)
    cm_slice = confusion_matrix(y_slice, y_pred_slice)
    acc = np.sum(np.diag(cm_slice)) / np.sum(cm_slice)
    print(f"验证 Petal Width = {slice_value:.2f} (Acc: {acc:.2f})")
    print(f"模式: {'3 类' if slice_value < 0.8 else '1 vs 2'}\nCM:\n{cm_slice}")

    # 绘图逻辑 (精确颜色映射)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#F5F3EE')
    ax.set_facecolor('#F5F3EE')

    colors_for_cmap = ['#98C1D9', '#E8D8C4', '#D0A5C0'] 

    if 0 in target_classes_for_plot and 1 in target_classes_for_plot: # 3类模式
        cmap_plot = ListedColormap(colors_for_cmap)
        norm_plot = BoundaryNorm([0, 1, 2, 3], cmap_plot.N)
    elif 1 in target_classes_for_plot: # 1 vs 2 模式 (只用绿和蓝)
        cmap_plot = ListedColormap([colors_for_cmap[1], colors_for_cmap[2]])
        norm_plot = BoundaryNorm([1, 2, 3], cmap_plot.N)
    else: # 只有 0 类
        cmap_plot = ListedColormap([colors_for_cmap[0]]); norm_plot = BoundaryNorm([0, 1], cmap_plot.N)

    # 填充等高线图，使用低透明度
    plt.contourf(xx, yy, Z, cmap=cmap_plot, norm=norm_plot, alpha=0.5)

    # 绘制数据点
    for i in range(3):
        idx = np.where(y_slice == i)
        plt.scatter(x_slice[idx, 0], x_slice[idx, 1], c=colors[i],
                    label=f'{labels[i]} (near slice)', 
                    edgecolor='#444444', 
                    linewidth=0.4,
                    alpha=0.8,
                    marker=MARKERS[i],
                    s=50)

    # 轴标签和标题
    plt.xlabel('Sepal Width', color='#333333'); plt.ylabel('Petal Length', color='#333333')
    plt.title(f'2D Validation Slice (Petal Width ≈ {slice_value:.2f})', 
              fontsize=14, fontweight='bold', color='#333333', pad=10)
    
    plt.legend(facecolor='#F5F3EE', edgecolor='#BBBBBB'); 
    # 使用虚线网格
    plt.grid(axis='y', linestyle='--', alpha=0.3, color='#BBBBBB')
    
    # 隐藏顶部和右部
    for spine in ['top','right']:
        ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    plt.show()

# 数据准备
iris = load_iris()
x_raw = iris.data
y = iris.target
x = x_raw[:, 1:] # 使用后三个特征 (Sepal Width, Petal Length, Petal Width)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# 模型训练
kernel = 1.0 * RBF([1.0, 1.0, 1.0])
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
gpc.fit(x_train, y_train)
print(f"Train accuracy: {gpc.score(x_train,y_train):.2f}")
print(f"Test accuracy: {gpc.score(x_test,y_test):.2f}")

# 绘制 3D 决策曲面
print("绘制 3D 决策曲面...")
plot_3d_decision_surface(gpc, x, y, PLOT_LABELS, PLOT_COLORS)

# 绘制 2D 切片验证图
print("绘制 2D 切片验证...")
# 检查 Setosa 区域 (低值)
plot_2d_slice(gpc, x, y, slice_value=0.3, colors=PLOT_COLORS, labels=PLOT_LABELS)
# 检查 Versicolor/Virginica 区域 (中高值)
plot_2d_slice(gpc, x, y, slice_value=1.5, colors=PLOT_COLORS, labels=PLOT_LABELS)