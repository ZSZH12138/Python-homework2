from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# 数据加载、预处理、模型训练
iris = load_iris()
x_raw1 = iris.data
y_raw = iris.target
x_raw2 = x_raw1[:, 1:]
mask = y_raw != 1
x = x_raw2[mask]
y = y_raw[mask]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
clf_gp = GaussianProcessClassifier(kernel=1.0 * RBF([1.0, 1.0, 1.0]), random_state=42)
clf_gp.fit(x_train, y_train)

# 创建网格
x1_min, x1_max =[x[:, i].min() - 0.5 for i in range(3)], [x[:, i].max() + 0.5 for i in range(3)]
x2_min, x2_max =[x[:, i].min() - 0.5 for i in [1,2,0]], [x[:, i].max() + 0.5 for i in [1,2,0]]
h = 0.05
xx1,xx2=[None,None,None],[None,None,None]
for i in range(3):
    xx1[i], xx2[i]= np.meshgrid(np.arange(x1_min[i], x1_max[i], h),
                                 np.arange(x2_min[i], x2_max[i], h))

# 由于z值用于展示Logit 所以只能其中一个特征切片法取定值
# 本处切三片 小中大
x3_vals=[[]for _ in range(3)]
for i,j in enumerate([2,0,1]):
    x3_vals[i] = [
        np.percentile(x[:,j], 25),
        np.mean(x[:, j]),
        np.percentile(x[:, j], 75) 
    ]
titles = ["Low slice", "Mean slice", "High slice"]
axenames=["Spenal Width","Petal Length","Petal Width"]
titles_las=[" of Spenal Width"," of Petal Length"," of Petal Width"]

# 计算Logit作为z值 并计算z轴范围
Z_list = [[]for _ in range(3)]
Z_lim=[]
for i,x3_vals_line in enumerate(x3_vals):
    for x3_fixed in x3_vals_line:
        X_grid = np.c_[xx1[i].ravel(), xx2[i].ravel(), np.full(xx1[i].size, x3_fixed)]
        P = clf_gp.predict_proba(X_grid)[:, 1]
        P = np.clip(P, 1e-10, 1 - 1e-10)
        Z_logit = np.log(P / (1 - P)).reshape(xx1[i].shape)
        Z_list[i].append(Z_logit)
    Z_min = min(z.min() for z in Z_list[i])
    Z_max = max(z.max() for z in Z_list[i])
    Z_range = max(abs(Z_min), abs(Z_max))
    Z_lim.append( (-Z_range, Z_range))
# 本处风格依旧与preview一致
BG_COLOR = '#F5F3EE'
TEXT_COLOR = '#333333'
SPINE_COLOR = '#444444'
WIREFRAME_COLOR = '#888888'
CONTOUR_LINE_COLOR = '#444444'
colors_for_cmap = ["#98C1D9", "#E8D8C4", "#D0A5C0"]
# 自定义颜色映射
custom_RdBu_cmap = LinearSegmentedColormap.from_list(
    "custom_RdBu", ["#98C1D9", "#E8D8C4", "#D0A5C0"]
)

# 绘图
# 设置全局背景色
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.style.use('default')
fig = plt.figure(figsize=(18, 16))

# 添加总标题
fig.suptitle(
    "Visualise 3D Probability Map",
    fontsize=18,
    fontweight='bold',
    color=TEXT_COLOR,
    y=0.985
)

cf_handle = None
idx=0
for sub in range(3):
    Z_proj = Z_lim[sub][0]
    Y_proj = x2_max[sub]
    for Z in Z_list[sub]:
        ax = fig.add_subplot(3, 3, idx + 1, projection='3d')
        # 子图背景色
        ax.set_facecolor(BG_COLOR)
        # 曲面网格
        ax.plot_wireframe(xx1[sub], xx2[sub], Z, color=WIREFRAME_COLOR, linewidth=0.5)

        # xOy投影
        # 捕获 contourf 结果，以便生成 colorbar
        cf_current = ax.contourf(xx1[sub], xx2[sub], Z,
                                 zdir='z', offset=Z_proj,
                                 cmap=custom_RdBu_cmap, alpha=0.6)
        # 只需捕获一次，用于生成 colorbar
        if cf_handle is None:
            cf_handle = cf_current

        # yOz投影
        yy_proj = np.linspace(x2_min[sub], x2_max[sub], Z.shape[0])
        zz_proj = np.linspace(Z_lim[sub][0], Z_lim[sub][1], Z.shape[0])
        YY, ZZ = np.meshgrid(yy_proj, zz_proj)
        Z_proj_left = np.interp(ZZ, (Z.min(), Z.max()), (Z.min(), Z.max()))
        ax.contourf(YY, ZZ, Z_proj_left,
                    zdir='x', offset=x1_max[sub],
                    cmap=custom_RdBu_cmap, alpha=0.6)

        # zOx投影
        xx_proj = np.linspace(x1_min[sub], x1_max[sub], Z.shape[0])
        XX, ZZ2 = np.meshgrid(xx_proj, zz_proj)
        Z_proj_right = np.interp(ZZ2, (Z.min(), Z.max()), (Z.min(), Z.max()))
        ax.contourf(XX, ZZ2, Z_proj_right,
                    zdir='y', offset=Y_proj,
                    cmap=custom_RdBu_cmap, alpha=0.6)

        # 轴限设置
        ax.set_xlim(x1_min[sub], x1_max[sub])
        ax.set_ylim(x2_min[sub], x2_max[sub])
        ax.set_zlim(Z_lim[sub])

        ax.set_xlabel(axenames[sub], fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel(axenames[sub+1] if sub<2 else axenames[0], fontsize=9, color=TEXT_COLOR)
        ax.set_zlabel("Logit", fontsize=9, color=TEXT_COLOR)
        if sub==0:
            ax.set_title(
                titles[idx%3]+titles_las[sub-1] if sub>0 else titles[idx%3]+titles_las[2],
                fontsize=11, fontweight='bold', color=TEXT_COLOR, pad=0
            )
        ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=7)
        ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=7)
        ax.tick_params(axis='z', colors=TEXT_COLOR, labelsize=7)

        # 子图背景色和网格
        ax.set_facecolor(BG_COLOR)
        ax.grid(axis='both', linestyle='--', alpha=0.3, color='#BBBBBB')

        # 轴颜色
        ax.xaxis.line.set_color(SPINE_COLOR)
        ax.yaxis.line.set_color(SPINE_COLOR)
        ax.zaxis.line.set_color(SPINE_COLOR)

        # 调整视图角度
        ax.view_init(elev=20, azim=-120)
        idx+=1

# 整体背景色
fig.patch.set_facecolor(BG_COLOR)

# 添加颜色条并排版，避免重叠
if cf_handle:
    # 颜色条的 Axes 位置：[左, 下, 宽, 高]，将其放置在最右边
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    # 使用 cax 参数将 colorbar 放置到新的 Axes 中
    cbar = fig.colorbar(cf_handle, cax=cbar_ax)
    cbar.set_label('Logit Value\n(Red:Virginica)\n(Blue:Setosa)', color=TEXT_COLOR, fontsize=10)
    cbar.ax.tick_params(colors=TEXT_COLOR)
    cbar.ax.set_facecolor(BG_COLOR)
    cbar.ax.yaxis.set_tick_params(direction='inout')
    # 调整颜色条的边框颜色
    cbar.outline.set_edgecolor(SPINE_COLOR)
    # 设置颜色条 Axes 的背景色
    cbar_ax.set_facecolor(BG_COLOR)

# 修正图例位置：将其放置到右上方
# 创建图例所需的色块
legend_patches = [
    # Setosa 是 Logit 负端（蓝色），Virginica 是 Logit 正端（粉色）
    Patch(facecolor=colors_for_cmap[0], edgecolor=SPINE_COLOR, alpha=0.8, label=r'Tendency to Setosa'),
    Patch(facecolor=colors_for_cmap[2], edgecolor=SPINE_COLOR, alpha=0.8, label=r'Tendency to Virginica')
]

plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.98]) # 调整布局矩形，给顶部和颜色条留出空间
plt.show()
