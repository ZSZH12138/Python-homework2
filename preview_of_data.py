import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# 加载整理数据集
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target
df["species"] = df["species"].apply(lambda x: iris.target_names[x])

# 提取各类的特征值
setosa = df[df["species"] == "setosa"]
versicolor = df[df["species"] == "versicolor"]
virginica = df[df["species"] == "virginica"]

# 定义画布与颜色
plt.style.use("seaborn-v0_8-white")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ["#98C1D9", "#E8D8C4", "#D0A5C0"]

# 提取共有多少个特征 方便进行遍历画图
features = iris.feature_names
markers=['o','s','D']

for i, feature in enumerate(features):
    data = [setosa[feature], versicolor[feature], virginica[feature]] # 数据提取

    # 箱图
    bp = axes[i].boxplot(
        data,
        patch_artist=True,
        widths=0.15,
        boxprops=dict(linewidth=1.2, color='#666666'),
        medianprops = dict(color='#E63946', linewidth=2.2),
        whiskerprops = dict(color='#CCCCCC', linewidth=1.2, linestyle='-'),
        capprops = dict(color='#CCCCCC', linewidth=1.2),
        showfliers=False,
        zorder=1 # 位于底层
    )

    vp = axes[i].violinplot(
        data,
        positions=[1, 2, 3],
        widths=0.6,
        showmeans=False,
        showextrema=False,
        showmedians=False
    )

    # 设置小提琴颜色（匹配你三类花的 colors）
    for violin, color in zip(vp['bodies'], colors):
        violin.set_facecolor(color)  # 使用低饱和浅色
        violin.set_alpha(0.3)
        violin.set_edgecolor('#BBBBBB')
        violin.set_linewidth(0.6)

    for group_index, y_vals in enumerate(data):
        xpos = group_index + 1  # 箱图位置为 1,2,3

        # 生成y_vals个随机数
        jitter = (np.random.rand(len(y_vals)) - 0.5) * 0.25
        # 进行jitter
        x = xpos + jitter

        axes[i].scatter(
            x,
            y_vals,
            s=20,
            alpha=0.65,
            edgecolor="#444444",
            facecolor=colors[group_index],
            linewidth=0.4,
            zorder=10,
            marker=markers[group_index]
        )

    # 给box上色 zip让迭代可以进行 要不然两个自变量没法迭代两个因变量
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9) # 柔光

    axes[i].set_title(
        feature.replace(" (cm)", ""),
        fontsize=14, fontweight='bold', color='#333333', pad=10
    )

    axes[i].set_xticklabels(
        ["setosa", "versicolor", "virginica"],
        fontsize=11,
        color='#333333'
    )

    axes[i].set_ylabel(feature, fontsize=11, color='#333333')

    # 子图背景色
    axes[i].set_facecolor('#F5F3EE')
    # 设置虚线网格
    axes[i].grid(axis='y', linestyle='--', alpha=0.3, color='#BBBBBB')

    # 隐藏顶部和右部
    for spine in ['top','right']:
        axes[i].spines[spine].set_visible(False)
    axes[i].spines['left'].set_color('#444444')
    axes[i].spines['bottom'].set_color('#444444')

# 整体背景色
fig.patch.set_facecolor('#F5F3EE')
plt.tight_layout()
plt.show()