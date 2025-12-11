from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = load_iris()
x_raw1 = iris.data          # 所有 4 个特征
y_raw = iris.target        # 0, 1, 2 三类
x_raw2=x_raw1[:,1:]

# 去掉第versicolor类
mask=y_raw!=1
x=x_raw2[mask]
y=y_raw[mask]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)
clf = LogisticRegression(C=100)
clf.fit(x_train, y_train)
# 评估正确性
print("Train accuracy:", clf.score(x_train, y_train))
print("Test accuracy:", clf.score(x_test, y_test))

# 获取系数 方便后面进行作图
w1, w2, w3 = clf.coef_[0]
b = clf.intercept_[0]

# 形成概率网格
grid_x = np.linspace(x[:,0].min(), x[:,0].max(), 30)
grid_y = np.linspace(x[:,1].min(), x[:,1].max(), 30)
grid_z = np.linspace(x[:,2].min(), x[:,2].max(), 30)
xx, yy, zz = np.meshgrid(grid_x, grid_y, grid_z)
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
probas = clf.predict_proba(grid_points)[:, 1]   # 类别virginica的概率 因为是二分法 保留一个即可
probas = probas.reshape(xx.shape)

# 依旧preview的风格
plt.style.use("seaborn-v0_8-white")
colors = ["#98C1D9", "#E8D8C4", "#D0A5C0"]
fig = plt.figure(figsize=(10, 8), facecolor='#F5F3EE')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#F5F3EE')
# 把setosa和virginica的散点图画出来
ax.scatter(
    x[y==0,0], x[y==0,1], x[y==0,2],
    c=colors[0], s=40, alpha=0.9, edgecolor="#444444", linewidth=0.5,marker='o',
    label="setosa"
)
ax.scatter(
    x[y==2,0], x[y==2,1], x[y==2,2],
    c=colors[1], s=40, alpha=0.9, edgecolor="#444444", linewidth=0.5,marker='D',
    label="virginica"
)

t_grid_x = np.linspace(x[:,0].min(), x[:,0].max(), 30)
t_grid_y = np.linspace(x[:,1].min(), x[:,1].max(), 30)
t_xx, t_yy = np.meshgrid(grid_x, grid_y)  # shape = (30,30)
bz=-(w1 * t_xx + w2 * t_yy + b) / w3 # 根据曲面方程反解bz
ax.plot_surface(t_xx, t_yy, bz, color='lightgrey', shade=False)

# 坐标轴风格
ax.set_xlabel("sepal length", fontsize=12, color='#333333', labelpad=10)
ax.set_ylabel("petal length", fontsize=12, color='#333333', labelpad=10)
ax.set_zlabel("petal width", fontsize=12, color='#333333', labelpad=10)

# 坐标轴颜色
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.label.set_color('#333333')
    axis.set_tick_params(colors='#333333')

# 图例
ax.legend(
    frameon=True,
    facecolor="#F5F3EE",
    edgecolor="#BBBBBB",
    fontsize=10
)

# 标题
ax.set_title(
    "Visualise 3D Boundary",
    fontsize=15, fontweight='bold', color='#333333', pad=20
)

plt.show()