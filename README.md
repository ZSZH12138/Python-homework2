# Python-homework2
This is the second homework of my python class.  
You need to install numpy, matplotlib and sklearn for these code.  
The first file is preview_of_data.py which shows the based style of all the graphs.  
Then comes the homework.  
Mission 1使用了不同的分类器对两个特征进行三分类 并将所有的分类结果用一张画布展示  
Mission 2使用了众多分类器的其中一个（线性逻辑回归） 在验证分类正确率达标（均为100%）后 绘制决策平面  
Mission 3依旧使用了众多分类器中的一个（非线性逻辑回归） 在验证分类正确率达标（训练集97% 验证集100%）后 确定一个特征作为切片 分别取25 50 75百分位数 x y轴为另外两个特征 z轴为Logit 进行绘图 然后轮换特征作切片 共9图1画布  
Mission 4-1 使用了Mission 3中的分类器 对不同的类别分别绘制决策曲面 将这些点牢牢裹住 在曲面内部的即为可预测点 但是实验过程中发现 绘制出来的曲面电脑承受不太住 卡的不行 所以下面又多加了一个切片来验证曲面绘制是否正确 由于Setosa的数据点比较偏 所以切片分成两个画布展示 一部分展示处于低值的Setosa 另一部分展示中高值的另外两类  
Mission 4-2 由于Mission 4-1实在卡的不像话 所以对mission 4-1采取了另一种方法mission 4-2 取消使用曲面展示 使用点云代替平面 这样虽然牺牲了视觉上的效果 但至少不卡了  
Mission 4-3 使用了Mission 3中的分类器和方法 但是由于Mission 3是二分类 一个数轴的正负本身就代表了两个类 这里我采取了一个比较笨的方法 我把每个类的概率分别拆开绘制 也就是说 你看到的每一个Logit值是具体某一类的概率 高则很可能是该类 低则很可能不是该类 其他方法与Mission 3 采取一致 总共3*9=27个图 我分成了三个画布取画 每个画布固定一个特征值作切片  
注意1：由于使用preview风格在这里表现得不是很好（等高图太浅了） 所以Mission 4-3我使用了较为难看的深色风格 不过信息表达是完全没问题的  
注意2：Mission 4-3的侧面等高线图不知道是中了什么诅咒 怎么画怎么失败 但是事实上 侧面等高线图的重要性远不如底部等高线 所以这里我只绘制了底部的等高线
