import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# 设置中文字体（Windows系统，SimHei字体支持中文）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 从 CSV 文件中读取数据，假设 12-3.csv 在当前目录下
kg_df = pd.read_csv('12-8 data.csv')

# 查看数据的前几行，确保数据已正确加载
print(kg_df.head())

# 从 pandas 数据框创建一个多重有向图
G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())

# 设置绘图大小
plt.figure(figsize=(12,12))

# 使用 spring_layout 函数布置节点位置
pos = nx.spring_layout(G)

# 绘制多重有向图
nx.draw(G, pos=pos, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, node_size=500, font_size=10, font_color='black', edge_color='gray')

# 显示绘图
plt.show()