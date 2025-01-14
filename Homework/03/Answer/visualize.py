import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# 设置字体以支持中文
font_path = "/Users/emptyblue/Documents/GitHub/gpt_academic/gpt_log/arxiv_cache/ttf_files/simsun.ttf"  # 替换为支持中文的字体路径
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 读取 CSV 数据, 来自 result.csv
data = pd.read_csv("result.csv", header=None).values

# 转换为 numpy 数组
data_array = np.array(data, dtype=np.int32)
# 添加标题
plt.title("PE+PI 结果", fontproperties=font_prop)
# 添加标签
plt.xlabel("第二个位置的汽车数量", fontproperties=font_prop)
plt.ylabel("第一个位置的汽车数量", fontproperties=font_prop)
# 创建颜色图
plt.imshow(data_array, cmap="viridis", aspect="auto", origin="lower")

# 添加颜色条
plt.colorbar()

# 保存图像
plt.savefig("policy_iteration.png")
