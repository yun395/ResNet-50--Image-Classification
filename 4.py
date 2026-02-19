import numpy as np

# 1. 指定你的.npy文件路径（替换为实际路径）
npy_path = "E:/shujuji/MIT67/feature_extractor/loc_224_npy/gameroom/sala_de_juegos_16_19_altavista.npy"

# 2. 读取.npy文件（核心操作）
data = np.load(npy_path)

# 3. 查看文件内容的关键信息
print("=== .npy文件基本信息 ===")
print(f"数据形状（shape）：{data.shape}")  # 特征维度，比如(1, 2048)对应2048维特征
print(f"数据类型（dtype）：{data.dtype}")  # 通常是float32
print(f"数据前5个值：{data[:5]}")  # 查看前5个特征值（按需调整）
print(f"完整数据：\n{data}")  # 打印全部数据（特征维度大时慎用）