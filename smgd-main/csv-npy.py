import pandas as pd
import numpy as np

# 先用pandas读入csv
# data = pd.read_csv("val_rs.csv")
# # 再使用numpy保存为npy
# np.save("val_rs.npy", data)

test1 = np.load('imagenet_class_to_idx.npy',allow_pickle=True)

test2 = np.load('val_rs.npy',allow_pickle=True)

print(test1)
print(test2)

