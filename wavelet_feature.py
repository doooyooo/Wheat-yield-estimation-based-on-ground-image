import os
import random

import pandas as pd
import numpy as np
import cv2
import pywt
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据路径
data_path = "总记录表格_processed.xlsx"
spike_folder = "hhh_photos/spike_resize"
mask_folder = "hhh_photos/spike_predict/segmented"  # 你的掩膜文件夹路径

# 读取数据
print("正在加载数据...")
data_original = pd.read_excel(data_path, sheet_name="Sheet5")

yield_column = data_original.columns[12]

all_ids = data_original.iloc[:, 2].values
yield_label = data_original.iloc[:, 12].values

def extract_wavelet_features(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        coeffs2 = pywt.dwt2(img, 'haar')
        cA, (cH, cV, cD) = coeffs2
        features = [
            np.mean(cA), np.std(cA),
            np.mean(cH), np.std(cH),
            np.mean(cV), np.std(cV),
            np.mean(cD), np.std(cD)
        ]
        return features
    except Exception as e:
        print(f"读取 {img_path} 出错: {e}")
        return None


def extract_wavelet_features_masked(img_path, mask_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            print(f"图像或掩膜读取失败: {img_path}, {mask_path}")
            return None
        _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        masked_img = cv2.bitwise_and(img, img, mask=binary_mask)

        coeffs2 = pywt.dwt2(masked_img, 'haar')
        cA, (cH, cV, cD) = coeffs2

        features = [
            np.mean(cA), np.std(cA),
            np.mean(cH), np.std(cH),
            np.mean(cV), np.std(cV),
            np.mean(cD), np.std(cD)
        ] 
        all_features = [
            cA.flatten(),
            cH.flatten(),
            cV.flatten(),
            cD.flatten()
        ]
        return features,all_features
    except Exception as e:
        print(f"处理出错: {e}")
        return None
results = []
cA_list, cH_list, cV_list, cD_list, yields_list = [], [], [], [], []
for idx, sample_id in enumerate(all_ids):
    matched_files = [f for f in os.listdir(spike_folder) if str(sample_id) in f]
    matched_masks = [f for f in os.listdir(mask_folder) if str(sample_id) in f]
    if matched_files and matched_masks:
        img_path = os.path.join(spike_folder, matched_files[0])
        mask_path = os.path.join(mask_folder, matched_masks[0])
        wavelet_features,all_features = extract_wavelet_features_masked(img_path, mask_path)
        # wavelet_features = extract_wavelet_features(img_path)
        if wavelet_features:
            results.append([sample_id, yield_label[idx]] + wavelet_features)
            cA_list.append(all_features[0])
            cH_list.append(all_features[1])
            cV_list.append(all_features[2])
            cD_list.append(all_features[3])
            yields_list.append(yield_label[idx])
    else:
        print(f"ID {sample_id} 找不到匹配图片或掩膜")
yields_list = np.array(yields_list)
components = {
    "cA": np.array(cA_list),
    "cH": np.array(cH_list),
    "cV": np.array(cV_list),
    "cD": np.array(cD_list)
}

columns = ["ID", "Yield",
           "cA_mean", "cA_std",
           "cH_mean", "cH_std",
           "cV_mean", "cV_std",
           "cD_mean", "cD_std"]
df_features = pd.DataFrame(results, columns=columns)

output_path = "小波特征与产量.xlsx"
df_features.to_excel(output_path, index=False)
print(f"已保存到 {output_path}")


plt.figure(figsize=(8, 6))
plt.scatter(df_features["cA_mean"], df_features["Yield"], alpha=0.7)
plt.xlabel("Wavelet Feature (cA_mean)")
plt.ylabel("Yield")
plt.title("Yield vs Wavelet Feature (cA_mean)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Yield_vs_WaveletFeature.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(df_features["cD_std"], df_features["Yield"], alpha=0.7)
plt.xlabel("Wavelet Feature (cD_std)")
plt.ylabel("Yield")
plt.title("Yield vs Wavelet Feature (cD_std)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Yield_vs_WaveletFeature.png", dpi=300)
plt.show()


corr_matrix = df_features.drop(columns=["ID"]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation between Yield and Wavelet Features")
plt.tight_layout()
plt.savefig("WaveletFeature_CorrelationHeatmap.png", dpi=300)
plt.show()

