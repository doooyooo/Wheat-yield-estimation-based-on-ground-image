import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from xgboost import XGBRegressor
from sklearn.svm import SVR
import torchvision.transforms as transforms
from data_loader import split_multi_view_data_by_interval, load_image_data,load_sigle_spike_image_data
from model.deepfea import extract_features_from_tensor,extract_and_average_features
from evaluator import plot_scatter,evaluate_multiple_models
from gridsea import tune_random_forest, tune_random_forest_fast, tune_xgboost, tune_xgboost_fast,tune_svr,tune_svr_fast
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def save_features_to_excel(train_features, test_features, train_ids, test_ids, filename="features.xlsx"):
    df_train = pd.DataFrame(train_features)
    df_train.insert(0, 'id', train_ids)
    df_train.insert(1, 'dataset', 'train')
    df_test = pd.DataFrame(test_features)
    df_test.insert(0, 'id', test_ids)
    df_test.insert(1, 'dataset', 'test')
    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    df_all.to_excel(filename, index=False)
    print(f"特征已保存到 {filename}")

def main(force_extract=False):
    spike_size = 1024
    # model_name = 'vit'
    # model_name = 'efficientnetb4'
    model_name = 'vgg'
    need_sea = 0
    force_extract = 0 #是否重新进行特征提取

    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据路径
    data_path = "总记录表格_processed.xlsx"
    spike_folder = "hhh_photos/spike_resize_1024_add"

    # 加载数据
    print("正在加载数据...")
    data_original = pd.read_excel(data_path,sheet_name="Sheet1")
    yield_column = data_original.columns[12]  
    data_original = data_original.dropna(subset=[yield_column])

    # 提取原始特征和图像路径
    id_column = data_original.iloc[:, 2]
    all_ids = id_column.values

    spatial_features = data_original.iloc[:, 18:27]
    wavelet_features = data_original.iloc[:, 27:35] 
    color_index_features = data_original.iloc[:, 35:47] 
    ndvi_features = data_original.iloc[:, 47:72]
    temp_features = data_original.iloc[:, 72:96]
    precip_features = data_original.iloc[:, 96:]
    original_features = np.hstack([spatial_features,wavelet_features,color_index_features,temp_features, precip_features])
    #特征优选结果：['spike_count', 'avg_spike_area', 'total_area', 'avg_perimeter', 'spatial_density', 'nni', 'spatial_entropy', 'morans_i', 'cA_mean', 'cA_std', 'cH_std', 'cV_std', 'cD_std', 'ExG_mean', 'DYGI_mean', 'GLI_mean', 'GLI_var', 'VEG_mean', 'VEG_var', 'TGI_mean', 'TGI_var', '2024_10_23_NDVI', '2024_12_22_NDVI', '2025_04_06_NDVI', '2025_04_21_NDVI', '2025_05_21_NDVI', 'Max_NDVI', 'Mean_NDVI', 'Std_NDVI', 'Rising_Slope', 'Amplitude', 'Mean_Temp', 'Nov_2024_Mean_Temp', 'Feb_2025_Min_Temp', 'Total_GDD_Base5', 'Dec_2024_Precip', 'Jun_2025_Precip', 'Light_Rain_Days', 'Avg_Drought_Length', 'Precip_CV', 'Precip_Concentration_Index']
    #根据特征优选结果选择指定特征
    # selected_features = [
    #     'spike_count', 'avg_spike_area', 'total_area', 'avg_perimeter', 'spatial_density',
    #     'nni', 'spatial_entropy', 'morans_i', 'cA_mean', 'cA_std', 'cH_std', 'cV_std',
    #     'cD_std', 'ExG_mean', 'DYGI_mean', 'GLI_mean', 'GLI_var', 'VEG_mean', 'VEG_var',
    #     'TGI_mean', 'TGI_var', '2024_10_23_NDVI', '2024_12_22_NDVI', '2025_04_06_NDVI',
    #     '2025_04_21_NDVI', '2025_05_21_NDVI', 'Max_NDVI', 'Mean_NDVI', 'Std_NDVI',
    #     'Rising_Slope', 'Amplitude', 'Mean_Temp', 'Nov_2024_Mean_Temp', 'Feb_2025_Min_Temp',
    #     'Total_GDD_Base5', 'Dec_2024_Precip', 'Jun_2025_Precip', 'Light_Rain_Days',
    #     'Avg_Drought_Length', 'Precip_CV', 'Precip_Concentration_Index'
    # ]
    # original_features = data_original[selected_features].values
    print("原始特征形状:", original_features.shape)
    yield_label = data_original.iloc[:, 12].values.reshape(-1, 1)  # 产量标签
    # 计算产量均值
    yield_mean = yield_label.mean()
    print(f"产量均值: {yield_mean}")
    spike_transform = transforms.Compose([
        transforms.Resize((spike_size, spike_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 转换为PyTorch张量
    original_features = torch.tensor(original_features, dtype=torch.float32)
    yield_label = torch.tensor(yield_label, dtype=torch.float32)
    split_ratio = [0.8, 0, 0.2]
    train_indices, valid_indices, test_indices = split_multi_view_data_by_interval(
            yield_label, num_intervals=num_intervals, split_ratio=split_ratio, visualize=True)
    train_ids = all_ids[train_indices]
    test_ids = all_ids[test_indices]
    # 加载特征数据
    train_original = original_features[train_indices]
    test_original = original_features[test_indices]
    train_yield = yield_label[train_indices]
    test_yield = yield_label[test_indices]
    
    scaler_original = MinMaxScaler()
    scaler_yield = MinMaxScaler()

    train_original_norm = scaler_original.fit_transform(train_original.numpy())
    test_original_norm = scaler_original.transform(test_original.numpy())

    train_yield_norm = scaler_yield.fit_transform(train_yield.numpy())
    test_yield_norm = scaler_yield.transform(test_yield.numpy())

    train_original_norm = torch.tensor(train_original_norm, dtype=torch.float32)
    test_original_norm = torch.tensor(test_original_norm, dtype=torch.float32)

    train_yield_norm = torch.tensor(train_yield_norm, dtype=torch.float32)
    test_yield_norm = torch.tensor(test_yield_norm, dtype=torch.float32)

    # 训练CNN模型并提取特征
    print("正在使用模型提取特征...")
    feature_dir = "saved_features"
    os.makedirs(feature_dir, exist_ok=True)
    all_spike_file = f"{feature_dir}/all_spike_{model_name}.npy"
    if not force_extract and os.path.exists(all_spike_file):
        all_spike_features = np.load(all_spike_file)
    else:
        all_spike = load_image_data(all_ids, spike_folder, spike_transform, suffix="_spike_1.jpg")
        # 提取所有特征
        all_spike_features = extract_features_from_tensor(
            all_spike, model_name, f"model/train_seg_{spike_size}_model/{model_name}_best_spike_paper3.pth")
        # 保存
        np.save(all_spike_file, all_spike_features)
        print(f"所有特征已保存到 {all_spike_file}")

    # 根据 train/test 划分
    train_spike_features = all_spike_features[train_indices]
    test_spike_features = all_spike_features[test_indices]

    print("训练特征形状:", train_spike_features.shape)
    print("测试特征形状:", test_spike_features.shape)

    train_image_features = np.hstack([train_spike_features])
    test_image_features = np.hstack([test_spike_features])

    feature_sets = {
        f"{model_name}图像特征": (train_image_features, test_image_features),
        f"{model_name}图像特征+原始特征": (np.hstack([train_image_features, train_original_norm.numpy()]),
                                           np.hstack([test_image_features, test_original_norm.numpy()])),
        "原始特征": (train_original_norm.numpy(), test_original_norm.numpy())
    }
    if need_sea:
        # 用最优参数替换默认模型
        print("正在进行参数搜索以减少过拟合...")
        tuned_rf = tune_random_forest_fast(np.hstack([train_image_features, train_original_norm.numpy()]), train_yield_norm.numpy().ravel())
        tuned_xgb = tune_xgboost_fast(np.hstack([train_image_features, train_original_norm.numpy()]), train_yield_norm.numpy().ravel())
        models = {
            "RandomForest": tuned_rf,
            "SVM": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "XGBoost": tuned_xgb
        }
    else:
        models = {
            "RandomForest": RandomForestRegressor(max_depth=10, max_features='log2', min_samples_leaf= 4, min_samples_split=5, n_estimators=200, random_state=seed),
            "SVM": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5),
            "XGBoost": XGBRegressor(subsample= 0.85, n_estimators=100, min_child_weight=5, max_depth=3, learning_rate=0.05, colsample_bytree=0.7),
            # "LightGBM":LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
            #                   subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1),

        }

    all_results, trained_models = evaluate_multiple_models(
        models,
        feature_sets,
        train_yield_norm.numpy(),
        test_yield_norm.numpy(),
        scaler_yield
    )

    train_yield_original = scaler_yield.inverse_transform(train_yield_norm.numpy())
    test_yield_original = scaler_yield.inverse_transform(test_yield_norm.numpy())

    for model_name1, results in all_results.items():
        for feature_set, result in results.items():
            plot_scatter(
                test_yield_original,
                result['test_pred'],
                f"{model_name1}模型-{feature_set}"
            )

    plt.figure(figsize=(14, 8))
    bar_width = 0.15
    index = np.arange(len(feature_sets))

    # 为每个模型准备数据
    rrmse_data = {model: [] for model in models}
    for model_name1 in models:
        for feature_set in feature_sets:
            rrmse_data[model_name1].append(all_results[model_name1][feature_set]['test_rrmse'])

    colors = ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099']
    for i, (model_name1, rrmse_values) in enumerate(rrmse_data.items()):
        plt.bar(index + i * bar_width, rrmse_values, bar_width, label=model_name1, color=colors[i])

    plt.xlabel('特征组合')
    plt.ylabel('rRMSE (%)')
    plt.title('不同模型和特征组合的测试集rRMSE对比')
    plt.xticks(index + bar_width * 2, list(feature_sets.keys()))
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    print("正在保存结果到CSV...")
    results_list = []

    for model_name1, feature_results in all_results.items():
        for feature_set, result in feature_results.items():

            for i in range(len(train_yield_original)):
                results_list.append({
                    'id': train_ids[i],  # 新增ID字段
                    'dataset': 'train',
                    'model': model_name1,
                    'feature_set': feature_set,
                    'true_yield': train_yield_original[i][0],
                    'predicted_yield': result['train_pred'][i][0]
                })

            for i in range(len(test_yield_original)):
                results_list.append({
                    'id': test_ids[i],  # 新增ID字段
                    'dataset': 'test',
                    'model': model_name1,
                    'feature_set': feature_set,
                    'true_yield': test_yield_original[i][0],
                    'predicted_yield': result['test_pred'][i][0]
                })

    results_df = pd.DataFrame(results_list)
    csv_filename = 'all_model_predictions_only_temp_pre.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"所有模型预测结果已保存到 {csv_filename}")
    print('所有模型评估完成！')


if __name__ == "__main__":
    main(force_extract=False)