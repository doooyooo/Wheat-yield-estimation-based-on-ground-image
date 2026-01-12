import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from data_loader import split_multi_view_data_by_interval

seed = 42
np.random.seed(seed)

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 2,
    'axes.unicode_minus': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

COLORS = {
    'dark_blue': '#0077B6',
    'accent_orange': '#F4A261',
    'text_dark': '#28282B',
    'group_spatial': '#1f77b4',      
    'group_wavelet': '#ff7f0e',      
    'group_color_index': '#2ca02c', 
}

def save_figure(fig, filename, dpi=300):
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{filename}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')


def run_feature_selection():
    # ------------------ Parameter Settings ------------------
    data_path = "总记录表格_processed.xlsx"
    sheet_name = "Sheet1"

    start_col_index = 18  # Feature start column
    yield_col_index = 12  # Yield column index

    print("Loading and preprocessing data...")
    data_original = pd.read_excel(data_path, sheet_name=sheet_name)

    # Drop rows with missing yield values
    data_original = data_original.dropna(subset=[data_original.columns[yield_col_index]])
    data_original = data_original[data_original.iloc[:, 0] != 0]
    data_original = data_original[data_original.iloc[:, 1] != 'XX']
    print(f"Processed data shape: {data_original.shape}")

    # Get all feature names
    feature_names = data_original.columns[start_col_index:].tolist()

    # Feature groups
    spatial_features = data_original.iloc[:, 18:27]  # S to AA
    wavelet_features = data_original.iloc[:, 27:35]  # AB to AI
    color_index_features = data_original.iloc[:, 35:47]  # Visible indices only

    # Combine features
    original_features = np.hstack(
        [spatial_features, wavelet_features, color_index_features]
    )
    print("Original feature shape:", original_features.shape)

    yield_label = data_original.iloc[:, yield_col_index].values.reshape(-1, 1)

    split_ratio = [0.8, 0, 0.2]
    yield_label1 = torch.tensor(yield_label, dtype=torch.float32)
    print("Splitting data...")
    train_indices, _, test_indices = split_multi_view_data_by_interval(
        yield_label1, num_intervals=3, split_ratio=split_ratio, visualize=True, seed=seed)

    X_train = original_features[train_indices]
    X_test = original_features[test_indices]
    y_train = yield_label[train_indices]
    y_test = yield_label[test_indices]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    y_train_norm = scaler_y.fit_transform(y_train)
    y_test_norm = scaler_y.transform(y_test)

    print(f"Train feature shape: {X_train_norm.shape}")
    print(f"Test feature shape: {X_test_norm.shape}")
    print("\nRunning RFECV for feature selection...")

    estimator = RandomForestRegressor(
        max_depth=10, max_features='log2',
        min_samples_leaf=4, min_samples_split=5,
        n_estimators=200, random_state=seed
    )

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    rfecv.fit(X_train_norm, y_train_norm.ravel())

    print("\nFeature selection completed!")
    print(f"Optimal number of features: {rfecv.n_features_}")

    fig, ax = plt.subplots(figsize=(8, 7))

    x_values = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
    scores = rfecv.cv_results_['mean_test_score']

    ax.plot(x_values, scores, color=COLORS['dark_blue'], linewidth=2, marker='o', markersize=4)

    optimal_idx = np.argmax(scores)
    ax.plot(optimal_idx + 1, scores[optimal_idx], 'o', color=COLORS['accent_orange'], markersize=8)

    ax.axvline(x=optimal_idx + 1, color=COLORS['text_dark'], linestyle='--', linewidth=1)
    ax.text(optimal_idx + 1, scores[optimal_idx], f'  {optimal_idx + 1}',
            ha='left', va='center', fontsize=12, color=COLORS['text_dark'])

    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cross-Validation Score')
    ax.set_title('RFE Feature Selection Performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'rfecv_results')
    plt.show()

    print("\n--- Selected Features ---")
    selected_features_mask = rfecv.support_
    selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_features_mask) if selected]
    print(selected_feature_names)
    print(f"\nTotal selected features: {len(selected_feature_names)}")

    print("\n--- Model Evaluation with Selected Features ---")
    X_train_selected = rfecv.transform(X_train_norm)
    X_test_selected = rfecv.transform(X_test_norm)

    final_model = RandomForestRegressor(
        max_depth=10, max_features='log2',
        min_samples_leaf=4, min_samples_split=5,
        n_estimators=200, random_state=seed
    )
    final_model.fit(X_train_selected, y_train_norm.ravel())

    y_pred_norm = final_model.predict(X_test_selected).reshape(-1, 1)
    y_test_true = scaler_y.inverse_transform(y_test_norm)
    y_test_pred = scaler_y.inverse_transform(y_pred_norm)

    r2 = r2_score(y_test_true, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
    rrmse = (rmse / np.mean(y_test_true)) * 100

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"rRMSE: {rrmse:.2f}%")

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_test_true, y_test_pred, c=COLORS['dark_blue'], s=30, alpha=0.8)

    min_val = min(y_test_true.min(), y_test_pred.min())
    max_val = max(y_test_true.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], '--', color=COLORS['text_dark'], linewidth=1, alpha=0.8)

    ax.set_xlabel('Observed')
    ax.set_ylabel('Predicted')
    ax.set_title('Observed vs. Predicted')
    ax.grid(True, alpha=0.3)

    ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {len(y_test_true)}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, 'prediction_results')
    plt.show()

    # ------------------ Feature Importance ------------------
    print("\n--- Feature Importance by Random Forest ---")
    importances = final_model.feature_importances_
    feature_names_selected = selected_feature_names

    feature_importance_df = pd.DataFrame({
        'feature': feature_names_selected,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\nTop 10 important features:")
    print(feature_importance_df.head(10))

    fig, ax = plt.subplots(figsize=(8, 10)) 

    feature_group_map = {}
    for name in feature_names_selected:
        if name in data_original.columns[18:27].tolist():
            feature_group_map[name] = COLORS['group_spatial']
        elif name in data_original.columns[27:35].tolist():
            feature_group_map[name] = COLORS['group_wavelet']
        elif name in data_original.columns[35:47].tolist():
            feature_group_map[name] = COLORS['group_color_index']

    bar_colors = [feature_group_map[f] for f in feature_importance_df['feature']]

    ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color=bar_colors)
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(f'Selected Feature Importance')
    ax.grid(axis='x', alpha=0.3)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['group_spatial']),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['group_wavelet']),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['group_color_index']),
    ]
    ax.legend(legend_handles, ['Spatial', 'Wavelet', 'Visible Vegetation Index'], title='Feature Group')

    plt.tight_layout()
    save_figure(fig, 'feature_importance_all_features')
    plt.show()

    print("\n--- Aggregated Importance by Feature Groups ---")
    feature_groups = {
        'Spatial': data_original.columns[18:27].tolist(),
        'Wavelet': data_original.columns[27:35].tolist(),
        'Visible Vegetation Index': data_original.columns[35:47].tolist()
    }

    aggregated_importance = {}
    for group_name, group_features in feature_groups.items():
        selected_and_in_group = feature_importance_df[
            feature_importance_df['feature'].isin(group_features)
        ]
        aggregated_importance[group_name] = selected_and_in_group['importance'].sum()

    aggregated_df = pd.DataFrame(
        list(aggregated_importance.items()),
        columns=['Feature Group', 'Total Importance']
    ).sort_values(by='Total Importance', ascending=False)

    group_colors = [
        COLORS['group_spatial'] if group == 'Spatial' else
        COLORS['group_wavelet'] if group == 'Wavelet' else
        COLORS['group_color_index']
        for group in aggregated_df['Feature Group']
    ]

    print("\nAggregated importance by groups:")
    print(aggregated_df)

    fig, ax = plt.subplots(figsize=(6, 8))

    ax.bar(aggregated_df['Feature Group'], aggregated_df['Total Importance'],
           color=group_colors, width=0.6)

    ax.set_xlabel('Feature Group')
    ax.set_ylabel('Total Importance')
    ax.set_title('Feature Group Importance')
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(aggregated_df['Total Importance']):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    save_figure(fig, 'group_importance')
    plt.show()


if __name__ == "__main__":
    run_feature_selection()