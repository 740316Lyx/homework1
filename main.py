import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold  # 导入KFold用于Stacking
from sklearn.ensemble import RandomForestRegressor, StackingRegressor  # 导入StackingRegressor
from sklearn.linear_model import Lasso, LinearRegression  # 导入LinearRegression作为Stacking的元模型
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
from sklearn.impute import KNNImputer
import time
import numpy as np

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['font.serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 数据层面：数据获取、探索与预处理 ---

def load_and_preprocess_data(file_path):
    """
    加载数据并进行初步预处理。
    包括加载、查看基本信息、处理列名、删除无用列、检查缺失值，并处理缺失值。
    新增特征工程。
    """
    try:
        df = pd.read_csv(file_path)
        print("数据加载成功！")
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。请确保文件在正确的目录下。")
        exit()

    print("\n--- 数据概览 ---")
    print("数据前5行:")
    print(df.head())

    print("\n数据基本信息:")
    df.info()

    print("\n描述性统计:")
    print(df.describe())

    # 清理列名：去除可能存在的空格
    original_columns = df.columns.tolist()
    df.columns = df.columns.str.strip()
    if original_columns != df.columns.tolist():
        print("\n已清理列名（去除首尾空格）。")

    # 删除 'Serial No.' 列，因为它通常不是预测的有用特征
    if 'Serial No.' in df.columns:
        df = df.drop('Serial No.', axis=1)
        print("已删除 'Serial No.' 列。")
    else:
        print("'Serial No.' 列不存在，无需删除。")

    print("\n原始缺失值检查:")
    print(df.isnull().sum())

    # --- 缺失值处理 ---
    print("\n--- 开始处理缺失值（使用 KNNImputer） ---")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print("警告: 数据中没有数值型列，无法使用 KNNImputer。")
    else:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print("数值型列的缺失值已使用 KNNImputer 填充。")

    print("\n处理后缺失值检查:")
    print(df.isnull().sum())
    print("--- 缺失值处理完成 ---")

    # --- 新增：特征工程 ---
    print("\n--- 开始特征工程 ---")
    # 示例1: 创建 GRE 和 TOEFL 的平均分或总分
    if 'GRE Score' in df.columns and 'TOEFL Score' in df.columns:
        df['GRE_TOEFL_Avg'] = (df['GRE Score'] + df['TOEFL Score']) / 2
        print("已创建新特征 'GRE_TOEFL_Avg'。")

    # 示例2: 研究经验与大学评级的交互特征
    # 假设Research为0/1，University Rating为1-5
    if 'Research' in df.columns and 'University Rating' in df.columns:
        df['Research_x_UniRating'] = df['Research'] * df['University Rating']
        print("已创建新特征 'Research_x_UniRating'。")

    print("--- 特征工程完成 ---")
    print("\n--- 数据预处理完成 ---")
    return df


def visualize_data(df):
    """
    绘制数据分布和特征相关性图。
    包括目标变量分布、单个特征分布、特征与目标变量的散点图以及特征相关性热力图。
    """
    print("\n--- 数据可视化 ---")

    # 绘制目标变量的分布 (假设 'Chance of Admit' 是目标变量)
    target_col = 'Chance of Admit'
    if target_col in df.columns:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(df[target_col], kde=True, bins=20, color='skyblue')
        plt.title('录取概率分布 (直方图 & KDE)', fontsize=14)
        plt.xlabel('录取概率', fontsize=12)
        plt.ylabel('频率 / 密度', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        sns.boxplot(y=df[target_col], color='lightcoral')
        plt.title('录取概率分布 (箱线图)', fontsize=14)
        plt.ylabel('录取概率', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"警告: '{target_col}' 列未找到，无法绘制目标变量分布图。")

    # --- 绘制单个特征的分布图 (Histplot/KDE) ---
    print("\n--- 绘制单个特征分布图 ---")
    features_to_plot = df.drop(columns=[target_col], errors='ignore').columns.tolist()

    n_cols_dist = 3
    n_rows_dist = (len(features_to_plot) + n_cols_dist - 1) // n_cols_dist
    plt.figure(figsize=(n_cols_dist * 8, n_rows_dist * 6))

    for i, col in enumerate(features_to_plot):
        plt.subplot(n_rows_dist, n_cols_dist, i + 1)
        sns.histplot(df[col], kde=True, color='purple' if col.startswith('Research') else 'green')
        plt.title(f'{col} 分布', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('频率 / 密度', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(pad=4.0)
    plt.show()
    print("单个特征分布图绘制完成。")

    # --- 特征与目标变量的散点图 ---
    print("\n--- 绘制特征与目标变量的散点图 ---")
    if target_col in df.columns:
        n_cols_scatter = 3
        n_rows_scatter = (len(features_to_plot) + n_cols_scatter - 1) // n_cols_scatter
        plt.figure(figsize=(n_cols_scatter * 10, n_rows_scatter * 8))

        for i, col in enumerate(features_to_plot):
            plt.subplot(n_rows_scatter, n_cols_scatter, i + 1)
            sns.scatterplot(x=df[col], y=df[target_col], alpha=0.6, color='darkblue')
            plt.title(f'{col} vs. {target_col}', fontsize=14)
            plt.xlabel(col, fontsize=12)
            plt.ylabel(target_col, fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout(pad=4.0)
        plt.show()
        print("特征与目标变量散点图绘制完成。")
    else:
        print("警告: 目标变量列不存在，无法绘制特征与目标变量散点图。")

    # --- 绘制特征相关性热力图 ---
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['number'])

    sns.heatmap(numeric_df.corr(),
                annot=True,
                cmap='viridis',
                fmt=".2f",
                linewidths=.5,
                linecolor='black',
                cbar_kws={'label': '相关系数'}
                )

    plt.title('特征相关性热力图', fontsize=16, pad=20)
    plt.xlabel('数据特征', fontsize=12)
    plt.ylabel('数据特征', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.figtext(0.96, 0.05,
                "相关系数范围：\n"
                "  1.0: 完美正相关\n"
                "  0.0: 无线性相关\n"
                " -1.0: 完美负相关\n\n"
                "颜色越深（黄绿色）：正相关越强\n"
                "颜色越浅（紫色）：负相关越强",
                ha='left', fontsize=9, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})
    plt.show()
    print("数据可视化完成。")


# --- B. 模型评估的全面性与模型解释性优化 ---

def train_and_evaluate_single_model(model_name, estimator, param_grid, X_train, y_train, X_test, y_test, X_train_cols,
                                    model_save_path_prefix):
    """
    训练并评估单个模型，包括超参数调优、性能评估、残差分析和特征重要性分析。
    此版本已移除所有 SHAP、Permutation Importance 和 LIME 解释功能。
    """
    print(f"\n--- 正在训练和评估模型: {model_name} ---")
    start_time = time.time()

    print(f"开始进行 {model_name} 的 GridSearchCV 超参数调优，这可能需要一些时间...")
    grid_search = GridSearchCV(estimator=estimator,
                               param_grid=param_grid,
                               cv=5,
                               scoring='r2',
                               n_jobs=-1,  # GridSearchCV 内部仍然可以并行，这是它自身的功能
                               verbose=1)

    grid_search.fit(X_train, y_train)

    end_time = time.time()
    training_duration = end_time - start_time
    print(f"\n{model_name} 超参数调优完成！耗时: {training_duration:.2f} 秒")
    print(f"{model_name} 最佳超参数: {grid_search.best_params_}")
    print(f"{model_name} 最佳 R-squared (在交叉验证中): {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_
    print(f"已使用最佳超参数训练 {model_name} 最终模型。")

    current_model_save_path = f"{model_save_path_prefix}_{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(best_model, current_model_save_path)
    print(f"\n{model_name} 最佳模型已保存为 {current_model_save_path}")

    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- {model_name} 模型在测试集上的评估结果 ---")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"决定系数 (R-squared): {r2:.4f}")

    # 可视化预测结果与真实值的对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('实际录取概率', fontsize=12)
    plt.ylabel('预测录取概率', fontsize=12)
    plt.title(f'{model_name}: 实际录取概率 vs. 预测录取概率', fontsize=14)
    plt.grid(True)
    plt.show()

    # --- 残差分析 ---
    print(f"\n--- {model_name} 模型残差分析 ---")
    residuals = y_test - y_pred

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)  # 残差分布图
    sns.histplot(residuals, kde=True, color='orange')
    plt.title(f'{model_name}: 残差分布', fontsize=14)
    plt.xlabel('残差 (实际值 - 预测值)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.axvline(0, color='red', linestyle='--', label='残差为0')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)  # 残差 vs 预测值散点图
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, color='blue')
    plt.axhline(0, color='red', linestyle='--', label='残差为0')
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title(f'{model_name}: 残差 vs. 预测值', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("残差分析图绘制完成。")

    # 特征重要性分析或系数分析 (保留模型自带的特征重要性或系数)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_, index=X_train_cols)
        feature_importances = feature_importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
        plt.title(f'{model_name}: 特征重要性', fontsize=14)
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        plt.show()
    elif hasattr(best_model, 'coef_'):
        print(f"\n--- {model_name} 模型系数 (特征重要性) ---")
        coefs = pd.Series(best_model.coef_, index=X_train_cols)
        coefs = coefs.sort_values(key=abs, ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=coefs.values, y=coefs.index, palette='coolwarm')
        plt.title(f'{model_name}: 特征系数 (重要性)', fontsize=14)
        plt.xlabel('系数大小', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.tight_layout()
        plt.show()
        print(coefs)
    else:
        print(f"当前模型 {model_name} 不支持直接获取特征重要性或系数。")

    return {
        'model_name': model_name,
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'cv_r2': grid_search.best_score_,
        'test_mse': mse,
        'test_rmse': rmse,
        'test_r2': r2,
        'training_duration': training_duration,
        'y_pred_test': y_pred  # 返回测试集预测结果，用于集成
    }


# --- 4. 验证层面：模型应用与未来展望 ---

def predict_new_data(model, X_train_cols, new_data_raw, model_name="Selected Model"):
    """
    使用训练好的模型进行新数据的预测。
    新数据需要进行与训练数据相同的特征工程。
    """
    print(f"\n--- {model_name} 对新数据进行预测 ---")

    # 对新数据进行与训练数据相同的特征工程
    new_data_processed = new_data_raw.copy()
    if 'GRE Score' in new_data_processed.columns and 'TOEFL Score' in new_data_processed.columns:
        new_data_processed['GRE_TOEFL_Avg'] = (new_data_processed['GRE Score'] + new_data_processed['TOEFL Score']) / 2
    if 'Research' in new_data_processed.columns and 'University Rating' in new_data_processed.columns:
        new_data_processed['Research_x_UniRating'] = new_data_processed['Research'] * new_data_processed[
            'University Rating']

    # 确保新数据的列与训练数据一致
    for col in X_train_cols:
        if col not in new_data_processed.columns:
            new_data_processed[col] = 0.0  # 假设为数值特征，填充0
            print(f"警告: 新数据中缺少训练数据中的列: '{col}'，已填充0。请检查特征工程步骤。")

    extra_cols = set(new_data_processed.columns) - set(X_train_cols)
    if extra_cols:
        print(f"警告: 新数据中有多余的列: {extra_cols}。将删除这些列。")
        new_data_processed = new_data_processed.drop(columns=list(extra_cols))

    new_data_final = new_data_processed[X_train_cols]

    print("\n待预测的新数据 (已处理):")
    print(new_data_final)

    # 进行预测
    predictions = model.predict(new_data_final)

    print("\n预测的录取概率:")
    for i, prob in enumerate(predictions):
        print(f"学生 {i + 1}: {prob:.4f}")

    new_data_raw_with_pred = new_data_raw.copy()
    new_data_raw_with_pred['Predicted Chance of Admit'] = predictions
    print("\n带有预测结果的原始新数据:")
    print(new_data_raw_with_pred)
    print("新数据预测完成。")
    return predictions


# --- 主程序入口 ---
if __name__ == "__main__":
    data_file = './Admission_Predict.csv'
    model_save_path_prefix = 'admission_model'

    # 1. 数据层面：加载、预处理与可视化
    df = load_and_preprocess_data(data_file)
    visualize_data(df.copy())

    target_column = 'Chance of Admit'
    if target_column not in df.columns:
        print(f"错误：数据中未找到目标列 '{target_column}'。请检查文件内容。")
        exit()

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集目标形状: {y_train.shape}")
    print(f"测试集目标形状: {y_test.shape}")

    # --- 定义要对比的单一模型和它们的超参数网格 ---
    # 调整超参数范围以加快训练速度，以便在Colab等环境中测试
    models_configs = [
        {
            "name": "Random Forest",
            "estimator": RandomForestRegressor(random_state=42),
            "param_grid": {
                'n_estimators': [100, 200],
                'max_features': ['sqrt', 0.6],
                'max_depth': [10, 20]
            }
        },
        {
            "name": "XGBoost",
            "estimator": xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42, n_jobs=-1),
            "param_grid": {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        {
            "name": "Lasso Regression",
            "estimator": Lasso(random_state=42, max_iter=2000),
            "param_grid": {
                'alpha': [0.001, 0.01]
            }
        }
    ]

    # --- 定义新数据，供后续所有模型预测使用 ---
    new_data_for_prediction = pd.DataFrame({
        'GRE Score': [320, 310, 335, 300, 340],
        'TOEFL Score': [110, 105, 118, 95, 120],
        'University Rating': [4, 3, 5, 2, 5],
        'SOP': [3.5, 3.0, 4.5, 2.5, 5.0],
        'LOR': [3.0, 3.5, 4.0, 2.0, 5.0],
        'CGPA': [9.0, 8.5, 9.5, 7.5, 9.9],
        'Research': [1, 0, 1, 0, 1]
    })

    print("\n--- 开始串行训练和评估所有单一模型 ---")
    model_results_list = []
    for model_info in models_configs:
        results = train_and_evaluate_single_model(
            model_name=model_info["name"],
            estimator=model_info["estimator"],
            param_grid=model_info["param_grid"],
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            X_train_cols=X_train_columns,
            model_save_path_prefix=model_save_path_prefix
        )
        model_results_list.append(results)

    # --- 模型性能最终对比 (单一模型) ---
    print("\n\n--- 单一模型性能最终对比 ---")
    results_df = pd.DataFrame(model_results_list)
    comparison_df = results_df[['model_name', 'test_r2', 'test_rmse', 'training_duration', 'best_params']]
    print(comparison_df.to_string())

    # 可视化模型 R2 分数对比 (单一模型)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model_name', y='test_r2', data=comparison_df, palette='viridis')
    plt.title('单一模型在测试集上的R-squared (R2) 对比', fontsize=14)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('R-squared (R2)', fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- 集成预测：简单平均法 (Averaging Ensemble) ---
    print("\n--- 开始进行所有单一模型在测试集上的集成预测 (简单平均法) ---")
    all_test_preds = []
    for model_res in model_results_list:
        all_test_preds.append(model_res['y_pred_test'])

    ensemble_avg_test_pred = np.mean(all_test_preds, axis=0)
    ensemble_avg_mse = mean_squared_error(y_test, ensemble_avg_test_pred)
    ensemble_avg_rmse = ensemble_avg_mse ** 0.5
    ensemble_avg_r2 = r2_score(y_test, ensemble_avg_test_pred)

    print(f"\n--- 集成模型 (简单平均) 在测试集上的评估结果 ---")
    print(f"均方误差 (MSE): {ensemble_avg_mse:.4f}")
    print(f"均方根误差 (RMSE): {ensemble_avg_rmse:.4f}")
    print(f"决定系数 (R-squared): {ensemble_avg_r2:.4f}")

    # 将集成模型 (简单平均) 的结果添加到对比DataFrame
    ensemble_avg_results_row = pd.DataFrame([{
        'model_name': 'Ensemble (Avg)',
        'test_r2': ensemble_avg_r2,
        'test_rmse': ensemble_avg_rmse,
        'training_duration': 0,  # 集成本身无训练时间
        'best_params': 'N/A'
    }])
    comparison_df = pd.concat([comparison_df, ensemble_avg_results_row], ignore_index=True)

    # --- 集成预测：堆叠法 (StackingRegressor) ---
    print("\n--- 开始进行堆叠法集成 (StackingRegressor) ---")

    # 获取所有训练好的最佳单一模型
    estimators = []
    for model_res in model_results_list:
        estimators.append((model_res['model_name'].lower().replace(' ', '_'), model_res['best_model']))

    # 定义元模型 (meta-learner)
    # 线性回归是一个常用的简单元模型，也可以尝试其他模型如Ridge, Lasso, RandomForest等
    meta_learner = LinearRegression()

    # 创建StackingRegressor
    # cv=5 表示使用5折交叉验证生成OOF预测来训练元模型
    # n_jobs=-1 可以在Stacking内部并行化基础模型的训练和预测
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),  # 显式传入KFold对象
        n_jobs=-1,  # 可以在Stacking内部并行化基础模型的训练
        verbose=1
    )

    stacking_start_time = time.time()
    stacking_model.fit(X_train, y_train)
    stacking_end_time = time.time()
    stacking_training_duration = stacking_end_time - stacking_start_time
    print(f"堆叠模型训练完成！耗时: {stacking_training_duration:.2f} 秒")

    # 在测试集上评估堆叠模型
    stacking_test_pred = stacking_model.predict(X_test)
    stacking_mse = mean_squared_error(y_test, stacking_test_pred)
    stacking_rmse = stacking_mse ** 0.5
    stacking_r2 = r2_score(y_test, stacking_test_pred)

    print(f"\n--- 堆叠模型 (Stacking) 在测试集上的评估结果 ---")
    print(f"均方误差 (MSE): {stacking_mse:.4f}")
    print(f"均方根误差 (RMSE): {stacking_rmse:.4f}")
    print(f"决定系数 (R-squared): {stacking_r2:.4f}")

    # 将堆叠模型的结果添加到对比DataFrame
    stacking_results_row = pd.DataFrame([{
        'model_name': 'Ensemble (Stacking)',
        'test_r2': stacking_r2,
        'test_rmse': stacking_rmse,
        'training_duration': stacking_training_duration,
        'best_params': 'N/A (Stacking)'
    }])
    comparison_df = pd.concat([comparison_df, stacking_results_row], ignore_index=True)

    print("\n--- 所有模型 (单一及集成) 性能对比完成。---")
    print(comparison_df.to_string())

    # 重新绘制模型 R2 分数对比图，包含所有单一模型和集成模型
    plt.figure(figsize=(12, 7))
    sns.barplot(x='model_name', y='test_r2', data=comparison_df.sort_values(by='test_r2', ascending=False),
                palette='viridis')
    plt.title('所有模型与集成模型在测试集上的R-squared (R2) 对比', fontsize=14)
    plt.xlabel('模型', fontsize=12)
    plt.ylabel('R-squared (R2)', fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # --- 最终选择最佳模型进行新数据预测 ---
    best_overall_model_row = comparison_df.loc[comparison_df['test_r2'].idxmax()]
    final_model_name = best_overall_model_row['model_name']

    print(f"\n根据测试集 R-squared，选择的最佳预测模型是: {final_model_name}")

    if final_model_name == 'Ensemble (Avg)':
        # 对新数据进行所有单一模型的预测并取平均
        all_new_preds_single_models = []
        for model_res in model_results_list:
            model_instance = model_res['best_model']
            # predict_new_data 函数内部会处理特征工程和列顺序
            # 这里的 new_data_for_prediction 需要是一个原始副本，predict_new_data会对其进行内部处理
            current_model_new_pred = predict_new_data(model_instance, X_train_columns, new_data_for_prediction.copy(),
                                                      model_name=f"{model_res['model_name']}")
            all_new_preds_single_models.append(current_model_new_pred)

        final_new_predictions_ensemble_avg = np.mean(all_new_preds_single_models, axis=0)
        print("\n--- 最终集成模型 (简单平均) 对新数据的预测结果 ---")
        new_data_raw_final = new_data_for_prediction.copy()
        new_data_raw_final['Predicted Chance of Admit'] = final_new_predictions_ensemble_avg
        print(new_data_raw_final)

    elif final_model_name == 'Ensemble (Stacking)':
        print("\n--- 最终集成模型 (Stacking) 对新数据的预测结果 ---")
        # 直接使用训练好的 stacking_model 进行预测
        final_new_predictions_stacking = predict_new_data(stacking_model, X_train_columns,
                                                          new_data_for_prediction.copy(),
                                                          model_name="Ensemble (Stacking)")

    else:
        # 如果最佳是单一模型，则加载并使用该模型进行预测
        best_single_model_instance = results_df.loc[results_df['model_name'] == final_model_name, 'best_model'].iloc[0]
        predict_new_data(best_single_model_instance, X_train_columns, new_data_for_prediction.copy(),
                         model_name=f"最佳单一模型 ({final_model_name})")

    print("\n所有预测任务完成。")