import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, confusion_matrix, classification_report

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_PATH = WORKSPACE_DIR / "data" / "raw" / "painel_final.csv"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "stress_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Macro series (fallback)
macro_data = {
    'Data': ['2015-12-01', '2016-03-01', '2016-06-01', '2016-09-01', '2016-12-01',
             '2017-03-01', '2017-06-01', '2017-09-01', '2017-12-01',
             '2018-03-01', '2018-06-01', '2018-09-01', '2018-12-01',
             '2019-03-01', '2019-06-01', '2019-09-01', '2019-12-01',
             '2020-03-01', '2020-06-01', '2020-09-01', '2020-12-01',
             '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01',
             '2022-03-01', '2022-06-01', '2022-09-01', '2022-12-01',
             '2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01',
             '2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01',
             '2025-03-01', '2025-06-01', '2025-09-01', '2025-12-01'],
    'Desemprego': [9.0, 10.9, 11.3, 11.8, 12.0, 13.7, 13.0, 12.4, 11.8, 13.1, 12.4, 11.9, 11.6, 12.7, 12.0, 11.8, 11.0, 12.2, 13.3, 14.6, 13.9, 14.7, 14.1, 12.6, 11.1, 11.1, 9.3, 8.7, 7.9, 8.8, 8.0, 7.7, 7.4, 7.9, 6.9, 6.4, 6.2, 6.6, 5.6, 5.6, 5.1],
    'Selic': [14.25, 14.25, 14.25, 14.25, 13.75, 12.25, 10.25, 8.25, 7.00, 6.50, 6.50, 6.50, 6.50, 6.50, 6.50, 5.50, 4.50, 3.75, 2.25, 2.00, 2.00, 2.75, 4.25, 5.75, 9.25, 11.75, 13.25, 13.75, 13.75, 13.75, 13.75, 12.75, 11.75, 10.75, 10.50, 10.50, 11.25, 13.25, 14.25, 15.00, 15.00]
}

def analyze_thresholds():
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df.sort_values(['Instituicao', 'Data'], inplace=True)
    
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    df = df.merge(df_macro, on='Data', how='left')
    df['Desemprego'] = df.groupby('Instituicao')['Desemprego'].ffill().bfill()
    df['Selic'] = df.groupby('Instituicao')['Selic'].ffill().bfill()
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())

    LAG = 1
    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    df_model = df.copy()
    features = []
    for f in core_features:
        col_name = f'{f}_lag{LAG}'
        df_model[col_name] = df_model.groupby('Instituicao')[f].shift(LAG)
        features.append(col_name)

    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Estresse_Alto_P90'] = (df_model['NPL'] > threshold_p90).astype(int)
    inter_col = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'
    df_model[inter_col] = df_model[f'RWA_Operacional_lag{LAG}'] * df_model[f'Alavancagem_lag{LAG}']
    features.append(inter_col)
    
    df_clean = df_model.dropna(subset=features + ['Estresse_Alto_P90']).copy()
    X = df_clean[features]
    y = df_clean['Estresse_Alto_P90']
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = sm.add_constant(X_scaled)
    
    model = sm.Logit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
    y_prob = model.predict(X_scaled)

    # Threshold Optimization
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Youden's J = Sensitivity + Specificity - 1
    # Specificity = 1 - FPR
    # J = TPR - FPR
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold_youden = thresholds[best_idx]
    
    # Precision, Recall, F1 Optimization
    precisions, recalls, pr_thresholds = precision_recall_curve(y, y_prob)
    # pr_thresholds is one smaller than precisions and recalls
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold_f1 = pr_thresholds[best_f1_idx] if best_f1_idx < len(pr_thresholds) else pr_thresholds[-1]

    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Performance vs Threshold
    plt.subplot(1, 2, 1)
    # Filter thresholds to 0-1 range for plotting
    plot_thresholds = np.linspace(0, 1, 100)
    res = []
    for t in plot_thresholds:
        y_p = (y_prob > t).astype(int)
        cm = confusion_matrix(y, y_p)
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        youden = recall - (cm[0,1] / (cm[0,1] + cm[0,0]))
        res.append({'t': t, 'Recall': recall, 'Precision': precision, 'F1': f1, 'Youden J': youden})
    
    res_df = pd.DataFrame(res)
    plt.plot(res_df['t'], res_df['Recall'], label='Recall (Sensitivity)')
    plt.plot(res_df['t'], res_df['Precision'], label='Precision')
    plt.plot(res_df['t'], res_df['F1'], label='F1 Score', linestyle='--')
    plt.axvline(best_threshold_youden, color='red', linestyle=':', label=f'Best Youden ({best_threshold_youden:.2f})')
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.50)')
    plt.title("Metrics vs Classification Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    
    # Plot 2: ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={roc_auc_score(y, y_prob):.3f})')
    plt.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Optimal Point (J={j_scores[best_idx]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.title("ROC Curve & Optimal Point")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "threshold_optimization.png")
    
    print(f"Optimal Threshold (Youden's J): {best_threshold_youden:.4f}")
    print(f"Optimal Threshold (F1-Max): {best_threshold_f1:.4f}")
    
    # Comparison table at optimal threshold
    def get_metrics(t):
        y_p = (y_prob > t).astype(int)
        report = classification_report(y, y_p, output_dict=True)
        return {
            'Threshold': t,
            'Recall': report['1']['recall'],
            'Precision': report['1']['precision'],
            'F1': report['1']['f1-score'],
            'Accuracy': report['accuracy']
        }
    
    comparison = pd.DataFrame([get_metrics(0.5), get_metrics(best_threshold_youden), get_metrics(best_threshold_f1)])
    comparison['Method'] = ['Default', 'Youden J', 'F1-Max']
    print("\nComparison Table:")
    print(comparison[['Method', 'Threshold', 'Recall', 'Precision', 'F1', 'Accuracy']].to_string(index=False))

    comparison.to_csv(OUTPUT_DIR / "threshold_comparison_results.csv", index=False)

if __name__ == "__main__":
    analyze_thresholds()
