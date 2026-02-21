import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

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

def optimize_lags():
    print("Running Lag Optimization (AIC/BIC analysis)...")
    
    df_raw = pd.read_csv(DATA_PATH)
    df_raw['Data'] = pd.to_datetime(df_raw['Data'], errors='coerce')
    df_raw.sort_values(['Instituicao', 'Data'], inplace=True)
    
    # Merge Macro
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    df_raw = df_raw.merge(df_macro, on='Data', how='left')
    df_raw['Desemprego'] = df_raw.groupby('Instituicao')['Desemprego'].ffill().bfill()
    df_raw['Selic'] = df_raw.groupby('Instituicao')['Selic'].ffill().bfill()
    df_raw['NPL_Volatility_8Q'] = df_raw.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())
    df_raw['NPL_Volatility_8Q'] = df_raw['NPL_Volatility_8Q'].fillna(df_raw['NPL_Volatility_8Q'].mean())

    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    
    results = []
    
    for lag in [1, 2, 3, 4]:
        print(f"Testing Lag: {lag}...")
        df_model = df_raw.copy()
        lagged_features = []
        for f in core_features:
            col_name = f'{f}_lag{lag}'
            df_model[col_name] = df_model.groupby('Instituicao')[f].shift(lag)
            lagged_features.append(col_name)
        
        # Interaction for this lag
        inter_col = f'RWA_Operacional_lag{lag}_x_Alavancagem_lag{lag}'
        df_model[inter_col] = df_model[f'RWA_Operacional_lag{lag}'] * df_model[f'Alavancagem_lag{lag}']
        features = lagged_features + [inter_col]
        
        # Target
        threshold_p90 = df_model['NPL'].quantile(0.90)
        df_model['Target'] = (df_model['NPL'] > threshold_p90).astype(int)
        
        # Clean
        df_clean = df_model.dropna(subset=features + ['Target']).copy()
        
        # Train
        X = df_clean[features]
        y = df_clean['Target']
        X_scaled = (X - X.mean()) / X.std()
        X_scaled = sm.add_constant(X_scaled)
        
        model = sm.Logit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
        
        # Metrics
        y_prob = model.predict(X_scaled)
        auc = roc_auc_score(y, y_prob)
        
        results.append({
            'Lag': lag,
            'AIC': model.aic,
            'BIC': model.bic,
            'AUC': auc,
            'N_Obs': model.nobs
        })

    results_df = pd.DataFrame(results)
    print("\nLag Optimization Results:")
    print(results_df.to_string(index=False))
    
    results_df.to_csv(OUTPUT_DIR / "lag_optimization_results.csv", index=False)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=results_df, x='Lag', y='AIC', marker='o', label='AIC')
    sns.lineplot(data=results_df, x='Lag', y='BIC', marker='s', label='BIC')
    plt.title("Information Criteria vs Lag Length")
    plt.xlabel("Lag (Quarters)")
    plt.ylabel("Value (Lower is Better)")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=results_df, x='Lag', y='AUC', marker='D', color='red')
    plt.title("AUC-ROC vs Lag Length")
    plt.xlabel("Lag (Quarters)")
    plt.ylabel("AUC (Higher is Better)")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lag_optimization_plot.png")
    print(f"\nSaved analysis plot to {OUTPUT_DIR / 'lag_optimization_plot.png'}")

if __name__ == "__main__":
    optimize_lags()
