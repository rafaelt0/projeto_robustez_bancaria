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

def implement_fixed_effects():
    print("Implementing Fixed Effects Logit model...")
    
    # 1. Load and Prepare Data
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'])
    
    # Merge Macro and Fill
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    df = df.merge(df_macro, on='Data', how='left')
    for col in ['Desemprego', 'Selic', 'PIB', 'Spread']:
        if col in df.columns:
            df[col] = df.groupby('Instituicao')[col].ffill().bfill()
            
    # Add Volatility
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    # 2. Setup Regression Components
    LAG = 1
    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    
    df_model = df.copy()
    lagged_features = []
    for f in core_features:
        col_name = f'{f}_lag{LAG}'
        df_model[col_name] = df_model.groupby('Instituicao')[f].shift(LAG)
        lagged_features.append(col_name)
    
    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Target'] = (df_model['NPL'] > threshold_p90).astype(int)
    
    inter_col = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'
    df_model[inter_col] = df_model[f'RWA_Operacional_lag{LAG}'] * df_model[f'Alavancagem_lag{LAG}']
    
    features = lagged_features + [inter_col]
    
    # 3. Handle Fixed Effects Logic (LSDV)
    # Identify institutions that HAVE variance in Target (necessary for FE)
    inst_variance = df_model.groupby('Instituicao')['Target'].std()
    eligible_inst = inst_variance[inst_variance > 0].index.tolist()
    
    print(f"Total Institutions: {df_model['Instituicao'].nunique()}")
    print(f"Institutions with outcome variance (eligible for FE): {len(eligible_inst)}")
    
    df_fe = df_model[df_model['Instituicao'].isin(eligible_inst)].dropna(subset=features + ['Target']).copy()
    
    # Create Dummies
    # We leave one out to avoid multicollinearity (the Intercept will be the base)
    fe_dummies = pd.get_dummies(df_fe['Instituicao'], prefix='FE', drop_first=True)
    
    # Scale core features
    X_core = df_fe[features]
    X_scaled = (X_core - X_core.mean()) / X_core.std()
    
    # Combine Scaled Features + Institution Dummies
    X = pd.concat([X_scaled, fe_dummies], axis=1)
    X = sm.add_constant(X)
    X = X.astype(float) # Fix: Ensure no object types (like bool from dummies)
    y = df_fe['Target'].astype(float)
    
    # 4. Train Model
    print("Fitting High-Dimensional Fixed Effects Logit...")
    # BFGS is usually better for many parameters
    model = sm.Logit(y, X).fit(method='bfgs', maxiter=1000, disp=False)
    
    # 5. Extract Institution Effects (Intercepts)
    # The 'const' is the intercept for the 'dropped' institution
    # The FE_xxx coefficients are the DIFFERENCES from the base
    base_inst = fe_dummies.columns[0].replace('FE_', '') # Approximation
    params = model.params
    
    # Calculate Absolue Intercepts for Ranking
    fe_results = []
    # Identify the base institution (the one dropped by get_dummies(drop_first=True))
    all_inst = sorted(eligible_inst)
    dropped_inst = all_inst[0] # Usually the first alphabetical one
    
    global_const = params['const']
    fe_results.append({'Instituicao': dropped_inst, 'Fixed_Effect_Intercept': global_const})
    
    for inst in all_inst[1:]:
        col_name = f'FE_{inst}'
        if col_name in params:
            fe_results.append({'Instituicao': inst, 'Fixed_Effect_Intercept': global_const + params[col_name]})
            
    fe_df = pd.DataFrame(fe_results).sort_values('Fixed_Effect_Intercept', ascending=False)
    # Note: High Intercept = structurally HIGHER risk bank
    fe_df['Intrinsic_Robustness'] = -fe_df['Fixed_Effect_Intercept']
    
    # 6. Performance
    y_prob = model.predict(X)
    auc = roc_auc_score(y, y_prob)
    print(f"Fixed Effects Model AUC-ROC: {auc:.4f}")
    
    # 7. Save and Report
    fe_df.to_csv(OUTPUT_DIR / "institution_fixed_effects.csv", index=False)
    print(f"Fixed Effects saved to {OUTPUT_DIR / 'institution_fixed_effects.csv'}")
    
    # Visualizing Top and Bottom Intrinsic Robustness
    plt.figure(figsize=(10, 8))
    top_bottom = pd.concat([fe_df.head(10), fe_df.tail(10)])
    sns.barplot(data=top_bottom, x='Intrinsic_Robustness', y='Instituicao', palette='RdYlGn')
    plt.title("Intrinsic Robustness (Fixed Effects Intercepts)\nHigh = Structurally Lower Stress Risk")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fixed_effects_ranking.png")
    
if __name__ == "__main__":
    implement_fixed_effects()
