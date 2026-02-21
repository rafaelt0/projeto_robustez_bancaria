import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_PATH = WORKSPACE_DIR / "data" / "raw" / "painel_final.csv"

# Macro data subset
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

def generate_econometrica_table():
    # 1. Prepare Data (FE version)
    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'])
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    df = df.merge(df_macro, on='Data', how='left')
    for col in ['Desemprego', 'Selic', 'PIB', 'Spread']:
        df[col] = df.groupby('Instituicao')[col].ffill().bfill()
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(8, 4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())
    
    LAG = 1
    core_features = ['Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    df_model = df.copy()
    features = []
    for f in core_features:
        name = f'{f}_lag{LAG}'
        df_model[name] = df_model.groupby('Instituicao')[f].shift(LAG)
        features.append(name)
    
    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Target'] = (df_model['NPL'] > threshold_p90).astype(int)
    
    # Filter for FE
    inst_variance = df_model.groupby('Instituicao')['Target'].std()
    eligible = inst_variance[inst_variance > 0].index
    df_fe = df_model[df_model['Instituicao'].isin(eligible)].dropna(subset=features + ['Target']).copy()
    
    # Regression
    dummies = pd.get_dummies(df_fe['Instituicao'], prefix='FE', drop_first=True)
    X_main = df_fe[features]
    X_scaled = (X_main - X_main.mean()) / X_main.std()
    X = pd.concat([X_scaled, dummies], axis=1).astype(float)
    X = sm.add_constant(X)
    y = df_fe['Target'].astype(float)
    
    res = sm.Logit(y, X).fit(method='bfgs', maxiter=1000, disp=False)
    
    # Extract only main variables for the table
    summary_data = []
    for f in ['const'] + features:
        coef = res.params[f]
        std_err = res.bse[f]
        p_val = res.pvalues[f]
        
        stars = ""
        if p_val < 0.01: stars = "***"
        elif p_val < 0.05: stars = "**"
        elif p_val < 0.10: stars = "*"
        
        summary_data.append([f.replace('_lag1', ''), f"{coef:.4f}{stars}", f"({std_err:.4f})"])

    print("\nTABLE 1")
    print("DETERMINANTS OF BANKING STRESS: LOGIT FIXED EFFECTS ESTIMATES")
    print("-" * 60)
    print(f"{'Variable':<25} {'Coefficient':<15} {'(Std. Err.)':<15}")
    print("-" * 60)
    for row in summary_data:
        print(f"{row[0]:<25} {row[1]:<15}")
        print(f"{'':<25} {row[2]:<15}")
    print("-" * 60)
    print(f"Num. Observations:      {int(res.nobs)}")
    print(f"Pseudo R-Squared:       {res.prsquared:.4f}")
    print(f"Log-Likelihood:         {res.llf:.2f}")
    print(f"Number of Entities:     {len(eligible)}")
    print(f"Fixed Effects:          YES (Institution)")
    print("-" * 60)
    print("Notes: *** p<0.01, ** p<0.05, * p<0.1. Standard errors in parentheses.")

if __name__ == "__main__":
    generate_econometrica_table()
