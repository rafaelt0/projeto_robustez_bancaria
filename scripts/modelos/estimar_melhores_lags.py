import pandas as pd
import numpy as np

# 1. Carregar Dados
painel_path = 'd:/projeto_robustez_bancaria/dados/brutos/painel_final.csv'
df = pd.read_csv(painel_path, sep=';', decimal=',', encoding='latin1')

df.columns = [
    'Instituicao', 'Data', 'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional',
    'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'Extra_Empty', 'NPL'
]
df.drop(columns=['Extra_Empty'], inplace=True)

def parse_num(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

numeric_cols = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'NPL']
for col in numeric_cols:
    df[col] = df[col].apply(parse_num)

df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
df.dropna(subset=['Data'], inplace=True)
df.sort_values(['Instituicao', 'Data'], inplace=True)

# Define Target
threshold = df['NPL'].quantile(0.75)
df['Estresse_Alto'] = (df['NPL'] > threshold).astype(int)

# Logistic Regression Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

def calculate_log_likelihood(X, Y, weights):
    z = np.dot(X, weights)
    h = sigmoid(z)
    epsilon = 1e-15
    return np.sum(Y * np.log(h + epsilon) + (1 - Y) * np.log(1 - h + epsilon))

def fit_logit(X, Y, lr=0.1, iterations=5000):
    n_features = X.shape[1]
    weights = np.zeros((n_features, 1))
    for _ in range(iterations):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - Y)) / Y.size
        weights -= lr * gradient
    return weights

# Lag Selection Logic
features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread']
lags_to_test = [1, 2, 3, 4]
results = []

for k in lags_to_test:
    df_temp = df.copy()
    lag_cols = []
    for f in features:
        col_name = f'{f}_lag{k}'
        df_temp[col_name] = df_temp.groupby('Instituicao')[f].shift(k)
        lag_cols.append(col_name)
    
    df_clean = df_temp.dropna(subset=lag_cols + ['Estresse_Alto']).copy()
    if df_clean.empty:
        continue

    X = df_clean[lag_cols].values
    Y = df_clean['Estresse_Alto'].values.reshape(-1, 1)

    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

    w = fit_logit(X_scaled, Y)
    ll_model = calculate_log_likelihood(X_scaled, Y, w)
    
    p_null = np.mean(Y)
    ll_null = len(Y) * (p_null * np.log(p_null + 1e-15) + (1 - p_null) * np.log(1 - p_null + 1e-15))
    
    n = len(Y)
    num_params = X_scaled.shape[1]
    
    aic = 2 * num_params - 2 * ll_model
    bic = num_params * np.log(n) - 2 * ll_model
    r2_pseudo = 1 - (ll_model / ll_null)
    
    results.append({
        'Lag': k,
        'AIC': aic,
        'BIC': bic,
        'Pseudo_R2': r2_pseudo
    })

df_res = pd.DataFrame(results)
df_res.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/metricas_selecao_lags.csv', sep=';', decimal=',', index=False)
