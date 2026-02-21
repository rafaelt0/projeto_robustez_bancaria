import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("ADICIONANDO AS 2 MACROS MAIS IMPORTANTES: DESEMPREGO E SELIC")
print("="*100)

# 1. Carregar dados existentes
painel_path = r'd:\projeto_robustez_bancaria\dados\brutos\painel_final.csv'
df = pd.read_csv(painel_path)

df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
df.sort_values(['Instituicao', 'Data'], inplace=True)

# 2. DEFINIR SÉRIES MACRO HISTÓRICAS (Trimestrais)
# As datas no CSV são geralmente o primeiro dia do trimestre (ex: 2018-03-01 para Q1)
macro_data = {
    'Data': [
        '2015-12-01', '2016-03-01', '2016-06-01', '2016-09-01', '2016-12-01',
        '2017-03-01', '2017-06-01', '2017-09-01', '2017-12-01',
        '2018-03-01', '2018-06-01', '2018-09-01', '2018-12-01',
        '2019-03-01', '2019-06-01', '2019-09-01', '2019-12-01',
        '2020-03-01', '2020-06-01', '2020-09-01', '2020-12-01',
        '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01',
        '2022-03-01', '2022-06-01', '2022-09-01', '2022-12-01',
        '2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01',
        '2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01',
        '2025-03-01', '2025-06-01', '2025-09-01', '2025-12-01'
    ],
    'Desemprego': [
        9.0, 10.9, 11.3, 11.8, 12.0,
        13.7, 13.0, 12.4, 11.8,
        13.1, 12.4, 11.9, 11.6,
        12.7, 12.0, 11.8, 11.0,
        12.2, 13.3, 14.6, 13.9,
        14.7, 14.1, 12.6, 11.1,
        11.1, 9.3, 8.7, 7.9,
        8.8, 8.0, 7.7, 7.4,
        7.9, 6.9, 6.4, 6.2,
        6.6, 5.6, 5.6, 5.1
    ],
    'Selic': [
        14.25, 14.25, 14.25, 14.25, 13.75,
        12.25, 10.25, 8.25, 7.00,
        6.50, 6.50, 6.50, 6.50,
        6.50, 6.50, 5.50, 4.50,
        3.75, 2.25, 2.00, 2.00,
        2.75, 4.25, 5.75, 9.25,
        11.75, 13.25, 13.75, 13.75,
        13.75, 13.75, 12.75, 11.75,
        10.75, 10.50, 10.50, 11.25,
        13.25, 14.25, 15.00, 15.00
    ]
}

df_macro = pd.DataFrame(macro_data)
df_macro['Data'] = pd.to_datetime(df_macro['Data'])

# 3. Mesclar com o painel principal
df = df.merge(df_macro, on='Data', how='left')

# Preencher possiveis gaps macro (se houver datas proximas)
df['Desemprego'] = df.groupby('Instituicao')['Desemprego'].ffill().bfill()
df['Selic'] = df.groupby('Instituicao')['Selic'].ffill().bfill()

print(f"Dados macro carregados. Amostra:\n{df_macro.tail(5)}")

# 4. Preparar Variáveis NPL dinâmicas (Growth e Vol) - Precisamos delas para o modelo completo
df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())

# 5. Criar Lags
LAG = 1
core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread']
new_features = ['Desemprego', 'Selic', 'NPL_Volatility_8Q']

lag_cols = []
for f in core_features + new_features:
    col_name = f'{f}_lag{LAG}'
    df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
    lag_cols.append(col_name)

# Critério de Estresse P90
threshold_p90 = df['NPL'].quantile(0.90)
df['Estresse_Alto_P90'] = (df['NPL'] > threshold_p90).astype(int)

# 6. Interação
df[f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'] = df[f'RWA_Operacional_lag{LAG}'] * df[f'Alavancagem_lag{LAG}']
interaction_col = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'

df_clean = df.dropna(subset=lag_cols + [interaction_col, 'Estresse_Alto_P90']).copy()
print(f"\nObservacoes apos lags e limpeza: {len(df_clean)}")

# 7. MODELAGEM
y = df_clean['Estresse_Alto_P90']
X = df_clean[lag_cols + [interaction_col]]

# Normalizar
X_scaled = (X - X.mean()) / X.std()
X_scaled = sm.add_constant(X_scaled)

print(f"\n{'='*100}")
print(f"AJUSTANDO MODELO COM DESEMPREGO E SELIC")
print(f"{'='*100}")

model = sm.Logit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=True)
print(model.summary())

# 8. Avaliação
y_prob = model.predict(X_scaled)
auc = roc_auc_score(y, y_prob)
threshold_decision = 0.175
y_pred = (y_prob > threshold_decision).astype(int)

print(f"\nAUC-ROC Final: {auc:.4f}")
print(f"Pseudo R2: {model.prsquared:.4f}")
print("\nRelatorio de Classificacao:")
print(classification_report(y, y_pred))

# 9. Verificar VIF para ver se Selic/Spread/Desemprego conflitam
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])][1:] # pula constante

print("\n" + "="*50)
print("ANALISE DE MULTICOLINEARIDADE (VIF)")
print("="*50)
print(vif_data.sort_values('VIF', ascending=False).to_string(index=False))

# 10. Salvar Resultados
df_clean['Prob_Estresse'] = y_prob
df_clean['Score_Robustez'] = -np.log(y_prob / (1-y_prob + 1e-10))

# Ranking Final
ranking = df_clean.groupby('Instituicao')['Score_Robustez'].mean().reset_index()
ranking.columns = ['Instituicao', 'Score_Robustez']
ranking.sort_values('Score_Robustez', ascending=False, inplace=True)

df_clean.to_csv(r'd:\projeto_robustez_bancaria\dados\processados\painel_com_macros_final.csv', index=False)
ranking.to_csv(r'd:\projeto_robustez_bancaria\dados\processados\ranking_com_macros_final.csv', index=False)

print("\nArquivos salvos: dados/processados/painel_com_macros_final.csv e ranking_com_macros_final.csv")
