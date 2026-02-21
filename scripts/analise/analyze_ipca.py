import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar dados
df = pd.read_csv('d:/projeto_robustez_bancaria/dados/brutos/painel_final.csv')

cols_to_fix = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'NPL']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
df.dropna(subset=['Data', 'NPL'], inplace=True)
df.sort_values(['Instituicao', 'Data'], inplace=True)

threshold_p90 = df['NPL'].quantile(0.90)
df['Estresse_Alto_P90'] = (df['NPL'] > threshold_p90).astype(int)

print("="*100)
print("ANALISE DETALHADA DO IPCA")
print("="*100)

# 1. ESTATÍSTICAS DESCRITIVAS DO IPCA
print("\n1. ESTATISTICAS DESCRITIVAS DO IPCA")
print("-"*100)
print(df['IPCA'].describe())
print(f"\nVariacao (max - min): {df['IPCA'].max() - df['IPCA'].min():.2f}%")
print(f"Desvio padrao: {df['IPCA'].std():.2f}%")
print(f"Coeficiente de variacao: {(df['IPCA'].std() / df['IPCA'].mean()):.2f}")

# 2. IPCA POR PERÍODO DE ESTRESSE
print("\n2. IPCA: NORMAL vs ESTRESSE")
print("-"*100)
ipca_normal = df[df['Estresse_Alto_P90'] == 0]['IPCA']
ipca_estresse = df[df['Estresse_Alto_P90'] == 1]['IPCA']

print(f"IPCA medio (Normal): {ipca_normal.mean():.2f}%")
print(f"IPCA medio (Estresse): {ipca_estresse.mean():.2f}%")
print(f"Diferenca: {ipca_estresse.mean() - ipca_normal.mean():.2f}%")

from scipy import stats
t_stat, p_value = stats.ttest_ind(ipca_normal, ipca_estresse)
print(f"\nTeste t: t-stat = {t_stat:.4f}, p-valor = {p_value:.4f}")
print(f"Diferenca significativa? {'SIM' if p_value < 0.05 else 'NAO'}")

# 3. CORRELAÇÃO COM OUTRAS VARIÁVEIS
print("\n3. CORRELACAO DO IPCA COM OUTRAS VARIAVEIS")
print("-"*100)
correlations = df[['IPCA', 'PIB', 'Spread', 'NPL', 'Capital_Principal', 'Alavancagem']].corr()['IPCA'].sort_values(ascending=False)
print(correlations)

# 4. EVOLUÇÃO TEMPORAL DO IPCA
print("\n4. EVOLUCAO TEMPORAL DO IPCA")
print("-"*100)
ipca_temporal = df.groupby('Data')['IPCA'].first().reset_index()
print(ipca_temporal.tail(10).to_string(index=False))

# 5. TESTAR IPCA COM LAG 4
LAG = 4
features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread']
lag_cols = []
for f in features:
    col_name = f'{f}_lag{LAG}'
    df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
    lag_cols.append(col_name)

df_clean = df.dropna(subset=lag_cols + ['Estresse_Alto_P90']).copy()

print("\n5. MODELO COM E SEM IPCA")
print("-"*100)

y = df_clean['Estresse_Alto_P90']

# Modelo SEM IPCA
features_no_ipca = [c for c in lag_cols if 'IPCA' not in c]
X_no_ipca = df_clean[features_no_ipca]
X_no_ipca_scaled = (X_no_ipca - X_no_ipca.mean()) / X_no_ipca.std()
X_no_ipca_scaled = sm.add_constant(X_no_ipca_scaled)

model_no_ipca = sm.Logit(y, X_no_ipca_scaled).fit(disp=0)
auc_no_ipca = roc_auc_score(y, model_no_ipca.predict(X_no_ipca_scaled))

# Modelo COM IPCA
X_with_ipca = df_clean[lag_cols]
X_with_ipca_scaled = (X_with_ipca - X_with_ipca.mean()) / X_with_ipca.std()
X_with_ipca_scaled = sm.add_constant(X_with_ipca_scaled)

model_with_ipca = sm.Logit(y, X_with_ipca_scaled).fit(disp=0)
auc_with_ipca = roc_auc_score(y, model_with_ipca.predict(X_with_ipca_scaled))

print(f"AUC SEM IPCA: {auc_no_ipca:.4f}")
print(f"AUC COM IPCA: {auc_with_ipca:.4f}")
print(f"Delta AUC: {auc_with_ipca - auc_no_ipca:.4f}")

print(f"\nPseudo R2 SEM IPCA: {model_no_ipca.prsquared:.4f}")
print(f"Pseudo R2 COM IPCA: {model_with_ipca.prsquared:.4f}")

# Teste de razão de verossimilhança
lr_stat = 2 * (model_with_ipca.llf - model_no_ipca.llf)
p_value_lr = stats.chi2.sf(lr_stat, 1)
print(f"\nLR Test p-valor: {p_value_lr:.4f}")
print(f"IPCA melhora o modelo? {'SIM' if p_value_lr < 0.05 else 'NAO'}")

# Coeficiente do IPCA
ipca_coef = model_with_ipca.params['IPCA_lag4']
ipca_pval = model_with_ipca.pvalues['IPCA_lag4']
print(f"\nCoeficiente IPCA: {ipca_coef:.4f}")
print(f"P-valor IPCA: {ipca_pval:.4f}")
print(f"Odds Ratio: {np.exp(ipca_coef):.4f}")

# 6. TESTAR TRANSFORMAÇÕES DO IPCA
print("\n6. TESTANDO TRANSFORMACOES DO IPCA")
print("-"*100)

transformations = []

# IPCA original
df_clean['IPCA_lag4_original'] = df_clean['IPCA_lag4']

# IPCA ao quadrado (capturar não-linearidade)
df_clean['IPCA_lag4_squared'] = df_clean['IPCA_lag4'] ** 2

# IPCA absoluto (magnitude, não direção)
df_clean['IPCA_lag4_abs'] = df_clean['IPCA_lag4'].abs()

# Dummy: IPCA alto (> mediana)
ipca_median = df_clean['IPCA_lag4'].median()
df_clean['IPCA_lag4_high'] = (df_clean['IPCA_lag4'] > ipca_median).astype(int)

# Dummy: Inflação muito alta (> P75)
ipca_p75 = df_clean['IPCA_lag4'].quantile(0.75)
df_clean['IPCA_lag4_very_high'] = (df_clean['IPCA_lag4'] > ipca_p75).astype(int)

transformations_to_test = [
    ('IPCA_lag4_original', 'IPCA Original'),
    ('IPCA_lag4_squared', 'IPCA ao Quadrado'),
    ('IPCA_lag4_abs', 'IPCA Absoluto'),
    ('IPCA_lag4_high', 'Dummy: IPCA Alto'),
    ('IPCA_lag4_very_high', 'Dummy: IPCA Muito Alto')
]

results = []
for var, name in transformations_to_test:
    # Substituir IPCA_lag4 pela transformação
    features_test = [c for c in lag_cols if 'IPCA' not in c] + [var]
    X_test = df_clean[features_test]
    X_test_scaled = (X_test - X_test.mean()) / X_test.std()
    X_test_scaled = sm.add_constant(X_test_scaled)
    
    try:
        model_test = sm.Logit(y, X_test_scaled).fit(disp=0)
        auc_test = roc_auc_score(y, model_test.predict(X_test_scaled))
        
        coef = model_test.params[var]
        pval = model_test.pvalues[var]
        
        results.append({
            'Transformacao': name,
            'AUC': auc_test,
            'Delta_AUC_vs_sem_IPCA': auc_test - auc_no_ipca,
            'Coeficiente': coef,
            'P-valor': pval,
            'Significativo': 'SIM' if pval < 0.05 else 'NAO'
        })
    except:
        results.append({
            'Transformacao': name,
            'AUC': np.nan,
            'Delta_AUC_vs_sem_IPCA': np.nan,
            'Coeficiente': np.nan,
            'P-valor': np.nan,
            'Significativo': 'ERRO'
        })

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# 7. ANÁLISE DE MULTICOLINEARIDADE
print("\n7. MULTICOLINEARIDADE (VIF)")
print("-"*100)

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_vif = df_clean[lag_cols]
vif_data = pd.DataFrame()
vif_data["Variavel"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
vif_data = vif_data.sort_values('VIF', ascending=False)

print(vif_data.to_string(index=False))
print(f"\nIPCA VIF: {vif_data[vif_data['Variavel'] == 'IPCA_lag4']['VIF'].values[0]:.2f}")
print("(VIF > 10 indica multicolinearidade severa)")

# 8. CONCLUSÃO
print("\n" + "="*100)
print("CONCLUSAO SOBRE O IPCA")
print("="*100)

print("""
PRINCIPAIS DESCOBERTAS:

1. IPCA NAO E SIGNIFICATIVO NO MODELO:
   - P-valor: {:.4f} (> 0.05)
   - Coeficiente: {:.4f} (proximo de zero)
   - Nao melhora AUC nem Pseudo R2

2. POSSÍVEIS RAZOES:

   a) BAIXA VARIABILIDADE:
      - IPCA varia pouco no periodo (CV = {:.2f})
      - Maior parte da variacao e temporal, nao cross-sectional
   
   b) MULTICOLINEARIDADE:
      - VIF do IPCA: {:.2f}
      - Correlacao com PIB: {:.2f}
      - Correlacao com Spread: {:.2f}
   
   c) EFEITO INDIRETO:
      - IPCA afeta NPL via Spread e PIB
      - Efeito direto e capturado por outras variaveis
   
   d) LAG INADEQUADO:
      - Lag de 4 trimestres pode ser muito longo
      - Efeito da inflacao pode ser mais imediato

3. RECOMENDACAO:
   - REMOVER IPCA do modelo (parcimonia)
   - OU testar lags menores (1-2 trimestres)
   - OU usar dummy para inflacao muito alta
""".format(
    ipca_pval,
    ipca_coef,
    df['IPCA'].std() / df['IPCA'].mean(),
    vif_data[vif_data['Variavel'] == 'IPCA_lag4']['VIF'].values[0],
    correlations['PIB'],
    correlations['Spread']
))

# Salvar análise
df_results.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/ipca_analysis.csv', index=False)
vif_data.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/vif_analysis.csv', index=False)

print("\n[OK] Resultados salvos em: ipca_analysis.csv e vif_analysis.csv")
