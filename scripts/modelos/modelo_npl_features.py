import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("MODELO FINAL COM NPL GROWTH E VOLATILIDADE")
print("="*100)

# 1. Carregar dados
painel_path = 'd:/projeto_robustez_bancaria/dados/brutos/painel_final.csv'
df = pd.read_csv(painel_path)

cols_to_fix = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'NPL']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
df.dropna(subset=['Data', 'NPL'], inplace=True)
df.sort_values(['Instituicao', 'Data'], inplace=True)

# 2. Definir threshold P90
threshold_p90 = df['NPL'].quantile(0.90)
df['Estresse_Alto_P90'] = (df['NPL'] > threshold_p90).astype(int)

print(f"\nThreshold P90: {threshold_p90*100:.2f}%")
print(f"Observacoes totais: {len(df)}")

# 3. CRIAR FEATURES DE NPL GROWTH E VOLATILIDADE
print(f"\n{'='*100}")
print(f"CRIANDO FEATURES DE NPL")
print(f"{'='*100}")

# NPL Growth (variação percentual trimestre a trimestre)
df['NPL_Growth_QoQ'] = df.groupby('Instituicao')['NPL'].pct_change()

# NPL Growth (variação percentual ano a ano - 4 trimestres)
df['NPL_Growth_YoY'] = df.groupby('Instituicao')['NPL'].pct_change(periods=4)

# NPL Volatilidade (desvio padrão móvel de 4 trimestres)
df['NPL_Volatility_4Q'] = df.groupby('Instituicao')['NPL'].transform(
    lambda x: x.rolling(window=4, min_periods=2).std()
)

# NPL Volatilidade (desvio padrão móvel de 8 trimestres - 2 anos)
df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(
    lambda x: x.rolling(window=8, min_periods=4).std()
)

# NPL Delta (diferença absoluta trimestre a trimestre)
df['NPL_Delta'] = df.groupby('Instituicao')['NPL'].diff()

# NPL Aceleração (segunda derivada - mudança na taxa de crescimento)
df['NPL_Acceleration'] = df.groupby('Instituicao')['NPL_Growth_QoQ'].diff()

print("\nFeatures de NPL criadas:")
print("1. NPL_Growth_QoQ: Crescimento trimestral (%)")
print("2. NPL_Growth_YoY: Crescimento anual (%)")
print("3. NPL_Volatility_4Q: Volatilidade 4 trimestres")
print("4. NPL_Volatility_8Q: Volatilidade 8 trimestres")
print("5. NPL_Delta: Variacao absoluta trimestral")
print("6. NPL_Acceleration: Aceleracao do crescimento")

# Estatísticas descritivas
print(f"\n{'='*100}")
print(f"ESTATISTICAS DESCRITIVAS")
print(f"{'='*100}")

npl_features_stats = df[['NPL_Growth_QoQ', 'NPL_Growth_YoY', 'NPL_Volatility_4Q', 
                          'NPL_Volatility_8Q', 'NPL_Delta', 'NPL_Acceleration']].describe()
print(npl_features_stats)

# 4. Criar lags das variáveis originais (SEM IPCA)
LAG = 4
features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread']
lag_cols = []
for f in features:
    col_name = f'{f}_lag{LAG}'
    df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
    lag_cols.append(col_name)

# Criar lags das features de NPL
npl_features = ['NPL_Growth_QoQ', 'NPL_Growth_YoY', 'NPL_Volatility_4Q', 'NPL_Volatility_8Q', 'NPL_Delta', 'NPL_Acceleration']
for f in npl_features:
    col_name = f'{f}_lag{LAG}'
    df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
    lag_cols.append(col_name)

# Criar interação RWA_Operacional x Alavancagem
df['RWA_Operacional_lag4_x_Alavancagem_lag4'] = df['RWA_Operacional_lag4'] * df['Alavancagem_lag4']

df_clean = df.dropna(subset=lag_cols + ['Estresse_Alto_P90']).copy()

print(f"\nObservacoes apos lags e features NPL: {len(df_clean)}")

# 5. TESTAR CADA FEATURE DE NPL INDIVIDUALMENTE
print(f"\n{'='*100}")
print(f"TESTANDO FEATURES DE NPL INDIVIDUALMENTE")
print(f"{'='*100}")

y = df_clean['Estresse_Alto_P90']

# Modelo base (sem features de NPL)
base_features = [c for c in lag_cols if 'NPL' not in c]
X_base = df_clean[base_features + ['RWA_Operacional_lag4_x_Alavancagem_lag4']]
X_base_scaled = (X_base - X_base.mean()) / X_base.std()
X_base_scaled = sm.add_constant(X_base_scaled)

model_base = sm.Logit(y, X_base_scaled).fit(disp=0)
auc_base = roc_auc_score(y, model_base.predict(X_base_scaled))

print(f"\nModelo BASE (sem NPL features):")
print(f"  AUC-ROC: {auc_base:.4f}")
print(f"  Pseudo R2: {model_base.prsquared:.4f}")

# Testar cada feature de NPL
npl_feature_lags = [f'{f}_lag{LAG}' for f in npl_features]
results = []

for npl_feat in npl_feature_lags:
    try:
        X_test = df_clean[base_features + ['RWA_Operacional_lag4_x_Alavancagem_lag4', npl_feat]]
        X_test_scaled = (X_test - X_test.mean()) / X_test.std()
        X_test_scaled = sm.add_constant(X_test_scaled)
        
        model_test = sm.Logit(y, X_test_scaled).fit(disp=0, maxiter=1000)
        auc_test = roc_auc_score(y, model_test.predict(X_test_scaled))
        
        # Teste de razão de verossimilhança
        from scipy import stats
        lr_stat = 2 * (model_test.llf - model_base.llf)
        p_value = stats.chi2.sf(lr_stat, 1)
        
        coef = model_test.params[npl_feat]
        pval = model_test.pvalues[npl_feat]
        
        results.append({
            'Feature': npl_feat.replace('_lag4', ''),
            'AUC': auc_test,
            'Delta_AUC': auc_test - auc_base,
            'Pseudo_R2': model_test.prsquared,
            'Delta_R2': model_test.prsquared - model_base.prsquared,
            'Coeficiente': coef,
            'P-valor': pval,
            'LR_Test_P-valor': p_value,
            'Significativo': 'SIM' if p_value < 0.05 else 'NAO'
        })
    except Exception as e:
        results.append({
            'Feature': npl_feat.replace('_lag4', ''),
            'AUC': np.nan,
            'Delta_AUC': np.nan,
            'Pseudo_R2': np.nan,
            'Delta_R2': np.nan,
            'Coeficiente': np.nan,
            'P-valor': np.nan,
            'LR_Test_P-valor': np.nan,
            'Significativo': f'ERRO: {str(e)[:30]}'
        })

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Delta_AUC', ascending=False)

print(f"\n{df_results.to_string(index=False)}")

# 6. MODELO FINAL COM MELHORES FEATURES DE NPL
print(f"\n{'='*100}")
print(f"MODELO FINAL COM MELHORES NPL FEATURES")
print(f"{'='*100}")

# Selecionar features significativas
significant_npl = df_results[df_results['Significativo'] == 'SIM']
print(f"\nFeatures NPL significativas: {len(significant_npl)}")

if len(significant_npl) > 0:
    print(significant_npl[['Feature', 'Delta_AUC', 'P-valor']].to_string(index=False))
    
    # Criar modelo com as 2 melhores features de NPL
    best_npl_features = [f"{row['Feature']}_lag4" for idx, row in significant_npl.head(2).iterrows()]
    
    final_features = base_features + ['RWA_Operacional_lag4_x_Alavancagem_lag4'] + best_npl_features
    
    X_final = df_clean[final_features]
    X_final_scaled = (X_final - X_final.mean()) / X_final.std()
    X_final_scaled = sm.add_constant(X_final_scaled)
    
    model_final = sm.Logit(y, X_final_scaled).fit(method='bfgs', maxiter=1000, disp=True)
    
    print(f"\n{'='*100}")
    print(f"RESUMO DO MODELO FINAL")
    print(f"{'='*100}")
    print(model_final.summary())
    
    # Métricas
    threshold_decision = 0.175
    y_pred_prob = model_final.predict(X_final_scaled)
    y_pred_class = (y_pred_prob > threshold_decision).astype(int)
    
    auc_final = roc_auc_score(y, y_pred_prob)
    
    print(f"\n{'='*100}")
    print(f"METRICAS DE PERFORMANCE (Threshold = {threshold_decision})")
    print(f"{'='*100}")
    print(f"AUC-ROC: {auc_final:.4f} (Base: {auc_base:.4f}, Delta: +{auc_final - auc_base:.4f})")
    print(f"Pseudo R2: {model_final.prsquared:.4f} (Base: {model_base.prsquared:.4f}, Delta: +{model_final.prsquared - model_base.prsquared:.4f})")
    print(f"\nMatriz de Confusao:")
    print(confusion_matrix(y, y_pred_class))
    print(f"\nRelatorio de Classificacao:")
    print(classification_report(y, y_pred_class, target_names=['Normal', 'Estresse']))
    
    # Salvar resultados
    df_clean['Prob_Estresse_Final'] = y_pred_prob
    df_clean['Log_Odds_Estresse_Final'] = np.log(y_pred_prob / (1 - y_pred_prob + 1e-10))
    df_clean['Score_Robustez_Final'] = -df_clean['Log_Odds_Estresse_Final']
    
    # Ranking
    ranking = df_clean.groupby('Instituicao').agg({
        'Score_Robustez_Final': 'mean',
        'Prob_Estresse_Final': 'mean',
        'NPL': 'mean'
    }).reset_index()
    ranking.columns = ['Instituicao', 'Score_Robustez', 'Prob_Estresse_Media', 'NPL_Medio']
    ranking.sort_values('Score_Robustez', ascending=False, inplace=True)
    
    print(f"\n{'='*100}")
    print(f"RANKING DE ROBUSTEZ (Top 20)")
    print(f"{'='*100}")
    print(ranking.head(20).to_string(index=False))
    
    # Salvar
    df_clean.to_csv('d:/projeto_robustez_bancaria/dados/processados/modelo_npl_features_painel.csv', index=False)
    ranking.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/modelo_npl_features_ranking.csv', index=False)
    df_results.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/npl_features_analysis.csv', index=False)
    
    print(f"\n{'='*100}")
    print(f"ARQUIVOS SALVOS")
    print(f"{'='*100}")
    print("- modelo_npl_features_painel.csv")
    print("- modelo_npl_features_ranking.csv")
    print("- npl_features_analysis.csv")
    
else:
    print("\nNenhuma feature de NPL foi significativa!")

# 7. COMPARAÇÃO FINAL
print(f"\n{'='*100}")
print(f"COMPARACAO: MODELO BASE vs MODELO COM NPL FEATURES")
print(f"{'='*100}")

if len(significant_npl) > 0:
    comparison = pd.DataFrame({
        'Metrica': ['AUC-ROC', 'Pseudo R2', 'Num Variaveis'],
        'Modelo Base': [auc_base, model_base.prsquared, len(base_features) + 1],
        'Modelo com NPL Features': [auc_final, model_final.prsquared, len(final_features)]
    })
    print(comparison.to_string(index=False))
    
    print(f"\n{'='*100}")
    print(f"CONCLUSAO")
    print(f"{'='*100}")
    print(f"""
Adicionar NPL Growth e Volatilidade:
- Melhoria AUC: +{(auc_final - auc_base):.4f} ({((auc_final - auc_base)/auc_base)*100:.2f}%)
- Melhoria Pseudo R2: +{(model_final.prsquared - model_base.prsquared):.4f}
- Features NPL significativas: {len(significant_npl)}
- Melhores features: {', '.join([row['Feature'] for idx, row in significant_npl.head(2).iterrows()])}

INTERPRETACAO:
NPL Growth e Volatilidade capturam a DINAMICA TEMPORAL do risco de credito,
complementando as variaveis de nivel (RWA, Capital, etc).
""")

print(f"\n{'='*100}")
print(f"MODELO COM NPL FEATURES CONCLUIDO!")
print(f"{'='*100}")
