import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from itertools import combinations

# 1. Carregar dados
painel_path = 'd:/projeto_robustez_bancaria/dados/brutos/painel_final.csv'
df = pd.read_csv(painel_path)

# Garantir tipos numéricos
cols_to_fix = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'NPL']
for col in cols_to_fix:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
df.dropna(subset=['Data', 'NPL'], inplace=True)
df.sort_values(['Instituicao', 'Data'], inplace=True)

# 2. Definir estresse P90
threshold_p90 = df['NPL'].quantile(0.90)
df['Estresse_Alto_P90'] = (df['NPL'] > threshold_p90).astype(int)

# 3. Criar Lags
LAG = 4
features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread']
lag_cols = []
for f in features:
    col_name = f'{f}_lag{LAG}'
    df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
    lag_cols.append(col_name)

df_clean = df.dropna(subset=lag_cols + ['Estresse_Alto_P90']).copy()

print("="*100)
print("TESTE DE INTERACOES ENTRE VARIAVEIS")
print("="*100)
print(f"\nObservacoes: {len(df_clean)}")
print(f"Threshold P90: {threshold_p90*100:.2f}%")

# 4. MODELO BASE (sem interações)
print("\n" + "="*100)
print("MODELO BASE (SEM INTERACOES)")
print("="*100)

X_base = df_clean[lag_cols]
y = df_clean['Estresse_Alto_P90']

X_base_scaled = (X_base - X_base.mean()) / X_base.std()
X_base_scaled = sm.add_constant(X_base_scaled)

model_base = sm.Logit(y, X_base_scaled).fit(disp=0)
auc_base = roc_auc_score(y, model_base.predict(X_base_scaled))

print(f"AUC-ROC: {auc_base:.4f}")
print(f"Pseudo R2: {model_base.prsquared:.4f}")
print(f"Log-Likelihood: {model_base.llf:.2f}")

# 5. TESTAR INTERAÇÕES TEORICAMENTE RELEVANTES
print("\n" + "="*100)
print("TESTANDO INTERACOES TEORICAMENTE RELEVANTES")
print("="*100)

# Interações a testar (baseadas em teoria econômica)
interactions = [
    ('Capital_Principal_lag4', 'Alavancagem_lag4', 'Capital x Alavancagem'),
    ('Capital_Principal_lag4', 'PIB_lag4', 'Capital x Ciclo Economico'),
    ('Alavancagem_lag4', 'Spread_lag4', 'Alavancagem x Custo de Funding'),
    ('RWA_Credito_lag4', 'PIB_lag4', 'Risco de Credito x Ciclo Economico'),
    ('Capital_Principal_lag4', 'RWA_Credito_lag4', 'Capital x Risco de Credito'),
    ('Spread_lag4', 'IPCA_lag4', 'Spread x Inflacao'),
    ('RWA_Operacional_lag4', 'Alavancagem_lag4', 'Risco Operacional x Alavancagem'),
    ('Capital_Principal_lag4', 'IPCA_lag4', 'Capital x Inflacao')
]

interaction_results = []

for var1, var2, name in interactions:
    # Criar termo de interação
    df_clean[f'{var1}_x_{var2}'] = df_clean[var1] * df_clean[var2]
    
    # Preparar features com interação
    X_int = df_clean[lag_cols + [f'{var1}_x_{var2}']]
    X_int_scaled = (X_int - X_int.mean()) / X_int.std()
    X_int_scaled = sm.add_constant(X_int_scaled)
    
    # Ajustar modelo
    try:
        model_int = sm.Logit(y, X_int_scaled).fit(disp=0, maxiter=1000)
        auc_int = roc_auc_score(y, model_int.predict(X_int_scaled))
        
        # Teste de razão de verossimilhança
        lr_stat = 2 * (model_int.llf - model_base.llf)
        from scipy import stats
        p_value = stats.chi2.sf(lr_stat, 1)  # 1 grau de liberdade (1 variável adicionada)
        
        # Coeficiente da interação
        interaction_coef = model_int.params[f'{var1}_x_{var2}']
        interaction_pval = model_int.pvalues[f'{var1}_x_{var2}']
        
        interaction_results.append({
            'Interacao': name,
            'AUC': auc_int,
            'Delta_AUC': auc_int - auc_base,
            'Pseudo_R2': model_int.prsquared,
            'Delta_R2': model_int.prsquared - model_base.prsquared,
            'Coef_Interacao': interaction_coef,
            'P-valor_Interacao': interaction_pval,
            'LR_Test_P-valor': p_value,
            'Significativo': 'SIM' if p_value < 0.05 else 'NAO'
        })
    except:
        interaction_results.append({
            'Interacao': name,
            'AUC': np.nan,
            'Delta_AUC': np.nan,
            'Pseudo_R2': np.nan,
            'Delta_R2': np.nan,
            'Coef_Interacao': np.nan,
            'P-valor_Interacao': np.nan,
            'LR_Test_P-valor': np.nan,
            'Significativo': 'ERRO'
        })

df_interactions = pd.DataFrame(interaction_results)
df_interactions = df_interactions.sort_values('Delta_AUC', ascending=False)

print(df_interactions.to_string(index=False))

# 6. MODELO COM MELHOR INTERAÇÃO
print("\n" + "="*100)
print("MODELO COM MELHOR INTERACAO")
print("="*100)

best_interaction = df_interactions.iloc[0]
print(f"\nMelhor interacao: {best_interaction['Interacao']}")
print(f"Delta AUC: {best_interaction['Delta_AUC']:.4f}")
print(f"P-valor (LR Test): {best_interaction['LR_Test_P-valor']:.4f}")

if best_interaction['Delta_AUC'] > 0.001:  # Melhoria significativa
    # Recriar modelo com melhor interação
    var1, var2, _ = interactions[df_interactions.index[0]]
    df_clean[f'{var1}_x_{var2}'] = df_clean[var1] * df_clean[var2]
    
    X_best = df_clean[lag_cols + [f'{var1}_x_{var2}']]
    X_best_scaled = (X_best - X_best.mean()) / X_best.std()
    X_best_scaled = sm.add_constant(X_best_scaled)
    
    model_best = sm.Logit(y, X_best_scaled).fit(disp=0)
    
    print("\nResumo do modelo:")
    print(model_best.summary())
    
    # Métricas
    y_pred_prob = model_best.predict(X_best_scaled)
    y_pred_class = (y_pred_prob > 0.175).astype(int)
    
    auc_score = roc_auc_score(y, y_pred_prob)
    print(f"\n{'='*100}")
    print(f"METRICAS DE PERFORMANCE (Threshold = 0.175)")
    print(f"{'='*100}")
    print(f"AUC-ROC: {auc_score:.4f}")
    print(f"\nMatriz de Confusao:")
    print(confusion_matrix(y, y_pred_class))
    print(f"\nRelatorio de Classificacao:")
    print(classification_report(y, y_pred_class, target_names=['Normal', 'Estresse']))

# 7. TESTAR MODELO COM MÚLTIPLAS INTERAÇÕES SIGNIFICATIVAS
print("\n" + "="*100)
print("MODELO COM MULTIPLAS INTERACOES SIGNIFICATIVAS")
print("="*100)

significant_interactions = df_interactions[df_interactions['Significativo'] == 'SIM']
print(f"\nInteracoes significativas (p < 0.05): {len(significant_interactions)}")

if len(significant_interactions) > 0:
    print(significant_interactions[['Interacao', 'Delta_AUC', 'P-valor_Interacao']].to_string(index=False))
    
    # Criar modelo com todas as interações significativas
    interaction_cols = []
    for idx in significant_interactions.index:
        var1, var2, _ = interactions[idx]
        col_name = f'{var1}_x_{var2}'
        if col_name not in df_clean.columns:
            df_clean[col_name] = df_clean[var1] * df_clean[var2]
        interaction_cols.append(col_name)
    
    X_multi = df_clean[lag_cols + interaction_cols]
    X_multi_scaled = (X_multi - X_multi.mean()) / X_multi.std()
    X_multi_scaled = sm.add_constant(X_multi_scaled)
    
    try:
        model_multi = sm.Logit(y, X_multi_scaled).fit(disp=0, maxiter=1000)
        auc_multi = roc_auc_score(y, model_multi.predict(X_multi_scaled))
        
        print(f"\nAUC-ROC (modelo com multiplas interacoes): {auc_multi:.4f}")
        print(f"Delta AUC vs modelo base: {auc_multi - auc_base:.4f}")
        print(f"Pseudo R2: {model_multi.prsquared:.4f}")
        
        # Teste de razão de verossimilhança
        lr_stat = 2 * (model_multi.llf - model_base.llf)
        from scipy import stats
        p_value = stats.chi2.sf(lr_stat, len(interaction_cols))
        print(f"LR Test p-valor: {p_value:.4f}")
        
    except Exception as e:
        print(f"Erro ao ajustar modelo com multiplas interacoes: {e}")

# 8. RESUMO FINAL
print("\n" + "="*100)
print("RESUMO COMPARATIVO")
print("="*100)

summary = pd.DataFrame({
    'Modelo': ['Base (sem interacoes)', f'Com melhor interacao ({best_interaction["Interacao"]})'],
    'AUC-ROC': [auc_base, best_interaction['AUC']],
    'Pseudo R2': [model_base.prsquared, best_interaction['Pseudo_R2']],
    'Num Variaveis': [len(lag_cols), len(lag_cols) + 1]
})

print(summary.to_string(index=False))

# Salvar resultados
df_interactions.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/interaction_analysis.csv', index=False)
print("\n[OK] Resultados salvos em: interaction_analysis.csv")
