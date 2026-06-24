import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import os
import sys

def run_final_model():
    # 1. Configuracoes e Lags (Otimizados via AIC/BIC)
    LAG = 1
    DECISION_THRESHOLD = 0.6  # Threshold de decisao final
    
    print(f"\n{'='*100}")
    print(f"MODELO LOGIT FINAL RECOMENDADO (LAG {LAG} - OTIMIZADO)")
    print(f"{'='*100}")

    # 2. Carregar Dados
    raw_path = 'dados/brutos/painel_final.csv'
    if not os.path.exists(raw_path):
        print(f"Erro: Arquivo {raw_path} nao encontrado.")
        return

    df = pd.read_csv(raw_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    # Validacao Out-of-Time: Treino ate 2021, Teste desde 2022
    split_date = pd.to_datetime('2022-01-01')
    df_train = df[df['Data'] < split_date].copy()
    df_test = df[df['Data'] >= split_date].copy()

    # Definir threshold P90 BASEADO APENAS NO TREINO
    threshold_p90 = df_train['NPL'].quantile(0.90)
    df['Estresse_Alto_P90'] = (df['NPL'] > threshold_p90).astype(int)

    print(f"Periodo de Treino: {df_train['Data'].min().year} - {df_train['Data'].max().year}")
    print(f"Periodo de Teste: {df_test['Data'].min().year} - {df_test['Data'].max().year}")
    print(f"Threshold P90 (do Treino): {threshold_p90*100:.2f}%")
    
    # 3. Criar lags dinamicos
    features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego']
    lag_cols = []
    for f in features:
        col_name = f'{f}_lag{LAG}'
        df[col_name] = df.groupby('Instituicao')[f].shift(LAG)
        lag_cols.append(col_name)

    # 4. Termo de Interação Dinamico
    inter_name = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'
    df[inter_name] = df[f'RWA_Operacional_lag{LAG}'] * df[f'Alavancagem_lag{LAG}']
    
    features_final = lag_cols + [inter_name]

    # Limpeza e Scaling
    df_model = df.dropna(subset=features_final + ['Estresse_Alto_P90', 'Instituicao']).copy()
    
    # Separar Treino e Teste ja nos lags
    train = df_model[df_model['Data'] < split_date].copy()
    test = df_model[df_model['Data'] >= split_date].copy()

    X_train = train[features_final]
    y_train = train['Estresse_Alto_P90']
    X_test = test[features_final]
    y_test = test['Estresse_Alto_P90']

    # Robust Scaling (Z-score)
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    # Logging para Stress Test
    print(f"\nPARAMETROS_SCALING_MEAN: {X_mean.to_dict()}")
    print(f"PARAMETROS_SCALING_STD: {X_std.to_dict()}\n")

    # 5. Estimacao com Pesos (Balanceamento)
    X_train_const = sm.add_constant(X_train_scaled)
    counts = y_train.value_counts()
    weight_stress = counts[0] / counts[1]
    weights = y_train.apply(lambda x: weight_stress if x == 1 else 1.0)

    model_final = sm.GLM(y_train, X_train_const, family=sm.families.Binomial(), var_weights=weights).fit()
    print(model_final.summary())

    # 6. Performance
    X_test_const = sm.add_constant(X_test_scaled)
    probs_train = model_final.predict(X_train_const)
    probs_test = model_final.predict(X_test_const)

    auc_train = roc_auc_score(y_train, probs_train)
    auc_test = roc_auc_score(y_test, probs_test)
    
    preds_test = (probs_test > DECISION_THRESHOLD).astype(int)
    recall_test = recall_score(y_test, preds_test)
    prec_test = precision_score(y_test, preds_test, zero_division=0)

    print(f"\nPERFORMANCE (Lag {LAG}):")
    print(f"AUC TREINO: {auc_train:.4f}")
    print(f"AUC TESTE (OOT): {auc_test:.4f}")
    print(f"RECALL TESTE: {recall_test:.1%}")

    # 7. Salvar e Encerrar (Omitindo logs longos para brevidade)
    os.makedirs('resultados/relatorios', exist_ok=True)
    # Salvar coeficientes para tabelas latex
    coef_df = pd.DataFrame({
        'Variavel': model_final.params.index,
        'Coeficiente': model_final.params.values,
        'P-valor': model_final.pvalues.values,
        'StdErr': model_final.bse.values
    })
    coef_df.to_csv('resultados/relatorios/modelo_final_statistics.csv', index=False)
    
    # Exportar painel predito
    df_model['Prob_Estresse'] = model_final.predict(sm.add_constant((df_model[features_final] - X_mean) / X_std))
    df_model.to_csv('dados/processados/modelo_final_painel.csv', index=False)

if __name__ == "__main__":
    run_final_model()
