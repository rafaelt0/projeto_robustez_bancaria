import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def run_lag_optimization():
    print("="*80)
    print("OTIMIZACAO DE LAGS VIA AIC E BIC (AMOSTRA COMUM)")
    print("="*80)

    # 1. Carregar Dados
    raw_path = 'dados/brutos/painel_final.csv'
    if not os.path.exists(raw_path):
        print("Erro: painel_final.csv nao encontrado.")
        return

    df = pd.read_csv(raw_path)
    df['Data'] = pd.to_datetime(df['Data'])
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    features = [
        'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 
        'Capital_Principal', 'Alavancagem', 
        'PIB', 'Spread', 'Desemprego'
    ]

    # 2. Criar todos os lags primeiro para definir amostra comum
    all_lag_cols = []
    for lag in range(1, 9):
        for f in features:
            col_name = f'{f}_lag{lag}'
            df[col_name] = df.groupby('Instituicao')[f].shift(lag)
            all_lag_cols.append(col_name)
        df[f'Interaction_lag{lag}'] = df[f'RWA_Operacional_lag{lag}'] * df[f'Alavancagem_lag{lag}']
        all_lag_cols.append(f'Interaction_lag{lag}')

    # Definir Target (P90)
    threshold_p90 = df['NPL'].quantile(0.90)
    df['Target'] = (df['NPL'] > threshold_p90).astype(int)

    # Amostra Comum: Todas as observações que têm dados para o Lag 8
    df_common = df.dropna(subset=all_lag_cols + ['Target']).copy()
    print(f"Amostra comum para todos os lags: {len(df_common)} observações.")

    results = []

    # 3. Iterar Lags
    for lag in range(1, 9):
        print(f"Testando Lag {lag} na amostra comum...")
        lag_cols = [f'{f}_lag{lag}' for f in features]
        inter_col = f'Interaction_lag{lag}'
        
        X = df_common[lag_cols + [inter_col]]
        # Padronização (Z-score)
        X = (X - X.mean()) / X.std()
        X = sm.add_constant(X)
        y = df_common['Target']

        # Pesos para balanceamento
        counts = y.value_counts()
        weight_stress = counts[0] / counts[1]
        weights = y.apply(lambda x: weight_stress if x == 1 else 1.0)

        try:
            model = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=weights).fit()
            results.append({
                'Lag': lag,
                'AIC': model.aic,
                'BIC': model.bic_llf,
                'LLF': model.llf
            })
        except Exception as e:
            print(f"  Erro no lag {lag}: {e}")

    # 4. Analisar Resultados
    if not results:
        print("Erro: Nenhum modelo convergiu.")
        return
        
    df_res = pd.DataFrame(results)
    print("\nRESULTADOS (AMOSTRA FIXA):")
    print(df_res.to_string(index=False))

    best_aic = df_res.loc[df_res['AIC'].idxmin(), 'Lag']
    best_bic = df_res.loc[df_res['BIC'].idxmin(), 'Lag']

    print(f"\nMelhor Lag (AIC): {best_aic}")
    print(f"Melhor Lag (BIC): {best_bic}")

    # 5. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_res['Lag'], df_res['AIC'], label='AIC', marker='o')
    plt.plot(df_res['Lag'], df_res['BIC'], label='BIC', marker='x')
    plt.title('Seleção de Lag (Amostra Fixa)')
    plt.xlabel('Lag (Trimestres)')
    plt.ylabel('Valor do Critério')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('resultados/lags_optimization_fixed.png')
    print("\nGráfico salvo em: resultados/lags_optimization_fixed.png")

if __name__ == "__main__":
    run_lag_optimization()
