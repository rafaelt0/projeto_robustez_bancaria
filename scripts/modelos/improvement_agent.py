import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from utils_modeling import train_logit_model, prepare_data

# Configuração
DATA_PATH = Path(r"d:\projeto_robustez_bancaria\dados\brutos\painel_final.csv")
OUTPUT_DIR = Path(r"d:\projeto_robustez_bancaria\resultados\experiments")
EXPERIMENT_LOG = OUTPUT_DIR / "experiment_history.csv"

# Garantir que o diretório de saída exista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_experiment_cycle():
    print("Iniciando Ciclo do Agente de Melhoria de Modelo...")
    
    if not DATA_PATH.exists():
        print(f"Erro: Arquivo de dados não encontrado em {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Identificar colunas principais (sem lag)
    core_features = [
        'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 
        'Capital_Principal', 'Alavancagem', 'PIB', 'Spread',
        'Desemprego', 'Selic', 'NPL_Volatility_8Q'
    ]
    
    available_features = [f for f in core_features if f in df.columns]
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df = df.sort_values(['Instituicao', 'Data'])

    results_list = []

    # 1. EXPLORAÇÃO DE LAGS (T1 a T8)
    print(f"Experimentando com horizontes de Lag (1-8 trimestres)...")
    for lag in range(1, 9):
        df_clean, lag_cols = prepare_data(df, lag=lag, features_to_lag=available_features)
        res = train_logit_model(df_clean, lag_cols)
        
        if 'error' not in res:
            results_list.append({
                'timestamp': pd.Timestamp.now(),
                'experiment_type': 'lag_search',
                'lag': lag,
                'model_type': 'Logit',
                'features': ", ".join(lag_cols),
                'auc': res['auc'],
                'aic': res['aic'],
                'pseudo_r2': res['pseudo_r2'],
                'n_obs': res['n_obs']
            })
            print(f"Lag {lag}: AUC={res['auc']:.4f}")

    # 2. EXPLORAÇÃO DE INTERAÇÕES (no melhor lag descoberto)
    batch_df = pd.DataFrame(results_list)
    if not batch_df.empty:
        best_lag = int(batch_df.loc[batch_df['auc'].idxmax(), 'lag'])
    else:
        best_lag = 1

    print(f"Testando melhorias de interação no Lag {best_lag}...")
    df_clean, lag_cols = prepare_data(df, lag=best_lag, features_to_lag=available_features)
    
    interactions = [
        ('Capital_Principal', 'Selic'),
        ('Alavancagem', 'Desemprego'),
        ('RWA_Operacional', 'Alavancagem'),
        ('PIB', 'Capital_Principal'),
        ('Capital_Principal', 'NPL_Volatility_8Q'),
        ('Selic', 'Alavancagem'),
        ('IPCA', 'Spread'),
        ('Desemprego', 'PIB')
    ]

    for f1, f2 in interactions:
        l1, l2 = f"{f1}_lag{best_lag}", f"{f2}_lag{best_lag}"
        if l1 in df_clean.columns and l2 in df_clean.columns:
            df_curr = df_clean.copy()
            inter_name = f"{l1}_x_{l2}"
            df_curr[inter_name] = df_curr[l1] * df_curr[l2]
            
            res = train_logit_model(df_curr, lag_cols + [inter_name])
            if 'error' not in res:
                results_list.append({
                    'timestamp': pd.Timestamp.now(),
                    'experiment_type': f'interaction_{f1}_{f2}',
                    'lag': best_lag,
                    'model_type': 'Logit',
                    'features': ", ".join(lag_cols + [inter_name]),
                    'auc': res['auc'],
                    'aic': res['aic'],
                    'pseudo_r2': res['pseudo_r2'],
                    'n_obs': res['n_obs']
                })
                print(f"Interação {f1}x{f2}: AUC={res['auc']:.4f}")

    # 3. COMPARAÇÃO DE MODELOS (Logit vs Probit)
    print("Comparando Logit vs Probit...")
    df_clean, lag_cols = prepare_data(df, lag=best_lag, features_to_lag=available_features)
    X = df_clean[lag_cols]
    X_scaled = (X - X.mean()) / X.std()
    X_scaled = sm.add_constant(X_scaled)
    y = df_clean['Estresse_Alto_P90']
    
    try:
        probit_mod = sm.Probit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
        from sklearn.metrics import roc_auc_score
        y_prob = probit_mod.predict(X_scaled)
        auc_p = roc_auc_score(y, y_prob)
        results_list.append({
            'timestamp': pd.Timestamp.now(),
            'experiment_type': 'model_comparison',
            'lag': best_lag,
            'model_type': 'Probit',
            'features': ", ".join(lag_cols),
            'auc': auc_p,
            'aic': probit_mod.aic,
            'pseudo_r2': probit_mod.prsquared,
            'n_obs': len(df_clean)
        })
        print(f"Modelo Probit: AUC={auc_p:.4f}")
    except:
        print("Probit falhou em convergir.")

    # Salvar resultados
    new_results_df = pd.DataFrame(results_list)
    if EXPERIMENT_LOG.exists():
        old_log = pd.read_csv(EXPERIMENT_LOG)
        combined = pd.concat([old_log, new_results_df], ignore_index=True)
    else:
        combined = new_results_df
    
    combined.sort_values('auc', ascending=False, inplace=True)
    combined.to_csv(EXPERIMENT_LOG, index=False)
    print(f"Ciclo completo. Melhor AUC no histórico: {combined['auc'].max():.4f}")

if __name__ == "__main__":
    run_experiment_cycle()
