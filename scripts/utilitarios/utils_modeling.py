import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

def train_logit_model(df, features, target_col='Estresse_Alto_P90', constant=True):
    """
    Treina um modelo Logit e retorna o objeto do modelo e métricas.
    """
    X = df[features]
    y = df[target_col]
    
    # Escalar variáveis
    X_scaled = (X - X.mean()) / X.std()
    if constant:
        X_scaled = sm.add_constant(X_scaled)
    
    try:
        model = sm.Logit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
        
        # Métricas
        y_prob = model.predict(X_scaled)
        auc = roc_auc_score(y, y_prob)
        aic = model.aic
        bic = model.bic
        pseudo_r2 = model.prsquared
        
        # VIF (Fator de Inflação de Variância)
        # Pular constante se presente para cálculo do VIF
        X_vif = X_scaled.drop('const', axis=1) if 'const' in X_scaled.columns else X_scaled
        vifs = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        
        results = {
            'model': model,
            'auc': auc,
            'aic': aic,
            'bic': bic,
            'pseudo_r2': pseudo_r2,
            'vifs': dict(zip(X_vif.columns, vifs)),
            'n_obs': len(df)
        }
        return results
    except Exception as e:
        return {'error': str(e)}

def prepare_data(df, lag=4, features_to_lag=None):
    """
    Prepara os dados com defasagens (lags) e variável alvo.
    """
    if features_to_lag is None:
        features_to_lag = [
            'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 
            'Capital_Principal', 'Alavancagem', 'PIB', 'Spread',
            'Desemprego', 'Selic', 'NPL_Volatility_8Q'
        ]
    
    df_lagged = df.copy()
    lag_cols = []
    
    for f in features_to_lag:
        if f in df_lagged.columns:
            col_name = f'{f}_lag{lag}'
            df_lagged[col_name] = df_lagged.groupby('Instituicao')[f].shift(lag)
            lag_cols.append(col_name)
            
    # Alvo P90
    # O limiar deve ser calculado por trimestre para ser justo, ou global?
    # A produção atual usa quantil global no NPL. Manteremos isso para consistência.
    threshold_p90 = df_lagged['NPL'].quantile(0.90)
    df_lagged['Estresse_Alto_P90'] = (df_lagged['NPL'] > threshold_p90).astype(int)
    
    return df_lagged.dropna(subset=lag_cols + ['Estresse_Alto_P90']), lag_cols
