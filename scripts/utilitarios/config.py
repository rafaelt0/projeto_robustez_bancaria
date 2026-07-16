"""
Configuracao central do projeto de robustez bancaria.

Este modulo e a UNICA fonte de verdade para a especificacao do modelo:
defasagem (lag), variaveis explicativas, limiar do alvo, janela de validacao
e caminhos dos arquivos. Todos os scripts (modelagem, stress testing e
tabelas econometricas) devem importar daqui para evitar divergencias.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Especificacao do modelo (fonte unica de verdade)
# ---------------------------------------------------------------------------

# Horizonte preditivo: 4 trimestres = 12 meses.
LAG = 4

# Limiar do alvo (P90 do NPL). Calculado APENAS no periodo de treino.
P90_QUANTILE = 0.90

# Limiar de decisao aplicado sobre a PROBABILIDADE CALIBRADA (ver calibracao
# abaixo). Escolhido por VALIDACAO WALK-FORWARD (maximo F1 sobre as previsoes
# out-of-fold), e nao ajustado no proprio conjunto de teste. Com as probabilidades
# calibradas, e interpretavel: "sinalizar estresse se P(estresse) > 17%".
# Ver scripts/analise/walk_forward_validation.py.
DECISION_THRESHOLD = 0.17

# Validacao out-of-time: treino antes desta data, teste a partir dela.
SPLIT_DATE = "2022-01-01"

# Regularizacao L2 (ridge). Encolhe os coeficientes, cura a quase-separacao
# causada pelo porte do banco e reduz o overfitting. Calibrado por validacao
# out-of-time (melhor AUC-PR e menor gap treino/teste).
L2_ALPHA = 2.0

# Numero de reamostragens do bootstrap por instituicao (cluster bootstrap) usado
# para obter erros-padrao/p-valores do estimador regularizado.
BOOTSTRAP_N = 250

# Fracao final do periodo de treino (por data) usada para AJUSTAR a calibracao
# de probabilidades. Um fold temporalmente proximo do teste corrige a
# miscalibracao causada pela mudanca da taxa-base do evento ao longo do tempo
# (drift): calibrar no treino inteiro superestima a cauda de alto risco.
CALIBRATION_FOLD_FRAC = 0.20

# Variaveis explicativas de nivel (antes da defasagem).
#   - Composicao de risco (RWA): participacao de cada RWA no RWA total. Usar
#     participacoes (e nao niveis em R$) remove o proxy de PORTE que causava
#     quase-separacao (coeficiente explodindo) e preserva o perfil de risco.
#   - Porte: log do RWA total. Devolve o sinal legitimo de tamanho do banco de
#     forma limitada (log), sem reintroduzir separacao perfeita.
#   - Microprudenciais: Capital, Alavancagem (ja sao razoes na fonte).
#   - Dinamica temporal: Volatilidade do NPL (janela de 8 trimestres).
#   - Macroeconomicas:   PIB, Spread, Desemprego.
#
# Nota: a serie Selic esta vazia no painel atual. O coletor
# scripts/preparacao_dados/coletar_macros_bcb.py ja esta preparado para baixa-la
# (SGS 4189); apos rodar o coletor + integrar_painel_final.py num ambiente com
# acesso a api.bcb.gov.br, basta adicionar "Selic" a esta lista para inclui-la.
# Enquanto a coluna estiver vazia, NAO adicione (o modelo perde todas as linhas).
CORE_FEATURES = [
    "RWA_Credito_share",
    "RWA_Mercado_share",
    "RWA_Operacional_share",
    "log_RWA_Total",
    "Capital_Principal",
    "Alavancagem",
    "NPL",                # nivel do NPL defasado (persistencia da inadimplencia)
    "NPL_Volatility_8Q",
    "PIB",
    "Spread",
    "Desemprego",
]

# Colunas de RWA em nivel (R$) usadas para derivar participacoes e porte.
RWA_LEVEL_COLS = ["RWA_Credito", "RWA_Mercado", "RWA_Operacional"]

# Termo de interacao (amplificacao nao linear de risco):
# participacao de RWA Operacional x Alavancagem.
INTERACTION = ("RWA_Operacional_share", "Alavancagem")

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PANEL = BASE_DIR / "dados" / "brutos" / "painel_final.csv"
PROCESSED_PANEL = BASE_DIR / "dados" / "processados" / "modelo_final_painel.csv"

REPORTS_DIR = BASE_DIR / "resultados" / "relatorios"
GRAPHICS_DIR = BASE_DIR / "resultados" / "graficos"

STATISTICS_CSV = REPORTS_DIR / "modelo_final_statistics.csv"
SCALING_CSV = REPORTS_DIR / "modelo_final_scaling.csv"
CALIBRATION_CSV = REPORTS_DIR / "modelo_final_calibration.csv"
PERFORMANCE_CSV = REPORTS_DIR / "modelo_final_performance.csv"
RANKING_CSV = REPORTS_DIR / "modelo_final_ranking.csv"


# ---------------------------------------------------------------------------
# Nomes derivados e construcao de features (compartilhados entre scripts)
# ---------------------------------------------------------------------------

def lag_name(feature: str) -> str:
    """Nome da coluna defasada para uma feature (ex.: 'PIB' -> 'PIB_lag4')."""
    return f"{feature}_lag{LAG}"


def interaction_name() -> str:
    """Nome da coluna do termo de interacao defasado."""
    a, b = INTERACTION
    return f"{lag_name(a)}_x_{lag_name(b)}"


def feature_columns():
    """Lista final de colunas explicativas (defasadas + interacao), na ordem do modelo."""
    return [lag_name(f) for f in CORE_FEATURES] + [interaction_name()]


def add_npl_volatility(df):
    """Adiciona a coluna NPL_Volatility_8Q (desvio-padrao movel de 8 trimestres do NPL).

    Requer que ``df`` esteja ordenado por Instituicao/Data. Preenche os valores
    iniciais ausentes com a media da propria serie para preservar observacoes.
    """
    vol = df.groupby("Instituicao")["NPL"].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    df["NPL_Volatility_8Q"] = vol.fillna(vol.mean())
    return df


def add_rwa_features(df):
    """Deriva as features de composicao e porte a partir dos RWA em nivel.

    Cria:
      - RWA_Total: soma dos RWA (Credito + Mercado + Operacional);
      - {RWA}_share: participacao de cada RWA no total (perfil de risco);
      - log_RWA_Total: porte do banco em escala logaritmica.

    Usar participacoes em vez de niveis em R$ elimina o proxy de porte que
    provocava quase-separacao no logit.
    """
    import numpy as np

    total = df[RWA_LEVEL_COLS].sum(axis=1)
    df["RWA_Total"] = total
    safe_total = total.replace(0, np.nan)
    for col in RWA_LEVEL_COLS:
        df[f"{col}_share"] = df[col] / safe_total
    df["log_RWA_Total"] = np.log(safe_total)
    return df


def prepare_panel(df):
    """Ordena e adiciona as features derivadas (volatilidade do NPL e RWA).

    Ponto unico de preparo do painel usado por todos os scripts, garantindo a
    mesma construcao de variaveis em modelagem, econometria e stress testing.
    """
    df = df.sort_values(["Instituicao", "Data"])
    df = add_npl_volatility(df)
    df = add_rwa_features(df)
    return df


def calibrated_prob(linear_predictor, a, b):
    """Aplica calibracao de Platt sobre o preditor linear (log-odds bruto).

    Retorna sigmoid(a * lp + b), a probabilidade calibrada. Como e monotonica
    em ``linear_predictor``, preserva a ordenacao (AUC) do modelo.
    """
    import numpy as np

    z = np.clip(a * np.asarray(linear_predictor) + b, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def build_lagged_features(df):
    """Constroi as colunas defasadas e o termo de interacao definidos na config.

    ``df`` deve conter as colunas de ``CORE_FEATURES`` (exceto NPL_Volatility_8Q,
    que e criada por :func:`add_npl_volatility`) e estar ordenado por
    Instituicao/Data. Retorna ``(df, feature_cols)``.
    """
    for feat in CORE_FEATURES:
        df[lag_name(feat)] = df.groupby("Instituicao")[feat].shift(LAG)

    a, b = INTERACTION
    df[interaction_name()] = df[lag_name(a)] * df[lag_name(b)]

    return df, feature_columns()
