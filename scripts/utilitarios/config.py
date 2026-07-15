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

# Limiar de decisao para classificar estresse a partir da probabilidade prevista.
DECISION_THRESHOLD = 0.60

# Validacao out-of-time: treino antes desta data, teste a partir dela.
SPLIT_DATE = "2022-01-01"

# Variaveis explicativas de nivel (antes da defasagem).
#   - Microprudenciais: RWA (Credito/Mercado/Operacional), Capital, Alavancagem
#   - Dinamica temporal: Volatilidade do NPL (janela de 8 trimestres)
#   - Macroeconomicas:   PIB, Spread, Desemprego
#
# Nota: a serie Selic esta ausente no painel (coluna 100% vazia na fonte de
# dados), portanto NAO e utilizada. O Spread bancario e a variavel financeira
# macro efetivamente disponivel.
CORE_FEATURES = [
    "RWA_Credito",
    "RWA_Mercado",
    "RWA_Operacional",
    "Capital_Principal",
    "Alavancagem",
    "NPL_Volatility_8Q",
    "PIB",
    "Spread",
    "Desemprego",
]

# Termo de interacao (amplificacao nao linear de risco): RWA Operacional x Alavancagem.
INTERACTION = ("RWA_Operacional", "Alavancagem")

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
