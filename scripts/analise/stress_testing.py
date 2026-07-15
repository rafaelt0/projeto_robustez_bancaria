"""
Analise de stress testing.

Aplica choques macroeconomicos e prudenciais sobre a base mais recente e mede
o impacto na probabilidade de estresse de cada instituicao.

Os coeficientes e parametros de escala NAO sao embutidos aqui: sao lidos dos
arquivos gerados por scripts/modelos/modelo_final_recomendado.py. Assim, ao
reestimar o modelo, o stress test reflete automaticamente os novos parametros.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utilitarios"))
import config


def load_model_params():
    """Carrega coeficientes e parametros de escala dos CSVs do modelo final."""
    if not config.STATISTICS_CSV.exists() or not config.SCALING_CSV.exists():
        raise FileNotFoundError(
            "Parametros do modelo nao encontrados. Rode primeiro "
            "scripts/modelos/modelo_final_recomendado.py."
        )
    stats = pd.read_csv(config.STATISTICS_CSV)
    coefs = dict(zip(stats["Variavel"], stats["Coeficiente"]))

    scaling = pd.read_csv(config.SCALING_CSV)
    X_mean = dict(zip(scaling["Variavel"], scaling["Mean"]))
    X_std = dict(zip(scaling["Variavel"], scaling["Std"]))
    return coefs, X_mean, X_std


def apply_shock(value, shock):
    """Aplica um choque (('add', x) ou ('mul', x)) a um valor de nivel."""
    if shock is None:
        return value
    kind, amount = shock
    return value + amount if kind == "add" else value * amount


def calculate_risk(row, coefs, X_mean, X_std, shocks=None):
    """Probabilidade de estresse para uma instituicao sob um cenario de choques.

    ``shocks`` e um dict {feature: ('add'|'mul', valor)} sobre as variaveis de
    nivel de config.CORE_FEATURES.
    """
    shocks = shocks or {}

    # Valores de nivel (com choques) por feature.
    levels = {}
    for feat in config.CORE_FEATURES:
        base = row.get(feat, np.nan)
        levels[feat] = apply_shock(base, shocks.get(feat))

    # Vetor defasado esperado pelos coeficientes (chaves com sufixo _lag{LAG}).
    v = {config.lag_name(feat): levels[feat] for feat in config.CORE_FEATURES}
    a, b = config.INTERACTION
    v[config.interaction_name()] = v[config.lag_name(a)] * v[config.lag_name(b)]

    # Padronizacao com os parametros do treino; imputa media quando ausente.
    log_odds = coefs.get("const", 0.0)
    for k, val in v.items():
        mean_val = X_mean.get(k, 0.0)
        std_val = X_std.get(k, 1.0) or 1.0
        if pd.isnull(val):
            val = mean_val
        scaled = (val - mean_val) / std_val
        log_odds += coefs.get(k, 0.0) * scaled

    return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20, 20)))


def run_stress_testing():
    print("\n" + "=" * 80)
    print("ANALISE DE STRESS TESTING (PIB, SPREAD, CAPITAL, DESEMPREGO)")
    print("=" * 80)

    if not config.RAW_PANEL.exists():
        print(f"Erro: {config.RAW_PANEL} nao encontrado.")
        return

    coefs, X_mean, X_std = load_model_params()

    df = pd.read_csv(config.RAW_PANEL)
    raw_cols = config.RWA_LEVEL_COLS + [
        "Capital_Principal", "Alavancagem", "PIB", "Spread", "Desemprego", "NPL",
    ]
    for c in raw_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Data"] = pd.to_datetime(df["Data"])
    df.sort_values(["Instituicao", "Data"], inplace=True)
    for c in ["PIB", "Spread", "Desemprego"]:
        if c in df.columns:
            df[c] = df[c].ffill()
    # Deriva volatilidade do NPL e features de RWA (participacoes + porte).
    df = config.add_npl_volatility(df)
    df = config.add_rwa_features(df)

    last_date = df["Data"].max()
    df_base = df[df["Data"] == last_date].copy().dropna(subset=["Instituicao", "RWA_Credito"])
    print(f"Base do stress test: {last_date.date()} ({len(df_base)} instituicoes)")

    # Cenarios de choque
    cenario_severo = {
        "PIB": ("add", -3.0),
        "Spread": ("add", 2.0),
        "Capital_Principal": ("mul", 0.85),
        "Desemprego": ("add", 2.0),
    }
    cenario_sistemico = {
        "PIB": ("add", -6.0),
        "Spread": ("add", 5.0),
        "Capital_Principal": ("mul", 0.75),
        "Desemprego": ("add", 5.0),
    }

    df_base["Prob_Baseline"] = df_base.apply(
        lambda r: calculate_risk(r, coefs, X_mean, X_std), axis=1
    )
    df_base["Prob_Stress_Severo"] = df_base.apply(
        lambda r: calculate_risk(r, coefs, X_mean, X_std, cenario_severo), axis=1
    )
    df_base["Prob_Stress_Interm"] = df_base.apply(
        lambda r: calculate_risk(r, coefs, X_mean, X_std, cenario_sistemico), axis=1
    )

    df_base["Score_Baseline"] = -np.log(
        df_base["Prob_Baseline"] / (1 - df_base["Prob_Baseline"] + 1e-10)
    )
    df_base["Score_Stress"] = -np.log(
        df_base["Prob_Stress_Interm"] / (1 - df_base["Prob_Stress_Interm"] + 1e-10)
    )
    df_base["Queda_Resiliencia"] = df_base["Score_Stress"] - df_base["Score_Baseline"]

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    df_base.to_csv(config.REPORTS_DIR / "stress_test_results.csv", index=False)
    generate_latex_table(df_base)


def generate_latex_table(df):
    df_sorted = df.sort_values("Queda_Resiliencia")
    top_affected = df_sorted.head(15)
    least_affected = df_sorted.tail(10).iloc[::-1]

    latex = """
% ==========================================================
% TABELA DE STRESS TESTING (CENARIOS COM DESEMPREGO)
% ==========================================================
\\begin{table}[htbp]
  \\centering
  \\caption{Analise de Sensibilidade (Cenarios: Severo vs Crise Sistemica [+5\\% Desemprego])}
  \\label{tab:stress_test}
  \\begin{tabular}{lcccc}
    \\hline
    \\textbf{Instituicao} & \\textbf{Baseline} & \\textbf{Severo} & \\textbf{Sistemica} & \\textbf{Impacto} \\\\
    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel A: 15 Instituicoes Mais Vulneraveis (Menor Resiliencia)}} \\\\
    \\hline
"""
    for _, row in top_affected.iterrows():
        inst = row["Instituicao"].replace("&", "\\&").replace("_", "\\_")[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"

    latex += """    \\hline
    \\multicolumn{5}{l}{\\textbf{Painel B: 10 Instituicoes Mais Resilientes (Maior Estabilidade)}} \\\\
    \\hline
"""
    for _, row in least_affected.iterrows():
        inst = row["Instituicao"].replace("&", "\\&").replace("_", "\\_")[:30]
        latex += f"    {inst} & {row['Prob_Baseline']:.1%} & {row['Prob_Stress_Severo']:.1%} & {row['Prob_Stress_Interm']:.1%} & {row['Queda_Resiliencia']:.2f} \\\\\n"

    latex += "    \\hline\n  \\end{tabular}\n\\end{table}\n"
    with open(config.REPORTS_DIR / "tabela_stress_test.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    print("Tabela LaTeX de Stress Test gerada.")


if __name__ == "__main__":
    run_stress_testing()
