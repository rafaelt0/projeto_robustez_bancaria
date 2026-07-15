"""
Modelo Logit final recomendado (P90, horizonte de 12 meses).

Este script e a UNICA fonte de coeficientes, parametros de escala e metricas
do projeto. Todos os artefatos gravados aqui (statistics, scaling, performance,
ranking) sao consumidos pelos demais scripts (stress testing e tabelas LaTeX),
que nunca devem reimplementar ou embutir estes valores.

Especificacao (lag, variaveis, limiar) vem de scripts/utilitarios/config.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utilitarios"))
import config


def run_final_model():
    print(f"\n{'='*100}")
    print(f"MODELO LOGIT FINAL RECOMENDADO (LAG {config.LAG} - HORIZONTE 12 MESES)")
    print(f"{'='*100}")

    if not config.RAW_PANEL.exists():
        print(f"Erro: Arquivo {config.RAW_PANEL} nao encontrado.")
        return

    # 1. Carregar e preparar dados
    df = pd.read_csv(config.RAW_PANEL)
    df["Data"] = pd.to_datetime(df["Data"])
    df.sort_values(["Instituicao", "Data"], inplace=True)
    df = config.add_npl_volatility(df)

    # 2. Split out-of-time
    split_date = pd.to_datetime(config.SPLIT_DATE)
    df_train_raw = df[df["Data"] < split_date]

    # Limiar P90 definido APENAS no treino (evita vazamento do futuro).
    threshold_p90 = df_train_raw["NPL"].quantile(config.P90_QUANTILE)
    df["Estresse_Alto_P90"] = (df["NPL"] > threshold_p90).astype(int)

    print(f"Periodo de Treino: {df_train_raw['Data'].min().year} - {df_train_raw['Data'].max().year}")
    print(f"Periodo de Teste:  {df[df['Data'] >= split_date]['Data'].min().year} - {df['Data'].max().year}")
    print(f"Threshold P90 (do Treino): {threshold_p90*100:.2f}%")

    # 3. Features defasadas + interacao (definidas na config)
    df, features_final = config.build_lagged_features(df)

    df_model = df.dropna(subset=features_final + ["Estresse_Alto_P90", "Instituicao"]).copy()
    train = df_model[df_model["Data"] < split_date].copy()
    test = df_model[df_model["Data"] >= split_date].copy()

    X_train, y_train = train[features_final], train["Estresse_Alto_P90"]
    X_test, y_test = test[features_final], test["Estresse_Alto_P90"]

    # 4. Padronizacao (Z-score) com parametros do TREINO aplicados ao teste
    X_mean = X_train.mean()
    X_std = X_train.std()
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    # 5. Estimacao com pesos de balanceamento
    counts = y_train.value_counts()
    weight_stress = counts[0] / counts[1]
    weights = y_train.apply(lambda v: weight_stress if v == 1 else 1.0)

    X_train_const = sm.add_constant(X_train_scaled)
    model_final = sm.GLM(
        y_train, X_train_const, family=sm.families.Binomial(), var_weights=weights
    ).fit()
    print(model_final.summary())

    # 6. Performance
    X_test_const = sm.add_constant(X_test_scaled)
    probs_train = model_final.predict(X_train_const)
    probs_test = model_final.predict(X_test_const)
    preds_test = (probs_test > config.DECISION_THRESHOLD).astype(int)

    # Pseudo R2 de McFadden (modelo nulo com os mesmos pesos)
    null_model = sm.GLM(
        y_train, np.ones(len(y_train)), family=sm.families.Binomial(), var_weights=weights
    ).fit()
    pseudo_r2 = 1 - (model_final.llf / null_model.llf)

    metrics = {
        "AUC_Train": roc_auc_score(y_train, probs_train),
        "AUC_Test": roc_auc_score(y_test, probs_test),
        "PR2_Train": pseudo_r2,
        "Recall_Test": recall_score(y_test, preds_test, zero_division=0),
        "Precision_Test": precision_score(y_test, preds_test, zero_division=0),
        "F1_Test": f1_score(y_test, preds_test, zero_division=0),
        "Acc_Test": accuracy_score(y_test, preds_test),
    }

    print(f"\nPERFORMANCE (Lag {config.LAG}, threshold {config.DECISION_THRESHOLD}):")
    print(f"  AUC Treino:     {metrics['AUC_Train']:.4f}")
    print(f"  AUC Teste (OOT):{metrics['AUC_Test']:.4f}")
    print(f"  Pseudo R2:      {metrics['PR2_Train']:.4f}")
    print(f"  Recall Teste:   {metrics['Recall_Test']:.1%}")

    # 7. Persistir artefatos (fonte unica para os demais scripts)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.GRAPHICS_DIR, exist_ok=True)

    # 7a. Coeficientes (inclui Z_stat, exigido por gerar_tabelas_latex.py)
    coef_df = pd.DataFrame(
        {
            "Variavel": model_final.params.index,
            "Coeficiente": model_final.params.values,
            "StdErr": model_final.bse.values,
            "Z_stat": model_final.tvalues.values,
            "P-valor": model_final.pvalues.values,
        }
    )
    coef_df.to_csv(config.STATISTICS_CSV, index=False)

    # 7b. Parametros de escala (consumidos pelo stress testing)
    scaling_df = pd.DataFrame(
        {"Variavel": X_mean.index, "Mean": X_mean.values, "Std": X_std.values}
    )
    scaling_df.to_csv(config.SCALING_CSV, index=False)

    # 7c. Metricas de performance
    pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    ).to_csv(config.PERFORMANCE_CSV, index=False)

    # 7d. Painel com probabilidade prevista
    df_model["Prob_Estresse"] = model_final.predict(
        sm.add_constant((df_model[features_final] - X_mean) / X_std)
    )
    os.makedirs(config.PROCESSED_PANEL.parent, exist_ok=True)
    df_model.to_csv(config.PROCESSED_PANEL, index=False)

    # 7e. Ranking de robustez por instituicao
    ranking = (
        df_model.groupby("Instituicao")["Prob_Estresse"]
        .mean()
        .reset_index()
        .rename(columns={"Prob_Estresse": "Prob_Estresse_Media"})
    )
    p = ranking["Prob_Estresse_Media"].clip(1e-6, 1 - 1e-6)
    ranking["Score_Robustez"] = -np.log(p / (1 - p))  # maior = mais robusto
    ranking = ranking.sort_values("Prob_Estresse_Media").reset_index(drop=True)
    ranking.to_csv(config.RANKING_CSV, index=False)

    # 8. Grafico de diagnostico (curva ROC out-of-time)
    fpr, tpr, _ = roc_curve(y_test, probs_test)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['AUC_Test']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC - Validacao Out-of-Time")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.GRAPHICS_DIR / "modelo_final_diagnostics.png", dpi=150)
    plt.close()

    print("\nArtefatos gravados em resultados/relatorios/ e resultados/graficos/.")


if __name__ == "__main__":
    run_final_model()
