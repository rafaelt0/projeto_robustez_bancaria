"""
Modelo Logit final recomendado (P90, horizonte de 12 meses).

Este script e a UNICA fonte de coeficientes, parametros de escala e metricas
do projeto. Todos os artefatos gravados aqui (statistics, scaling, performance,
ranking) sao consumidos pelos demais scripts (stress testing e tabelas LaTeX),
que nunca devem reimplementar ou embutir estes valores.

Especificacao (lag, variaveis, limiar, regularizacao) vem de
scripts/utilitarios/config.py. Notas metodologicas:

  - As variaveis de RWA entram como PARTICIPACOES (share do RWA total) mais o
    log do RWA total, evitando a quase-separacao que niveis em R$ provocavam.
  - A estimacao usa regularizacao L2 (ridge), que encolhe os coeficientes e
    estabiliza o modelo. Como o estimador regularizado nao fornece erros-padrao
    analiticos, a inferencia (erro-padrao, z, p-valor) e obtida por BOOTSTRAP
    POR INSTITUICAO (cluster bootstrap), respeitando a estrutura de painel.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utilitarios"))
import config

RANDOM_SEED = 42


def fit_ridge(X_const, y, weights, alpha_vec):
    """Ajusta um Logit (GLM binomial) com penalidade L2 (ridge)."""
    return sm.GLM(
        y, X_const, family=sm.families.Binomial(), var_weights=weights
    ).fit_regularized(alpha=alpha_vec, L1_wt=0.0)


def weighted_loglik(y, p, w):
    """Log-verossimilhanca binomial ponderada."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p)))


def bootstrap_inference(X_const, y, groups, alpha_vec, n_boot):
    """Erros-padrao/z/p do estimador ridge via cluster bootstrap por instituicao."""
    Xmat = X_const.values
    yvec = y.values
    insts = np.array(list(groups.keys()))
    rng = np.random.default_rng(RANDOM_SEED)

    boot = []
    for _ in range(n_boot):
        chosen = rng.choice(len(insts), size=len(insts), replace=True)
        rows = np.concatenate([groups[insts[i]] for i in chosen])
        yb = yvec[rows]
        n_pos = yb.sum()
        if n_pos == 0 or n_pos == len(yb):  # resample degenerado
            continue
        wb = np.where(yb == 1, (len(yb) - n_pos) / n_pos, 1.0)
        try:
            rb = fit_ridge(Xmat[rows], yb, wb, alpha_vec)
            boot.append(np.asarray(rb.params, dtype=float))
        except Exception:
            continue

    boot = np.array(boot)
    se = boot.std(axis=0, ddof=1)
    return se, boot.shape[0]


def run_final_model():
    print(f"\n{'='*100}")
    print(f"MODELO LOGIT FINAL (LAG {config.LAG} - 12 MESES, RWA EM COMPOSICAO + RIDGE)")
    print(f"{'='*100}")

    if not config.RAW_PANEL.exists():
        print(f"Erro: Arquivo {config.RAW_PANEL} nao encontrado.")
        return

    # 1. Carregar e preparar dados (ordenacao + volatilidade NPL + features de RWA)
    df = pd.read_csv(config.RAW_PANEL)
    df["Data"] = pd.to_datetime(df["Data"])
    df = config.prepare_panel(df)

    # 2. Split out-of-time e limiar P90 definido APENAS no treino
    split_date = pd.to_datetime(config.SPLIT_DATE)
    threshold_p90 = df[df["Data"] < split_date]["NPL"].quantile(config.P90_QUANTILE)
    df["Estresse_Alto_P90"] = (df["NPL"] > threshold_p90).astype(int)

    print(f"Periodo de Treino: {df[df['Data'] < split_date]['Data'].min().year} - {df[df['Data'] < split_date]['Data'].max().year}")
    print(f"Periodo de Teste:  {df[df['Data'] >= split_date]['Data'].min().year} - {df['Data'].max().year}")
    print(f"Threshold P90 (do Treino): {threshold_p90*100:.2f}%")

    # 3. Features defasadas + interacao (definidas na config)
    df, features_final = config.build_lagged_features(df)

    df_model = df.dropna(subset=features_final + ["Estresse_Alto_P90", "Instituicao"]).copy()
    train = df_model[df_model["Data"] < split_date].copy().reset_index(drop=True)
    test = df_model[df_model["Data"] >= split_date].copy()

    X_train, y_train = train[features_final], train["Estresse_Alto_P90"]
    X_test, y_test = test[features_final], test["Estresse_Alto_P90"]

    # 4. Padronizacao (Z-score) com parametros do TREINO aplicados ao teste
    X_mean, X_std = X_train.mean(), X_train.std()
    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled = (X_test - X_mean) / X_std

    # 5. Estimacao ridge com pesos de balanceamento.
    #    A penalidade nao se aplica ao intercepto (primeiro coeficiente).
    counts = y_train.value_counts()
    weight_stress = counts[0] / counts[1]
    weights = y_train.apply(lambda v: weight_stress if v == 1 else 1.0)

    X_train_const = sm.add_constant(X_train_scaled)
    X_test_const = sm.add_constant(X_test_scaled)
    alpha_vec = np.array([0.0] + [config.L2_ALPHA] * len(features_final))

    model_final = fit_ridge(X_train_const, y_train, weights, alpha_vec)
    params = model_final.params

    # 6. Inferencia por cluster bootstrap (erro-padrao, z, p-valor)
    print(f"\nRodando cluster bootstrap por instituicao (N={config.BOOTSTRAP_N})...")
    groups = train.groupby("Instituicao").indices
    se, n_ok = bootstrap_inference(X_train_const, y_train, groups, alpha_vec, config.BOOTSTRAP_N)
    z_stat = params.values / se
    p_val = 2 * (1 - norm.cdf(np.abs(z_stat)))
    print(f"  reamostragens validas: {n_ok}/{config.BOOTSTRAP_N}")

    # 7. Performance
    probs_train = np.asarray(model_final.predict(X_train_const))
    probs_test = np.asarray(model_final.predict(X_test_const))
    preds_test = (probs_test > config.DECISION_THRESHOLD).astype(int)

    # Pseudo R2 de McFadden com log-verossimilhanca ponderada
    p0 = np.average(y_train, weights=weights)
    ll_model = weighted_loglik(y_train.values, probs_train, weights.values)
    ll_null = weighted_loglik(y_train.values, np.full(len(y_train), p0), weights.values)
    pseudo_r2 = 1 - ll_model / ll_null

    metrics = {
        "AUC_Train": roc_auc_score(y_train, probs_train),
        "AUC_Test": roc_auc_score(y_test, probs_test),
        "AUCPR_Test": average_precision_score(y_test, probs_test),
        "PR2_Train": pseudo_r2,
        "Recall_Test": recall_score(y_test, preds_test, zero_division=0),
        "Precision_Test": precision_score(y_test, preds_test, zero_division=0),
        "F1_Test": f1_score(y_test, preds_test, zero_division=0),
        "Acc_Test": accuracy_score(y_test, preds_test),
    }

    print(f"\nPERFORMANCE (Lag {config.LAG}, alpha {config.L2_ALPHA}, threshold {config.DECISION_THRESHOLD}):")
    print(f"  AUC-ROC Treino:  {metrics['AUC_Train']:.4f}")
    print(f"  AUC-ROC Teste:   {metrics['AUC_Test']:.4f}  (gap {metrics['AUC_Train']-metrics['AUC_Test']:.3f})")
    print(f"  AUC-PR Teste:    {metrics['AUCPR_Test']:.4f}  (baseline {y_test.mean():.4f})")
    print(f"  Pseudo R2:       {metrics['PR2_Train']:.4f}")
    print(f"  Recall Teste:    {metrics['Recall_Test']:.1%}")
    print(f"  Maior |coef|:    {params.drop('const').abs().max():.3f}")

    # 8. Persistir artefatos (fonte unica para os demais scripts)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    os.makedirs(config.GRAPHICS_DIR, exist_ok=True)

    coef_df = pd.DataFrame(
        {
            "Variavel": params.index,
            "Coeficiente": params.values,
            "StdErr": se,
            "Z_stat": z_stat,
            "P-valor": p_val,
        }
    )
    coef_df.to_csv(config.STATISTICS_CSV, index=False)

    scaling_df = pd.DataFrame(
        {"Variavel": X_mean.index, "Mean": X_mean.values, "Std": X_std.values}
    )
    scaling_df.to_csv(config.SCALING_CSV, index=False)

    pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    ).to_csv(config.PERFORMANCE_CSV, index=False)

    df_model["Prob_Estresse"] = np.asarray(
        model_final.predict(sm.add_constant((df_model[features_final] - X_mean) / X_std))
    )
    os.makedirs(config.PROCESSED_PANEL.parent, exist_ok=True)
    df_model.to_csv(config.PROCESSED_PANEL, index=False)

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

    # 9. Grafico de diagnostico (curva ROC out-of-time)
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
