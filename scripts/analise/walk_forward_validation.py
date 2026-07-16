"""
Validacao walk-forward (janela expansiva) do modelo Logit de estresse.

Em vez de um unico corte treino/teste (que depende de um periodo especifico e
permite escolher hiperparametros espiando o teste), reestima o modelo a cada
trimestre: treina em tudo que antecede o trimestre-alvo e preve aquele trimestre.
As previsoes out-of-fold (OOF) de todos os folds sao agregadas para uma
estimativa de performance mais honesta e menos dependente de um unico periodo.

Serve tambem para ESCOLHER o limiar de decisao por validacao (maximo F1 sobre as
previsoes OOF), evitando ajustar o threshold no proprio conjunto de teste.

Uso: python scripts/analise/walk_forward_validation.py
Saida: resultados/relatorios/walk_forward_folds.csv e um resumo no console.
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utilitarios"))
import config

# Data a partir da qual comeca a avaliacao (garante treino inicial suficiente).
EVAL_START = "2020-01-01"


def _fit_predict(train, test, feats, alpha):
    """Treina o modelo (mesma receita do modelo final) e preve o fold de teste.

    Retorna (y_true, prob_calibrada) ou None se o treino for degenerado.
    """
    thr = train["NPL"].quantile(config.P90_QUANTILE)
    y_tr = (train["NPL"] > thr).astype(int)
    y_te = (test["NPL"] > thr).astype(int)
    if y_tr.sum() < 5 or y_tr.nunique() < 2:
        return None

    mean, std = train[feats].mean(), train[feats].std()
    X_tr = sm.add_constant((train[feats] - mean) / std)
    X_te = sm.add_constant((test[feats] - mean) / std, has_constant="add")

    w = y_tr.map({0: 1.0, 1: (y_tr == 0).sum() / (y_tr == 1).sum()})
    alpha_vec = np.array([0.0] + [alpha] * len(feats))
    model = sm.GLM(
        y_tr, X_tr, family=sm.families.Binomial(), var_weights=w
    ).fit_regularized(alpha=alpha_vec, L1_wt=0.0)

    lp_tr = X_tr.values @ model.params.values
    lp_te = X_te.values @ model.params.values

    # Calibracao de Platt no fold recente do treino (mesma logica do modelo final)
    cut = train["Data"].quantile(1 - config.CALIBRATION_FOLD_FRAC)
    cal_mask = (train["Data"] > cut).values
    if cal_mask.sum() < 30 or y_tr.values[cal_mask].sum() < 3:
        cal_mask = np.ones(len(y_tr), dtype=bool)
    platt = LogisticRegression(C=1e6).fit(
        lp_tr[cal_mask].reshape(-1, 1), y_tr.values[cal_mask]
    )
    prob = config.calibrated_prob(lp_te, platt.coef_[0, 0], platt.intercept_[0])
    return y_te.values, np.asarray(prob)


def run_walk_forward(alpha=None):
    alpha = config.L2_ALPHA if alpha is None else alpha
    print("=" * 80)
    print(f"VALIDACAO WALK-FORWARD (janela expansiva, alpha={alpha})")
    print("=" * 80)

    if not config.RAW_PANEL.exists():
        print(f"Erro: {config.RAW_PANEL} nao encontrado.")
        return

    df = pd.read_csv(config.RAW_PANEL)
    df["Data"] = pd.to_datetime(df["Data"])
    df = config.prepare_panel(df)
    df, feats = config.build_lagged_features(df)
    dm = df.dropna(subset=feats + ["NPL"]).copy()

    cutoffs = [q for q in sorted(dm["Data"].unique()) if q >= pd.to_datetime(EVAL_START)]

    fold_rows, Y, P = [], [], []
    for c in cutoffs:
        train = dm[dm["Data"] < c]
        test = dm[dm["Data"] == c]
        if len(test) == 0:
            continue
        res = _fit_predict(train, test, feats, alpha)
        if res is None:
            continue
        y_te, prob = res
        Y.append(y_te)
        P.append(prob)
        fold_rows.append(
            {
                "trimestre": pd.Timestamp(c).date(),
                "n_treino": len(train),
                "n_teste": len(test),
                "positivos": int(y_te.sum()),
                "auc": roc_auc_score(y_te, prob) if y_te.sum() and y_te.sum() < len(y_te) else np.nan,
                "brier": brier_score_loss(y_te, prob),
            }
        )

    Y = np.concatenate(Y)
    P = np.concatenate(P)

    # Selecao do limiar por CV: maximo F1 sobre as previsoes OOF agregadas.
    best_t, best_f1 = config.DECISION_THRESHOLD, -1.0
    for t in np.arange(0.05, 0.60, 0.01):
        f1 = f1_score(Y, (P > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, round(float(t), 2)
    preds = (P > best_t).astype(int)

    print(f"\nFolds avaliados: {len(fold_rows)} | previsoes OOF: {len(Y)} | positivos: {Y.sum()}")
    print("\n--- Performance agregada (out-of-fold) ---")
    print(f"  AUC-ROC:  {roc_auc_score(Y, P):.4f}")
    print(f"  AUC-PR:   {average_precision_score(Y, P):.4f}  (baseline {Y.mean():.4f})")
    print(f"  Brier:    {brier_score_loss(Y, P):.4f}")
    print(f"\n--- Limiar escolhido por CV (max F1 OOF) = {best_t} ---")
    print(f"  Precisao: {precision_score(Y, preds, zero_division=0):.1%}")
    print(f"  Recall:   {recall_score(Y, preds, zero_division=0):.1%}")
    print(f"  F1:       {best_f1:.3f}")

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(config.REPORTS_DIR / "walk_forward_folds.csv", index=False)
    print(f"\nDetalhe por fold salvo em: {config.REPORTS_DIR / 'walk_forward_folds.csv'}")
    return best_t


if __name__ == "__main__":
    run_walk_forward()
