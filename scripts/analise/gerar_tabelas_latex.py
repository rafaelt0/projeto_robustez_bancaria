"""
GERADOR DE TABELAS LATEX PARA O MODELO FINAL
Este script formata os resultados do modelo Logit consolidado (com macros e volatilidade)
em tabelas padrão para artigos acadêmicos.
"""

import pandas as pd
import numpy as np

# 1. Dados do modelo consolidado (baseado nos logs da execução anterior)
model_results = [
    {"Variable": "Constant", "Coef": -16.987, "StdErr": 2.202, "PVal": 0.000},
    {"Variable": "RWA Credit (lag 4)", "Coef": -51.632, "StdErr": 7.847, "PVal": 0.000},
    {"Variable": "RWA Market (lag 4)", "Coef": -0.562, "StdErr": 1.845, "PVal": 0.761},
    {"Variable": "RWA Operational (lag 4)", "Coef": 3.822, "StdErr": 2.670, "PVal": 0.152},
    {"Variable": "Core Capital (lag 4)", "Coef": -0.340, "StdErr": 0.108, "PVal": 0.002},
    {"Variable": "Leverage Ratio (lag 4)", "Coef": -0.691, "StdErr": 0.157, "PVal": 0.000},
    {"Variable": "GDP Growth (lag 4)", "Coef": -0.428, "StdErr": 0.187, "PVal": 0.022},
    {"Variable": "Interest Spread (lag 4)", "Coef": -0.025, "StdErr": 0.121, "PVal": 0.835},
    {"Variable": "Unemployment Rate (lag 4)", "Coef": 0.142, "StdErr": 0.204, "PVal": 0.486},
    {"Variable": "Taxa Selic (lag 4)", "Coef": 0.356, "StdErr": 0.162, "PVal": 0.028},
    {"Variable": "NPL Volatility (8Q lag 4)", "Coef": 0.215, "StdErr": 0.058, "PVal": 0.000},
    {"Variable": "Interaction: RWA Op x Leverage", "Coef": 3.405, "StdErr": 0.561, "PVal": 0.000}
]

def get_stars(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

# Gerar Tabela 1: Coeficientes do Modelo Final
latex_table1 = r"""
\begin{table}[htbp]
  \centering
  \caption{Estimated Coefficients for the Consolidate Logit Stress Model (P90)}
  \label{tab:logit_results}
  \begin{tabular}{lcccc}
    \hline
    \textbf{Variable} & \textbf{Coefficient} & \textbf{Std. Error} & \textbf{z-statistic} & \textbf{P-value} \\
    \hline
"""

for res in model_results:
    stars = get_stars(res["PVal"])
    z_stat = res["Coef"] / res["StdErr"]
    line = f"    {res['Variable']} & {res['Coef']:.4f}{stars} & {res['StdErr']:.4f} & {z_stat:.3f} & {res['PVal']:.4f} \\\\\n"
    latex_table1 += line

latex_table1 += r"""    \hline
    \multicolumn{5}{l}{\textit{Note: *** p<0.01, ** p<0.05, * p<0.1. Threshold: P90 of NPL Index.}} \\
    \multicolumn{5}{l}{\textit{Number of observations: 2,505. Pseudo R2: 0.2621. AUC-ROC: 0.8655.}} \\
    \hline
  \end{tabular}
\end{table}
"""

# Gerar Tabela 2: Métricas de Performance (Latex)
latex_table2 = r"""
\begin{table}[htbp]
  \centering
  \caption{Model Predictive Performance (Threshold = 0.175)}
  \label{tab:performance}
  \begin{tabular}{lc}
    \hline
    \textbf{Metric} & \textbf{Value} \\
    \hline
    Area Under the ROC Curve (AUC) & 0.8655 \\
    Pseudo R-squared (McFadden) & 0.2621 \\
    Log-Likelihood & -477.65 \\
    Recall (Sensitivity) & 61.1\% \\
    Precision & 36.3\% \\
    F1-Score & 0.455 \\
    Overall Accuracy & 90.1\% \\
    \hline
  \end{tabular}
\end{table}
"""

# Salvar no arquivo central de tabelas
output_path = 'd:/projeto_robustez_bancaria/tabelas_final_latex.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("% ==========================================================\n")
    f.write("% TABELA DE RESULTADOS DO MODELO LOGIT FINAL (CONSOLIDADO)\n")
    f.write("% ==========================================================\n\n")
    f.write(latex_table1)
    f.write("\n\n")
    f.write("% ==========================================================\n")
    f.write("% TABELA DE PERFORMANCE E METRICAS\n")
    f.write("% ==========================================================\n\n")
    f.write(latex_table2)

print(f"Tabelas LaTeX geradas com sucesso em: {output_path}")
print("\n--- Visualizacao da Tabela de Coeficientes ---")
print(latex_table1)
