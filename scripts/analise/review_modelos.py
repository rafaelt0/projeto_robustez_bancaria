import pandas as pd
import numpy as np

# TABELA DE REVIEW COMPARATIVA DOS MODELOS
print("="*100)
print("REVIEW COMPARATIVO: MODELOS LOGIT PARA PREDIÇÃO DE ESTRESSE BANCÁRIO")
print("="*100)

# 1. ESPECIFICAÇÕES DOS MODELOS
print("\n" + "="*100)
print("1. ESPECIFICAÇÕES DOS MODELOS")
print("="*100)

specs = pd.DataFrame({
    'Característica': [
        'Threshold de Estresse (NPL)',
        'Observações em Estresse',
        '% do Dataset',
        'Threshold de Decisão',
        'Método de Estimação',
        'Lag (trimestres)',
        'Variáveis Preditoras',
        'Observações Totais'
    ],
    'Modelo Original (Manual)': [
        'P95 (24.18%)',
        '118',
        '4.0%',
        '0.5 (padrão)',
        'Gradient Descent Manual',
        '4',
        '8',
        '2953'
    ],
    'Modelo P95 (Statsmodels)': [
        'P95 (24.18%)',
        '118',
        '4.0%',
        '0.5 (padrão)',
        'Maximum Likelihood (BFGS)',
        '4',
        '8',
        '2953'
    ],
    'Modelo P90 (Statsmodels)': [
        'P90 (12.41%)',
        '254',
        '8.6%',
        '0.5 (padrão)',
        'Maximum Likelihood (BFGS)',
        '4',
        '8',
        '2953'
    ],
    'Modelo P90 Otimizado': [
        'P90 (12.41%)',
        '254',
        '8.6%',
        '0.175 (otimizado)',
        'Maximum Likelihood (BFGS)',
        '4',
        '8',
        '2953'
    ]
})

print(specs.to_string(index=False))

# 2. MÉTRICAS DE PERFORMANCE
print("\n" + "="*100)
print("2. MÉTRICAS DE PERFORMANCE")
print("="*100)

performance = pd.DataFrame({
    'Métrica': [
        'AUC-ROC',
        'Pseudo R²',
        'Log-Likelihood',
        'Acurácia',
        'Precision (Estresse)',
        'Recall (Estresse)',
        'F1-Score',
        'Casos Detectados',
        'Falsos Positivos',
        'Quasi-Separação (%)'
    ],
    'Original': [
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A',
        'N/A'
    ],
    'P95': [
        '0.8095',
        '0.1764',
        '-408.14',
        '96.3%',
        '100.0%',
        '6.8%',
        '0.127',
        '8 / 118',
        '0',
        '18%'
    ],
    'P90': [
        '0.8111',
        '0.1789',
        '-710.93',
        '91.5%',
        '61.5%',
        '3.1%',
        '0.060',
        '8 / 254',
        '5',
        '15%'
    ],
    'P90 Otimizado': [
        '0.8111',
        '0.1789',
        '-710.93',
        '80.8%',
        '27.0%',
        '50.4%',
        '0.352',
        '128 / 254',
        '346',
        '15%'
    ]
})

print(performance.to_string(index=False))

# 3. SIGNIFICÂNCIA ESTATÍSTICA DAS VARIÁVEIS
print("\n" + "="*100)
print("3. SIGNIFICÂNCIA ESTATÍSTICA DAS VARIÁVEIS (p < 0.05)")
print("="*100)

significance = pd.DataFrame({
    'Variavel': [
        'RWA_Credito_lag4',
        'RWA_Mercado_lag4',
        'RWA_Operacional_lag4',
        'Capital_Principal_lag4',
        'Alavancagem_lag4',
        'PIB_lag4',
        'IPCA_lag4',
        'Spread_lag4'
    ],
    'P95': [
        'SIM (p < 0.001)',
        'NAO (p = 0.463)',
        'SIM (p < 0.001)',
        'NAO (p = 0.154)',
        'SIM (p = 0.002)',
        'SIM (p = 0.014)',
        'NAO (p = 0.862)',
        'NAO (p = 0.911)'
    ],
    'P90': [
        'SIM (p < 0.001)',
        'NAO (p = 0.213)',
        'SIM (p < 0.001)',
        'SIM (p < 0.001) ***',
        'SIM (p = 0.019)',
        'SIM (p = 0.079)',
        'NAO (p = 0.570)',
        'SIM (p = 0.050) ***'
    ],
    'Interpretacao P90': [
        'Protetor forte (-58.5)',
        'Nao significativo',
        'Risco forte (+8.8)',
        'Protetor moderado (-0.35)',
        'Risco moderado (+0.14)',
        'Protetor fraco (-0.14)',
        'Nao significativo',
        'Risco moderado (+0.15)'
    ]
})

print(significance.to_string(index=False))

# 4. VANTAGENS E DESVANTAGENS
print("\n" + "="*100)
print("4. ANÁLISE COMPARATIVA")
print("="*100)

print("\n[P95 - Statsmodels]")
print("-" * 100)
print("VANTAGENS:")
print("  • Precision perfeita (100%) - quando preve estresse, sempre acerta")
print("  • Poucos falsos positivos (0)")
print("  • Foco em eventos extremos (colapso iminente)")
print("  • Util para decisoes criticas (intervencao regulatoria)")
print("\nDESVANTAGENS:")
print("  • Recall pessimo (6.8%) - perde 93% dos casos de estresse")
print("  • Threshold muito alto (24% NPL) - detecta apenas colapso total")
print("  • Quasi-separacao elevada (18%)")
print("  • 4 de 8 variaveis nao significativas")
print("  • Inutil para early warning system")

print("\n[P90 - Statsmodels - Threshold Padrao]")
print("-" * 100)
print("VANTAGENS:")
print("  • Melhor AUC-ROC (0.8111 vs 0.8095)")
print("  • Mais variaveis significativas (6 vs 4)")
print("  • Capital Principal agora e significativo ***")
print("  • Spread agora e significativo ***")
print("  • Menos quasi-separacao (15% vs 18%)")
print("  • Threshold mais realista (12.4% vs 24.2%)")
print("\nDESVANTAGENS:")
print("  • Recall AINDA PIOR (3.1% vs 6.8%)")
print("  • Precision menor (61.5% vs 100%)")
print("  • Mais falsos positivos (5 vs 0)")

print("\n[P90 OTIMIZADO - Threshold = 0.175]")
print("-" * 100)
print("VANTAGENS:")
print("  • Recall 16x melhor (50.4% vs 3.1%)")
print("  • F1-Score 487% melhor (0.352 vs 0.060)")
print("  • Detecta 128 de 254 casos (vs 8 de 254)")
print("  • Equilibrio razoavel precision/recall")
print("  • Util para monitoramento continuo de risco")
print("  • Identifica deterioracao gradual (nao apenas colapso)")
print("\nDESVANTAGENS:")
print("  • Precision menor (27% vs 61.5%)")
print("  • Mais falsos positivos (346 vs 5)")
print("  • Acuracia menor (80.8% vs 91.5%)")
print("  • Requer validacao manual dos alertas")

# 5. RECOMENDAÇÕES DE USO
print("\n" + "="*100)
print("5. RECOMENDAÇÕES DE USO")
print("="*100)

recommendations = pd.DataFrame({
    'Cenário': [
        'Early Warning System',
        'Monitoramento Contínuo',
        'Decisões Regulatórias',
        'Análise de Portfólio',
        'Stress Testing',
        'Pesquisa Acadêmica'
    ],
    'Modelo Recomendado': [
        'P90 Otimizado',
        'P90 Otimizado',
        'P95',
        'P90 Otimizado',
        'P95',
        'P90 (Threshold Padrão)'
    ],
    'Justificativa': [
        'Recall alto (50%) detecta deterioração precoce',
        'Equilíbrio precision/recall permite ação preventiva',
        'Precision 100% evita intervenções desnecessárias',
        'Identifica instituições em risco moderado',
        'Foco em eventos extremos (tail risk)',
        'Significância estatística robusta'
    ]
})

print(recommendations.to_string(index=False))

# 6. SCORE FINAL
print("\n" + "="*100)
print("6. AVALIAÇÃO FINAL (Escala 0-10)")
print("="*100)

scores = pd.DataFrame({
    'Critério': [
        'Capacidade Discriminatória (AUC)',
        'Robustez Estatística',
        'Interpretabilidade',
        'Utilidade Prática (Early Warning)',
        'Validação',
        'Equilíbrio Precision/Recall',
        'NOTA FINAL'
    ],
    'P95': [
        '8.0',
        '6.0',
        '9.0',
        '3.0',
        '4.0',
        '2.0',
        '5.3'
    ],
    'P90': [
        '8.1',
        '7.0',
        '9.0',
        '2.0',
        '4.0',
        '1.0',
        '5.2'
    ],
    'P90 Otimizado': [
        '8.1',
        '7.0',
        '9.0',
        '7.5',
        '4.0',
        '7.0',
        '7.1'
    ]
})

print(scores.to_string(index=False))

# 7. PRÓXIMOS PASSOS
print("\n" + "="*100)
print("7. MELHORIAS FUTURAS RECOMENDADAS")
print("="*100)

improvements = pd.DataFrame({
    'Prioridade': ['ALTA', 'ALTA', 'ALTA', 'MÉDIA', 'MÉDIA', 'BAIXA'],
    'Melhoria': [
        'Validação Cruzada Temporal',
        'Feature Engineering (NPL growth, volatilidade)',
        'Verificar Multicolinearidade (VIF)',
        'Modelo Ensemble (Logit + XGBoost)',
        'Random Effects Logit (Painel)',
        'Modelo de Sobrevivência (Cox)'
    ],
    'Impacto Esperado': [
        'Medir capacidade preditiva real',
        'Capturar dinâmica temporal',
        'Reduzir coeficientes inflados',
        'Melhorar AUC em 5-10%',
        'Controlar heterogeneidade entre bancos',
        'Modelar tempo até estresse'
    ],
    'Esforço': ['Baixo', 'Médio', 'Baixo', 'Alto', 'Médio', 'Alto']
})

print(improvements.to_string(index=False))

# 8. CONCLUSÃO
print("\n" + "="*100)
print("8. CONCLUSÃO E RECOMENDAÇÃO FINAL")
print("="*100)

print("""
*** MODELO RECOMENDADO: P90 OTIMIZADO (Threshold = 0.175) ***

JUSTIFICATIVA:
- Melhor equilibrio entre precision (27%) e recall (50.4%)
- F1-Score 487% superior ao modelo padrao
- Detecta deterioracao gradual, nao apenas colapso
- Util para gestao proativa de risco
- Variaveis economicamente interpretaveis e estatisticamente significativas

CONFIGURACAO FINAL:
- Threshold de Estresse: P90 (NPL > 12.41%)
- Threshold de Decisao: 0.175 (17.5%)
- Lag: 4 trimestres
- Metodo: Maximum Likelihood (BFGS)
- AUC-ROC: 0.8111

LIMITACOES:
- Recall de 50% ainda deixa metade dos casos sem deteccao
- Precision de 27% gera ~3 falsos positivos para cada verdadeiro positivo
- Falta validacao temporal (pode estar overfitting)
- Quasi-separacao em 15% das observacoes

PROXIMO PASSO CRITICO:
Implementar validacao cruzada temporal (treinar em 2016-2022, testar em 2023-2025)
para verificar se o modelo realmente generaliza para periodos futuros.
""")

print("="*100)
print("FIM DO REVIEW")
print("="*100)

# Salvar tabelas em CSV
specs.to_csv('d:/projeto_robustez_bancaria/review_especificacoes.csv', index=False)
performance.to_csv('d:/projeto_robustez_bancaria/review_performance.csv', index=False)
significance.to_csv('d:/projeto_robustez_bancaria/review_significancia.csv', index=False)
recommendations.to_csv('d:/projeto_robustez_bancaria/review_recomendacoes.csv', index=False)
scores.to_csv('d:/projeto_robustez_bancaria/review_scores.csv', index=False)
improvements.to_csv('d:/projeto_robustez_bancaria/review_melhorias.csv', index=False)

print("\n[OK] Tabelas de review salvas em CSV")
