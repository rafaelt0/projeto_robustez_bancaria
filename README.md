# 🏦 Projeto de Análise de Estresse e Robustez Bancária (Logit P90)

Este projeto implementa um modelo econométrico Logit para prever estresse bancário em instituições financeiras brasileiras, utilizando dados prudenciais do Banco Central do Brasil (BCB) e indicadores macroeconômicos do IBGE.

## 📁 Estrutura de Pastas

- **`/data`**: Armazenamento de dados.
    - `/raw`: Dados brutos originais (`painel_final.csv`).
    - `/processed`: Dados limpos, com lags, macros e probabilidades calculadas.
- **`/scripts`**: Motores de execução em Python.
    - `models/`: Modelos preditivos e de machine learning.
        - `modelo_com_macros.py`: O modelo final consolidado.
        - `modelo_npl_features.py`: Análise da dinâmica do NPL.
    - `analysis/`: Scripts de análise estatística e geração de tabelas.
        - `gerar_tabelas_latex.py`: Utilitário para formatação acadêmica.
    - `data_prep/`: Scripts de preparação e limpeza de dados.
    - `utils/`: Funções utilitárias compartilhadas.
- **`/docs`**: Documentação técnica e tabelas em LaTeX para o paper.
- **`/outputs`**: Resultados visuais e relatórios de performance.

## 📊 Resumo do Modelo Final (Logit P90)

- **Target**: Estresse Bancário (NPL > 12.41%).
- **Horizonte de Previsão**: 12 meses (Lag 4 trimestres).
- **Variáveis Chave**:
    - **Micro**: RWA (Crédito, Mercado, Op), Capital Principal, Alavancagem.
    - **Temporal**: Volatilidade do NPL (8 trimestres).
    - **Macro**: PIB, Taxa Selic.
    - **Interação**: Risco Operacional x Alavancagem (Non-linear risk amplification).

## 🚀 Performance
- **AUC-ROC**: 0.8655
- **Pseudo R2**: 0.2621
- **Recall (@0.175)**: 61.1%

## 🛠️ Como Executar
Sempre execute o script final de modelagem para atualizar os rankings:
```bash
python scripts/modelos/modelo_com_macros.py
```

---
*Desenvolvido em colaboração com Antigravity AI.*
