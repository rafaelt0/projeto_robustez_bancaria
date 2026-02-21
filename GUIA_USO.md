# Guia de Uso - Projeto de Robustez Bancária

Este documento descreve como configurar e executar o projeto de análise de robustez bancária.

## 1. Estrutura do Projeto

O projeto está organizado da seguinte forma:

- `dados/`: Contém os dados brutos, processados e consolidados.
- `scripts/`: Contém todos os códigos em Python.
  - `preparacao_dados/`: Scripts para coleta e limpeza de dados.
  - `modelos/`: Scripts para treinamento e execução do modelo Logit.
  - `analise/`: Scripts para geração de resultados, tabelas e testes adicionais.
  - `utilitarios/`: Funções de suporte e auxiliares.
- `resultados/`: Contém relatórios em CSV e gráficos gerados.
- `documentacao/`: Arquivos LaTeX e PDF do artigo científico.

## 2. Ordem de Execução Recomendada

Para reproduzir a análise completa do zero, siga esta ordem:

### Etapa 1: Coleta e Consolidação de Dados
1. `scripts/preparacao_dados/script_scraping_var_dependentes.py`: Baixa os dados do IF.data (BCB).
2. `scripts/preparacao_dados/coletar_macros_bcb.py`: Coleta indicadores macroeconômicos (PIB, SELIC, etc.).
3. `scripts/preparacao_dados/consolidar_painel_datas.py`: Consolida os arquivos baixados em um único painel de dados (`dados/consolidados/painel_completo.csv`).

### Etapa 2: Modelagem
4. `scripts/modelos/modelo_final_recomendado.py`: Executa o modelo Logit com as variáveis selecionadas e interação RWA/Alavancagem. Gera o ranking de robustez inicial.

### Etapa 3: Análise e Relatórios
5. `scripts/analise/gerar_tabelas_latex.py`: Formata os resultados do modelo em tabelas prontas para uso no LaTeX.
6. `scripts/analise/stress_testing.py`: (Opcional) Realiza testes de estresse com base no modelo ajustado.

## 3. Requisitos de Ambiente

- Python 3.10+
- Bibliotecas: `pandas`, `numpy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`.

Para instalar as dependências:
```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```

## 4. Resultados Principais

Os resultados mais importantes podem ser encontrados em:
- `resultados/relatorios/modelo_final_ranking.csv`: Ranking de robustez das instituições.
- `resultados/graficos/modelo_final_diagnostics.png`: Curva ROC e diagnósticos do modelo.
- `tabelas_final_latex.txt`: Código LaTeX das tabelas para o artigo.
