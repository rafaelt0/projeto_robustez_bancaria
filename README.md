# 🏦 Projeto de Análise de Estresse e Robustez Bancária (Logit P90)

Modelo econométrico Logit em painel para previsão de estresse bancário em instituições financeiras brasileiras, utilizando dados prudenciais do Banco Central do Brasil (BCB) e indicadores macroeconômicos.

---

## 🎯 Objetivo

Estimar a probabilidade de deterioração financeira das instituições com horizonte preditivo de 12 meses, integrando:

- Indicadores prudenciais (RWA, Capital, Alavancagem)
- Dinâmica do NPL
- Variáveis macroeconômicas (PIB, Selic)

---

## 📁 Estrutura do Projeto

O projeto utiliza nomes de diretórios em português para compatibilidade com o código-fonte:

```text
projeto_robustez_bancaria/
│
├── dados/
│   ├── brutos/               # Dados brutos originais
│   ├── processados/          # Dados tratados e com features
│   └── consolidados/         # Painéis consolidados
│
├── scripts/
│   ├── modelos/              # Modelos econométricos
│   ├── analise/              # Geração de tabelas, gráficos e testes
│   ├── preparacao_dados/     # Limpeza e transformação (Scraping BCB)
│   └── utilitarios/          # Funções auxiliares
│
├── documentacao/             # Documentação técnica e LaTeX
├── resultados/               # Resultados e gráficos
│   ├── relatorios/
│   ├── graficos/
│   └── stress_tests/
└── README.md
```

---

## 🚀 Como Executar

### 1. Requisitos
- Python 3.10+
- `pip` e `venv` (opcional, mas recomendado)

### 2. Instalação
```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Execução Rápida
Para rodar a análise principal e gerar o ranking de robustez:
```bash
python scripts/modelos/modelo_final_recomendado.py
```

Para gerar as tabelas LaTeX para o artigo:
```bash
python scripts/analise/gerar_tabelas_latex.py
```

Para realizar os testes de estresse:
```bash
python scripts/analise/stress_testing.py
```

---

## 📊 Especificação do Modelo Final (Logit P90)

> A especificação é definida em um único ponto — `scripts/utilitarios/config.py` —
> e é consumida por todos os scripts (modelagem, stress testing e tabelas). Não há
> parâmetros duplicados ou embutidos manualmente.

**Target:**  
Estresse Bancário (NPL acima do percentil 90). O limiar é estimado **apenas no
período de treino** (≈ 16.76% na base atual), evitando vazamento do futuro.

**Horizonte de Previsão:**  
12 meses (Lag de 4 trimestres)

### 🔎 Variáveis Explicativas

**Microprudenciais**
- RWA Crédito
- RWA Mercado
- RWA Operacional
- Capital Principal
- Alavancagem

**Dinâmica Temporal**
- Volatilidade do NPL (janela móvel de 8 trimestres)

**Macroeconômicas**
- Crescimento do PIB
- Spread bancário
- Desemprego

> ℹ️ A série **Selic** está ausente na base de dados (coluna vazia na fonte) e,
> portanto, não é utilizada. O Spread bancário é a variável financeira macro
> efetivamente disponível.

**Interações**
- RWA Operacional × Alavancagem  
  (Amplificação não linear de risco)

---

## 📈 Performance

Validação *out-of-time* (treino até 2021, teste a partir de 2022), limiar de
decisão 0.60. Valores regenerados automaticamente em
`resultados/relatorios/modelo_final_performance.csv`.

| Métrica | Valor |
|---------|-------|
| AUC-ROC (Treino) | 0.8739 |
| AUC-ROC (Teste OOT) | 0.7602 |
| Pseudo R² (McFadden) | 0.3821 |
| Recall (Teste @0.60) | 53.6% |
