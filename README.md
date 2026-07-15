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

**Composição de risco (RWA)**
- Participação do RWA de Crédito no RWA total
- Participação do RWA de Mercado no RWA total
- Participação do RWA Operacional no RWA total

> ℹ️ Os RWA entram como **participações** (e não em nível, R$) para remover o
> proxy de *porte* que provocava quase-separação no logit. O porte legítimo é
> capturado separadamente pelo **log do RWA total**.

**Porte e microprudenciais**
- Log do RWA total (tamanho do banco)
- Capital Principal
- Alavancagem

**Dinâmica Temporal**
- NPL defasado (persistência da inadimplência — preditor mais forte)
- Volatilidade do NPL (janela móvel de 8 trimestres)

**Macroeconômicas**
- Crescimento do PIB
- Spread bancário
- Desemprego

> ℹ️ A série **Selic** está ausente na base de dados (coluna vazia na fonte) e,
> portanto, não é utilizada. O Spread bancário é a variável financeira macro
> efetivamente disponível.

**Interações**
- RWA Operacional (participação) × Alavancagem  
  (Amplificação não linear de risco)

**Estimação:** Logit (GLM binomial) com **regularização L2 (ridge)** para conter
a quase-separação e o overfitting. Como o estimador regularizado não fornece
erros-padrão analíticos, a inferência (erro-padrão, z, p-valor) é obtida por
**bootstrap por instituição** (respeitando a estrutura de painel).

**Calibração:** o balanceamento por pesos desloca as probabilidades brutas para
cima, então aplica-se **calibração de Platt** (logística de 2 parâmetros sobre o
log-odds, ajustada no treino). Por ser monotônica, preserva a ordenação (AUC) e
recupera probabilidades interpretáveis — a probabilidade prevista passa a refletir
a taxa-base real do evento.

---

## 📈 Performance

Validação *out-of-time* (treino até 2021, teste a partir de 2022), limiar de
decisão calibrado 0.25. Valores regenerados automaticamente em
`resultados/relatorios/modelo_final_performance.csv`.

| Métrica | Valor |
|---------|-------|
| AUC-ROC (Treino) | 0.9266 |
| AUC-ROC (Teste OOT) | 0.8076 |
| AUC-PR (Teste OOT) | 0.2349 *(baseline 0.059)* |
| Pseudo R² (McFadden) | 0.2493 |
| Brier (Teste OOT) | 0.0649 |
| Recall (Teste @0.25) | 41.6% |

> ✅ **Calibração:** probabilidade média prevista ≈ 0.094 vs. taxa real ≈ 0.059
> (Brier ≈ 0.065 após Platt). As probabilidades — inclusive as do stress test —
> podem ser lidas como probabilidades reais de estresse, não apenas como score
> relativo.

> 📈 **Enriquecimento de dados:** a inclusão do **NPL defasado** (persistência da
> inadimplência) elevou a AUC-ROC de 0.74 → 0.81 e a AUC-PR de 0.18 → 0.23. A
> coleta de séries adicionais do BCB (ex.: Selic) depende de acesso à API do
> Banco Central, indisponível no ambiente atual — deve ser executada via
> `scripts/preparacao_dados/` em rede liberada.
