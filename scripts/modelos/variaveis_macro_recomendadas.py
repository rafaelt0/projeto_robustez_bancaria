"""
VARIAVEIS MACROECONOMICAS ADICIONAIS PARA MODELO DE ESTRESSE BANCARIO

Este documento lista variaveis macro relevantes para prever estresse bancario,
baseado em literatura academica e pratica de supervisao prudencial.
"""

print("="*100)
print("VARIAVEIS MACROECONOMICAS RECOMENDADAS PARA MODELO DE ESTRESSE BANCARIO")
print("="*100)

variaveis_recomendadas = """

================================================================================
1. VARIAVEIS DE CICLO ECONOMICO (Alta Prioridade)
================================================================================

a) TAXA DE DESEMPREGO
   - Fonte: IBGE (PNAD Continua)
   - Frequencia: Trimestral
   - Justificativa: Desemprego alto -> inadimplencia de credito pessoal/consignado
   - Relacao esperada: POSITIVA (mais desemprego = mais estresse)
   - Literatura: Bernanke & Gertler (1989), Pesaran et al. (2006)
   - Disponibilidade: SIDRA/IBGE (Tabela 6381)

b) PRODUCAO INDUSTRIAL (Variacao %)
   - Fonte: IBGE
   - Frequencia: Mensal (agregar para trimestral)
   - Justificativa: Proxy para atividade economica real, afeta credito corporativo
   - Relacao esperada: NEGATIVA (queda producao = mais estresse)
   - Literatura: Borio & Lowe (2002)
   - Disponibilidade: SIDRA/IBGE (Tabela 8888)

c) INDICE DE CONFIANCA DO CONSUMIDOR (ICC-FGV)
   - Fonte: FGV/IBRE
   - Frequencia: Mensal
   - Justificativa: Antecede consumo e inadimplencia
   - Relacao esperada: NEGATIVA (baixa confianca = mais estresse)
   - Literatura: Ludvigson (2004)
   - Disponibilidade: FGV Dados (API)

================================================================================
2. VARIAVEIS DE MERCADO FINANCEIRO (Alta Prioridade)
================================================================================

d) TAXA SELIC (%)
   - Fonte: Banco Central do Brasil
   - Frequencia: Diaria (usar media trimestral)
   - Justificativa: Custo de funding, politica monetaria
   - Relacao esperada: POSITIVA (Selic alta = mais estresse)
   - Literatura: Adrian & Shin (2010)
   - Disponibilidade: BCB (Serie 432)
   - NOTA: Pode substituir/complementar Spread

e) CAMBIO (BRL/USD - Variacao %)
   - Fonte: Banco Central do Brasil
   - Frequencia: Diaria (usar media ou volatilidade trimestral)
   - Justificativa: Bancos com exposicao cambial, importadores
   - Relacao esperada: POSITIVA (depreciacao = mais estresse)
   - Literatura: Kaminsky & Reinhart (1999)
   - Disponibilidade: BCB (Serie 1)

f) VOLATILIDADE DO IBOVESPA (Desvio Padrao Trimestral)
   - Fonte: B3
   - Frequencia: Diaria (calcular vol. trimestral)
   - Justificativa: Proxy para incerteza/risco sistemico
   - Relacao esperada: POSITIVA (alta volatilidade = mais estresse)
   - Literatura: Adrian & Brunnermeier (2016)
   - Disponibilidade: Yahoo Finance, B3

================================================================================
3. VARIAVEIS DE CREDITO E LIQUIDEZ (Media Prioridade)
================================================================================

g) CRESCIMENTO DO CREDITO (% YoY)
   - Fonte: BCB (Serie 20539 - Saldo Operacoes de Credito)
   - Frequencia: Mensal
   - Justificativa: Expansao rapida de credito precede crises
   - Relacao esperada: POSITIVA (crescimento excessivo = mais estresse futuro)
   - Literatura: Schularick & Taylor (2012), Jorda et al. (2013)
   - Disponibilidade: BCB-SGS

h) CREDIT-TO-GDP GAP (Desvio da Tendencia)
   - Fonte: Calcular a partir de BCB + IBGE
   - Frequencia: Trimestral
   - Justificativa: Indicador de BIS para risco sistemico
   - Relacao esperada: POSITIVA (gap alto = bolha de credito)
   - Literatura: BIS (2010), Drehmann et al. (2011)
   - Disponibilidade: Calcular manualmente

i) SPREAD BANCARIO MEDIO (%)
   - Fonte: BCB (Serie 20783)
   - Frequencia: Mensal
   - Justificativa: Custo do credito, risco percebido
   - Relacao esperada: POSITIVA (spread alto = mais risco)
   - Disponibilidade: BCB-SGS
   - NOTA: JA INCLUIDO no modelo atual

================================================================================
4. VARIAVEIS DE PRECOS DE ATIVOS (Media Prioridade)
================================================================================

j) PRECO DE IMOVEIS (Indice FipeZap)
   - Fonte: FipeZap
   - Frequencia: Mensal
   - Justificativa: Colateral de credito imobiliario
   - Relacao esperada: NEGATIVA (queda precos = mais estresse)
   - Literatura: Mian & Sufi (2009)
   - Disponibilidade: FipeZap (API)

k) PRECO DE COMMODITIES (Indice CRB ou Petroleo Brent)
   - Fonte: Bloomberg, FRED
   - Frequencia: Diaria
   - Justificativa: Brasil e exportador, afeta termos de troca
   - Relacao esperada: NEGATIVA (queda commodities = mais estresse)
   - Literatura: Cashin et al. (2004)
   - Disponibilidade: FRED, Bloomberg

================================================================================
5. VARIAVEIS FISCAIS E EXTERNAS (Baixa Prioridade)
================================================================================

l) DIVIDA PUBLICA / PIB (%)
   - Fonte: BCB, Tesouro Nacional
   - Frequencia: Mensal
   - Justificativa: Risco soberano, crowding out
   - Relacao esperada: POSITIVA (divida alta = mais risco)
   - Literatura: Reinhart & Rogoff (2010)
   - Disponibilidade: BCB (Serie 4513)

m) BALANCA COMERCIAL (USD milhoes)
   - Fonte: MDIC, BCB
   - Frequencia: Mensal
   - Justificativa: Pressao cambial, liquidez externa
   - Relacao esperada: NEGATIVA (deficit = mais estresse)
   - Disponibilidade: BCB (Serie 22707)

n) RESERVAS INTERNACIONAIS (USD bilhoes)
   - Fonte: BCB
   - Frequencia: Mensal
   - Justificativa: Capacidade de intervencao cambial
   - Relacao esperada: NEGATIVA (mais reservas = menos estresse)
   - Disponibilidade: BCB (Serie 3546)

================================================================================
6. VARIAVEIS DERIVADAS (Engenharia de Features)
================================================================================

o) HIATO DO PRODUTO (Output Gap)
   - Calcular: PIB real - PIB potencial (filtro HP)
   - Justificativa: Ciclo economico, pressoes inflacionarias
   - Relacao esperada: NEGATIVA (recessao = mais estresse)

p) CURVA DE JUROS (Spread DI Futuro 360d - Selic)
   - Fonte: B3
   - Justificativa: Expectativas de politica monetaria
   - Relacao esperada: POSITIVA (inversao = recessao)

q) VOLATILIDADE CAMBIAL (Desvio Padrao 90 dias)
   - Calcular: Rolling std do cambio
   - Justificativa: Incerteza, risco de mercado
   - Relacao esperada: POSITIVA

r) INDICE DE CONDICOES FINANCEIRAS (FCI)
   - Combinar: Selic, Cambio, Spread, Ibovespa
   - Justificativa: Medida agregada de stress financeiro
   - Relacao esperada: POSITIVA

================================================================================
7. RECOMENDACOES FINAIS
================================================================================

PRIORIDADE ALTA (Adicionar primeiro):
1. Taxa de Desemprego
2. Taxa Selic (se nao usar Spread)
3. Cambio (BRL/USD - variacao ou volatilidade)
4. Crescimento do Credito (% YoY)
5. Volatilidade do Ibovespa

PRIORIDADE MEDIA (Testar depois):
6. Producao Industrial
7. Preco de Imoveis (FipeZap)
8. Credit-to-GDP Gap
9. Indice de Confianca do Consumidor

PRIORIDADE BAIXA (Opcional):
10. Divida Publica / PIB
11. Preco de Commodities
12. Balanca Comercial

VARIAVEIS DERIVADAS (Feature Engineering):
- Hiato do Produto
- Curva de Juros (Spread DI)
- Volatilidade Cambial
- Indice de Condicoes Financeiras

================================================================================
8. FONTES DE DADOS
================================================================================

BANCO CENTRAL DO BRASIL (BCB):
- SGS (Sistema Gerenciador de Series Temporais): https://www3.bcb.gov.br/sgspub
- API: https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json

IBGE:
- SIDRA: https://sidra.ibge.gov.br/
- API: https://servicodados.ibge.gov.br/api/docs

FGV:
- FGV Dados: https://portalibre.fgv.br/
- ICC, IPC, IGP-M

B3:
- Dados historicos: http://www.b3.com.br/pt_br/market-data-e-indices/

IPEADATA:
- Portal: http://www.ipeadata.gov.br/
- Agregador de series macro

FRED (Federal Reserve Economic Data):
- https://fred.stlouisfed.org/
- Commodities, indices internacionais

================================================================================
9. EXEMPLO DE IMPLEMENTACAO
================================================================================

# Exemplo: Adicionar Taxa de Desemprego ao modelo

import pandas as pd
import requests

# 1. Baixar dados do IBGE (PNAD Continua - Taxa de Desemprego)
url = "https://servicodados.ibge.gov.br/api/v3/agregados/6381/periodos/202001|202012/variaveis/4099"
response = requests.get(url)
data = response.json()

# 2. Processar e adicionar ao painel
df_desemprego = pd.DataFrame(...)  # processar JSON
df_painel = df_painel.merge(df_desemprego, on='Data', how='left')

# 3. Criar lag
df_painel['Desemprego_lag4'] = df_painel.groupby('Instituicao')['Desemprego'].shift(4)

# 4. Adicionar ao modelo
features = [..., 'Desemprego_lag4']

================================================================================
10. CUIDADOS E LIMITACOES
================================================================================

MULTICOLINEARIDADE:
- PIB, Desemprego, Producao Industrial sao altamente correlacionados
- Usar VIF para detectar (VIF > 10 = problema)
- Considerar PCA para reduzir dimensionalidade

FREQUENCIA DOS DADOS:
- Alinhar frequencias (mensal -> trimestral)
- Usar medias, somas ou ultimos valores conforme apropriado

DISPONIBILIDADE TEMPORAL:
- Verificar se series tem dados desde 2015
- Algumas series (FipeZap, ICC) podem ter inicio posterior

CAUSALIDADE:
- Variaveis macro sao endogenas (bidirecionalidade)
- Usar lags para mitigar (4 trimestres)

OVERFITTING:
- Nao adicionar muitas variaveis (curse of dimensionality)
- Usar regularizacao (Lasso, Ridge) se necessario
- Validacao cruzada temporal e essencial

"""

print(variaveis_recomendadas)

# Criar tabela resumo
import pandas as pd

resumo = pd.DataFrame({
    'Variavel': [
        'Taxa de Desemprego',
        'Taxa Selic',
        'Cambio (BRL/USD)',
        'Crescimento do Credito',
        'Volatilidade Ibovespa',
        'Producao Industrial',
        'Preco de Imoveis (FipeZap)',
        'Credit-to-GDP Gap',
        'Indice Confianca Consumidor',
        'Divida Publica / PIB',
        'Preco Commodities',
        'Hiato do Produto',
        'Curva de Juros (Spread DI)',
        'Volatilidade Cambial'
    ],
    'Prioridade': [
        'ALTA', 'ALTA', 'ALTA', 'ALTA', 'ALTA',
        'MEDIA', 'MEDIA', 'MEDIA', 'MEDIA',
        'BAIXA', 'BAIXA',
        'DERIVADA', 'DERIVADA', 'DERIVADA'
    ],
    'Fonte': [
        'IBGE', 'BCB', 'BCB', 'BCB', 'B3',
        'IBGE', 'FipeZap', 'BCB+IBGE', 'FGV',
        'BCB', 'FRED',
        'Calcular', 'B3', 'Calcular'
    ],
    'Relacao_Esperada': [
        'POSITIVA', 'POSITIVA', 'POSITIVA', 'POSITIVA', 'POSITIVA',
        'NEGATIVA', 'NEGATIVA', 'POSITIVA', 'NEGATIVA',
        'POSITIVA', 'NEGATIVA',
        'NEGATIVA', 'POSITIVA', 'POSITIVA'
    ],
    'Disponibilidade': [
        'Facil', 'Facil', 'Facil', 'Facil', 'Media',
        'Facil', 'Media', 'Dificil', 'Media',
        'Facil', 'Facil',
        'Dificil', 'Media', 'Facil'
    ]
})

print("\n" + "="*100)
print("RESUMO DAS VARIAVEIS RECOMENDADAS")
print("="*100)
print(resumo.to_string(index=False))

# Salvar recomendacoes
resumo.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/variaveis_macro_recomendadas.csv', index=False)

print("\n" + "="*100)
print("PROXIMOS PASSOS")
print("="*100)
print("""
1. Coletar dados das variaveis de ALTA prioridade
2. Alinhar frequencias (mensal -> trimestral)
3. Criar lags (4 trimestres)
4. Testar cada variavel individualmente (Delta AUC)
5. Verificar multicolinearidade (VIF)
6. Selecionar melhor conjunto de variaveis
7. Validar com dados out-of-sample (2023-2025)
""")

print("\n[OK] Recomendacoes salvas em: variaveis_macro_recomendadas.csv")
