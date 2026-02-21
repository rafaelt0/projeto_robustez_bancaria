import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar Dados
painel_path = 'd:/projeto_robustez_bancaria/dados/brutos/painel_final.csv'
df = pd.read_csv(painel_path, sep=';', decimal=',', encoding='latin1')

# 2. Processar NPL
npl_col = [c for c in df.columns if 'NPL' in c][0]
def parse_num(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

df['NPL_Val'] = df[npl_col].apply(parse_num)
npl_data = df['NPL_Val'].dropna() * 100 

# 3. Criar Gráfico
plt.figure(figsize=(10, 6), facecolor='#0f172a')
ax = plt.axes()
ax.set_facecolor('#0f172a')

plt.hist(npl_data, bins=30, color='#38bdf8', edgecolor='#1e293b', alpha=0.8)

plt.title('Distribuição dos Índices NPL Bancários', color='#f8fafc', fontsize=16, pad=20)
plt.xlabel('Índice NPL (%)', color='#94a3b8', fontsize=12)
plt.ylabel('Frequência (Observações Trimestrais)', color='#94a3b8', fontsize=12)

plt.xticks(color='#94a3b8')
plt.yticks(color='#94a3b8')
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

stats_text = (f"Média: {npl_data.mean():.2f}%\n"
              f"Mediana: {npl_data.median():.2f}%\n"
              f"Máximo: {npl_data.max():.2f}%")
plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, color='#f8fafc',
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='#1e293b', alpha=0.5, edgecolor='#334155'))

plt.grid(axis='y', color='#334155', linestyle='--', alpha=0.3)

# Salvar
output_img = 'd:/projeto_robustez_bancaria/resultados/graficos/distribuicao_npl.png'
plt.tight_layout()
plt.savefig(output_img, dpi=300, facecolor='#0f172a')
print(f"Gráfico gerado em: {output_img}")
