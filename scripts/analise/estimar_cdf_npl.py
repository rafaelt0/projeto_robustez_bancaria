import pandas as pd
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
npl_data = df['NPL_Val'].dropna().values
npl_data_sorted = np.sort(npl_data)

# 3. Calcular CDF Empírica
def calculate_cdf(data, x_range):
    cdf_vals = [np.sum(data <= x) / len(data) for x in x_range]
    return np.array(cdf_vals)

npl_range = np.linspace(0, npl_data.max(), 11)
cdf_values = calculate_cdf(npl_data_sorted, npl_range)

# 4. Tabela
cdf_table = pd.DataFrame({
    'Indice NPL (%)': [f"{x*100:.2f}%" for x in npl_range],
    'Probabilidade Acumulada (CDF)': [f"{v:.4f}" for v in cdf_values],
    'Interpretacao': [f"{v*100:.1f}% das observações são <= {x*100:.2f}%" for x, v in zip(npl_range, cdf_values)]
})

print("\n--- Tabela da Função de Distribuição Acumulada (CDF) ---")
print(cdf_table.to_string(index=False))

# 5. Exportar
cdf_table.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/tabela_cdf_npl.csv', sep=';', decimal=',', index=False, encoding='latin1')
print(f"\nTabela CDF salva.")
