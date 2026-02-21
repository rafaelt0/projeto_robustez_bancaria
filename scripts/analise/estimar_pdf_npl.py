import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

# 1. Carregar Dados
painel_path = 'dados/brutos/painel_final.csv'
df = pd.read_csv(painel_path, sep=';', decimal=',', encoding='latin1')

# 2. Processar NPL
npl_col = [c for c in df.columns if 'NPL' in c][0]
def parse_num(val):
    if isinstance(val, str):
        val = val.replace('.', '').replace(',', '.')
    return pd.to_numeric(val, errors='coerce')

df['NPL_Val'] = df[npl_col].apply(parse_num)
npl_data = df['NPL_Val'].dropna().values

# 3. Estimar PDF usando KDE
kde = gaussian_kde(npl_data)
npl_range = np.linspace(0, npl_data.max(), 11)
pdf_values = kde.evaluate(npl_range)

# 4. Tabela
pdf_table = pd.DataFrame({
    'Indice NPL (%)': [f"{x*100:.2f}%" for x in npl_range],
    'Densidade PDF Estimada': [f"{v:.4f}" for v in pdf_values]
})

print("\n--- Tabela da Função de Densidade de Probabilidade (PDF) ---")
print(pdf_table.to_string(index=False))

# 5. Exportar
pdf_table.to_csv('resultados/relatorios/tabela_pdf_npl.csv', sep=';', decimal=',', index=False, encoding='latin1')
print(f"\nTabela PDF salva.")
