import pandas as pd

df = pd.read_csv(r'D:\downloads_bcb\painel_bcb_capital.csv', low_memory=False)
sample_rows = df[df['Data'].str.contains('2024-03-01', na=False)]
if sample_rows.empty:
    print("Data 2024-03-01 nao encontrada!")
    exit()

sample = sample_rows.iloc[0]

print("--- Colunas com valor nao zero para 2024-03-01 ---\n")
for col in df.columns:
    val = sample[col]
    if pd.notna(val) and val != '0' and val != 0 and val != '0,00%':
        print(f"COL: {col} | VAL: {val}")
