import pandas as pd

df = pd.read_csv(r'D:\downloads_bcb\painel_bcb_capital.csv', low_memory=False)
df['Instituicao'] = df.iloc[:, 0].str.upper() # A primeira coluna costuma ser Instituicao

target = df[(df['Data'].str.contains('2024-03-01')) & (df['Instituicao'].str.contains('BB - PRUDENCIAL', na=False))]

print(f"Linhas encontradas: {len(target)}")
if len(target) > 0:
    for col in target.columns:
        if 'capital principal' in col.lower():
            print(f"COL: {col} | VAL: {target[col].iloc[0]}")
