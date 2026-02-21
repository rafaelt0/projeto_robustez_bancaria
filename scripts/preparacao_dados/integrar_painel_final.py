import pandas as pd
import subprocess
import os

def executar_pipeline():
    print("="*60)
    print("EXECUTANDO PIPELINE DE INTEGRACAO")
    print("="*60)

    # 1. Atualizar Macro
    subprocess.run(["python", "scripts/coletar_macros_bcb.py"], check=True)

    # 2. Carregar Bases
    path_npl = 'dados.csv' # Output do fix_dados.py
    path_macro = 'indicadores_macro_sgs.csv'

    df_npl = pd.read_csv(path_npl, sep=';', decimal=',', encoding='latin1')
    df_macro = pd.read_csv(path_macro, sep=';', decimal=',', encoding='latin1')

    # 3. Merge Final
    # No df_macro, a chave é 'Data_Key' (DD/MM/YYYY)
    # No df_npl, a chave é 'Data' (DD/MM/YYYY)
    
    painel_final = pd.merge(
        df_npl, 
        df_macro, 
        left_on='Data', 
        right_on='Data_Key', 
        how='left'
    )

    # Limpeza: remover a chave duplicada
    if 'Data_Key' in painel_final.columns:
        painel_final = painel_final.drop(columns=['Data_Key'])

    # 4. Exportar
    output_path = 'painel_final.csv'
    painel_final.to_csv(output_path, sep=';', decimal=',', index=False, encoding='latin1')
    
    print(f"\n[SUCESSO] Painel Final Atualizado: {output_path}")
    print(f"Colunas Macro Presentes: { [c for c in painel_final.columns if 'Trimestral' in c] }")

if __name__ == "__main__":
    executar_pipeline()
