import pandas as pd
import requests
import json
from datetime import datetime

def fetch_sgs_series(series_code, start_date='01/01/2015'):
    url = f"http://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={start_date}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            df['data'] = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = pd.to_numeric(df['valor'])
            return df
    except Exception as e:
        print(f"Erro na serie {series_code}: {e}")
    return None

def coletar_indicadores_macro():
    print("="*60)
    print("COLETANDO INDICADORES MACRO (SGS/BCB) - VERSAO CORRIGIDA")
    print("="*60)

    # 1. Coleta
    df_ipca = fetch_sgs_series(433)
    df_pib = fetch_sgs_series(22109)
    df_spread = fetch_sgs_series(20786)

    # 2. Harmonizacao para o FIM do trimestre (QE)
    # PIB
    df_pib.set_index('data', inplace=True)
    df_pib_trim = df_pib['valor'].resample('QE').last().reset_index()
    df_pib_trim.columns = ['Data', 'Crescimento_PIB_Trimestral']

    # IPCA (Acumulado)
    df_ipca.set_index('data', inplace=True)
    df_ipca_trim = (1 + df_ipca['valor']/100).resample('QE').prod() - 1
    df_ipca_trim = (df_ipca_trim * 100).reset_index()
    df_ipca_trim.columns = ['Data', 'IPCA_Trimestral']

    # Spread (Media)
    df_spread.set_index('data', inplace=True)
    df_spread_trim = df_spread['valor'].resample('QE').mean().reset_index()
    df_spread_trim.columns = ['Data', 'Spread_Medio_Trimestral']

    # 3. Merge Único por Data
    macro_panel = df_pib_trim.merge(df_ipca_trim, on='Data', how='outer')
    macro_panel = macro_panel.merge(df_spread_trim, on='Data', how='outer')
    
    # Formatar data para match com painel bancario (usando dia 01/MM/YYYY)
    macro_panel['Data_Key'] = macro_panel['Data'].apply(lambda x: f"01/{x.month:02d}/{x.year}")
    
    # Remover coluna Data original para evitar duplicatas no merge final
    macro_panel = macro_panel.drop(columns=['Data'])

    # 4. Salvar
    output_path = 'dados/brutos/indicadores_macro_sgs.csv'
    macro_panel.to_csv(output_path, sep=';', decimal=',', index=False, encoding='latin1')
    
    print(f"\n[OK] Indicadores macro salvos com chave unica em: {output_path}")

if __name__ == "__main__":
    coletar_indicadores_macro()
