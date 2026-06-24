import pandas as pd
import os
import numpy as np
import subprocess
import json

def fetch_sgs_series(series_code, start_date='01/01/2009'):
    # URL para JSON do Banco Central
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={start_date}"
    
    try:
        print(f"Buscando serie {series_code} via curl...")
        res = subprocess.run(['curl', '-s', '-L', '-A', 'Mozilla/5.0', url], capture_output=True, text=True)
        
        if res.returncode == 0 and res.stdout:
            data = json.loads(res.stdout)
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df['data'] = pd.to_datetime(df['data'], dayfirst=True)
                df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
                return df[['data', 'valor']]
    except Exception as e:
        print(f"  Erro na serie {series_code}: {e}")
    
    return None

def coletar_indicadores_macro():
    print("="*60)
    print("COLETANDO INDICADORES MACRO (PIB, IPCA, SPREAD, DESEMPREGO)")
    print("="*60)

    output_path = 'dados/brutos/indicadores_macro_sgs.csv'
    
    series_map = {
        'IPCA_Trimestral': 433,
        'PIB': 22109,
        'Spread': 20786,
        'Desemprego': 24369
    }

    # Baixar e processar cada uma individualmente
    dfs = []
    for name, code in series_map.items():
        df_raw = fetch_sgs_series(code)
        if df_raw is not None:
            df_raw.set_index('data', inplace=True)
            
            if name == 'IPCA_Trimestral':
                res = (1 + df_raw['valor']/100).resample('QE').prod() - 1
                df_trim = (res * 100).to_frame(name)
            elif name == 'PIB':
                df_trim = df_raw['valor'].resample('QE').last().to_frame(name)
            else:
                df_trim = df_raw['valor'].resample('QE').mean().to_frame(name)
            
            dfs.append(df_trim)
            print(f"  [OK] {name} baixada.")
        else:
            print(f"⚠️ {name} falhou.")

    if dfs:
        # Concatenar todos os trimestres baseados no Index de Data
        df_final = pd.concat(dfs, axis=1).reset_index()
        df_final.rename(columns={'data': 'Data'}, inplace=True)
        
        # Criar a chave para o merge (01/MM/YYYY)
        df_final['Data_Key'] = df_final['Data'].apply(lambda x: f"01/{x.month:02d}/{x.year}")
        
        # Limpar Nomes para o padrão do painel
        df_final.rename(columns={
            'IPCA_Trimestral': 'IPCA',
            'PIB': 'PIB',
            'Spread': 'Spread',
            'Desemprego': 'Desemprego'
        }, inplace=True)
        
        # Salvar
        df_final.to_csv(output_path, sep=';', decimal=',', index=False, encoding='latin1')
        print(f"\n[FIM] Sincronização concluída: {output_path}")
    else:
        print("❌ Nenhuma serie macro foi baixada.")

if __name__ == "__main__":
    coletar_indicadores_macro()
