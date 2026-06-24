import pandas as pd
import os

def executar_pipeline():
    print("="*60)
    print("INTEGRANDO NOVAS VARIAVEIS MACRO (SELIC/DESEMPREGO)")
    print("="*60)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path_painel = os.path.join(base_dir, 'dados/brutos/painel_final.csv')
    path_macro = os.path.join(base_dir, 'dados/brutos/indicadores_macro_sgs.csv')

    if not os.path.exists(path_painel):
        print(f"Erro: {path_painel} nao encontrado.")
        return

    # 1. Carregar Painel Atual
    # O painel final atual usa virgula como separador e ponto decimal (padrao pandas output anterior)
    df_painel = pd.read_csv(path_painel)
    
    # 2. Carregar Novas Macros
    # O script de coleta salva com ';' e ','
    df_macro = pd.read_csv(path_macro, sep=';', decimal=',', encoding='latin1')
    
    # 3. Preparar para o Merge
    # Remover colunas macro ja existentes para evitar duplicatas, exceto Data_Key
    cols_to_remove = [c for c in df_painel.columns if c in ['PIB', 'IPCA', 'Spread', 'Selic', 'Desemprego']]
    # Mas queremos manter os nomes do modelo final: PIB, IPCA, Spread, Selic, Desemprego
    # O df_macro tem nomes longos: Crescimento_PIB_Trimestral, etc.
    
    macro_rename = {
        'Crescimento_PIB_Trimestral': 'PIB',
        'IPCA_Trimestral': 'IPCA',
        'Spread_Medio_Trimestral': 'Spread',
        'Selic_Media_Trimestral': 'Selic',
        'Desemprego_Trimestral': 'Desemprego'
    }
    df_macro = df_macro.rename(columns=macro_rename)
    
    # Colunas originais do IF.data no painel
    cols_bancarias = ['Instituicao', 'Data', 'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'NPL']
    df_base = df_painel[[c for c in cols_bancarias if c in df_painel.columns]].copy()

    # Formatar Data no painel para match (garantir DD/MM/YYYY)
    # No painel a data costuma ser YYYY-MM-DD se lida como datetime, ou str
    df_base['Data_DT'] = pd.to_datetime(df_base['Data'])
    df_base['Data_Match'] = df_base['Data_DT'].apply(lambda x: f"01/{x.month:02d}/{x.year}")

    # 4. Merge
    painel_novo = pd.merge(
        df_base,
        df_macro,
        left_on='Data_Match',
        right_on='Data_Key',
        how='left'
    )

    # 5. Limpeza Final
    # Remover chaves auxiliares
    painel_novo.drop(columns=['Data_DT', 'Data_Match', 'Data_Key'], inplace=True, errors='ignore')
    
    # Reordenar colunas
    cols_ordem = ['Instituicao', 'Data', 'RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'IPCA', 'Spread', 'Selic', 'Desemprego', 'NPL']
    painel_novo = painel_novo[[c for c in cols_ordem if c in painel_novo.columns]]

    # 6. Salvar
    painel_novo.to_csv(path_painel, index=False)
    
    print(f"\n[SUCESSO] Painel Integrado salvo em: {path_painel}")
    print(f"Colunas Finais: {painel_novo.columns.tolist()}")
    print(f"Stats Selic - Nulls: {painel_novo['Selic'].isnull().sum()}, Mean: {painel_novo['Selic'].mean()}")

if __name__ == "__main__":
    executar_pipeline()
