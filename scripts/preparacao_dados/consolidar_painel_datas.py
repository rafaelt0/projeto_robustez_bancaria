import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

def parse_date_from_filename(filename):
    """Extrai a data do nome do arquivo (formato: bcb_MM_YYYY_...)"""
    match = re.search(r'bcb_(\d{2})_(\d{4})_', filename)
    if match:
        month, year = match.groups()
        return f"{year}-{month}-01"
    return None

def flatten_multiindex_columns(df, sep=" | "):
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []

        for col in df.columns:
            top = str(col[0]).strip()
            bottom = str(col[-1]).strip()

            if top == bottom:
                new_cols.append(top)
            else:
                new_cols.append(f"{top}{sep}{bottom}")

        df.columns = new_cols
    else:
        df.columns = [str(col).strip() for col in df.columns]

    return df



def read_csv_robust(file_path):
    """
    Lê CSV com tratamento robusto para diferentes formatos e encodings
    """
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    separators = [';', ',', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                # Tentar ler com header multi-level
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, header=[0, 1])
                if len(df.columns) > 1:
                    df = flatten_multiindex_columns(df)
                    return df
            except:
                pass
            
            try:
                # Tentar ler com header simples
                df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                if len(df.columns) > 1:
                    df = flatten_multiindex_columns(df)
                    return df
            except:
                pass
    
    raise ValueError(f"Não foi possível ler o arquivo {file_path.name}")

def consolidate_reports_by_date():
    """
    Consolida todos os relatórios CSV por data-base
    Mantém as 5 primeiras colunas apenas do primeiro arquivo
    Trata multi-level headings mantendo apenas o nível mais baixo
    """
    
    download_dir = Path(r"D:\downloads_bcb")
    output_dir = Path("./dados/consolidados")
    output_dir.mkdir(exist_ok=True)
    
    if not download_dir.exists():
        print(f"❌ Diretório {download_dir} não encontrado!")
        print(f"   Execute primeiro o script de download.")
        return None
    
    # Agrupar arquivos por data
    files_by_date = {}
    
    print("=" * 70)
    print("CONSOLIDANDO RELATÓRIOS POR DATA")
    print("=" * 70)
    
    all_files = list(download_dir.glob('*.csv'))
    print(f"✓ Total de arquivos CSV encontrados: {len(all_files)}\n")
    
    for csv_file in sorted(all_files):
        date_str = parse_date_from_filename(csv_file.name)
        if date_str:
            if date_str not in files_by_date:
                files_by_date[date_str] = []
            files_by_date[date_str].append(csv_file)
        else:
            print(f"⚠️  Não foi possível extrair data de: {csv_file.name}")
    
    print(f"✓ Total de datas encontradas: {len(files_by_date)}")
    print(f"✓ Total de arquivos agrupados: {sum(len(v) for v in files_by_date.values())}\n")
    
    # Lista para armazenar todos os dataframes consolidados
    all_consolidated = []
    stats = {
        'dates_processed': 0,
        'dates_failed': 0,
        'total_files': 0,
        'files_failed': 0
    }
    
    # Processar cada data
    for idx, (date_str, files) in enumerate(sorted(files_by_date.items()), 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(files_by_date)}] DATA: {date_str}")
        print(f"{'='*70}")
        print(f"Relatórios encontrados: {len(files)}")
        
        try:
            consolidated_df = None
            first_5_cols = None
            successful_merges = 0
            
            for file_idx, file_path in enumerate(files):
                stats['total_files'] += 1
                
                try:
                    # Ler CSV
                    df = read_csv_robust(file_path)
                    
                    # Remover linhas completamente vazias
                    df = df.dropna(how='all')
                    
                    # Remover colunas completamente vazias
                    df = df.dropna(axis=1, how='all')
                    
                    if len(df) == 0:
                        print(f"  ⚠️  Arquivo {file_idx + 1} está vazio: {file_path.name}")
                        stats['files_failed'] += 1
                        continue
                    
                    # Se é o primeiro arquivo, manter todas as colunas
                    if consolidated_df is None:
                        consolidated_df = df.copy()
                        first_5_cols = list(df.columns[:5])
                        print(f"  ✓ Base criada: {len(df)} linhas × {len(df.columns)} colunas")
                        print(f"    ID columns: {first_5_cols[:3]}...")
                        successful_merges += 1
                    else:
                        # Para arquivos seguintes, pegar apenas colunas a partir da 6ª
                        if len(df.columns) > 5:
                            new_cols = list(df.columns[5:])
                            
                            # Verificar se as primeiras 5 colunas são compatíveis
                            current_first_5 = list(df.columns[:5])
                            
                            # Merge nas primeiras 5 colunas
                            try:
                                # Renomear as primeiras 5 colunas para garantir compatibilidade
                                df_merge = df.copy()
                                for i in range(min(5, len(df.columns))):
                                    df_merge.columns.values[i] = first_5_cols[i]
                                
                                # Fazer merge
                                consolidated_df = consolidated_df.merge(
                                    df_merge[first_5_cols + new_cols],
                                    on=first_5_cols,
                                    how='outer',
                                    suffixes=('', f'_dup{file_idx}')
                                )
                                
                                print(f"  ✓ Relatório {file_idx + 1}: +{len(new_cols)} colunas | Total: {len(consolidated_df.columns)} colunas")
                                successful_merges += 1
                                
                            except Exception as e:
                                print(f"  ❌ Erro no merge do arquivo {file_idx + 1}: {str(e)[:100]}")
                                stats['files_failed'] += 1
                        else:
                            print(f"  ⚠️  Arquivo {file_idx + 1} tem apenas {len(df.columns)} colunas (esperado > 5)")
                            stats['files_failed'] += 1
                
                except Exception as e:
                    print(f"  ❌ Erro ao ler arquivo {file_idx + 1}: {str(e)[:100]}")
                    print(f"     Arquivo: {file_path.name}")
                    stats['files_failed'] += 1
                    continue
            
            if consolidated_df is not None and len(consolidated_df) > 0:
                # Adicionar coluna de data
                consolidated_df['data_base'] = date_str
                
                # Mover data_base para segunda coluna
                cols = list(consolidated_df.columns)
                cols.insert(1, cols.pop(cols.index('data_base')))
                consolidated_df = consolidated_df[cols]
                
                # Salvar arquivo consolidado por data
                output_file = output_dir / f"consolidado_{date_str}.csv"
                consolidated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
                
                # Calcular estatísticas
                rows, cols = consolidated_df.shape
                size_mb = output_file.stat().st_size / (1024 * 1024)
                
                print(f"\n  ✅ CONSOLIDADO SALVO")
                print(f"     • Arquivo: {output_file.name}")
                print(f"     • Dimensões: {rows:,} linhas × {cols} colunas")
                print(f"     • Tamanho: {size_mb:.2f} MB")
                print(f"     • Merges bem-sucedidos: {successful_merges}/{len(files)}")
                
                # Adicionar à lista geral
                all_consolidated.append(consolidated_df)
                stats['dates_processed'] += 1
            else:
                print(f"  ❌ Nenhum dado consolidado para esta data")
                stats['dates_failed'] += 1
        
        except Exception as e:
            print(f"  ❌ Erro ao consolidar data {date_str}: {str(e)}")
            stats['dates_failed'] += 1
            continue
    
    # Criar painel completo (empilhando todas as datas)
    if all_consolidated:
        print(f"\n{'='*70}")
        print("CRIANDO PAINEL COMPLETO")
        print(f"{'='*70}")
        
        panel_df = pd.concat(all_consolidated, ignore_index=True)
        
        # Converter data_base para datetime
        panel_df['data_base'] = pd.to_datetime(panel_df['data_base'])
        
        # Ordenar por data e primeira coluna
        id_col = panel_df.columns[0]
        panel_df = panel_df.sort_values(['data_base', id_col])
        
        # Salvar painel completo
        panel_file = output_dir / "painel_completo.csv"
        panel_df.to_csv(panel_file, index=False, encoding='utf-8-sig')
        
        # Calcular estatísticas finais
        rows, cols = panel_df.shape
        size_mb = panel_file.stat().st_size / (1024 * 1024)
        
        print(f"\n✅ PAINEL COMPLETO CRIADO!")
        print(f"\n📊 ESTATÍSTICAS FINAIS:")
        print(f"   • Arquivo: {panel_file.name}")
        print(f"   • Total de observações: {rows:,}")
        print(f"   • Total de colunas: {cols}")
        print(f"   • Tamanho do arquivo: {size_mb:.2f} MB")
        print(f"   • Período: {panel_df['data_base'].min().strftime('%Y-%m')} até {panel_df['data_base'].max().strftime('%Y-%m')}")
        print(f"   • Entidades únicas: {panel_df[id_col].nunique():,}")
        
        # Estatísticas de processamento
        print(f"\n📈 PROCESSAMENTO:")
        print(f"   • Datas processadas com sucesso: {stats['dates_processed']}/{len(files_by_date)}")
        print(f"   • Datas com falha: {stats['dates_failed']}")
        print(f"   • Arquivos processados: {stats['total_files'] - stats['files_failed']}/{stats['total_files']}")
        print(f"   • Arquivos com falha: {stats['files_failed']}")
        
        # Estatísticas por data
        obs_per_date = panel_df.groupby('data_base').size()
        print(f"\n📊 OBSERVAÇÕES POR PERÍODO:")
        print(f"   • Mínimo: {obs_per_date.min():,} observações")
        print(f"   • Máximo: {obs_per_date.max():,} observações")
        print(f"   • Média: {obs_per_date.mean():.0f} observações")
        print(f"   • Mediana: {obs_per_date.median():.0f} observações")
        
        # Mostrar primeiras e últimas datas
        print(f"\n📅 PRIMEIRAS 5 DATAS:")
        for date, count in obs_per_date.head().items():
            print(f"   • {date.strftime('%Y-%m')}: {count:,} observações")
        
        print(f"\n📅 ÚLTIMAS 5 DATAS:")
        for date, count in obs_per_date.tail().items():
            print(f"   • {date.strftime('%Y-%m')}: {count:,} observações")
        
        # Informações sobre as colunas
        print(f"\n📋 ESTRUTURA DAS COLUNAS:")
        print(f"   • Coluna ID (1ª): {id_col}")
        print(f"   • Coluna Data (2ª): data_base")
        print(f"   • Primeiras 10 colunas:")
        for i, col in enumerate(panel_df.columns[:10], 1):
            print(f"      {i}. {col}")
        if cols > 10:
            print(f"      ... e mais {cols - 10} colunas")
        
        # Salvar metadados
        metadata_file = output_dir / "metadata.txt"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write(f"PAINEL COMPLETO - METADADOS\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Data de criação: {pd.Timestamp.now()}\n")
            f.write(f"Arquivo: {panel_file.name}\n\n")
            f.write(f"DIMENSÕES:\n")
            f.write(f"  Observações: {rows:,}\n")
            f.write(f"  Colunas: {cols}\n")
            f.write(f"  Tamanho: {size_mb:.2f} MB\n\n")
            f.write(f"PERÍODO:\n")
            f.write(f"  Início: {panel_df['data_base'].min()}\n")
            f.write(f"  Fim: {panel_df['data_base'].max()}\n")
            f.write(f"  Entidades únicas: {panel_df[id_col].nunique():,}\n\n")
            f.write(f"COLUNAS:\n")
            for i, col in enumerate(panel_df.columns, 1):
                f.write(f"  {i}. {col}\n")
        
        print(f"\n✅ Metadados salvos em: {metadata_file}")
        print(f"\n{'='*70}\n")
        
        return panel_df
    
    else:
        print(f"\n❌ NENHUM DADO FOI CONSOLIDADO!")
        print(f"\nESTATÍSTICAS DE ERRO:")
        print(f"   • Datas com falha: {stats['dates_failed']}/{len(files_by_date)}")
        print(f"   • Arquivos com falha: {stats['files_failed']}/{stats['total_files']}")
        return None

if __name__ == "__main__":
    panel_df = consolidate_reports_by_date()
    
    if panel_df is not None:
        print("\n✅ Processo concluído com sucesso!")
        print(f"\n📁 Arquivos criados:")
        print(f"   • ./dados/consolidados/painel_completo.csv")
        print(f"   • ./dados/consolidados/consolidado_YYYY-MM-DD.csv (um por data)")
        print(f"   • ./dados/consolidados/metadata.txt")
    else:
        print("\n❌ Processo finalizado com erros.")