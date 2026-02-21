import asyncio
from playwright.async_api import async_playwright
from pathlib import Path
import re
from datetime import datetime

async def download_bcb_data_all():
    """
    Script para baixar TODOS os relatórios do IF.data do Banco Central do Brasil
    Para TODAS as datas disponíveis: mantém a primeira instituição e baixa todos os relatórios
    """
    # Configurar diretório de download
    download_dir = Path(r"D:\downloads_bcb")
    download_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        # Iniciar navegador
        browser = await p.chromium.launch(
            headless=False,  # Mude para True para rodar sem interface
            slow_mo=50
        )
        
        context = await browser.new_context(
            accept_downloads=True
        )
        
        page = await context.new_page()
        
        # Navegar para a página
        print("Acessando IF.data...")
        await page.goto("https://www3.bcb.gov.br/ifdata")
        await page.wait_for_load_state("networkidle")
        
        # 1. OBTER TODAS AS DATAS DISPONÍVEIS (SEM FILTRO)
        print("=== OBTENDO TODAS AS DATAS DISPONÍVEIS ===")
        
        await page.click('#btnDataBase')
        await page.wait_for_timeout(1000)
        
        database_options = page.locator('#ulDataBase > li')
        db_count = await database_options.count()
        
        # Extrair TODAS as datas válidas
        database_items = []
        for i in range(db_count):
            text = await database_options.nth(i).text_content()
            text = text.strip()
            
            # Filtrar apenas datas válidas (formato: MM/YYYY)
            if text and re.search(r'\d{2}/\d{4}', text):
                database_items.append({
                    'index': i,
                    'text': text
                })
        
        print(f"✅ Total de datas encontradas: {len(database_items)}")
        if database_items:
            print(f"📅 Primeira data: {database_items[0]['text']}")
            print(f"📅 Última data: {database_items[-1]['text']}\n")
        else:
            print("⚠️  Nenhuma data encontrada!")
            await browser.close()
            return
        
        # Fechar dropdown
        await page.keyboard.press('Escape')
        await page.wait_for_timeout(500)
        
        # Contadores
        total_downloads = 0
        total_errors = 0
        dates_processed = 0
        
        # 2. ITERAR POR CADA DATA
        for idx, db_item in enumerate(database_items):
            print(f"\n{'='*70}")
            print(f"[{idx + 1}/{len(database_items)}] DATA: {db_item['text']}")
            print(f"{'='*70}")
            
            try:
                # PASSO 1: Selecionar data-base
                await page.click('#btnDataBase')
                await page.wait_for_timeout(500)
                
                await page.locator('#ulDataBase > li').nth(db_item['index']).click()
                await page.wait_for_timeout(2000)
                print(f"  ✓ Data selecionada: {db_item['text']}")
                
                # PASSO 2: Selecionar tipo de instituição (PRIMEIRA OPÇÃO - MANTÉM FIXA)
                await page.click('#btnTipoInst')
                await page.wait_for_timeout(1000)
                
                institution_options = page.locator('#ulTipoInst > li')
                inst_count = await institution_options.count()
                
                selected_institution = None
                
                if inst_count > 0:
                    # Pegar primeira opção válida
                    for i in range(inst_count):
                        inst_text = await institution_options.nth(i).text_content()
                        inst_text = inst_text.strip()
                        
                        if inst_text and not inst_text.startswith('--'):
                            await institution_options.nth(i).click()
                            await page.wait_for_timeout(2000)
                            selected_institution = inst_text
                            print(f"  ✓ Instituição (FIXA): {inst_text}")
                            break
                    
                    if not selected_institution:
                        print("  ⚠️  Nenhuma instituição válida encontrada")
                        await page.keyboard.press('Escape')
                        total_errors += 1
                        continue
                else:
                    print("  ⚠️  Nenhuma instituição disponível")
                    await page.keyboard.press('Escape')
                    total_errors += 1
                    continue
                
                # PASSO 3: OBTER TODOS OS RELATÓRIOS DISPONÍVEIS
                await page.click('#btnRelatorio')
                await page.wait_for_timeout(1000)
                
                report_options = page.locator('#ulRelatorio > li')
                rep_count = await report_options.count()
                
                # Coletar todos os cinco primeiros relatórios
                report_items = []
                for i in range(4,5):
                    rep_text = await report_options.nth(i).text_content()
                    rep_text = rep_text.strip()
                    
                    if rep_text and not rep_text.startswith('--'):
                        report_items.append({
                            'index': i,
                            'text': rep_text
                        })
                
                if not report_items:
                    print("  ⚠️  Nenhum relatório disponível")
                    await page.keyboard.press('Escape')
                    total_errors += 1
                    continue
                
                print(f"  📋 Total de relatórios encontrados: {len(report_items)}")
                
                # Fechar dropdown de relatórios
                await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
                
                # PASSO 4: ITERAR POR CADA RELATÓRIO
                date_downloads = 0
                for rep_idx, report_item in enumerate(report_items):
                    print(f"\n  → [{rep_idx + 1}/{len(report_items)}] Relatório: {report_item['text'][:60]}...")
                    
                    try:
                        # Abrir dropdown de relatório
                        await page.click('#btnRelatorio')
                        await page.wait_for_timeout(500)
                        
                        # Selecionar relatório específico
                        await page.locator('#ulRelatorio > li').nth(report_item['index']).click()
                        await page.wait_for_timeout(2000)
                        
                        # Baixar CSV
                        csv_link = page.locator('a:has-text("CSV")').first
                        
                        if await csv_link.count() > 0:
                            try:
                                async with page.expect_download(timeout=30000) as download_info:
                                    await csv_link.click()
                                
                                download = await download_info.value
                                
                                # Criar nome seguro e descritivo para o arquivo
                                safe_date = re.sub(r'[^\w\-]', '_', db_item['text'])
                                safe_inst = re.sub(r'[^\w\-]', '_', selected_institution)[:30]  # Limitar tamanho
                                safe_report = re.sub(r'[^\w\-]', '_', report_item['text'])[:50]  # Limitar tamanho
                                
                                filename = f"bcb_{safe_date}_{safe_inst}_{safe_report}.csv"
                                
                                save_path = download_dir / filename
                                await download.save_as(str(save_path))
                                
                                # Verificar tamanho do arquivo
                                size_kb = save_path.stat().st_size / 1024
                                print(f"    ✅ Download OK ({size_kb:.1f} KB)")
                                total_downloads += 1
                                date_downloads += 1
                                
                            except Exception as e:
                                print(f"    ❌ Erro no download: {str(e)}")
                                total_errors += 1
                        else:
                            print(f"    ⚠️  Link CSV não encontrado")
                            total_errors += 1
                        
                        # Pequena pausa entre downloads
                        await page.wait_for_timeout(800)
                        
                    except Exception as e:
                        print(f"    ❌ Erro ao processar relatório: {str(e)}")
                        total_errors += 1
                        
                        # Tentar fechar dropdown
                        try:
                            await page.keyboard.press('Escape')
                            await page.wait_for_timeout(300)
                        except:
                            pass
                
                dates_processed += 1
                print(f"\n  📊 Resumo da data {db_item['text']}: {date_downloads} arquivo(s) baixado(s)")
                
                # Pausa entre datas
                await page.wait_for_timeout(500)
                
            except Exception as e:
                print(f"  ❌ Erro ao processar data {db_item['text']}: {str(e)}")
                total_errors += 1
                
                # Tentar resetar estado
                try:
                    await page.keyboard.press('Escape')
                    await page.wait_for_timeout(500)
                except:
                    pass
                
                continue
        
        # Resumo final detalhado
        print(f"\n{'='*70}")
        print("✅ PROCESSO CONCLUÍDO!")
        print(f"{'='*70}")
        print(f"📊 Estatísticas Gerais:")
        print(f"   • Total de datas disponíveis: {len(database_items)}")
        print(f"   • Datas processadas com sucesso: {dates_processed}")
        print(f"   • Downloads bem-sucedidos: {total_downloads}")
        print(f"   • Erros encontrados: {total_errors}")
        
        if dates_processed > 0:
            avg_per_date = total_downloads / dates_processed
            print(f"   • Média de arquivos por data: {avg_per_date:.1f}")
        
        print(f"\n📁 Diretório: {download_dir.absolute()}")
        
        # Listar arquivos baixados
        files = sorted(download_dir.glob('*.csv'))
        print(f"📄 Total de arquivos salvos: {len(files)}")
        
        if files:
            total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
            print(f"💾 Tamanho total: {total_size:.2f} MB")
            
            print(f"\n📋 Primeiros 10 arquivos:")
            for f in files[:10]:
                size_kb = f.stat().st_size / 1024
                print(f"   • {f.name[:70]}... ({size_kb:.1f} KB)")
            
            if len(files) > 10:
                print(f"   • ... e mais {len(files) - 10} arquivo(s)")
            
            print(f"\n📋 Últimos 5 arquivos:")
            for f in files[-5:]:
                size_kb = f.stat().st_size / 1024
                print(f"   • {f.name[:70]}... ({size_kb:.1f} KB)")
        
        print(f"{'='*70}\n")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(download_bcb_data_all())

async def download_bcb_data():
    """
    Script para baixar TODOS os relatórios do IF.data do Banco Central do Brasil
    Para cada data (12/2015 até 12/2024): mantém a primeira instituição e baixa todos os relatórios disponíveis
    """
    # Configurar diretório de download
    download_dir = Path(r"D:\downloads_bcb")
    download_dir.mkdir(exist_ok=True)
    
    async with async_playwright() as p:
        # Iniciar navegador
        browser = await p.chromium.launch(
            headless=False,  # Mude para True para rodar sem interface
            slow_mo=50
        )
        
        context = await browser.new_context(
            accept_downloads=True
        )
        
        page = await context.new_page()
        
        # Navegar para a página
        print("Acessando IF.data...")
        await page.goto("https://www3.bcb.gov.br/ifdados/index2024.html")
        await page.wait_for_load_state("networkidle")
        
        # Clicar na aba "Dados de 2000 a 2024"
        try:
            await page.click('text="Dados de 2000 a 2024"')
            await page.wait_for_timeout(2000)
            print("✓ Aba 'Dados de 2000 a 2024' selecionada\n")
        except:
            print("Já estava na aba correta\n")
        
        # 1. OBTER TODAS AS DATAS DISPONÍVEIS E FILTRAR
        print("=== OBTENDO LISTA DE DATAS (12/2015 - 12/2024) ===")
        
        await page.click('#btnDataBase')
        await page.wait_for_timeout(1000)
        
        database_options = page.locator('#ulDataBase > li')
        db_count = await database_options.count()
        
        # Extrair textos e filtrar
        database_items = []
        for i in range(db_count):
            text = await database_options.nth(i).text_content()
            text = text.strip()
            
            # Filtrar apenas datas válidas (formato: MM/YYYY)
            if text and re.search(r'\d{2}/\d{4}', text):
                # Converter para datetime para comparação
                try:
                    date_obj = datetime.strptime(text, '%m/%Y')
                    start_date = datetime(2015, 12, 1)  # 12/2015
                    end_date = datetime(2024, 12, 31)    # 12/2024
                    
                    # Filtrar apenas datas no intervalo
                    if start_date <= date_obj <= end_date:
                        database_items.append({
                            'index': i,
                            'text': text,
                            'date': date_obj
                        })
                except ValueError:
                    # Se não conseguir parsear, ignora
                    continue
        
        # Ordenar por data (mais recente primeiro ou mais antiga primeiro)
        database_items.sort(key=lambda x: x['date'], reverse=True)  # Mude para False para ordem crescente
        
        print(f"Total de datas no intervalo (12/2015 - 12/2024): {len(database_items)}")
        if database_items:
            print(f"Primeira data a processar: {database_items[0]['text']}")
            print(f"Última data a processar: {database_items[-1]['text']}\n")
        else:
            print("⚠️  Nenhuma data encontrada no intervalo especificado!")
            await browser.close()
            return
        
        # Fechar dropdown
        await page.keyboard.press('Escape')
        await page.wait_for_timeout(500)
        
        # Contadores
        total_downloads = 0
        total_errors = 0
        
        # 2. ITERAR POR CADA DATA
        for idx, db_item in enumerate(database_items):
            print(f"\n{'='*70}")
            print(f"[{idx + 1}/{len(database_items)}] DATA: {db_item['text']}")
            print(f"{'='*70}")
            
            try:
                # PASSO 1: Selecionar data-base
                await page.click('#btnDataBase')
                await page.wait_for_timeout(500)
                
                await page.locator('#ulDataBase > li').nth(db_item['index']).click()
                await page.wait_for_timeout(2000)
                print(f"  ✓ Data selecionada: {db_item['text']}")
                
                # PASSO 2: Selecionar tipo de instituição (PRIMEIRA OPÇÃO - MANTÉM FIXA)
                await page.click('#btnTipoInst')
                await page.wait_for_timeout(1000)
                
                institution_options = page.locator('#ulTipoInst > li')
                inst_count = await institution_options.count()
                
                selected_institution = None
                
                if inst_count > 0:
                    # Pegar primeira opção válida
                    for i in range(inst_count):
                        inst_text = await institution_options.nth(i).text_content()
                        inst_text = inst_text.strip()
                        
                        if inst_text and not inst_text.startswith('--'):
                            await institution_options.nth(i).click()
                            await page.wait_for_timeout(2000)
                            selected_institution = inst_text
                            print(f"  ✓ Instituição (FIXA): {inst_text}")
                            break
                    
                    if not selected_institution:
                        print("  ⚠️  Nenhuma instituição válida encontrada")
                        await page.keyboard.press('Escape')
                        total_errors += 1
                        continue
                else:
                    print("  ⚠️  Nenhuma instituição disponível")
                    await page.keyboard.press('Escape')
                    total_errors += 1
                    continue
                
                # PASSO 3: OBTER TODOS OS RELATÓRIOS DISPONÍVEIS
                await page.click('#btnRelatorio')
                await page.wait_for_timeout(1000)
                
                report_options = page.locator('#ulRelatorio > li')
                rep_count = await report_options.count()
                
                # Coletar todos os relatórios válidos
                report_items = []
                for i in range(4,5):
                    rep_text = await report_options.nth(i).text_content()
                    rep_text = rep_text.strip()
                    
                    if rep_text and not rep_text.startswith('--'):
                        report_items.append({
                            'index': i,
                            'text': rep_text
                        })
                
                if not report_items:
                    print("  ⚠️  Nenhum relatório disponível")
                    await page.keyboard.press('Escape')
                    total_errors += 1
                    continue
                
                print(f"  📋 Total de relatórios encontrados: {len(report_items)}")
                
                # Fechar dropdown de relatórios
                await page.keyboard.press('Escape')
                await page.wait_for_timeout(500)
                
                # PASSO 4: ITERAR POR CADA RELATÓRIO
                for rep_idx, report_item in enumerate(report_items):
                    print(f"\n  → [{rep_idx + 1}/{len(report_items)}] Relatório: {report_item['text']}")
                    
                    try:
                        # Abrir dropdown de relatório
                        await page.click('#btnRelatorio')
                        await page.wait_for_timeout(500)
                        
                        # Selecionar relatório específico
                        await page.locator('#ulRelatorio > li').nth(report_item['index']).click()
                        await page.wait_for_timeout(2000)
                        
                        # Baixar CSV
                        csv_link = page.locator('a:has-text("CSV")').first
                        
                        if await csv_link.count() > 0:
                            try:
                                async with page.expect_download(timeout=30000) as download_info:
                                    await csv_link.click()
                                
                                download = await download_info.value
                                
                                # Criar nome seguro e descritivo para o arquivo
                                safe_date = re.sub(r'[^\w\-]', '_', db_item['text'])
                                safe_inst = re.sub(r'[^\w\-]', '_', selected_institution)
                                safe_report = re.sub(r'[^\w\-]', '_', report_item['text'])
                                
                                filename = f"bcb_{safe_date}_{safe_inst}_{safe_report}.csv"
                                
                                save_path = download_dir / filename
                                await download.save_as(str(save_path))
                                
                                # Verificar tamanho do arquivo
                                size_kb = save_path.stat().st_size / 1024
                                print(f"    ✅ Download OK: {filename} ({size_kb:.1f} KB)")
                                total_downloads += 1
                                
                            except Exception as e:
                                print(f"    ❌ Erro no download: {str(e)}")
                                total_errors += 1
                        else:
                            print(f"    ⚠️  Link CSV não encontrado")
                            total_errors += 1
                        
                        # Pequena pausa entre downloads
                        await page.wait_for_timeout(800)
                        
                    except Exception as e:
                        print(f"    ❌ Erro ao processar relatório: {str(e)}")
                        total_errors += 1
                        
                        # Tentar fechar dropdown
                        try:
                            await page.keyboard.press('Escape')
                            await page.wait_for_timeout(300)
                        except:
                            pass
                
                # Pausa entre datas
                await page.wait_for_timeout(500)
                
            except Exception as e:
                print(f"  ❌ Erro ao processar data {db_item['text']}: {str(e)}")
                total_errors += 1
                
                # Tentar resetar estado
                try:
                    await page.keyboard.press('Escape')
                    await page.wait_for_timeout(500)
                except:
                    pass
                
                continue
        
        # Resumo final
        print(f"\n{'='*70}")
        print("✅ PROCESSO CONCLUÍDO!")
        print(f"{'='*70}")
        print(f"📊 Estatísticas:")
        print(f"   • Período: 12/2015 - 12/2024")
        print(f"   • Total de datas processadas: {len(database_items)}")
        print(f"   • Downloads bem-sucedidos: {total_downloads}")
        print(f"   • Erros encontrados: {total_errors}")
        print(f"\n📁 Diretório: {download_dir.absolute()}")
        
        # Listar arquivos baixados
        files = sorted(download_dir.glob('*.csv'))
        print(f"📄 Total de arquivos salvos: {len(files)}")
        
        if files:
            total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)  # MB
            print(f"💾 Tamanho total: {total_size:.2f} MB")
            
            print(f"\n📋 Primeiros 10 arquivos:")
            for f in files[:10]:
                size_kb = f.stat().st_size / 1024
                print(f"   • {f.name} ({size_kb:.1f} KB)")
            
            if len(files) > 10:
                print(f"   • ... e mais {len(files) - 10} arquivo(s)")
        
        print(f"{'='*70}\n")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(download_bcb_data())