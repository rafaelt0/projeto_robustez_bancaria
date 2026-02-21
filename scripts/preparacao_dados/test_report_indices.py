import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import os
from pathlib import Path

# Paths
OUTPUT_DIR = Path(r"D:\downloads_bcb_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

async def download_test_reports():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()
        
        print("Acessando IF.data...")
        await page.goto('https://www3.bcb.gov.br/ifdata', timeout=60000)
        
        # Selecionar Data: 12/2023 (exemplo recente e estável)
        await page.click('#btnDataBase')
        await page.wait_for_timeout(1000)
        # 12/2023 costuma ser o primeiro ou segundo se 2024 não estiver completo, 
        # mas vamos tentar clicar no texto literal se possível ou no ID
        # No IF.data os itens são li dentro de ul
        await page.click('li:has-text("12/2023")')
        await page.wait_for_timeout(1000)
        
        # Tipo de Instituição: Conglomerados Prudenciais e Instituições Independentes
        await page.click('#btnTipoInst')
        await page.wait_for_timeout(1000)
        await page.click('li:has-text("Conglomerados Prudenciais e Instituições Independentes")')
        await page.wait_for_timeout(1000)
        
        # Listar Relatórios
        await page.click('#btnRelatorio')
        await page.wait_for_timeout(1000)
        
        report_options = page.locator('#ulRelatorio > li')
        count = await report_options.count()
        print(f"Total de relatórios encontrados: {count}")
        
        for i in range(min(5, count)):
            text = await report_options.nth(i).text_content()
            text = text.strip()
            print(f"Relatório {i}: {text}")
            
            # Tentar baixar os dois primeiros (Resumo e Ativo geralmente)
            if i in [0, 1, 2]:
                print(f"Baixando {text}...")
                await report_options.nth(i).click()
                await page.wait_for_timeout(2000)
                
                async with page.expect_download() as download_info:
                    await page.click('#btnExportarCSV')
                
                download = await download_info.value
                filename = f"test_report_{i}_{text.replace(' ', '_')}.csv"
                await download.save_as(OUTPUT_DIR / filename)
                print(f"Salvo em {filename}")
                
                # Re-abrir dropdown para o próximo loop
                await page.click('#btnRelatorio')
                await page.wait_for_timeout(1000)

        await browser.close()

if __name__ == "__main__":
    try:
        asyncio.run(download_test_reports())
    except Exception as e:
        print(f"Erro: {e}")
