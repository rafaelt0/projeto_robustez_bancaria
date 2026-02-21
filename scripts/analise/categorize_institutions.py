import pandas as pd
import numpy as np
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
RANKING_PATH = WORKSPACE_DIR / "outputs" / "stress_tests" / "ranking_robusto_consenso.csv"
SCORES_PATH = WORKSPACE_DIR / "outputs" / "stress_tests" / "ranking_final_robustez_ajustada.csv"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapping Rules
MAPPING = {
    'Bancos Múltiplos e Comerciais': [
        'BB', 'ITAU', 'BRADESCO', 'SANTANDER', 'BANRISUL', 'SAFRA', 'INTER', 
        'AGIBANK', 'ABC-BRASIL', 'DAYCOVAL', 'VOTORANTIM', 'BANCO MASTER',
        'BANESTES', 'BCO DO EST. DE SE', 'MÁXIMA', 'BCO A.J. RENNER', 'SOFISA',
        'BBM', 'ORIGINAL', 'BANCO DIGIMAIS', 'BANCO TOPÁZIO', 'BANCO SEMEAR',
        'BCO CLASSICO', 'BANCO VOITER', 'BCO GUANABARA', 'BS2', 'BCO TRIANGULO',
        'BCO FIBRA', 'BANCO BARI', 'INDUSVAL', 'PINE', 'LECCA', 'ANDBANK',
        'CONFIDENCE', 'BANCO PAULISTA', 'INBURSA', 'SOCIAL BANK', 'MERCANTIL DO BRASIL',
        'INDUSTRIAL DO BRASIL', 'RENDIMENTO', 'BANCO C6', 'LETSBANK', 'GERADOR', 'ALFA',
        'BRB', 'BMG', 'PAN'
    ],
    'Caixa': [
        'CAIXA ECONÔMICA FEDERAL'
    ],
    'Instituições de Pagamento (IP)': [
        'NU PAGAMENTOS', 'MERCADO PAGO', 'PICPAY', 'STONE', 'CIELO', 'NEON', 
        'PAGSEGURO', 'RECARGAPAY', 'EFI S.A.', 'BANQI', 'MAGALUPAY', 'SHPP', 
        'DOCK', 'ASAAS', 'AME DIGITAL', 'PAGUEVELOZ', 'BULLLA', 'RECARGAPAY',
        'SUMUP', 'FISERV', 'BRINKS PAY', 'PINBANK', 'PRUDENCIAL IP', 'EDENRED'
    ],
    'Bancos de Montadoras': [
        'BMW', 'VOLKSWAGEN', 'VOLVO', 'MERCEDES-BENZ', 'TOYOTA', 'HONDA', 
        'JOHN DEERE', 'CATERPILLAR', 'SCANIA', 'PSA FINANCE', 'YAMAHA', 'STELLANTIS', 'GM'
    ],
    'Bancos de Desenvolvimento e Fomento': [
        'BNDES', 'BCO DO NORDESTE', 'BD REGIONAL DO EXTREMO SUL', 
        'AGENCIA FOMENTO', 'DESENBAHIA', 'AF DO ESTADO DE SC', 'AF PARANÁ'
    ],
    'Bancos de Investimento e Corretoras': [
        'BTG PACTUAL', 'GOLDMAN SACHS', 'MORGAN STANLEY', 'XP', 'GENIAL', 
        'BR PARTNERS', 'UBS', 'CREDIT SUISSE', 'BOFA MERRILL LYNCH', 'JP MORGAN',
        'BNY MELLON', 'BOCOM', 'MIRAE', 'GENIAL', 'VORTX', 'INFRA INVESTIMENTOS',
        'CODEPE', 'SOCIA BANK BANCO MULTIPLO', 'INTRA INVESTIMENTOS', 'BRASIL PLURAL',
        'MAF-BRL TRUST', 'TULLETT PREBON', 'OURINVEST', 'TRINUS', 'MODAL', 'BR-CAPITAL',
        'SOCOPA', 'LASTRO RDV', 'H.H.PICCHIONI', 'TRAVELEX', 'PLANNER'
    ],
    'Financeiras (CFI / SCFI)': [
        'CREFISA', 'GRAZZIOTIN', 'OMNI', 'SOROCRED', 'DACASA', 'PORTO SEGURO', 
        'FACTA', 'GAZINCRED', 'ZEMA', 'SAX S.A.', 'COBUCCIO SCFI', 'SANTANA S.A.',
        'LECCA', 'AVISTA', 'CREDSYSTEM', 'GAZIN', 'DM FINANCEIRA', 'HS FINANCEIRA',
        'BBC S.A. - PRUDENCIAL', 'RANDON', 'RODOBENS', 'GOLCRED', 'FATOR', 'BARIGUI',
        'COOPER CARD', 'STARK', 'RPW S/A', 'PARANÁ BANCO', 'VR - PRUDENCIAL', 'CREDIARE',
        'PERNAMBUCANAS', 'SINOSSERRA', 'WILL', 'CREDI SHOP', 'ESTRELA MINEIRA', 'BMP',
        'PROVINCIA', 'SANTINVEST', 'CPCI', 'MONEY PLUS'
    ],
    'Sociedades de Crédito Direto (SCD)': [
        'QI SCD', 'STARK SCD', 'EAGLE SCD', 'FFCRED SCD', 'BARU SCD', 'STARK SCD'
    ],
    'Bancos Cooperativos': [
        'BANCO SICOOB', 'BANCOOB', 'SICREDI', 'BCO COOPERATIVO SICREDI'
    ],
    'Bancos Estrangeiros (Atacado)': [
        'HSBC', 'ING', 'BARCLAYS', 'MIZUHO', 'SUMITOMO MITSUI', 'BANK OF CHINA', 
        'SOCIETE GENERALE', 'DEUTSCHE', 'CITIBANK', 'SCOTIABANK', 'CREDIT AGRICOLE',
        'BNP PARIBAS', 'HAITONG', 'MIZUHO', 'J.P. MORGAN', 'MITSUBISHI', 'CAIXA GERAL',
        'CCB', 'KDB', 'BANIF', 'SCOTIABANK', 'UBS', 'CREDIT SUISSE', 'GOLDMAN SACHS',
        'MORGAN STANLEY', 'BOFA MERRILL LYNCH', 'JP MORGAN CHASE'
    ]
}

def categorize(name):
    name_upper = str(name).upper()
    
    # Priority 1: Keyword Detection
    if ' IP ' in name_upper or name_upper.endswith(' IP'): return 'Instituições de Pagamento (IP)'
    if 'SCFI' in name_upper: return 'Financeiras (CFI / SCFI)'
    if 'CFI' in name_upper: return 'Financeiras (CFI / SCFI)'
    if 'SCD' in name_upper: return 'Sociedades de Crédito Direto (SCD)'
    if 'FOMENTO' in name_upper: return 'Bancos de Desenvolvimento e Fomento'
    
    # Priority 2: Mapping Lookup
    for category, members in MAPPING.items():
        for member in members:
            if member in name_upper:
                return category
                
    # Default
    return 'Outros / Não Categorizados'

def run_categorization():
    if not SCORES_PATH.exists():
        print(f"Error: {SCORES_PATH} not found.")
        return

    df = pd.read_csv(SCORES_PATH)
    df['Categoria'] = df['Instituicao'].apply(categorize)
    
    # Grouped Analysis
    analysis = df.groupby('Categoria').agg({
        'Score_Final_Robustez': ['mean', 'min', 'max', 'count'],
        'Baseline': 'mean',
        'Worst_Stress_Score': 'mean'
    }).reset_index()
    
    analysis.columns = ['Categoria', 'Score_Medio', 'Score_Min', 'Score_Max', 'Num_Instituicoes', 'Baseline_Medio', 'Worst_Medio']
    analysis = analysis.sort_values('Score_Medio', ascending=False)
    
    # Save Results
    df.to_csv(WORKSPACE_DIR / "outputs" / "stress_tests" / "ranking_categorizado.csv", index=False)
    analysis.to_csv(OUTPUT_DIR / "robustez_por_categoria.csv", index=False)
    
    print("Categorization complete.")
    print("\nSummary by Category:")
    print(analysis[['Categoria', 'Score_Medio', 'Num_Instituicoes']].to_string(index=False))

    # Visualization
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=analysis, x='Score_Medio', y='Categoria', palette='viridis')
    plt.title("Robustez Média por Categoria de Instituição")
    plt.xlabel("Score Médio de Robustez (Baseline + Stress)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robustez_categoria_plot.png")

if __name__ == "__main__":
    run_categorization()
