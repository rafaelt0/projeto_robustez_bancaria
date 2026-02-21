import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_RAW = WORKSPACE_DIR / "data" / "raw" / "painel_final.csv"
STRESS_RESULTS = WORKSPACE_DIR / "outputs" / "stress_tests" / "stress_test_results.csv"
ROBUST_RANK = WORKSPACE_DIR / "outputs" / "stress_tests" / "ranking_robusto_consenso.csv"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "institutional_profiles" / "BRB"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INSTITUICAO = "BRB - PRUDENCIAL"

def generate_brb_panel():
    print(f"Generating panel for {INSTITUICAO}...")
    
    # 1. Load Data
    df_hist = pd.read_csv(DATA_RAW)
    df_stress = pd.read_csv(STRESS_RESULTS)
    df_rank = pd.read_csv(ROBUST_RANK)
    
    # 2. Filter for BRB
    brb_hist = df_hist[df_hist['Instituicao'] == INSTITUICAO].copy()
    brb_hist['Data'] = pd.to_datetime(brb_hist['Data'])
    brb_hist = brb_hist.sort_values('Data')
    
    brb_stress = df_stress[df_stress['Instituicao'] == INSTITUICAO].copy()
    brb_rank_info = df_rank[df_rank['Instituicao'] == INSTITUICAO].iloc[0]
    
    # 3. Create Plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Evolution of NPL and Capital
    plt.subplot(2, 2, 1)
    ax1 = sns.lineplot(data=brb_hist, x='Data', y='NPL', label='NPL %', color='red', marker='o')
    plt.ylabel('NPL (%)')
    ax2 = ax1.twinx()
    sns.lineplot(data=brb_hist, x='Data', y='Capital_Principal', label='Capital Principal', color='blue', marker='x', ax=ax2)
    plt.ylabel('Capital (R$)')
    plt.title("Evolution: NPL vs Capital")
    
    # Plot 2: Evolution of Leverage
    plt.subplot(2, 2, 2)
    sns.lineplot(data=brb_hist, x='Data', y='Alavancagem', color='green', marker='s')
    plt.title("Trend: Leverage (Alavancagem)")
    
    # Plot 3: Stress Test Scenarios
    plt.subplot(2, 2, 3)
    sns.barplot(data=brb_stress, x='Scenario', y='Robustness_Score', palette='viridis')
    plt.axhline(0, color='black', linewidth=1)
    plt.title("Robustness Score by Scenario")
    plt.xticks(rotation=15)
    
    # Plot 4: Rank Summary Info
    plt.subplot(2, 2, 4)
    plt.axis('off')
    info_text = (
        f"INSTITUTIONAL PROFILE: {INSTITUICAO}\n\n"
        f"Robust Consensus Rank: {brb_rank_info['Average_Rank']:.1f}\n"
        f"Baseline Rank (pos): {brb_rank_info['Baseline_Rank']:.0f}\n"
        f"Worst-Case Rank (pos): {brb_rank_info['Worst_Stress_Rank']:.0f}\n\n"
        f"Current NPL (latest): {brb_hist['NPL'].iloc[-1]:.2f}%\n"
        f"Current Leverage: {brb_hist['Alavancagem'].iloc[-1]:.2f}\n"
        f"Current Capital: R$ {brb_hist['Capital_Principal'].iloc[-1]:,.2f}\n"
    )
    plt.text(0.1, 0.5, info_text, fontsize=12, family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "brb_profile_dashboard.png")
    
    # 4. Save CSV summary
    brb_stress.to_csv(OUTPUT_DIR / "brb_stress_summary.csv", index=False)
    brb_hist.to_csv(OUTPUT_DIR / "brb_historical_clean.csv", index=False)
    
    print(f"Panel success! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_brb_panel()
