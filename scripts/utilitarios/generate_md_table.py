import pandas as pd
from pathlib import Path

# Paths
OUTPUT_DIR = Path(r"d:\projeto_robustez_bancaria\resultados\stress_tests")
SENSITIVITY_FILE = OUTPUT_DIR / "alpha_sensitivity_analysis.csv"
ROBUST_FILE = OUTPUT_DIR / "ranking_robusto_consenso.csv"

def generate_markdown_table():
    if not SENSITIVITY_FILE.exists() or not ROBUST_FILE.exists():
        print("Missing data files.")
        return

    # Load sensitivity and robust ranking
    df_sens = pd.read_csv(SENSITIVITY_FILE)
    df_robust = pd.read_csv(ROBUST_FILE)

    # Values of alpha to include in the table
    target_alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Pivot sensitivity data to get ranks as columns for specific alphas
    df_pivot = df_sens[df_sens['Alpha'].round(1).isin(target_alphas)].pivot(
        index='Instituicao', columns='Alpha', values='Rank'
    )
    
    # Clean column names
    df_pivot.columns = [f"alpha={a:.1f}" for a in df_pivot.columns]
    
    # Merge with Robust Ranking
    final_table = df_robust[['Instituicao', 'Average_Rank']].merge(
        df_pivot, left_on='Instituicao', right_index=True
    )
    
    # Sort by robustness and take top 15
    final_table = final_table.sort_values('Average_Rank').head(15)
    
    # Rename Average_Rank to Robust Cons.
    final_table = final_table.rename(columns={'Average_Rank': 'Robust Cons.'})
    
    # Round ranks
    for col in final_table.columns:
        if col != 'Instituicao':
            final_table[col] = final_table[col].round(1)

    # Convert to Markdown Table
    md_table = final_table.to_markdown(index=False)
    
    print("--- GENERATED TABLE ---")
    print(md_table)
    print("--- END ---")

if __name__ == "__main__":
    generate_markdown_table()
