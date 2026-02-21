
import pandas as pd
import os

def clean_institution_name(name):
    """Removes the suffix ' - PRUDENCIAL' from institution names."""
    return name.replace(' - PRUDENCIAL', '')

def get_rating(score):
    """Assigns a rating based on the Robustness Score."""
    if score >= 30:
        return 'AAA'
    elif score >= 15:
        return 'AA'
    elif score >= 10:
        return 'A'
    elif score >= 5:
        return 'BBB'
    elif score >= 2:
        return 'BB'
    elif score >= 0:
        return 'B'
    else:
        return 'CCC'

def generate_latex_ranking():
    # Define paths - using absolute paths based on the project structure
    base_path = r'd:\projeto_robustez_bancaria'
    input_path = os.path.join(base_path, 'outputs', 'stress_tests', 'ranking_categorizado.csv')
    output_path = os.path.join(base_path, 'outputs', 'reports', 'tabela_ranking_completo.tex')
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # Load data
    try:
        df = pd.read_csv(input_path)
        
        # Load stress probability data (Use Current Baseline Probability, not Historical Average)
        # We use stress_test_results.csv which contains the specific probs for each scenario including Baseline
        prob_path = os.path.join(base_path, 'outputs', 'stress_tests', 'stress_test_results.csv')
        
        if os.path.exists(prob_path):
            df_stress_results = pd.read_csv(prob_path)
            
            # Filter for Baseline scenario to get current underlying probability
            df_baseline = df_stress_results[df_stress_results['Scenario'] == 'Baseline'].copy()
            
            # Rename for merge
            df_baseline = df_baseline[['Instituicao', 'Prob_Stress']]
            df_baseline = df_baseline.rename(columns={'Prob_Stress': 'Prob_Estresse_Atual'})
            
            # Merge on Instituicao
            df = pd.merge(df, df_baseline, on='Instituicao', how='left')
            
            # Use the new column
            prob_column = 'Prob_Estresse_Atual'
        else:
            print(f"Warning: Probability file not found at {prob_path}")
            df['Prob_Estresse_Atual'] = 0.0
            prob_column = 'Prob_Estresse_Atual'

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Sort by Score_Final_Robustez descending just in case
    if 'Score_Final_Robustez' in df.columns:
        df = df.sort_values(by='Score_Final_Robustez', ascending=False).reset_index(drop=True)
    
    # Add Rank column
    df['Rank'] = df.index + 1

    # Add Rating column
    if 'Score_Final_Robustez' in df.columns:
        df['Rating'] = df['Score_Final_Robustez'].apply(get_rating)
    else:
        df['Rating'] = 'N/A'

    # Clean names
    if 'Instituicao' in df.columns:
        df['Instituicao_Clean'] = df['Instituicao'].apply(clean_institution_name)
    else:
        print("Error: 'Instituicao' column not found.")
        return

    # Select and rename columns for the table
    # Columns: Rank, Institution, Category, Rating, Robustness Score, Stress Probability
    if 'Categoria' not in df.columns:
        df['Categoria'] = 'N/A' # Fallback
        
    latex_content = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{longtable}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{a4paper, margin=1in}",
        r"\begin{document}",
        r"",
        r"%\section*{Ranking Completo de Robustez Bancária}",
        r"",
        r"\begin{longtable}{clllcc}", # Added column for Rating
        r"\caption{Ranking Completo de Robustez Bancária} \label{tab:ranking_completo} \\",
        r"\toprule",
        r"\textbf{Pos.} & \textbf{Instituição} & \textbf{Categoria} & \textbf{Rating} & \textbf{Score} & \textbf{Prob. Estresse} \\",
        r"\midrule",
        r"\endfirsthead",
        r"",
        r"\multicolumn{6}{c}%", # Updated column count
        r"{{\tablename\ \thetable{} -- continuação da página anterior}} \\",
        r"\toprule",
        r"\textbf{Pos.} & \textbf{Instituição} & \textbf{Categoria} & \textbf{Rating} & \textbf{Score} & \textbf{Prob. Estresse} \\",
        r"\midrule",
        r"\endhead",
        r"",
        r"\midrule",
        r"\multicolumn{6}{r}{{Continua na próxima página}} \\", # Updated column count
        r"\endfoot",
        r"",
        r"\bottomrule",
        r"\endlastfoot",
        r""
    ]

    for _, row in df.iterrows():
        rank = row['Rank']
        name = row['Instituicao_Clean'].replace('&', '\\&') # Escape specialized latex characters
        category = row['Categoria'].replace('&', '\\&')
        rating = row['Rating']
        score = f"{row['Score_Final_Robustez']:.3f}"
        
        # Format probability
        prob_val = row.get(prob_column, 0.0)
        if pd.isna(prob_val):
            prob_val = 0.0
            
        prob_pct = prob_val * 100
        if prob_val < 0.0001 and prob_val > 0:
             prob_str = "$<0.01\\%$"
        else:
             prob_str = f"{prob_pct:.2f}\\%"
        
        latex_content.append(f"{rank} & {name} & {category} & {rating} & {score} & {prob_str} \\\\")

    latex_content.extend([
        r"\end{longtable}",
        r"\end{document}"
    ])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_content))
    
    print(f"LaTeX tables successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_latex_ranking()
