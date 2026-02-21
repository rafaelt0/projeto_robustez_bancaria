import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score

# Paths
WORKSPACE_DIR = Path(r"d:\projeto_robustez_bancaria")
DATA_PATH = WORKSPACE_DIR / "data" / "raw" / "painel_final.csv"
OUTPUT_DIR = WORKSPACE_DIR / "outputs" / "stress_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Macro series (fallback identical to production model)
macro_data = {
    'Data': ['2015-12-01', '2016-03-01', '2016-06-01', '2016-09-01', '2016-12-01',
             '2017-03-01', '2017-06-01', '2017-09-01', '2017-12-01',
             '2018-03-01', '2018-06-01', '2018-09-01', '2018-12-01',
             '2019-03-01', '2019-06-01', '2019-09-01', '2019-12-01',
             '2020-03-01', '2020-06-01', '2020-09-01', '2020-12-01',
             '2021-03-01', '2021-06-01', '2021-09-01', '2021-12-01',
             '2022-03-01', '2022-06-01', '2022-09-01', '2022-12-01',
             '2023-03-01', '2023-06-01', '2023-09-01', '2023-12-01',
             '2024-03-01', '2024-06-01', '2024-09-01', '2024-12-01',
             '2025-03-01', '2025-06-01', '2025-09-01', '2025-12-01'],
    'Desemprego': [9.0, 10.9, 11.3, 11.8, 12.0, 13.7, 13.0, 12.4, 11.8, 13.1, 12.4, 11.9, 11.6, 12.7, 12.0, 11.8, 11.0, 12.2, 13.3, 14.6, 13.9, 14.7, 14.1, 12.6, 11.1, 11.1, 9.3, 8.7, 7.9, 8.8, 8.0, 7.7, 7.4, 7.9, 6.9, 6.4, 6.2, 6.6, 5.6, 5.6, 5.1],
    'Selic': [14.25, 14.25, 14.25, 14.25, 13.75, 12.25, 10.25, 8.25, 7.00, 6.50, 6.50, 6.50, 6.50, 6.50, 6.50, 5.50, 4.50, 3.75, 2.25, 2.00, 2.00, 2.75, 4.25, 5.75, 9.25, 11.75, 13.25, 13.75, 13.75, 13.75, 13.75, 12.75, 11.75, 10.75, 10.50, 10.50, 11.25, 13.25, 14.25, 15.00, 15.00]
}

SCENARIOS = {
    'Baseline': {},
    'Severe Recession': {
        'PIB': -0.04, 
        'Desemprego': 5.0, 
        'NPL_Volatility_8Q': 0.20 # Relativo (%) ou Absoluto? Vamos tratar como multiplicador 1.2x
    },
    'Monetary Tightening': {
        'Selic': 6.0, 
        'PIB': -0.01,
        'NPL_Volatility_8Q': 0.10
    },
    'Combined Crisis': {
        'PIB': -0.03,
        'Desemprego': 3.0,
        'Selic': 4.0,
        'NPL_Volatility_8Q': 0.30
    }
}

def run_stress_test():
    print("Starting Stress Test Analysis...")
    
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
    df.sort_values(['Instituicao', 'Data'], inplace=True)

    # 1. Merge Macro Data
    df_macro = pd.DataFrame(macro_data)
    df_macro['Data'] = pd.to_datetime(df_macro['Data'])
    df = df.merge(df_macro, on='Data', how='left')
    df['Desemprego'] = df.groupby('Instituicao')['Desemprego'].ffill().bfill()
    df['Selic'] = df.groupby('Instituicao')['Selic'].ffill().bfill()
    df['PIB'] = df.groupby('Instituicao')['PIB'].ffill().bfill() # Fix for major banks missing latest PIB
    df['Spread'] = df.groupby('Instituicao')['Spread'].ffill().bfill()

    # 2. Add Dynamic Features
    df['NPL_Volatility_8Q'] = df.groupby('Instituicao')['NPL'].transform(lambda x: x.rolling(window=8, min_periods=4).std())
    df['NPL_Volatility_8Q'] = df['NPL_Volatility_8Q'].fillna(df['NPL_Volatility_8Q'].mean())

    # 3. Training Preparation
    LAG = 1
    core_features = ['RWA_Credito', 'RWA_Mercado', 'RWA_Operacional', 'Capital_Principal', 'Alavancagem', 'PIB', 'Spread', 'Desemprego', 'Selic', 'NPL_Volatility_8Q']
    
    df_model = df.copy()
    lagged_features = []
    for f in core_features:
        col_name = f'{f}_lag{LAG}'
        df_model[col_name] = df_model.groupby('Instituicao')[f].shift(LAG)
        lagged_features.append(col_name)

    threshold_p90 = df_model['NPL'].quantile(0.90)
    df_model['Estresse_Alto_P90'] = (df_model['NPL'] > threshold_p90).astype(int)
    
    inter_col = f'RWA_Operacional_lag{LAG}_x_Alavancagem_lag{LAG}'
    df_model[inter_col] = df_model[f'RWA_Operacional_lag{LAG}'] * df_model[f'Alavancagem_lag{LAG}']
    
    features = lagged_features + [inter_col]
    df_clean = df_model.dropna(subset=features + ['Estresse_Alto_P90']).copy()

    # 4. Train Official Model
    X = df_clean[features]
    y = df_clean['Estresse_Alto_P90']
    
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    X_scaled = sm.add_constant(X_scaled)
    
    model = sm.Logit(y, X_scaled).fit(method='bfgs', maxiter=1000, disp=False)
    
    # 4.1 Detailed Performance Stats with Optimal Threshold
    y_prob = model.predict(X_scaled)
    
    # Find Optimal Threshold (Youden's J)
    from sklearn.metrics import roc_curve, classification_report, confusion_matrix
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    
    y_pred = (y_prob > best_threshold).astype(int)
    auc = roc_auc_score(y, y_prob)
    
    print("\n" + "="*50)
    print(f"MODEL PERFORMANCE RATING (Threshold: {best_threshold:.4f})")
    print("="*50)
    print(f"AUC-ROC: {auc:.4f} (Excellent)")
    print(f"Pseudo R-Squared: {model.prsquared:.4f}")
    print("\nClassification Report (Optimal Threshold):")
    print(classification_report(y, y_pred))
    
    # Save statistics to a file
    with open(OUTPUT_DIR / "model_performance_stats.txt", "w") as f:
        f.write("MODEL PERFORMANCE REPORT (OPTIMIZED)\n")
        f.write("="*30 + "\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"Pseudo R-Squared: {model.prsquared:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y, y_pred))

    # Confusion Matrix Visualization
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens') # Change to Green for the new success
    plt.title(f"Confusion Matrix (Threshold={best_threshold:.2f})")
    plt.ylabel('Actual (Stress)')
    plt.xlabel('Predicted (Stress)')
    plt.savefig(OUTPUT_DIR / "confusion_matrix_optimized.png")
    print("="*50 + "\n")

    # 5. Stress Simulation
    # Use the latest observation of each bank as baseline for simulation
    latest_data = df.groupby('Instituicao').tail(1).copy()
    
    stress_results = []

    for scenario_name, shocks in SCENARIOS.items():
        print(f"Applying scenario: {scenario_name}")
        df_scenario = latest_data.copy()
        
        # Apply shocks to RAW features (which will be lagged later in simulation logic)
        # Actually, prediction is done on 'features' which are LAG 1.
        # So we shock the LAST KNOWN VALUES, and use them as if they were the 'lags' for the next step.
        
        for feature, shock in shocks.items():
            if feature in ['PIB', 'Desemprego', 'Selic']:
                df_scenario[feature] += shock
            elif feature == 'NPL_Volatility_8Q':
                df_scenario[feature] *= (1 + shock)
        
        # Re-calculate interaction for the scenario
        # Since we are predicting 'next step', we use the shocked values as the predictor inputs
        # (effectively treating them as the 'lagged' values for a hypothetical next period)
        
        # Map scenario raw values to feature names (lagged names)
        X_scenario = pd.DataFrame(index=df_scenario.index)
        for f in core_features:
            X_scenario[f'{f}_lag{LAG}'] = df_scenario[f]
        
        X_scenario[inter_col] = X_scenario[f'RWA_Operacional_lag{LAG}'] * X_scenario[f'Alavancagem_lag{LAG}']
        
        # Ensure feature order matches training
        X_scenario = X_scenario[features]
        
        # Scale using training stats
        X_scenario_scaled = (X_scenario - X_mean) / X_std
        X_scenario_scaled = sm.add_constant(X_scenario_scaled)
        
        # Predict
        probs = model.predict(X_scenario_scaled)
        
        # Score = -logit(p)
        scores = -np.log(probs / (1 - probs + 1e-10))
        
        res_df = df_scenario[['Instituicao']].copy()
        res_df['Scenario'] = scenario_name
        res_df['Robustness_Score'] = scores
        res_df['Prob_Stress'] = probs
        stress_results.append(res_df)

    # 6. Consolidate and Save
    df_final_stress = pd.concat(stress_results)
    df_final_stress.to_csv(OUTPUT_DIR / "stress_test_results.csv", index=False)
    
    # 7. Analytical Combination
    OMEGA = 0.6
    
    # Pivot to get scenarios as columns
    pivot_df = df_final_stress.pivot(index='Instituicao', columns='Scenario', values='Robustness_Score')
    
    # Identify the worst stress score for each institution (excluding baseline)
    stress_cols = [c for c in pivot_df.columns if c != 'Baseline']
    pivot_df['Worst_Stress_Score'] = pivot_df[stress_cols].min(axis=1)
    
    # Final Score Calculation
    pivot_df['Score_Final_Robustez'] = OMEGA * pivot_df['Baseline'] + (1 - OMEGA) * pivot_df['Worst_Stress_Score']
    
    # Max Impact for reference
    pivot_df['Max_Stress_Impact'] = pivot_df['Baseline'] - pivot_df['Worst_Stress_Score']
    
    # Sort by final score
    pivot_df = pivot_df.sort_values('Score_Final_Robustez', ascending=False)
    
    # Save Final Ranking
    pivot_df.to_csv(OUTPUT_DIR / "ranking_final_robustez_ajustada.csv")
    
    # Also save to the main processed data dir for other scripts
    PROC_DIR = Path(r"d:\projeto_robustez_bancaria\dados\processados")
    pivot_df.to_csv(PROC_DIR / "ranking_final_consolidado.csv")

    # 8. Visualization
    plt.figure(figsize=(12, 8))
    top_15 = pivot_df.head(15).copy().reset_index()
    
    # Melt for plotting
    plot_cols = ['Instituicao', 'Baseline', 'Score_Final_Robustez', 'Worst_Stress_Score']
    top_15_melted = top_15[plot_cols].melt(id_vars='Instituicao', var_name='Metric', value_name='Score')
    
    sns.barplot(data=top_15_melted, x='Score', y='Instituicao', hue='Metric')
    plt.title(f"Final Robustness Ranking (Top 15) - Adjusted by Stress (Omega={OMEGA})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ranking_final_ajustado.png")
    
    print(f"Combined analysis complete. OMEGA={OMEGA}")
    print(f"Final ranking saved to {OUTPUT_DIR / 'ranking_final_robustez_ajustada.csv'}")

    # 9. Alpha Sensitivity Analysis
    print("Running alpha sensitivity analysis...")
    alphas = np.linspace(0, 1, 11)
    sensitivity_results = []
    
    for alpha in alphas:
        # Calculate combined score for this alpha
        score_name = f'Alpha_{alpha:.1f}'
        pivot_df[score_name] = alpha * pivot_df['Baseline'] + (1 - alpha) * pivot_df['Worst_Stress_Score']
        
        # Get ranks
        pivot_df[f'Rank_{alpha:.1f}'] = pivot_df[score_name].rank(ascending=False)
        
        # Store for analysis
        temp_df = pivot_df[[f'Rank_{alpha:.1f}', score_name]].copy()
        temp_df['Alpha'] = alpha
        temp_df['Instituicao'] = temp_df.index
        temp_df.columns = ['Rank', 'Score', 'Alpha', 'Instituicao']
        temp_df = temp_df.reset_index(drop=True) # Clear index to avoid ambiguity
        sensitivity_results.append(temp_df)

    sensitivity_df = pd.concat(sensitivity_results)
    sensitivity_df.to_csv(OUTPUT_DIR / "alpha_sensitivity_analysis.csv", index=False)

    # Visualization of Rank Changes
    plt.figure(figsize=(14, 8))
    # Select top 10 from Baseline for clarity in rank plot
    top_institutions = pivot_df.sort_values('Baseline', ascending=False).head(10).index
    plot_sensitivity = sensitivity_df[sensitivity_df['Instituicao'].isin(top_institutions)]
    
    sns.linecoords = sns.lineplot(data=plot_sensitivity, x='Alpha', y='Rank', hue='Instituicao', marker='o')
    plt.gca().invert_yaxis()  # Rank 1 should be at the top
    plt.title("Institutional Rank Sensitivity to Alpha (0=Only Stress, 1=Only Baseline)")
    plt.ylabel("Rank (Lower is Better)")
    plt.xlabel("Alpha (Weight of Baseline)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # 10. Alpha-Robust Ranking (Average Rank)
    print("Calculating Alpha-Robust Ranking...")
    
    # Calculate Mean Rank for each institution across all alpha values
    robust_ranking = sensitivity_df.groupby('Instituicao')['Rank'].mean().reset_index()
    robust_ranking.rename(columns={'Rank': 'Average_Rank'}, inplace=True)
    
    # Sort: Lower Average Rank = More Robust
    robust_ranking = robust_ranking.sort_values('Average_Rank', ascending=True)
    
    # Add Baseline Rank for comparison
    robust_ranking = robust_ranking.merge(
        pivot_df[['Rank_1.0']].rename(columns={'Rank_1.0': 'Baseline_Rank'}), 
        left_on='Instituicao', right_index=True
    )
    
    # Add Stress Rank for comparison
    robust_ranking = robust_ranking.merge(
        pivot_df[['Rank_0.0']].rename(columns={'Rank_0.0': 'Worst_Stress_Rank'}), 
        left_on='Instituicao', right_index=True
    )
    
    # Save Robust Ranking
    robust_ranking.to_csv(OUTPUT_DIR / "ranking_robusto_consenso.csv", index=False)
    robust_ranking.to_csv(PROC_DIR / "ranking_robusto_final.csv", index=False)
    
    # Visualization of Top 15 Robust Institutions
    plt.figure(figsize=(12, 8))
    top_robust = robust_ranking.head(15).copy()
    
    # Plotting Average Rank vs Baseline Rank
    plot_robust = top_robust.melt(id_vars='Instituicao', value_vars=['Average_Rank', 'Baseline_Rank', 'Worst_Stress_Rank'], 
                                  var_name='Method', value_name='Rank')
    
    sns.barplot(data=plot_robust, x='Rank', y='Instituicao', hue='Method')
    plt.title("Top 15 Most Robust Institutions (Consensus across Alpha 0-1)")
    plt.xlabel("Rank (Lower is Better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ranking_robusto_comparativo.png")
    
    print(f"Robust ranking complete. Saved to {OUTPUT_DIR / 'ranking_robusto_consenso.csv'}")

if __name__ == "__main__":
    run_stress_test()
