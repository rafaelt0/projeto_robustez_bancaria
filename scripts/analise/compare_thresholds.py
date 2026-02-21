import pandas as pd
import numpy as np

# Carregar dados P90
df = pd.read_csv('d:/projeto_robustez_bancaria/dados/processados/painel_p90.csv')

y_true = df['Estresse_Alto_P90']
y_prob = df['Prob_Estresse_P90']

# Testar diferentes thresholds
thresholds = [0.05, 0.10, 0.175, 0.25, 0.30, 0.50]
results = []

for t in thresholds:
    y_pred = (y_prob > t).astype(int)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    results.append({
        'Threshold': t,
        'Precision': f'{precision:.1%}',
        'Recall': f'{recall:.1%}',
        'F1-Score': f'{f1:.4f}',
        'Accuracy': f'{accuracy:.1%}',
        'Casos Detectados': f'{tp} / {tp + fn}',
        'Falsos Positivos': fp,
        'Falsos Negativos': fn
    })

df_results = pd.DataFrame(results)

print("="*100)
print("COMPARACAO DE THRESHOLDS DE DECISAO (Modelo P90)")
print("="*100)
print(df_results.to_string(index=False))

print("\n" + "="*100)
print("ANALISE DETALHADA: THRESHOLD 0.25 vs THRESHOLD OTIMO (0.175)")
print("="*100)

# Comparação direta
comparison = pd.DataFrame({
    'Metrica': ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Casos Detectados', 'Falsos Positivos'],
    'Threshold 0.175 (Otimo)': ['27.0%', '50.4%', '0.3516', '80.8%', '128 / 254', '346'],
    'Threshold 0.25': ['31.5%', '11.4%', '0.1676', '90.2%', '29 / 254', '63'],
    'Diferenca': ['+4.5pp', '-39.0pp', '-0.184', '+9.4pp', '-99 casos', '-283']
})

print(comparison.to_string(index=False))

print("\n" + "="*100)
print("INTERPRETACAO")
print("="*100)

print("""
THRESHOLD 0.25 (Mais Conservador):
  VANTAGENS:
    - Precision maior (31.5% vs 27.0%)
    - Menos falsos positivos (63 vs 346)
    - Acuracia maior (90.2% vs 80.8%)
    - Alertas mais confiaveis (1 em 3 e verdadeiro vs 1 em 4)
  
  DESVANTAGENS:
    - Recall MUITO MENOR (11.4% vs 50.4%)
    - Detecta apenas 29 de 254 casos (vs 128 de 254)
    - Perde 88.6% dos casos de estresse
    - F1-Score 52% pior (0.168 vs 0.352)

THRESHOLD 0.175 (Otimo):
  VANTAGENS:
    - Recall 4.4x maior (50.4% vs 11.4%)
    - Detecta 128 casos (vs 29)
    - F1-Score 2x melhor (0.352 vs 0.168)
    - Melhor para early warning system
  
  DESVANTAGENS:
    - Precision menor (27% vs 31.5%)
    - Mais falsos positivos (346 vs 63)
    - Acuracia menor (80.8% vs 90.2%)

RECOMENDACAO:
  - Use 0.175 para MONITORAMENTO CONTINUO (prioriza deteccao)
  - Use 0.25 para DECISOES CRITICAS (prioriza confiabilidade)
  - Use 0.30-0.50 para INTERVENCOES REGULATORIAS (precision > 50%)
""")

# Salvar comparação
df_results.to_csv('d:/projeto_robustez_bancaria/resultados/relatorios/threshold_comparison.csv', index=False)
print("\nArquivo salvo: threshold_comparison.csv")
