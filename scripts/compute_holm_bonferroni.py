import pandas as pd
import numpy as np

for cohort, path in [
    ('OASIS Test (N=30)', r'D:\ml project\results\statistical_audit\oasis_test_paired_auc_differences.csv'),
    ('ADNI External (N=500)', r'D:\ml project\results\statistical_audit\adni_external_paired_auc_differences.csv')
]:
    df = pd.read_csv(path)
    df = df.sort_values('p_value').reset_index(drop=True)
    m = len(df)
    
    p_raw = df['p_value'].values
    p_holm = np.zeros(m)
    for i in range(m):
        p_holm[i] = (m - i) * p_raw[i]
    for i in range(1, m):
        p_holm[i] = max(p_holm[i], p_holm[i-1])
    p_holm = np.minimum(p_holm, 1.0)
    
    df['p_holm'] = np.round(p_holm, 4)
    df['sig_holm_005'] = df['p_holm'] < 0.05
    out_csv = path.replace('.csv', '_holm_corrected.csv')
    df.to_csv(out_csv, index=False)
    print(f'=== {cohort} (m={m}) ===')
    for _, row in df.iterrows():
        print(f"{row['model_A']} vs {row['model_B']}: delta={row['delta_auc_median']:+.4f}, p_raw={row['p_value']:.4f}, p_holm={row['p_holm']:.4f}, sig={row['sig_holm_005']}")
