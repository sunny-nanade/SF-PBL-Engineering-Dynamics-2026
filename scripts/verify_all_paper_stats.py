"""
Comprehensive verification of all paper statistics against raw CSV and Excel datasets.
"""
import pandas as pd
import numpy as np
import openpyxl
import re

csv_path = r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Frontiers_Final\data\DSM_Survey_Raw_PrePost_Anonymized.csv'
df = pd.read_csv(csv_path)

print("=" * 70)
print("1. PARTICIPANTS & SAMPLE SIZE")
print("=" * 70)
print(f"Total rows in paired dataset: N = {len(df)}")

print("\n" + "=" * 70)
print("2. SELF-EFFICACY (SE1 - SE10) & SE TOTAL")
print("=" * 70)
for i in range(1, 11):
    pre = df[f'Pre_SE{i}']
    post = df[f'Post_SE{i}']
    diff = post - pre
    d = diff.mean() / np.sqrt((pre.std(ddof=1)**2 + post.std(ddof=1)**2) / 2)
    t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(df)))
    print(f"SE{i:<2}: Pre M={pre.mean():.2f} (SD={pre.std(ddof=1):.2f}) | "
          f"Post M={post.mean():.2f} (SD={post.std(ddof=1):.2f}) | "
          f"Gain=+{diff.mean():.2f} | d={d:.2f} | t(52)={t:.2f}")

pre_tot = df[[f'Pre_SE{i}' for i in range(1, 11)]].mean(axis=1)
post_tot = df[[f'Post_SE{i}' for i in range(1, 11)]].mean(axis=1)
diff_tot = post_tot - pre_tot
d_tot = diff_tot.mean() / np.sqrt((pre_tot.std(ddof=1)**2 + post_tot.std(ddof=1)**2) / 2)
t_tot = diff_tot.mean() / (diff_tot.std(ddof=1) / np.sqrt(len(df)))
ci_lo = diff_tot.mean() - 1.96 * (diff_tot.std(ddof=1) / np.sqrt(len(df)))
ci_hi = diff_tot.mean() + 1.96 * (diff_tot.std(ddof=1) / np.sqrt(len(df)))
print("-" * 70)
print(f"SE TOTAL: Pre M={pre_tot.mean():.2f} (SD={pre_tot.std(ddof=1):.2f}) | "
      f"Post M={post_tot.mean():.2f} (SD={post_tot.std(ddof=1):.2f}) | "
      f"Gain=+{diff_tot.mean():.2f} (95% CI [{ci_lo:.2f}, {ci_hi:.2f}]) | "
      f"d={d_tot:.2f} | t(52)={t_tot:.2f}")

print("\n" + "=" * 70)
print("3. CONCEPTUAL KNOWLEDGE (PK1 - PK5) & PK TOTAL")
print("=" * 70)
for i in range(1, 6):
    pre_c = df[f'Pre_PK{i}_Correct'].mean() * 100
    post_c = df[f'Post_PK{i}_Correct'].mean() * 100
    print(f"PK{i}: Pre={pre_c:.1f}% correct ({int(df[f'Pre_PK{i}_Correct'].sum())}/{len(df)}) | "
          f"Post={post_c:.1f}% correct ({int(df[f'Post_PK{i}_Correct'].sum())}/{len(df)}) | "
          f"Gain=+{post_c-pre_c:.1f} pp")

pre_pk_tot = df[[f'Pre_PK{i}_Correct' for i in range(1, 6)]].sum(axis=1)
post_pk_tot = df[[f'Post_PK{i}_Correct' for i in range(1, 6)]].sum(axis=1)
diff_pk = post_pk_tot - pre_pk_tot
d_pk = diff_pk.mean() / np.sqrt((pre_pk_tot.std(ddof=1)**2 + post_pk_tot.std(ddof=1)**2) / 2)
t_pk = diff_pk.mean() / (diff_pk.std(ddof=1) / np.sqrt(len(df)))
# Cohort average normalized gain
g_hake = (post_pk_tot.mean() - pre_pk_tot.mean()) / (5.0 - pre_pk_tot.mean())
# Mean of individual normalized gains
g_indiv = np.mean([(post - pre)/(5.0 - pre) for pre, post in zip(pre_pk_tot, post_pk_tot) if pre < 5.0])
ci_pk_lo = diff_pk.mean() - 1.96 * (diff_pk.std(ddof=1) / np.sqrt(len(df)))
ci_pk_hi = diff_pk.mean() + 1.96 * (diff_pk.std(ddof=1) / np.sqrt(len(df)))
print("-" * 70)
print(f"PK TOTAL: Pre M={pre_pk_tot.mean():.2f} (SD={pre_pk_tot.std(ddof=1):.2f}) | "
      f"Post M={post_pk_tot.mean():.2f} (SD={post_pk_tot.std(ddof=1):.2f}) | "
      f"Gain=+{diff_pk.mean():.2f} (95% CI [{ci_pk_lo:.2f}, {ci_pk_hi:.2f}]) | "
      f"d={d_pk:.2f} | t(52)={t_pk:.2f} | Hake g={g_indiv:.2f} (cohort avg g={g_hake:.2f})")

print("\n" + "=" * 70)
print("4. PBL SATISFACTION (PBL1 - PBL10)")
print("=" * 70)
print(f"PBL Total: M={df['PBL_Total_Mean'].mean():.2f} (SD={df['PBL_Total_Mean'].std(ddof=1):.2f})")
print(f"PBL Value (PBL1-4): M={df['PBL_Value_Mean'].mean():.2f}")
print(f"PBL Scaffolding (PBL5-7): M={df['PBL_Scaffold_Mean'].mean():.2f}")
print(f"PBL Overall (PBL8-10): M={df['PBL_Overall_Mean'].mean():.2f}")

print("\n" + "=" * 70)
print("5. AI TOOL USAGE")
print("=" * 70)
ai_counts = df['AI_Tools_Used'].value_counts()
for tool, count in ai_counts.items():
    print(f"  {tool:<30}: n={count:<2} ({count/len(df)*100:.1f}%)")
ai_users = (df['AI_Tools_Used'] != 'None').sum()
print(f"Total AI users: {ai_users}/{len(df)} = {ai_users/len(df)*100:.1f}%")
print(f"Non-users: {(df['AI_Tools_Used'] == 'None').sum()}/{len(df)} = {(df['AI_Tools_Used'] == 'None').mean()*100:.1f}%")

print("\n" + "=" * 70)
print("6. QUALITATIVE THEMES")
print("=" * 70)
print("OE1 (Most Valuable Learning):")
for theme, count in df['OE1_Theme'].value_counts().items():
    print(f"  {theme:<45}: n={count} ({count/len(df)*100:.1f}%)")
print("\nOE2 (Suggested Changes):")
for theme, count in df['OE2_Theme'].value_counts().items():
    print(f"  {theme:<45}: n={count} ({count/len(df)*100:.1f}%)")
print("\nOE3 (Breakthrough Moments):")
for theme, count in df['OE3_Theme'].value_counts().items():
    print(f"  {theme:<45}: n={count} ({count/len(df)*100:.1f}%)")

print("\n" + "=" * 70)
print("7. COURSE OUTCOME ATTAINMENT (DIRECT vs INDIRECT)")
print("=" * 70)
co_map = {
    'CO1': ('PK1', ['SE1']),
    'CO2': ('PK2', ['SE2']),
    'CO3': ('PK3', ['SE3']),
    'CO4': ('PK4', ['SE4']),
    'CO5': ('PK5', ['SE5', 'SE6', 'SE7', 'SE8', 'SE9']),
}
for co, (pk, se_list) in co_map.items():
    direct_pre = df[f'Pre_{pk}_Correct'].mean() * 100
    direct_post = df[f'Post_{pk}_Correct'].mean() * 100
    indirect_pre = np.mean([(df[f'Pre_{s}'] >= 4).mean() * 100 for s in se_list])
    indirect_post = np.mean([(df[f'Post_{s}'] >= 4).mean() * 100 for s in se_list])
    print(f"{co}: Direct Pre={direct_pre:.1f}% -> Post={direct_post:.1f}% (Gain=+{direct_post-direct_pre:.1f}pp) | "
          f"Indirect Pre={indirect_pre:.1f}% -> Post={indirect_post:.1f}% (Gain=+{indirect_post-indirect_pre:.1f}pp)")
