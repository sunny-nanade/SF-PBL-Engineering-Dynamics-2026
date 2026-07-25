"""Full rubric re-extraction and correct statistics computation."""
import openpyxl, numpy as np, re

wb = openpyxl.load_workbook(
    r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Exhibition_2026\Evaluation_Rubric_Sheets.xlsx',
    data_only=True)

judge_info = [
    ('1_Kamlesh_Panchal',  16, 'KP'),
    ('2_Hasti_Chandarana', 16, 'HC'),
    ('3_Nitu_Gupta',       16, 'NG'),
    ('4_Mahendra_Kane',    20, 'MK'),
    ('5_Parminder_Jandoo', 16, 'PJ'),
    ('6_Debasis_Dash',     16, 'DD'),
]
MAX = {'KP':16,'HC':16,'NG':16,'MK':20,'PJ':16,'DD':16}
roll_pattern = re.compile(r'^H\d+$')
all_data = {}

for sheet_name, max_marks, code in judge_info:
    ws = wb[sheet_name]
    judge_scores = {}
    for row in ws.iter_rows(min_row=8, values_only=True):
        cell0 = str(row[0]).strip() if row[0] is not None else ''
        if roll_pattern.match(cell0):
            vals = [row[i] for i in range(2, 6)]
            if all(isinstance(v, (int, float)) for v in vals):
                judge_scores[cell0] = sum(vals)
    all_data[code] = judge_scores

all_rolls = sorted(set(r for d in all_data.values() for r in d.keys()))
rows = []
for roll in all_rolls:
    d = {code: all_data[code].get(roll, None) for code in ['KP','HC','NG','MK','PJ','DD']}
    if all(v is not None for v in d.values()):
        total = sum(d.values())
        rec = dict(roll=roll, total=total)
        rec.update(d)
        rows.append(rec)

absent  = [r for r in rows if r['total'] == 0]
present = [r for r in rows if r['total']  > 0]

print("=" * 60)
print("EXHIBITION RUBRIC — GROUND TRUTH FROM EXCEL")
print("=" * 60)
print(f"Total rows in Excel (all sheets):  {len(rows)}")
print(f"Students with all scores = 0:      {len(absent)}")
for r in absent:
    print(f"  {r['roll']}  total={r['total']}")
print(f"Students with total > 0:           {len(present)}")

all_t  = np.array([r['total'] for r in rows])
pres_t = np.array([r['total'] for r in present])

print()
print(f"--- Including absent (N={len(rows)}) ---")
print(f"  M={all_t.mean():.4f}  SD={all_t.std(ddof=1):.4f}  min={all_t.min()}  max={all_t.max()}")

print()
print(f"--- Excluding absent / zero-scored (N={len(present)}) ---")
print(f"  M={pres_t.mean():.4f}  SD={pres_t.std(ddof=1):.4f}  min={pres_t.min()}  max={pres_t.max()}")

print()
print("--- Per-Judge Means (N excluding absent) ---")
codes = ['KP','HC','NG','MK','PJ','DD']
names = ['Kamlesh Panchal','Hasti Chandarana','Nitu Gupta',
         'Mahendra Kane','Parminder Jandoo','Debasis Dash']
running_sum = 0
for code, name in zip(codes, names):
    arr = np.array([r[code] for r in present])
    mx  = MAX[code]
    m   = arr.mean(); sd = arr.std(ddof=1)
    running_sum += m
    print(f"  {name:<22} ({code}) /{ mx}: M={m:.4f}  SD={sd:.4f}  pct={m/mx*100:.1f}%")
print(f"  SUM of judge means = {running_sum:.4f}  (total mean = {pres_t.mean():.4f})")
print(f"  Difference = {abs(running_sum - pres_t.mean()):.6f}  (should be ~0)")

print()
print("--- MANUSCRIPT vs CORRECT VALUES ---")
print(f"  Manuscript: N=53  M=79.62  SD=18.43")
print(f"  Excel:      N={len(present)}  M={pres_t.mean():.2f}  SD={pres_t.std(ddof=1):.2f}")
if abs(pres_t.mean() - 79.62) > 0.5:
    print("  >>> MISMATCH: Manuscript exhibition stats are INCORRECT")
    print(f"  >>> CORRECT: M={pres_t.mean():.2f}  SD={pres_t.std(ddof=1):.2f}  N={len(present)}")
else:
    print("  OK: Manuscript value matches Excel within 0.5 points")

print()
print("--- Per-Judge CORRECT values for Table 6 ---")
for code, name in zip(codes, names):
    arr = np.array([r[code] for r in present])
    mx  = MAX[code]
    lo  = arr.min(); hi = arr.max()
    print(f"  {name:<22}  M={arr.mean():.2f}  SD={arr.std(ddof=1):.2f}  Range={lo}-{hi}  /{mx}")
print(f"  COMBINED TOTAL        M={pres_t.mean():.2f}  SD={pres_t.std(ddof=1):.2f}  Range={pres_t.min()}-{pres_t.max()}  /100")
