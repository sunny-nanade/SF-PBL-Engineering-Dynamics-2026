"""
FINAL PRE-SUBMISSION HOLISTIC REVIEW
Checks 7 dimensions that could still trip a reviewer:
  1. Data provenance — pre-survey raw vs combined dataset consistency
  2. N-count discrepancies (N=55 enrolled / N=54 Excel / N=53 survey)
  3. Unsupported claims in manuscript (50-75% AI derivation, mode=2)
  4. Exhibition date discrepancy (Apr 25 photos vs Apr 26-27 stated)
  5. Cohen's d formula — is pooled-SD correct for paired data?
  6. Abstract word count vs Frontiers 250-word limit
  7. Hake g re-check with updated correct formula
"""
import pandas as pd, numpy as np
from scipy import stats

SEP  = "=" * 68
PASS = "[PASS]"
WARN = "[WARN]"
FAIL = "[FAIL]"

pre  = pd.read_csv(r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_PreSurvey.csv')
post = pd.read_csv(r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_PostSurvey.csv')
df   = pd.read_csv(r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_Raw_PrePost.csv')

lines = [SEP, "  FINAL PRE-SUBMISSION HOLISTIC REVIEW", SEP]
issues = []

# ─────────────────────────────────────────────────────────────────
# CHECK 1: Data Provenance — Does pre-survey raw match combined?
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 1] DATA PROVENANCE — PreSurvey.csv vs Raw_PrePost.csv")

pre_n  = len(pre)
comb_n = len(df)
lines.append(f"  PreSurvey.csv rows:       {pre_n}")
lines.append(f"  Raw_PrePost.csv rows:     {comb_n}")

# Check roll numbers match
pre_rolls  = set(pre['Roll_No'].astype(str))
comb_rolls = set(df['Roll_No'].astype(str))
only_in_pre  = pre_rolls - comb_rolls
only_in_comb = comb_rolls - pre_rolls

if only_in_pre:
    lines.append(f"  {WARN} Rolls in PreSurvey only: {only_in_pre}")
    issues.append(f"CHECK1: Rolls in PreSurvey only: {only_in_pre}")
if only_in_comb:
    lines.append(f"  {WARN} Rolls in Combined only:  {only_in_comb}")
    issues.append(f"CHECK1: Rolls in Combined only: {only_in_comb}")

# Compare SE1-SE10 pre values
common = sorted(pre_rolls & comb_rolls)
mismatches = []
for roll in common:
    pre_row  = pre[pre['Roll_No'].astype(str)==roll].iloc[0]
    comb_row = df[df['Roll_No'].astype(str)==roll].iloc[0]
    for i in range(1,11):
        pv = pre_row[f'Pre_SE{i}']
        cv = comb_row[f'Pre_SE{i}']
        if pv != cv:
            mismatches.append(f"Roll={roll} SE{i}: PreSurvey={pv} vs Combined={cv}")

if mismatches:
    lines.append(f"  {FAIL} SE item mismatches between PreSurvey and Combined ({len(mismatches)}):")
    for m in mismatches[:5]:
        lines.append(f"       {m}")
    issues.append(f"CHECK1: {len(mismatches)} SE item mismatches")
else:
    lines.append(f"  {PASS} All {len(common)} students: Pre_SE1-SE10 identical in both files")

# MCQ consistency
pk_mismatches = []
for roll in common:
    pre_row  = pre[pre['Roll_No'].astype(str)==roll].iloc[0]
    comb_row = df[df['Roll_No'].astype(str)==roll].iloc[0]
    for k in ['PK1','PK2','PK3','PK4','PK5']:
        pv = pre_row[f'Pre_{k}_Correct']
        cv = comb_row[f'Pre_{k}_Correct']
        if pv != cv:
            pk_mismatches.append(f"Roll={roll} {k}")

if pk_mismatches:
    lines.append(f"  {FAIL} MCQ mismatches: {pk_mismatches}")
    issues.append(f"CHECK1: MCQ mismatches")
else:
    lines.append(f"  {PASS} All Pre_PK items identical in both files")

# ─────────────────────────────────────────────────────────────────
# CHECK 2: N-count consistency
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 2] N-COUNT CONSISTENCY")

lines.append(f"  PreSurvey.csv:            N = {pre_n}  (enrolled + present pre-survey)")
lines.append(f"  PostSurvey.csv rows:      N = {len(post)}")
lines.append(f"  Combined PrePost.csv:     N = {comb_n}")
lines.append(f"  Rubric Excel (all):       N = 54  (53 present + 1 absent H005)")
lines.append(f"  Rubric Excel (present):   N = 53  (H005 excluded)")

if pre_n != comb_n:
    lines.append(f"  {WARN} PreSurvey ({pre_n}) != Combined ({comb_n}) — need to explain")
    issues.append(f"CHECK2: PreSurvey N={pre_n} != Combined N={comb_n}")
else:
    lines.append(f"  {PASS} PreSurvey and Combined have same N={pre_n}")

# H005 in survey data?
h005_in_survey = 'H005' in comb_rolls
lines.append(f"  H005 (absent from exhibition) in survey data: {h005_in_survey}")
if h005_in_survey:
    lines.append(f"  {WARN} H005 is in the SURVEY data (completed survey) but absent from EXHIBITION.")
    lines.append(f"         Manuscript says 'N=53 at exhibition' for survey too — need to clarify.")
    h005_row = df[df['Roll_No']=='H005']
    if len(h005_row):
        lines.append(f"         H005 SE_Mean Pre={h005_row['Pre_SE_Mean'].values[0]:.2f} Post={h005_row['Post_SE_Mean'].values[0]:.2f}")
    issues.append("CHECK2: H005 is in survey data but absent from exhibition — clarify N scope in methods")

# Manuscript states "N enrolled = 55"
lines.append(f"  PreSurvey rows = {pre_n}  | Manuscript states N enrolled = 55")
if pre_n != 55:
    lines.append(f"  {WARN} Mismatch: Pre-survey has {pre_n} rows but manuscript says 55 enrolled")
    issues.append(f"CHECK2: PreSurvey has {pre_n} students, manuscript says 55 enrolled")
else:
    lines.append(f"  {PASS} N enrolled = 55 confirmed")

# ─────────────────────────────────────────────────────────────────
# CHECK 3: Manuscript Claims vs Actual Data
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 3] UNSUPPORTED / UNVERIFIED MANUSCRIPT CLAIMS")

# Claim: "50–75% of mathematical derivation completed without AI"
ai_deriv = df['AI_Derivation_No_AI_Pct'].value_counts()
lines.append(f"  AI_Derivation_No_AI_Pct distribution:")
for val, cnt in ai_deriv.items():
    lines.append(f"    {val}: n={cnt} ({cnt/comb_n*100:.1f}%)")
# Check if "50-75%" range is actually the mode
top_val = ai_deriv.index[0]
lines.append(f"  Most common response: '{top_val}'")
if '50' in str(top_val) or '75' in str(top_val):
    lines.append(f"  {PASS} Claim '50-75% without AI' is consistent with data")
else:
    lines.append(f"  {WARN} Manuscript says '50-75%' but most common is '{top_val}'")
    issues.append(f"CHECK3: AI derivation claim: most common is '{top_val}' not '50-75%'")

# Claim: "mode = 2" for pre MCQ
pre_pk = df['Pre_PK_Total'].values
mode_val = pd.Series(pre_pk).mode()[0]
lines.append(f"\n  Pre-MCQ mode (manuscript claims 2): computed = {mode_val}")
if mode_val == 2:
    lines.append(f"  {PASS} Mode = 2 confirmed")
else:
    lines.append(f"  {WARN} Mode mismatch: manuscript says 2, computed = {mode_val}")
    issues.append(f"CHECK3: Pre-MCQ mode: manuscript=2, computed={mode_val}")

# Claim: abstract word count 267
abstract = """Engineering dynamics courses demand the simultaneous development of mathematical
modelling, computational implementation, and communication competencies yet
conventional lecture-based instruction rarely provides authentic contexts for their
integration. This study presents the design, implementation, and mixed-methods
evaluation of a Simulation-Focused Problem-Based Learning SF-PBL framework
embedded within a 15-week undergraduate Dynamic System Modeling DSM course
N 53 B.Tech Mechatronics Engineering Semester IV at an NBA-accredited
institution. The framework operationalises a four-phase computational pipeline
System Identification Mathematical Modelling Python Simulation Validation and
Communication scaffolded through an Explain-Try-Challenge instructional model and
assessed via a six-judge authentic evaluation panel. A pre-post mixed-methods design
measured self-efficacy 10-item Likert scale conceptual knowledge 5-item MCQ PBL
experience satisfaction AI tool usage and qualitative breakthrough reflections.
Results demonstrated significant improvement in self-efficacy pre M 2.92
SD 0.21 post M 3.80 SD 0.40 t 52 13.91 p 001
Cohen d 2.72 and knowledge gain pre M 1.96 SD 0.94 post
M 3.28 SD 1.25 t 52 11.32 p 001 d 1.20
Hake g 0.50. All ten self-efficacy items showed significant pre-post
improvement p 001. PBL satisfaction was high M 3.74 SD 0.30.
Eighty-five percent of students used AI tools primarily for simulation coding and
debugging. Qualitative themes identified mathematical validation of simulations PID
tuning and linking derivation to computational output as primary breakthrough moments.
Exhibition rubric scores across 17 projects evaluated by a six-judge expert panel
yielded M 82.62 100 SD 9.98 present students only N 53. These findings demonstrate that a
computationally-scaffolded SF-PBL framework can yield large learning gains in
engineering dynamics and provides a replicable model for OBE-aligned STEM institutions."""
wc = len(abstract.split())
lines.append(f"\n  Abstract word count (manuscript claims 267): approx {wc} words")
lines.append(f"  Frontiers STEM Education limit: 250 words for Original Research")
if wc > 250:
    lines.append(f"  {WARN} Abstract MAY exceed 250-word Frontiers limit (exact count depends on formatting)")
    issues.append(f"CHECK3: Abstract ~{wc} words — Frontiers limit is 250 for Original Research")
else:
    lines.append(f"  {PASS} Abstract within 250-word limit")

# ─────────────────────────────────────────────────────────────────
# CHECK 4: Cohen's d formula — paired vs independent
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 4] COHEN'S d FORMULA — PAIRED DESIGN")

se_pre  = df['Pre_SE_Mean'].values
se_post = df['Post_SE_Mean'].values
se_diff = se_post - se_pre
pk_pre  = df['Pre_PK_Total'].values.astype(float)
pk_post = df['Post_PK_Total'].values.astype(float)
pk_diff = pk_post - pk_pre

# Method 1: pooled SD (what we used)
d_se_pooled = (se_post.mean()-se_pre.mean()) / np.sqrt((se_pre.std(ddof=1)**2+se_post.std(ddof=1)**2)/2)
d_pk_pooled = (pk_post.mean()-pk_pre.mean()) / np.sqrt((pk_pre.std(ddof=1)**2+pk_post.std(ddof=1)**2)/2)

# Method 2: SD of difference scores (Lakens 2013 recommendation for paired)
d_se_diff = se_diff.mean() / se_diff.std(ddof=1)
d_pk_diff = pk_diff.mean() / pk_diff.std(ddof=1)

# Method 3: pre-test SD (Glass's delta)
d_se_glass = (se_post.mean()-se_pre.mean()) / se_pre.std(ddof=1)
d_pk_glass = (pk_post.mean()-pk_pre.mean()) / pk_pre.std(ddof=1)

lines.append(f"  SE Cohen's d comparison:")
lines.append(f"    Pooled SD (used in manuscript):    d = {d_se_pooled:.4f}")
lines.append(f"    SD of differences (Lakens 2013):   d = {d_se_diff:.4f}")
lines.append(f"    Pre-test SD (Glass delta):         d = {d_se_glass:.4f}")
lines.append(f"  PK Cohen's d comparison:")
lines.append(f"    Pooled SD (used in manuscript):    d = {d_pk_pooled:.4f}")
lines.append(f"    SD of differences (Lakens 2013):   d = {d_pk_diff:.4f}")
lines.append(f"    Pre-test SD (Glass delta):         d = {d_pk_glass:.4f}")
lines.append(f"  {WARN} Manuscript does not specify which d formula was used.")
lines.append(f"         Pooled SD is standard and defensible but must be stated explicitly.")
issues.append("CHECK4: Manuscript should specify 'Cohen's d computed using pooled SD' in methods")

# ─────────────────────────────────────────────────────────────────
# CHECK 5: Exhibition date
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 5] EXHIBITION DATE CONSISTENCY")
lines.append("  Photo filenames:    IMG_20260425_*.jpg  (April 25, 2026)")
lines.append("  README.md states:   Apr 26-27 2026  (NMIMS IUCEE)")
lines.append("  Manuscript §3.3:    'April 26–27, 2026'")
lines.append("  SF-PBL Social Media FOLDER_REFERENCE_GUIDE: 'InnovateLab 2026, Apr 25'")
lines.append(f"  {WARN} INCONSISTENCY: Photos dated Apr 25; manuscript/README say Apr 26-27.")
lines.append(f"         Likely explanation: Apr 25 = morning demo session; Apr 26-27 = formal exhibition.")
lines.append(f"         Or: The exhibition ran Apr 25-26 and the README has a typo.")
lines.append(f"         ACTION NEEDED: Confirm the actual exhibition date before submission.")
issues.append("CHECK5: Exhibition date inconsistency — photos Apr 25, manuscript says Apr 26-27. MUST VERIFY.")

# ─────────────────────────────────────────────────────────────────
# CHECK 6: Pre-survey N vs. enrolled N
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 6] H005 PARTICIPATION SCOPE — SURVEY vs EXHIBITION")

if 'H005' in df['Roll_No'].values:
    h5 = df[df['Roll_No']=='H005'].iloc[0]
    lines.append(f"  H005 IS in the survey data (Pre_SE_Mean={h5['Pre_SE_Mean']:.2f}, Post_SE_Mean={h5['Post_SE_Mean']:.2f})")
    lines.append(f"  H005 scored 0 in exhibition rubric (absent on day).")
    lines.append(f"  Manuscript says N=53 for BOTH survey AND exhibition.")
    lines.append(f"  ACTUAL situation:")
    lines.append(f"    Survey N:      {comb_n} (includes H005 who completed survey)")
    lines.append(f"    Exhibition N:  53 (excludes H005 who was absent)")
    if comb_n == 54:
        lines.append(f"  {FAIL} CRITICAL: Survey N=54 but manuscript reports N=53 throughout.")
        lines.append(f"         All survey statistics were computed on N=53. But PrePost CSV has {comb_n} rows.")
        issues.append(f"CHECK6: CRITICAL — Survey data has {comb_n} rows but all stats computed on N=53. Verify whether H005 completed the survey.")
    elif comb_n == 53:
        lines.append(f"  {PASS} Survey N=53 consistent — H005 did NOT complete the survey")
    else:
        lines.append(f"  {WARN} Unexpected N={comb_n}")

# ─────────────────────────────────────────────────────────────────
# CHECK 7: Frontiers word limits
# ─────────────────────────────────────────────────────────────────
lines.append("\n[CHECK 7] FRONTIERS SUBMISSION REQUIREMENTS")
lines.append("  Article type: Original Research")
lines.append("  Frontiers in Education / STEM Education section requirements:")
lines.append("    Abstract:         max 250 words")
lines.append("    Main text:        typically 4,000-8,000 words")
lines.append("    Keywords:         3-8 keywords")
lines.append("    References:       no stated limit")
lines.append("    Figures:          no stated limit")
lines.append(f"  Current manuscript keywords: 9  (exceeds max of 8)")
issues.append("CHECK7: Manuscript has 9 keywords; Frontiers typically allows max 8. Remove 1.")
lines.append(f"  {WARN} 9 keywords listed — Frontiers allows maximum 8.")
lines.append(f"  {WARN} Abstract may exceed 250 words (stated as 267 in manuscript).")
lines.append(f"         Must count precisely and trim to <=250 before submission.")

# ─────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────
lines.append("\n" + SEP)
lines.append("  FINAL REVIEW COMPLETE")
lines.append(f"  Issues requiring action before submission: {len(issues)}")
lines.append(SEP)
for i, iss in enumerate(issues, 1):
    lines.append(f"  {i}. {iss}")

report = "\n".join(lines)
print(report)

with open(r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Final_PreSubmission_Review.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\nSaved: DSM_Final_PreSubmission_Review.txt")
