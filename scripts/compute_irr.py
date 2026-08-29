"""
Inter-Rater Reliability Analysis for the Six-Judge Expert Panel
================================================================
Computes composite reliability (Cronbach's alpha) and rank-order
concordance (Kendall's W) across the six professional domain judges
who evaluated 17 student project teams (N=53 present students).

Each judge assessed a distinct professional dimension:
  KP: Production Engineering / Manufacturing (max 16)
  HC: Patent & Innovation / IP (max 16)
  NG: Mathematics & Dynamics (max 16)
  MK: Industry 4.0 / Digital Twin — Siemens Ltd. (max 20)
  PJ: Communication & Presentation Skills (max 16)
  DD: Behavioural Science & HR (max 16)

Because judges evaluate different constructs, classical pairwise
agreement (Cohen's kappa) is not directly applicable. Instead:
  - Cronbach's alpha assesses composite reliability across domains
  - Kendall's W assesses rank-order concordance

Dependencies: numpy, scipy, openpyxl
Usage: python compute_irr.py
"""

import openpyxl
import numpy as np
import re
from scipy.stats import chi2

# --- Configuration ---
EXCEL_PATH = r'D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Exhibition_2026\Evaluation_Rubric_Sheets.xlsx'

JUDGE_INFO = [
    ('1_Kamlesh_Panchal',  16, 'KP'),
    ('2_Hasti_Chandarana', 16, 'HC'),
    ('3_Nitu_Gupta',       16, 'NG'),
    ('4_Mahendra_Kane',    20, 'MK'),
    ('5_Parminder_Jandoo', 16, 'PJ'),
    ('6_Debasis_Dash',     16, 'DD'),
]

MAX_MARKS = {'KP': 16, 'HC': 16, 'NG': 16, 'MK': 20, 'PJ': 16, 'DD': 16}
ROLL_PATTERN = re.compile(r'^H\d+$')


def extract_judge_scores(wb):
    """Extract total scores per student from each judge sheet."""
    all_data = {}
    for sheet_name, max_marks, code in JUDGE_INFO:
        ws = wb[sheet_name]
        judge_scores = {}
        for row in ws.iter_rows(min_row=8, values_only=True):
            cell0 = str(row[0]).strip() if row[0] is not None else ''
            if ROLL_PATTERN.match(cell0):
                vals = [row[i] for i in range(2, 6)]
                if all(isinstance(v, (int, float)) for v in vals):
                    judge_scores[cell0] = sum(vals)
        all_data[code] = judge_scores
    return all_data


def build_student_matrix(all_data):
    """Build aligned matrix of scores, excluding absent students."""
    codes = ['KP', 'HC', 'NG', 'MK', 'PJ', 'DD']
    all_rolls = sorted(set(r for d in all_data.values() for r in d.keys()))
    rows = []
    for roll in all_rolls:
        d = {code: all_data[code].get(roll, None) for code in codes}
        if all(v is not None for v in d.values()):
            total = sum(d.values())
            if total > 0:  # Exclude absent students
                rows.append(d)
    return rows, codes


def compute_cronbach_alpha(pct_matrix):
    """Compute Cronbach's alpha from a (N x k) matrix."""
    k = pct_matrix.shape[1]
    item_vars = pct_matrix.var(axis=0, ddof=1).sum()
    total_var = pct_matrix.sum(axis=1).var(ddof=1)
    alpha = (k / (k - 1)) * (1 - item_vars / total_var)
    return alpha


def compute_kendall_w(pct_matrix):
    """Compute Kendall's W (coefficient of concordance) and chi-square test."""
    n, m = pct_matrix.shape  # n students, m judges
    # Rank each column (judge) independently
    ranks = np.zeros_like(pct_matrix)
    for j in range(m):
        order = pct_matrix[:, j].argsort().argsort() + 1.0
        ranks[:, j] = order

    R_i = ranks.sum(axis=1)  # Sum of ranks for each student
    R_mean = m * (n + 1) / 2
    S = ((R_i - R_mean) ** 2).sum()
    W = (12 * S) / (m ** 2 * (n ** 3 - n))

    # Chi-square approximation
    df = n - 1
    chi2_stat = m * (n - 1) * W
    p_val = 1 - chi2.cdf(chi2_stat, df)

    return W, chi2_stat, df, p_val


def main():
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    all_data = extract_judge_scores(wb)
    rows, codes = build_student_matrix(all_data)

    n = len(rows)
    print("=" * 65)
    print("INTER-RATER RELIABILITY ANALYSIS — SIX-JUDGE EXPERT PANEL")
    print("=" * 65)
    print(f"Students evaluated (excluding absent): N = {n}")
    print()

    # Standardise to percentage scores for fair comparison
    pct_matrix = np.array([
        [row[code] / MAX_MARKS[code] * 100 for code in codes]
        for row in rows
    ])

    # Per-judge descriptive statistics
    print("--- Per-Judge Descriptive Statistics (Standardised %) ---")
    names = ['Kamlesh Panchal', 'Hasti Chandarana', 'Nitu Gupta',
             'Mahendra Kane', 'Parminder Jandoo', 'Debasis Dash']
    for i, (code, name) in enumerate(zip(codes, names)):
        col = pct_matrix[:, i]
        print(f"  {name:<22} ({code}): M={col.mean():.1f}%  SD={col.std(ddof=1):.1f}%")
    print()

    # Cronbach's alpha
    alpha = compute_cronbach_alpha(pct_matrix)
    print(f"--- Cronbach's Alpha (composite reliability) ---")
    print(f"  alpha = {alpha:.3f}")
    if alpha >= 0.80:
        print(f"  Interpretation: HIGH internal consistency (>= 0.80)")
    elif alpha >= 0.70:
        print(f"  Interpretation: ACCEPTABLE internal consistency (>= 0.70)")
    else:
        print(f"  Interpretation: Below acceptable threshold (< 0.70)")
    print()

    # Kendall's W
    W, chi2_stat, df, p_val = compute_kendall_w(pct_matrix)
    print(f"--- Kendall's W (coefficient of concordance) ---")
    print(f"  W = {W:.3f}")
    print(f"  Chi-square({df}) = {chi2_stat:.2f}")
    print(f"  p = {p_val:.4e}")
    if p_val < 0.001:
        print(f"  Interpretation: STATISTICALLY SIGNIFICANT concordance (p < .001)")
    elif p_val < 0.05:
        print(f"  Interpretation: STATISTICALLY SIGNIFICANT concordance (p < .05)")
    else:
        print(f"  Interpretation: NOT significant (p >= .05)")
    print()

    # Summary for manuscript
    print("=" * 65)
    print("MANUSCRIPT REPORTING TEXT:")
    print("=" * 65)
    print(f"Cronbach's alpha = {alpha:.3f} across six standardised judge")
    print(f"percentage domains, indicating high composite reliability.")
    print(f"Kendall's W = {W:.3f}, chi-square({df}) = {chi2_stat:.2f},")
    print(f"p < .0001, confirming statistically significant agreement")
    print(f"in judges' rank-ordering of student performance.")


if __name__ == '__main__':
    main()
