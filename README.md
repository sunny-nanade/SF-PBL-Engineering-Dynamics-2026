# DSM SF-PBL Frontiers Manuscript — Package README
## Status: FINAL | All statistics audited & verified | 2026-07-25

---

## Files in This Package

| File | Purpose |
|------|---------|
| `DSM_Frontiers_Manuscript_Standalone.tex` | **USE THIS** — compiles immediately with standard LaTeX |
| `DSM_Frontiers_Manuscript.tex` | Frontiers-class version — needs `frontiersSCNS.cls` from Frontiers |
| `DSM_references.bib` | Complete bibliography (16 references, all verified) |
| `compile.ps1` | PowerShell compile script (Windows) |

---

## How to Compile (Option A — Standalone, Recommended)

### Prerequisites
Install **MiKTeX** (Windows): https://miktex.org/download
Or **TeX Live**: https://tug.org/texlive/

### Steps
1. Open PowerShell in this folder:
   ```
   cd D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Frontiers_Final
   ```
2. Run the compile script:
   ```powershell
   .\compile.ps1
   ```
   The PDF will open automatically on success.

### Manual compile (if script fails)
```bash
pdflatex DSM_Frontiers_Manuscript_Standalone.tex
bibtex   DSM_Frontiers_Manuscript_Standalone
pdflatex DSM_Frontiers_Manuscript_Standalone.tex
pdflatex DSM_Frontiers_Manuscript_Standalone.tex
```

---

## How to Compile (Option B — Official Frontiers Template)

1. Download the official Frontiers LaTeX template from:
   https://www.frontiersin.org/guidelines/author-guidelines
   (Section: "LaTeX submissions")
2. Extract `frontiersSCNS.cls` and `frontiersinHLTH.bst` into this folder
3. Use `DSM_Frontiers_Manuscript.tex` + same `DSM_references.bib`
4. Same 4-pass compile as above

---

## Key Verified Statistics (do NOT change these)

| Metric | Value | Source |
|--------|-------|--------|
| Self-Efficacy Pre M (SD) | 2.92 (0.21) | ✅ CSV verified |
| Self-Efficacy Post M (SD) | 3.80 (0.40) | ✅ CSV verified |
| SE Cohen's d | 2.72 (pooled SD) | ✅ CSV verified |
| SE t(52) | 13.91, p < .001 | ✅ CSV verified |
| PK Pre M (SD) | 1.96 (0.94) | ✅ CSV verified |
| PK Post M (SD) | 3.28 (1.25) | ✅ CSV verified |
| PK Cohen's d | 1.20 | ✅ CSV verified |
| Hake's g | 0.50 (medium) | ✅ CSV verified |
| PBL Total M (SD) | 3.74 (0.30) | ✅ CSV verified |
| Exhibition M (SD) | 82.62 (9.98) | ✅ Excel verified |
| Exhibition Range | 53–96 | ✅ Excel verified |
| Exhibition N | 53 (1 absent excluded) | ✅ Excel verified |
| AI users | 45/53 = 84.9% | ✅ CSV verified |
| Course dates | 2 Jan – 25 Apr 2026 | ✅ Confirmed |
| Exhibition date | 25 April 2026 | ✅ Photos + judge sheets |
| Keywords | 8 | ✅ Frontiers limit |
| Abstract words | 247 | ✅ Under 250 limit |

---

## Submission Checklist (Frontiers in Education)

- [x] Abstract ≤ 250 words (current: 247)
- [x] Keywords: 3–8 (current: 8)
- [x] All statistics independently verified from raw data
- [x] Cohen's d formula explicitly stated (pooled SD)
- [x] Hake's g classification correct (medium, bands defined)
- [x] Exhibition date confirmed (25 April 2026)
- [x] N enrolled vs N consenting clarified (55 / 53)
- [x] Table footnote for absent student
- [x] Ethics statement present
- [x] Data availability statement present
- [x] Author contributions present
- [x] Conflict of interest statement present
- [x] ORCID: 0000-0001-7098-1084
- [ ] Upload to Frontiers submission portal
- [ ] Select section: STEM Education
- [ ] Confirm article type: Original Research
- [ ] Upload supplementary data files if required

---

## Source Data Files

| File | Location |
|------|----------|
| Pre-survey raw | `D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_PreSurvey.csv` |
| Post-survey raw | `D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_PostSurvey.csv` |
| Combined dataset | `D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Survey_Raw_PrePost.csv` |
| Exhibition rubric | `D:\Sunny\Paper\DSM_Exhibition_Framework\DSM_Exhibition_2026\Evaluation_Rubric_Sheets.xlsx` |
| Audit scripts | `deep_audit.py`, `recompute_rubric.py`, `final_review.py` |

---

*All corrections applied 2026-07-25 by Antigravity audit.*
*For questions: sunny.nanade@nmims.edu*
