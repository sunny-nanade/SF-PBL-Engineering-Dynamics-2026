# A Computationally-Integrated SF-PBL Framework for Engineering Dynamics

**Status:** Revised Supplementary Data, Code, and Manuscript Repository (*Frontiers in Education*)

This repository contains the empirical datasets, analysis scripts, pedagogical instruments, and LaTeX source files supporting the revised research article: 
*"A Computationally-Integrated Simulation-Focused Problem-Based Learning Framework for Engineering Dynamics: Design, Implementation, and Mixed-Methods Evaluation in an OBE-Aligned Undergraduate Course"*

**Authors:** Sunny Nanade, Debasis Dash, Sudipto Sarkar, Koteswararao Anne  
**Affiliation:** Mukesh Patel School of Technology Management & Engineering, SVKM's NMIMS, Mumbai, India  

---

## 1. Repository Structure

The repository is organized to support full transparency and reproducibility of the research findings.

### `/data`
Contains the anonymized survey and evaluation data for the 53 participating students.
- `DSM_Survey_PreSurvey_Anonymized.csv`: Baseline conceptual knowledge and self-efficacy data.
- `DSM_Survey_PostSurvey_Anonymized.csv`: Post-intervention knowledge, self-efficacy, and PBL satisfaction data.
- `DSM_Survey_Raw_PrePost_Anonymized.csv`: The combined, paired dataset used for all statistical analyses.
- `Evaluation_Rubric_Sheets_Anonymized.xlsx`: The raw evaluation scores provided by six external industry and academic judges across 17 student project teams.

### `/scripts`
Python scripts required to reproduce the statistical findings presented in the manuscript.
- `compute_irr.py`: Computes composite reliability (Cronbach's alpha) and rank concordance (Kendall's W) across the six-judge expert panel.
- `deep_audit.py`: Computes paired t-tests, Cohen's d (pooled SD), and Hake's normalized gain.
- `recompute_rubric.py`: Aggregates the external judge evaluation rubric scores and verifies judge consistency.
- `verify_all_paper_stats.py`: Automated verification script cross-checking all reported numbers against raw datasets.
- `final_review.py`: A holistic validation script ensuring alignment between datasets.

### `/instruments_and_rubrics`
Original pedagogical documents and assessment tools utilized during the 15-week course implementation.
- `DSM_PrePost_Survey_Instrument.docx`: The complete 30-item survey instrument.
- `DSM_Evaluator_Rubric_Kit.docx`: The 5-dimension scoring rubric provided to the expert panel.
- `DSM_SF_PBL_Framework_Guide.docx`: The student project manual and timeline.
- `DSM_Presentation_Template.pptx`: The standardized presentation structure used by student teams during the exhibition.

### `/latex_manuscript`
The complete LaTeX source code and compiled manuscript PDF, formatted according to Frontiers guidelines.
- `DSM_Frontiers_Manuscript.tex`: Main manuscript LaTeX source file.
- `DSM_Frontiers_Manuscript.pdf`: Compiled manuscript PDF.
- `DSM_references.bib`: Complete bibliography database.
- `fig1_sfpbl_pipeline.jpg` through `fig4_qualitative_themes.jpg`: High-resolution figures.

### `/student_projects`
Raw, representative project submissions from the 17 student teams.
- Contains mathematical models, Python simulation code, Free Body Diagrams, and final validation reports for the engineering systems analyzed during the course.

---

## 2. Data Privacy and Ethics

In accordance with institutional research ethics and data privacy guidelines, all student-identifying information (such as names and contact details) has been removed from the public datasets. Student records are identified strictly by anonymized Roll Numbers (e.g., H079). The datasets retain 100% of the empirical scores, ratings, and qualitative responses required to replicate the statistical analyses.

## 3. Reproducibility

To replicate the statistical analyses:
1. Ensure Python 3.10+ is installed.
2. Install the required libraries: `pip install pandas scipy numpy openpyxl`
3. Execute the audit scripts located in the root directory or the `/scripts` directory (depending on your local clone structure).
   For example: `python deep_audit.py`

## 4. License

This repository is licensed under the MIT License. Researchers and educators are encouraged to use, adapt, and build upon the Simulation-Focused Problem-Based Learning (SF-PBL) framework, provided appropriate attribution is given to the original authors.
