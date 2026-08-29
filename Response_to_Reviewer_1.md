# Response to Reviewer 1

**Manuscript Title:** A Computationally-Integrated Simulation-Focused Problem-Based Learning Framework for Engineering Dynamics: Design, Implementation, and Mixed-Methods Evaluation in an OBE-Aligned Undergraduate Course  
**Manuscript ID:** 1948391  
**Journal:** *Frontiers in Education* (Section: STEM Education)  
**Authors:** Sunny Nanade, Debasis Dash, Sudipto Sarkar, Koteswararao Anne  

---

We sincerely thank Reviewer 1 for their thoughtful, rigorous, and constructive evaluation of our manuscript. We have carefully addressed all comments, revised the manuscript text, added new analyses where requested, and moderated our claims throughout. Below is our point-by-point response detailing the changes made. In the revised manuscript, all modifications are highlighted in blue.

---

### Comment R1.1: Originality in Relation to Prior PBL and Computational Approaches
> *"The originality of the proposed framework should be discussed more critically in relation to previous PBL and computational learning approaches."*

**Response:**  
We appreciate this important observation. In Section 1.2 of the revised manuscript, we have added a dedicated paragraph explicitly distinguishing the Simulation-Focused Problem-Based Learning (SF-PBL) framework from three established instructional models:

1. **Classical Medical PBL** \citep{hmelo2004problem}: Focuses on open-ended diagnostic reasoning without a structured computational implementation pipeline; the primary artefact is a clinical reasoning report rather than a mathematical derivation verified via numerical simulation.
2. **Project-Based Learning / Capstone Design** \citep{dym2005engineering}: Positions a physical prototype or CAD model as the primary deliverable, treating mathematical derivation as instrumental rather than as the central intellectual product.
3. **Standalone Computational Modelling Courses** \citep{justo2021enhancing}: Often pre-specify governing equations, requiring students to code rather than derive equations from first principles.

We have also cited a companion study from our institution investigating cross-disciplinary team formation in an immersive XR engineering course \citep{nanade2026cross} to provide institutional context on active-learning engineering initiatives.

**Location in Manuscript:** Section 1.2, Paragraph 4.

---

### Comment R1.2: Synthesis of the Theoretical Framework
> *"Some sections are excessively descriptive and would benefit from greater synthesis."*

**Response:**  
We thank the reviewer for this constructive suggestion. We have condensed the narrative descriptions of the three theoretical pillars in Section 2.1 and introduced a structured synthesis table (**Table 2: Theoretical--Pedagogical Synthesis**). This table directly maps each foundational learning theory (Hmelo-Silver's PBL cycle, Bandura's self-efficacy, Hake's interactive engagement, Vygotsky's ZPD, and Kolb's experiential learning cycle) to its corresponding SF-PBL phase, Explain-Try-Challenge (E-T-C) instructional stage, and assessment mechanism.

**Location in Manuscript:** Section 2.1, Table 2.

---

### Comment R1.3: Causal Claims and Single-Group Pre--Post Design
> *"Many conclusions continue to imply that the observed improvements are entirely attributable to the proposed framework. The discussion should be revised to better reflect the limitations of the research design."*

**Response:**  
We fully agree with the reviewer that a single-group pretest--posttest design \citep{campbell1963experimental} cannot establish exclusive causal attribution due to potential maturation effects, assessment familiarity, and concurrent course exposure. We have made comprehensive revisions to address this across the entire manuscript:

1. **Methodological Framing:** Added an explicit framing paragraph at the opening of Section 5 (Discussion) that frames all outcomes as evidence consistent with, rather than proof of, the framework's effectiveness.
2. **Systematic Tone Moderation:** Replaced causal phrasing (such as "confirms", "proves", "demonstrates", and "produced") with associative and conservative language (such as "is consistent with the interpretation that", "indicates", and "pre-to-post gains observed within this cohort") throughout the Abstract, Introduction, Discussion, and Conclusion.
3. **Boundary Condition:** Added explicit qualifiers affirming that the findings reflect the experience of a single cohort ($N = 53$) at one institution.

**Location in Manuscript:** Section 5 (Opening paragraph), Section 5.1, Section 5.4, Section 6.

---

### Comment R1.4: Sample Size and Institutional Generalisability
> *"The limitation regarding sample size and single-institution context should be emphasised consistently throughout the manuscript rather than only in the discussion."*

**Response:**  
We agree with the reviewer. We have explicitly stated the single-cohort, single-institution boundary condition across multiple sections:
- **Abstract:** Explicitly identified as a single NBA-accredited institution in India.
- **Section 1.4 (Contributions):** Clarified that empirical findings represent pre-to-post gains within a single cohort ($N = 53$).
- **Section 3.1 (Participants):** Added an explicit sentence cautioning that findings should be interpreted within this specific institutional and cultural context.
- **Section 6 (Conclusion):** Re-emphasised the single-cohort context and highlighted the necessity for multi-institutional replication.

**Location in Manuscript:** Abstract, Section 1.4, Section 3.1, Section 6.

---

### Comment R1.5: Discussion of the Large Self-Efficacy Effect Size ($d = 2.72$)
> *"The manuscript reports exceptionally large effect sizes... this issue deserves a more detailed discussion to avoid overinterpretation."*

**Response:**  
We thank the reviewer for highlighting this critical methodological point. In Section 5.1, we have significantly expanded our discussion to contextualise the reported Cohen's $d = 2.72$ (pooled SD):

1. **Baseline Range Restriction:** We explain that the baseline standard deviation ($SD_{\text{pre}} = 0.21$ on a 5-point scale) was exceptionally narrow because students uniformly reported low-to-moderate confidence prior to dynamics instruction. This narrow denominator mathematically amplifies standardised effect sizes.
2. **Alternative Effect Size Metrics:** We report alternative formulations for transparency:
   - Within-subjects $d_z$ (based on SD of paired differences): $d_z = 1.91$
   - Glass's $\Delta$ (using only baseline SD): $\Delta = 4.16$
3. **Absolute Mean Gain as Practical Metric:** We emphasise that the absolute gain ($+0.88$ points, 95% CI [0.75, 1.00]) represents a shift from approximately "Neutral" to "Agree", which is a meaningful pedagogical outcome regardless of the standardised metric used.

**Location in Manuscript:** Section 5.1, Paragraph 1.

---

### Comment R1.6: Inter-Rater Reliability for the Expert Panel
> *"No inter-rater reliability analysis is reported."*

**Response:**  
We appreciate this important comment. We have performed a formal inter-rater reliability analysis on the raw rubric data across the six expert judges who evaluated the 17 student teams ($N = 53$ present students):

1. **Methodological Note:** Because each judge represented a distinct professional domain (Production Engineering, Patent/IP, Mathematics, Industry 4.0/Siemens, Communication, Behavioural Science), judges assessed complementary rather than identical criteria. Therefore, composite reliability and rank concordance were evaluated.
2. **Empirical Results:**
   - **Cronbach's $\alpha = 0.829$** across the six standardised percentage domains, demonstrating high composite reliability of the multi-domain panel.
   - **Kendall's coefficient of concordance $W = 0.448$**, $\chi^2(52) = 139.63$, $p < .0001$, confirming statistically significant agreement in the judges' rank-ordering of student project quality.
3. **Repository Script:** A Python script (`scripts/compute_irr.py`) has been added to the public repository to ensure full reproducibility of these calculations.

**Location in Manuscript:** Section 3.6 (Data Analysis), Section 4.5 (Rubric Results), Section 5.7 (Limitations).

---

### Comment R1.7: Moderation of Universal Claims
> *"Claims regarding universal applicability, evidence-based effectiveness, and broad transferability should be moderated."*

**Response:**  
We have systematically reviewed the entire manuscript and removed all instances of overgeneralised wording:
- Replaced "universal breakthrough moments" with "commonly reported breakthrough moments" (Abstract, Section 4.6).
- Replaced "universal learning landmark" with "widely observed / common learning landmark" (Section 5.5).
- Moderated transferability claims in Section 5.6 to state that the framework is designed to be domain-adaptable and that transferability to other engineering disciplines warrants empirical investigation.
- Replaced "evidence-based model" with "empirically-evaluated model" throughout.

**Location in Manuscript:** Abstract, Section 5.5, Section 5.6, Section 6.
