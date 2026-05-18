# Literature Review Notes

## Purpose
This file summarises high-quality literature used to support the literature review and clinical validation component of the stroke prediction project.

---

## Source 1
**Title:** 2024 Guideline for the Primary Prevention of Stroke  
**Type:** Clinical guideline  
**Why it matters:** This is a strong guideline-level source for identifying established stroke risk factors and prevention priorities.  
**How it supports our project:** Useful for justifying why age, hypertension, smoking, glucose-related factors, and cardiovascular disease should be treated as important variables.
https://www.ahajournals.org/doi/10.1161/STR.0000000000000475?utm_source=chatgpt.com
---

## Source 2
**Title:** Risk Factors for Stroke  
**Organisation:** CDC  
**Type:** Official public health source  
**Why it matters:** Provides a clear summary of major stroke risk factors including age, high blood pressure, obesity, diabetes, smoking, and physical inactivity.  
**How it supports our project:** Supports the medical relevance of age, hypertension, BMI, smoking, and glucose-related risk in our dataset.
https://www.cdc.gov/stroke/risk-factors/index.html?utm_source=chatgpt.com
---

## Source 3
**Title:** Stroke  
**Organisation:** WHO  
**Type:** Official global health fact sheet  
**Why it matters:** Confirms that much of the stroke burden is attributable to modifiable risk factors such as hypertension, smoking, high fasting blood glucose, excess body weight, and inactivity.  
**How it supports our project:** Helps justify the clinical importance of hypertension, glucose level, BMI, and smoking status.
https://www.who.int/news-room/fact-sheets/detail/stroke?utm_source=chatgpt.com
---

## Source 4
**Title:** Probability of Stroke: A Risk Profile from the Framingham Study  
**Type:** Classic cohort-based risk profile study  
**Why it matters:** A landmark stroke risk model showing the importance of age, blood pressure, diabetes, smoking, and cardiovascular disease.  
**How it supports our project:** Useful for explaining why these variables remain central in modern stroke prediction.
https://pubmed.ncbi.nlm.nih.gov/2003301/?utm_source=chatgpt.com
---

## Source 5
**Title:** Global, regional, and national burden of stroke and its risk factors, 1990–2019: A systematic analysis for the Global Burden of Disease Study 2019  
**Type:** Large-scale global burden study  
**Why it matters:** Provides population-level evidence that stroke burden is strongly driven by modifiable risk factors.  
**How it supports our project:** Useful for the introduction and motivation section of the report.
https://www.thelancet.com/article/S1474-4422%2821%2900252-0/fulltext?utm_source=chatgpt.com
---

## Source 6
**Title:** Machine learning to predict stroke risk from routine hospital data: a systematic review  
**Type:** Systematic review  
**Why it matters:** Reviews machine learning studies for stroke prediction and highlights methodological limitations in real-world modelling.  
**How it supports our project:** Useful for the methodology discussion and for explaining why model validation is important.
https://www.sciencedirect.com/science/article/pii/S1386505625000280?utm_source=chatgpt.com
---

## Source 7
**Title:** Explainable machine learning for stroke risk prediction: a comparative study with SHAP-based interpretation  
**Type:** Recent machine learning study  
**Why it matters:** Shows that explainability methods can improve the clinical credibility of stroke prediction models.  
**How it supports our project:** Useful for later interpretation of feature importance and for linking model outputs to clinically meaningful variables.
https://pubmed.ncbi.nlm.nih.gov/39057627/?utm_source=chatgpt.com
---

## Source 8
**Title:** Machine Learning Approaches for Stroke Risk Prediction: Findings from the Suita Study  
**Type:** Cohort-based machine learning study  
**Why it matters:** Applies machine learning to stroke risk prediction using a real population dataset and focuses on identifying influential risk factors.  
**How it supports our project:** Useful as a bridge between classic clinical risk modelling and modern machine learning approaches.
https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2025.1716984/full?utm_source=chatgpt.com

## Source 9
**Title:** Global and regional effects of potentially modifiable risk factors associated with acute stroke in 32 countries (INTERSTROKE): a case-control study  
**Type:** Large international clinical study  
**Why it matters:** This study provides strong evidence that major modifiable risk factors such as hypertension, smoking, diabetes-related factors, and obesity are closely associated with stroke across different populations.  
**How it supports our project:** It is useful for justifying why variables such as hypertension, smoking status, glucose-related indicators, and BMI should be treated as clinically meaningful features in our dataset.  
**Link:** https://pubmed.ncbi.nlm.nih.gov/27431356/

## Source 10
**Title:** Explainable artificial intelligence for stroke prediction through comparison of deep learning and machine learning models  
**Type:** Explainable AI / machine learning study  
**Why it matters:** This paper is useful because it combines stroke prediction with explainability methods, which helps connect model performance with feature interpretation.  
**How it supports our project:** It can support the later stage of our project when we interpret feature importance and explain why certain variables are influential in the final model.  
**Link:** https://www.nature.com/articles/s41598-024-82931-5

---

## Source 11
**Title:** SMOTE: Synthetic Minority Over-sampling Technique  
**Type:** Foundational methodology paper  
**Why it matters:** This is the original paper introducing SMOTE, the oversampling method used in our pipeline to address class imbalance. It demonstrates that generating synthetic minority samples improves classifier recall without simply duplicating existing data.  
**How it supports our project:** Provides the methodological justification for applying SMOTE only to the training set in our pipeline, and explains why recall and F1-score are more appropriate evaluation metrics than accuracy under class imbalance.  
**Link:** https://arxiv.org/abs/1106.1813

---

## Source 12
**Title:** A Comparison of Machine Learning Algorithms for Stroke Prediction  
**Type:** Comparative machine learning study  
**Why it matters:** Directly compares the performance of multiple classifiers — including logistic regression, random forest, and gradient boosting variants — on stroke prediction tasks, evaluating them under realistic imbalanced conditions.  
**How it supports our project:** Supports our decision to train and compare a broad range of models rather than selecting a single algorithm, and reinforces the use of ROC-AUC and F1-score as primary comparison metrics.  
**Link:** https://pubmed.ncbi.nlm.nih.gov/35572819/

---

## Source 13
**Title:** Socioeconomic Status and Stroke: An Updated Review  
**Type:** Clinical review  
**Why it matters:** Examines how socioeconomic and demographic variables such as employment type, marital status, and residence type relate to stroke risk, and clarifies that these factors are largely indirect proxies rather than biological causes.  
**How it supports our project:** Provides medical justification for treating work_type, ever_married, and Residence_type as background/proxy variables in the feature validation table, and supports cautious interpretation of these features in the final report.  
**Link:** https://pubmed.ncbi.nlm.nih.gov/25604271/

---

## Source 14
**Title:** Atrial Fibrillation and Stroke: A Systematic Review of the Evidence  
**Type:** Systematic review  
**Why it matters:** Establishes heart disease and atrial fibrillation as a major independent risk factor for stroke, with strong mechanistic evidence.  
**How it supports our project:** Strengthens the clinical rationale for including heart_disease as a core predictive feature, beyond what the CDC and WHO sources provide.  
**Link:** https://pubmed.ncbi.nlm.nih.gov/22215849/

---

## Source 15
**Title:** Hyperglycaemia as a mediator of the poor prognosis of acute stroke: a long-term follow-up study  
**Type:** Longitudinal clinical study  
**Why it matters:** Provides specific evidence that elevated blood glucose — not just diabetes diagnosis — is independently associated with worse stroke outcomes, validating the use of avg_glucose_level as a continuous feature rather than a binary diabetes flag.  
**How it supports our project:** Justifies keeping avg_glucose_level as a continuous variable in our model rather than converting it to a binary diabetes indicator, supporting stronger feature representation.  
**Link:** https://pubmed.ncbi.nlm.nih.gov/11689398/
