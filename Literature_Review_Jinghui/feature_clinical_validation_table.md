# Feature Clinical Validation Table

This table classifies the dataset variables based on their level of clinical support in stroke risk research.

| Feature | Clinical Support | Interpretation Note |
|---|---|---|
| age | Strong | Age is a well-established stroke risk factor and should be treated as a core predictive variable. |
| hypertension | Very strong | Hypertension is one of the most important modifiable stroke risk factors. |
| heart_disease | Strong | Cardiovascular disease is strongly associated with stroke risk. |
| avg_glucose_level | Strong | Elevated glucose level is clinically relevant and linked to stroke risk. |
| bmi | Moderate | BMI is useful, but its effect may overlap with hypertension and glucose-related risks. |
| smoking_status | Strong | Smoking is a recognised behavioural risk factor for stroke. |
| gender | Moderate | Can be analysed, but should not be overstated as a primary modifiable factor. |
| ever_married | Weak to Moderate | Likely reflects demographic or social background rather than direct clinical causation. |
| work_type | Weak to Moderate | May capture indirect lifestyle or socioeconomic patterns. |
| Residence_type | Weak | Should be interpreted cautiously because it has limited direct clinical meaning. |

## Notes
- Core clinically supported variables: age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status.
- Background or proxy-like variables: ever_married, work_type, Residence_type.
- These proxy/background variables may still help prediction, but they should be interpreted cautiously in the final report.
