Research Project Context

This research investigates the use of fuzzy logic, machine learning, and explainable artificial intelligence for predicting drug-release behavior from polymer-based long-acting injectable formulations.

1. Research Domain and Problem

The research focuses on drug-loaded PLGA microparticles used in long-acting injectable drug-delivery systems.

PLGA stands for poly(lactide-co-glycolide), a biodegradable and biocompatible polymer commonly used to encapsulate drugs. After administration, the drug is gradually released from the PLGA microparticles over an extended period.

The drug-release behavior is affected by multiple interacting characteristics related to:

- The drug itself
- The PLGA polymer
- The formulation composition
- The manufacturing method
- The resulting microparticle properties
- The in-vitro drug-release conditions

Developing an effective long-acting formulation normally requires repeated and time-consuming laboratory experiments. An important challenge is predicting whether a formulation will produce:

- Rapid early drug release
- Sustained or delayed drug release
- A burst-like release profile
- A monophasic, biphasic, or triphasic release curve
- A suitable complete drug-release profile

Machine-learning models can learn the relationships between formulation characteristics and drug-release outcomes. However, these relationships are complex, multidimensional, nonlinear, and sometimes uncertain.

2. Benchmark Study

The principal benchmark paper is:

“Predicting early and complete drug release from long-acting injectables using explainable machine learning” by Karla N. Robles and Manar D. Samad, published in the International Journal of Pharmaceutics in 2026.

The benchmark study uses explainable machine learning to predict drug release from the static material and formulation characteristics of drug-loaded PLGA microparticles.

The benchmark study investigates three main prediction tasks:

Task 1: Early Drug-Release Prediction

This is a regression problem.

The objective is to predict the fractional or cumulative drug release at:

- 24 hours
- 48 hours
- 72 hours

The output is a numerical release value, usually represented as a fraction between 0 and 1.

For example:

0.35 means that approximately 35% of the drug has been released.

The benchmark models for this task include:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Task 2: Drug-Release Profile Classification

This is a binary classification problem.

Each normalized drug-release curve is characterized using the area under the curve, or AUC.

The two classes are:

- AUC ≤ 0.5
- AUC > 0.5

AUC ≤ 0.5 generally represents delayed, sustained, approximately linear, or more complex low-early-release profiles.

AUC > 0.5 generally represents profiles with greater release during the earlier part of the normalized release period and may include burst-biphasic behavior.

The AUC classification is a mathematical representation of the normalized release curve. It must not automatically be interpreted as a clinical or regulatory diagnosis of burst release.

The benchmark classifiers include:

- Logistic Regression
- Random Forest
- XGBoost

Task 3: Complete Drug-Release Curve Prediction

This is a multi-output or sequential regression problem.

The objective is to predict the full standardized drug-release curve using static drug, polymer, formulation, and microparticle characteristics.

The benchmark compares time-dependent and time-independent approaches.

The time-independent models attempt to predict the entire release curve without using time as an input feature and without requiring previous or partial drug-release measurements.

The benchmark models include:

- Time-dependent XGBoost
- XGBoost without time
- XGBoost Multi-Regressor
- Fully Connected Neural Network combined with LSTM
- Fully Connected Neural Network combined with GRU

3. Dataset

The original dataset was created by Bao et al. and published as:

“A dataset on formulation parameters and characteristics of drug-loaded PLGA microparticles.”

It was compiled from 113 scientific publications and contains:

- 321 in-vitro PLGA microparticle drug-release experiments
- 4,913 drug-release time points
- Multiple small-molecule drugs
- Formulation parameters
- Drug descriptors
- Polymer characteristics
- Microparticle characteristics
- Complete in-vitro drug-release profiles

The original dataset publication reports 89 drugs. The 2026 benchmark paper reports using 321 release profiles corresponding to 88 drugs. This difference may result from the benchmark study’s filtering, preparation, or drug-identity processing and should be checked when reproducing the experiments.

Each formulation represents one experimental drug-loaded PLGA microparticle formulation associated with a drug-release profile measured at multiple time points.

4. Main Input Features

The main input variables include:

A. Formulation Method

The emulsion method used to prepare the microparticles.

This is a categorical variable and may require one-hot encoding or another appropriate categorical representation.

B. Drug Molecular Weight — Drug MW

The molecular weight of the drug molecule.

C. Drug Topological Polar Surface Area — Drug TPSA

A molecular descriptor associated with the polar surface of the drug and its potential interactions with water and biological environments.

D. Drug LogP

The logarithmic partition coefficient describing the relative lipophilicity or hydrophilicity of the drug.

A higher LogP generally indicates greater affinity for lipophilic environments, while a lower LogP generally indicates greater affinity for aqueous environments.

E. Polymer Molecular Weight — Polymer MW

The molecular weight of the PLGA polymer carrier.

F. LA/GA Ratio

The molar ratio between lactide and glycolide in the PLGA polymer.

G. Initial Drug-to-Polymer Ratio — Initial DPR

The initial weight ratio of the drug to the polymer during formulation preparation.

H. Particle Size

The diameter of the resulting PLGA microparticles.

I. Encapsulation Efficiency — EE

The percentage of the initially added drug successfully encapsulated inside the microparticles.

J. Loading Capacity

The amount of drug relative to the total mass of the final drug-loaded microparticles.

K. Solubility Enhancer Concentration

The concentration of the solubility-enhancing agent used in the in-vitro release medium.

5. Benchmark Data Preparation

The benchmark study performs several preparation steps.

A. Categorical Encoding

The formulation method is treated as a nominal categorical variable and one-hot encoded.

B. Drug-Release Time Normalization

Each release profile can have a different total duration.

The duration of each profile is normalized to a range from 0 to 1 using Min-Max normalization:

normalized_time = (time - minimum_time) /
                  (maximum_time - minimum_time)

This creates a time-independent standardized representation of release progression.

C. Linear Interpolation

The original release profiles are measured at different and irregular time points.

Linear interpolation is used to:

- Estimate release values between observed points
- Produce uniformly sampled curves
- Create fixed-length output sequences
- Make curves from different studies more comparable

D. Early Release Extraction

Interpolated drug-release values are obtained at:

- 24 hours
- 48 hours
- 72 hours

These become the targets for the early-release regression experiments.

E. AUC Calculation

The area under each normalized drug-release curve is calculated.

The AUC is used to create the two release-profile classes:

- AUC ≤ 0.5
- AUC > 0.5

F. Feature Scaling

Numerical input features are standardized or scaled when required.

To avoid data leakage, preprocessing parameters must be fitted only on the training data and then applied to validation and test data.

G. Drug-Level Data Splitting

All formulations associated with the same drug identity should remain in the same fold.

A drug must not appear in both training and test sets.

The benchmark groups formulations according to drug identity using molecular structure information such as SMILES.

This grouping is important because a random row-level split could leak drug-specific physicochemical information into the test data and produce overly optimistic results.

H. Class-Imbalance Handling

For the binary classification experiment, the benchmark applies random undersampling to the majority class within the training data.

Resampling must not be applied to the validation or test sets.

6. Proposed Fuzzy-ML Methodology

The fuzzy methodology is currently a proposed experimental direction and has not yet been finalized.

The provisional approach is to use fuzzy logic as a feature-representation or feature-engineering method.

The main idea is to represent each numerical formulation characteristic using overlapping linguistic fuzzy sets such as:

- Low
- Medium
- High

In conventional crisp classification, a value belongs to only one interval.

For example:

- Values below 70 may be classified as Medium
- Values equal to or above 70 may be classified as High

This creates a hard cutoff even when two neighboring values are almost identical.

Fuzzy logic avoids this hard boundary by allowing the same value to belong to multiple fuzzy sets with different membership degrees.

For example, an encapsulation efficiency value of 75% might be represented as:

- EE_Low = 0.00
- EE_Medium = 0.30
- EE_High = 0.70

This means the value strongly belongs to the High set while retaining partial membership in the Medium set.

The process of converting a numerical value into fuzzy membership degrees is called fuzzification.

7. Membership Functions

Each numerical feature may be represented using membership functions such as:

- Triangular membership functions
- Trapezoidal membership functions
- Gaussian membership functions

The membership-function parameters may be determined through:

- Pharmaceutical domain knowledge
- Statistical properties of the dataset
- Minimum and maximum values
- Quantiles or percentiles
- Clustering
- Data-driven optimization
- A hybrid expert-driven and data-driven strategy

The final membership-function design has not yet been selected and must be experimentally evaluated.

Fuzzy membership functions should be fitted or defined without using information from the test set.

If their parameters depend on the data distribution, they should be calculated from the training data only within each cross-validation fold.

8. Crisp and Fuzzy Feature Representation

The original processed numerical features are referred to as crisp features.

For example:

Particle_Size = 45
Encapsulation_Efficiency = 75

After fuzzification, additional fuzzy features may be generated:

Particle_Size_Low
Particle_Size_Medium
Particle_Size_High

EE_Low
EE_Medium
EE_High

The current proposed approach is to combine:

1. Original crisp features
2. Generated fuzzy membership features

This produces a unified feature vector.

Example:

Original features:

- EE = 75
- Particle_Size = 45

Fuzzy features:

- EE_Low = 0.00
- EE_Medium = 0.30
- EE_High = 0.70
- Particle_Size_Low = 0.60
- Particle_Size_Medium = 0.40
- Particle_Size_High = 0.00

Unified feature vector:

- EE
- EE_Low
- EE_Medium
- EE_High
- Particle_Size
- Particle_Size_Low
- Particle_Size_Medium
- Particle_Size_High

The unified feature vector can then be used to train machine-learning or deep-learning models.

9. Experimental Comparison

The research should include controlled comparisons between models.

A. Baseline Representation

Use only the original crisp features.

B. Fuzzy-Only Representation

Use only fuzzy membership features, if this experiment is considered useful.

C. Hybrid Representation

Use both crisp and fuzzy features.

The comparison should determine whether fuzzy representation provides measurable improvements in:

- Predictive performance
- Robustness
- Generalization to unseen drugs
- Representation of nonlinear relationships
- Interpretability

All model comparisons should use the same data splits, preprocessing rules, evaluation procedure, and random seeds where possible.

10. Candidate Models

The provisional candidate models include:

- XGBoost for tabular regression and classification
- GRU-based models for complete release-curve prediction

Other models may be used as baselines or added later.

The exact fuzzy model has not yet been finalized.

The current proposal should not automatically be described as ANFIS, a Mamdani fuzzy inference system, a Sugeno system, or a fuzzy rule-based classifier unless the research team explicitly selects and implements one of these methods.

At the current stage, the approach is best described as:

A hybrid crisp and fuzzy feature-representation framework for explainable machine-learning prediction of PLGA microparticle drug-release behavior.

11. Explainability

Explainable AI will be used to investigate how formulation and fuzzy features influence model predictions.

The benchmark study uses SHAP, or Shapley Additive Explanations.

SHAP may be used to analyze:

- Global feature importance
- The direction of each feature’s effect
- Individual formulation predictions
- Feature influence at 24, 48, and 72 hours
- Feature influence on release-profile classification
- Feature importance across different parts of the complete release curve
- The importance of crisp features compared with fuzzy features

Possible IF–THEN rules may also be explored.

However, IF–THEN rules are not automatically generated merely by adding fuzzy features to XGBoost or GRU.

Rule extraction requires an additional explicit method, such as:

- A fuzzy rule-based inference system
- A neuro-fuzzy model
- A surrogate decision tree
- A dedicated rule-extraction algorithm
- Rules derived from interpretable membership functions and model explanations

Do not claim that fuzzy IF–THEN rules have been produced unless a specific rule-generation method has been implemented and validated.

12. Evaluation Metrics

For early drug-release regression:

- Root Mean Squared Error — RMSE
- Mean Absolute Error — MAE, if added
- Pearson correlation
- Coefficient of determination — R², if appropriate

For release-profile classification:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Confusion matrix

Because the classes are imbalanced, F1-score, recall, and class-specific results should be interpreted alongside accuracy and AUROC.

For complete release-curve prediction:

- RMSE across the full curve
- Correlation between actual and predicted curves
- Adjusted R² or another appropriate curve-level measure
- Performance within AUC ≤ 0.5 and AUC > 0.5 subgroups
- Visual comparison of representative predicted and actual curves

13. Research Question

The central research question is:

Can fuzzy representation of drug, polymer, formulation, and microparticle characteristics improve the prediction and interpretability of PLGA microparticle drug-release behavior compared with models trained using only the original crisp numerical features?

Possible subquestions include:

- Does fuzzification improve prediction at 24, 48, or 72 hours?
- Does it improve binary release-profile classification?
- Does it improve complete release-curve prediction?
- Which features benefit most from fuzzy representation?
- Do fuzzy features provide clearer and more scientifically meaningful explanations?
- Which membership-function design performs best?
- Does combining crisp and fuzzy features outperform either representation alone?
- Does the proposed method generalize to completely unseen drugs?

14. Important Research Constraints

When performing any task related to this project:

- Do not invent pharmaceutical interpretations.
- Clearly distinguish verified findings from hypotheses.
- Do not treat mathematical AUC classes as confirmed clinical burst-release labels.
- Prevent data leakage at every preprocessing stage.
- Keep all formulations of the same drug in the same cross-validation fold.
- Fit scaling, imputation, fuzzification parameters, feature selection, and resampling only on training data.
- Preserve the benchmark preprocessing and evaluation strategy when making comparisons.
- Report negative or insignificant results honestly.
- Do not assume that fuzzy features will necessarily improve accuracy.
- Do not claim the final fuzzy architecture has been selected.
- Do not claim IF–THEN rules are available unless rule extraction has actually been implemented.
- Explain all technical decisions clearly and provide reproducible code when requested.

15. Current Project Status

The benchmark paper and its dataset have been selected.

A preliminary hybrid fuzzy-ML methodology has been proposed and approved as a promising direction for initial experimentation.

However, the following decisions are still open:

- The exact membership-function type
- The number of fuzzy sets per feature
- How membership-function boundaries will be selected
- Whether all features or only selected features will be fuzzified
- Whether fuzzy features will replace or complement crisp features
- Whether a dedicated fuzzy-inference or neuro-fuzzy model will be used
- Which prediction task will be implemented first
- How IF–THEN rules will be generated
- The final model-selection and ablation-study design

Any suggested methodology should therefore be presented as a proposal to test rather than as an already finalized research design.