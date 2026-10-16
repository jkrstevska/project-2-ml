![Ironhack logo](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project 2 — Evaluation Rubric

*Machine Learning: From Data to Predictions · Data Science & Machine Learning*

This is the **single source of truth** for how Project 2 is graded. It is adapted from Ironhack's DSML project rubric and uses the same **0–3 scale** as the [Generic Lab Rubric](https://gist.github.com/ironhack-edu/f5cf405db1708c201ad774ee4516bc94) and the Project 1 rubric, so labs and projects are assessed on one house scale.

Read it on launch day. It tells you exactly what is being looked at, and there are no surprises in here that the [brief](README.md) does not already ask for.

## How the scale works

| Score | Meaning |
|---|---|
| **0 — Incomplete** | Not attempted, or attempted in a way that does not meet the requirement. |
| **1 — Fair** | Attempted, with real gaps. |
| **2 — Good** | Meets the requirement competently. **This is the target.** |
| **3 — Excellent** | Exceeds it — depth, rigour or polish beyond what was asked. |

Each level is a **checklist behind a threshold**: "all of these apply", "at least two of the following apply". Read the threshold before the bullets. You are placed at the highest level whose threshold you satisfy.

**Passing is a 2 on every criterion that applies to your project.** Two or three 3s is a strong project; 3s everywhere is rare and is not what you should be aiming for on a four-day project.

## The ten criteria

| # | Criterion | Domain | Where it is judged from |
|---|---|---|---|
| 1 | [Problem framing](#1--problem-framing) | Business Problem Solving | The framing at the top of notebook 01, the conclusions in notebook 03 |
| 2 | [Data preparation](#2--data-preparation) | Data Preparation | Notebooks 01 and 02 |
| 3 | [Exploratory analysis](#3--exploratory-analysis) | Data Analysis | Notebook 01 |
| 4 | [Modelling and validation](#4--modelling-and-validation) | Machine Learning | Notebook 02 |
| 5 | [Evaluation and interpretation](#5--evaluation-and-interpretation) | Machine Learning | Notebook 03 |
| 6 | [Data visualisation](#6--data-visualisation) | Data visualization and communication | Notebooks, slides |
| 7 | [Code quality and structure](#7--code-quality-and-structure) | Coding | `src/functions.py`, the repo as a whole |
| 8 | [Git and GitHub](#8--git-and-github) | Coding | Commit history |
| 9 | [Documentation](#9--documentation) | Coding | `README.md`, docstrings, comments |
| 10 | [Presentation and demo](#10--presentation-and-demo) | Data visualization and communication | Friday, 7 + 3 minutes |

Criteria **4** and **5** are specific to this project and carry most of the week's learning. Each says what it means for the **image option** where that differs. The other eight are the Ironhack DSML rubric criteria, carried over with the changes noted.

---

## 1 | Problem framing

**Learning outcome:** Turn a business problem into a prediction task — who or what is predicted, for whom, and what they would do with it — and choose a metric from the cost of each kind of mistake.

> Adapted from the DSML rubric's *Business problem solving*: research questions become a prediction task, a metric and a baseline.

**0 — Incomplete** · *all of these apply*
- There is no clear statement of what is predicted or why.
- No metric is chosen, or it is chosen with no reason.
- No conclusions are drawn for the business.

**1 — Fair** · *at least two of the following apply*
- The target is stated, but not who would use the prediction or what they would do with it.
- A metric is used, but it does not fit the problem — for example accuracy on a heavily imbalanced target with no discussion.
- There is no baseline, so a model's score cannot be judged.
- Conclusions restate the scores without saying what they mean for the business.

**2 — Good** · *all of these apply*
- **The prediction task is framed**: the target, the positive class for classification, who uses the prediction and what decision it informs.
- **The metric is justified** by which mistake costs more.
- **A baseline is scored** with the same validation as the models, and every result is read against it.
- Conclusions answer the original question, including whether the model is good enough to use.

**3 — Excellent** · *at least three of the following apply*
- The cost of each kind of mistake is made concrete — with numbers or a worked scenario — and drives the choice of metric and threshold.
- The conclusion includes a recommendation a non-technical stakeholder could act on.
- The framing is revisited at the end: what the model can and cannot be used for, and what would change the recommendation.
- A second, simpler question is answered along the way and used to support the main one.

---

## 2 | Data preparation

**Learning outcome:** Identify data quality issues in a dataset and apply appropriate data cleaning, wrangling, and manipulation techniques to address them, without letting information from the future or from the test set into training.

**0 — Incomplete** · *all of these apply*
- No relevant data quality issues found.
- No relevant implementation of data cleaning or wrangling techniques.

**1 — Fair** · *all of these apply*
- Some data quality issues identified, but they are irrelevant, unclear or incomplete.
- Cleaning and wrangling techniques were applied, but they are inadequate for the issues identified.

**2 — Good** · *at least three of the following apply*
- Identified the relevant data quality issues, with clear explanations of their impact on the models.
- Implemented appropriate cleaning techniques, consistently applied to most identified issues.
- **Identifiers and columns that would not be known at prediction time are dropped**, with the reason written down.
- Missing values handled with a stated strategy; imputation that learns from the data is done inside the pipeline (criterion 4).
- *Image option:* broken or misleading images checked, image size chosen and justified, and the train, validation and test sets kept apart.

**3 — Excellent** · *all of these apply*
- Identified most relevant data quality issues, with clear explanations of their impact.
- Implemented relevant techniques, consistently applied to **all** identified issues.
- Every column kept or dropped for a reason that can be read in the notebook, including the borderline ones.
- Missing data handled with the strategy fully justified — including where a missing value was kept as information.

---

## 3 | Exploratory analysis

**Learning outcome:** Apply Exploratory Data Analysis techniques to understand the data and the target, and to guide the modelling decisions that follow.

**0 — Incomplete** · *all of these apply*
- No basic EDA techniques used to analyse the data.
- Improper use of EDA for the data types present.

**1 — Fair** · *at least two of the following apply*
- Basic EDA techniques, but incomplete or inaccurate.
- The target's distribution is not looked at.
- Charts are produced, but they are not easy to understand or not interpreted.

**2 — Good** · *at least three of the following apply*
- **The target is analysed**: its distribution, and for classification the share of each class.
- Univariate analysis of the features that matter, and **bivariate analysis relating them to the target**.
- Findings are written in prose under the outputs, not left as bare cells.
- The EDA visibly informs a later decision — a transformation, a dropped column, a feature, a metric.
- *Image option:* a grid of examples per class, the class balance, image sizes, and a few hard or mislabelled-looking examples.

**3 — Excellent** · *all of these apply*
- Thorough univariate and bivariate analysis, using numerical measures and charts suited to each data type.
- Relationships between features are checked where they matter for the models (strong correlations, redundant columns).
- Every modelling decision that follows can be traced back to something shown in the EDA.

---

## 4 | Modelling and validation

**Learning outcome:** Build models with scikit-learn pipelines (or Keras for images), compare them fairly with cross-validation against a baseline, and tune them without touching the test set.

**Skill tags:** *Statistics & Machine Learning: Supervised & Unsupervised Learning*, *Model Evaluation*

**0 — Incomplete** · *all of these apply*
- One model, or models with no comparison.
- No baseline.
- The test set used for training or tuning, or no separate test set at all.

**1 — Fair** · *at least two of the following apply*
- Several models, but scored on different splits, or on the training data.
- Preprocessing that learns from the data — a scaler, an imputer, an encoder, feature selection — fitted on all the data before the split.
- Hyperparameters tuned by looking at test-set scores.
- Results reported with no spread, so a small difference cannot be told from noise.

**2 — Good** · *all of these apply*
- **The data is split once**, with a fixed `random_state` (stratified for classification), and the test set is not used until criterion 5.
- **Every step that learns from data is inside a `Pipeline`**, with a `ColumnTransformer` for the different column types.
- **At least three model families compared with cross-validation on the same folds**, against the baseline, with the mean and spread reported.
- **The best one or two tuned** with a search on the training set only.
- *Image option:* a fixed train, validation and test split; preprocessing and augmentation as layers inside the model; a CNN trained from scratch compared with a pretrained network used for transfer learning; training curves shown and early stopping used.

**3 — Excellent** · *at least three of the following apply*
- Feature engineering inside the pipeline that measurably improves the cross-validated score, with the improvement reported.
- The comparison covers a deliberate range — linear, tree-based and ensemble — and the reason the winner wins is discussed.
- Class imbalance handled deliberately (class weights, the metric, the threshold) and its effect measured.
- The simplest model within the noise of the best is preferred, and the choice is argued.
- *Image option:* fine-tuning of the pretrained network after training the new layer, with its effect measured, or a comparison of two pretrained networks.

---

## 5 | Evaluation and interpretation

**Learning outcome:** Evaluate a final model once on held-out data, read the result honestly, explain what drives its predictions and where it fails, and save it so it can be used.

**Skill tags:** *Statistics & Machine Learning: Model Evaluation*

**0 — Incomplete** · *all of these apply*
- No evaluation on held-out data.
- Scores reported with no interpretation.

**1 — Fair** · *at least two of the following apply*
- The test set is used several times, so the reported score is optimistic.
- Only one number is reported — no confusion matrix, no per-class or residual analysis.
- No attempt to explain what drives the predictions.
- For classification, the default threshold of 0.5 used without question.

**2 — Good** · *all of these apply*
- **The final model is evaluated once on the test set**, and the score is compared with the cross-validated estimate and with the baseline.
- **Errors are analysed**: a confusion matrix and the per-class results for classification; residuals and where they are largest for regression.
- For classification, **the threshold is chosen from cross-validated probabilities** on the training set, for a stated reason.
- **What drives the predictions is explained**, with permutation importance or an equivalent, and read with care.
- **The fitted pipeline is saved** and reloaded to predict new rows.
- *Image option:* a confusion matrix, per-class precision and recall, a gallery of misclassified images with a reading of why, and the model saved as a `.keras` file and reloaded to predict a new photo.

**3 — Excellent** · *at least three of the following apply*
- The errors are segmented — by group, by price range, by class — and the segments where the model should not be trusted are named.
- The interpretation is turned into business insight, not just a ranking of features.
- Uncertainty is shown, for example the spread across folds alongside the test score.
- Limitations are stated precisely: what data, time period or population the model holds for.

---

## 6 | Data visualisation

**Learning outcome:** Select and use appropriate data visualisation techniques that effectively communicate insights, and create informative visualisations using Python libraries.

**0 — Incomplete** · *at least two of these apply*
- No appropriate visualisation techniques used to communicate insights.
- Visualisations lack clarity; the metrics are not properly identified, defined or plotted.
- Visual design is not informative.

**1 — Fair** · *at least two of these apply*
- Fair use of visualisation to communicate insights, but chart-type selection and design need work.
- Visualisations have some clarity but need better organisation.
- Metrics need to be more clearly defined and plotted.
- Some plots lack detail or have formatting problems — missing titles, unlabelled axes, unreadable ticks.

**2 — Good** · *at least three of these apply*
- Solid use of visualisation for both the data (EDA) and the models: at least a model comparison and an error analysis chart.
- Charts are clear, well organised and support decision-making.
- Chart types are appropriate to the data, with clear labelling: title, axis labels, units.
- Each chart carries a written takeaway — what it shows, in one sentence.

**3 — Excellent** · *all of these apply*
- Exceptional use of visualisation to communicate insights effectively.
- Highly effective, with a clear and intuitive layout that conveys the key insight at a glance.
- Metrics expertly defined, measured and plotted, with great attention to detail.
- Model results shown in terms a stakeholder understands — money, customers, calls — not only in scores.

---

## 7 | Code quality and structure

**Learning outcome:** Write clean, modular, efficient code following best practices, and maintain a clean and logical project structure.

**0 — Incomplete** · *at least three of these apply*
- Much unused code left in the project.
- No functions.
- Naming conventions not applied, making the code hard to read.
- Many hard-coded values or global variables.
- No consistent approach to naming, structure and organisation of files and folders.

**1 — Fair** · *at least four of these apply*
- Some unused code left in the project.
- Functions are either too large or do several things at once.
- Naming conventions barely applied.
- Some hard-coded values or global variables.
- Some files and folders are organised appropriately; others need work.

**2 — Good** · *at least four of these apply*
- Little unused code left.
- **Functions are modular and reusable, and live in `src/functions.py`**, not pasted into cells.
- Naming conventions well applied.
- Few hard-coded values; the random seed and the paths are defined once.
- **The notebooks run top to bottom on a fresh kernel**, after `download_data.py`.

**3 — Excellent** · *all of these apply*
- No unused code left.
- Functions are cleanly modular and reusable, each doing one thing.
- Naming conventions applied consistently throughout.
- No hard-coded values or magic strings; paths and configuration are defined in one place.
- All files and folders organised appropriately, and the results are reproducible: the same numbers on a second run.

---

## 8 | Git and GitHub

**Learning outcome:** Save and track changes in the source code using Git and GitHub.

**0 — Incomplete** · *all of these apply*
- Zero or one commit in total.
- Commit messages provide no useful information.

**1 — Fair** · *all of these apply*
- At least two commits made during the project.
- Commit messages are unclear and ambiguous.

**2 — Good** · *at least two of these apply*
- Several commits made during the project, but fewer than one per project day.
- Commit messages are clear and accurately describe the changes.
- Working in a pair, both people have commits in the history.
- No data files and no large model files committed — `data/` and `models/` stayed ignored.

**3 — Excellent** · *all of these apply*
- **At least one commit per project day**, from each person if you are working in a pair.
- Atomic commits with accurate, precise descriptions, consistently.
- Branches used for development rather than committing everything straight to `main`.

---

## 9 | Documentation

**Learning outcome:** Document the project's features, configuration and technical specifications.

**0 — Incomplete** · *all of these apply*
- No attempt to document the project in a README.
- No documentation of the code or functions.

**1 — Fair** · *at least two of these apply*
- A partially completed README.
- Functions partially documented, with incomplete docstrings.
- Few comments explaining the rationale or logic behind the code.

**2 — Good** · *all of these apply*
- **A well-structured, clear README**, written for *your* project — the question, the data and its licence, the metric, the models compared, the final result against the baseline, and how to run it. The template brief has been replaced, not left in place.
- Functions documented with accurate docstrings.
- Enough comments to explain the rationale, logic and main ideas.

**3 — Excellent** · *all of these apply*
- A fully comprehensive, well-structured README that someone could follow from clone to prediction without asking a question.
- Functions documented with complete docstrings — parameters, returns, and the decisions the caller has to make.
- Clear, concise comments explaining purpose and functionality, addressing *why* rather than restating *what*.

---

## 10 | Presentation and demo

**Learning outcome:** Build a presentation and perform a demo to deliver your results.

Format: **7 minutes of slides plus a 3-minute live demo**, presented from your own machine by sharing your screen. Any slide tool. The demo loads your saved model and predicts for new data.

**0 — Incomplete** · *at least two of these apply*
- The presentation lacks clear structure and purpose, making the results hard to follow.
- The demo is poorly executed and does not communicate the results.
- No storytelling, making the presentation hard to engage with.

**1 — Fair** · *at least two of these apply*
- The presentation has some structure, but would benefit from better organisation and clearer presentation of the findings behind the conclusions.
- The demo communicates the results adequately, but could be more polished, with better pacing.
- Storytelling is present but not used effectively.

**2 — Good** · *at least three of these apply*
- Clear structure and purpose, effectively communicating the results.
- The demo is engaging, well rehearsed, and stays within the allocated time: the saved model is loaded and predicts for an example the audience can follow.
- Storytelling techniques are well incorporated and add to the audience's engagement.
- Conclusions and next steps are included.
- The metric and the baseline are explained so a non-technical listener knows whether the model is any good.

**3 — Excellent** · *at least four of these apply*
- Highly compelling, with a clear message and a well-structured narrative.
- Flawlessly executed demo, showcasing the results memorably and using the time proficiently.
- Storytelling expertly used to build a narrative that deepens the audience's understanding.
- Conclusions and next steps included, alongside the strengths and limitations of the model and recommendations for further work.
- Visualisation used in an innovative and meaningful way to support the findings.

---

## Bringing your own dataset

A pair working on their own data is assessed on **exactly these criteria** — there is no separate, easier or harder rubric.

The conditions in [`data/README.md`](data/README.md#bringing-your-own-dataset) are what keep that fair: a clear target, enough rows or images, and every feature known at the moment of prediction. Without them, criteria 4 and 5 have nothing honest to assess. Clear the dataset with your teacher on launch day.

## Attribution

Bank Marketing and the flower photos are under Creative Commons licences that require attribution, and Telco is under Apache 2.0. Name the source and the licence in your README and on your data slide; the exact lines to use are in [`data/README.md`](data/README.md). A missing attribution counts against criterion 9.

---

## A note on flexibility

Carried over from Ironhack's project rubric, and it applies here too:

> In this agnostic project rubric, it is important to acknowledge that each project may have unique characteristics and requirements. As such, it is understood that not all learning outcomes or criteria points listed in the rubric will be applicable to every project.
>
> Students should focus on relevant outcomes and criteria for self-assessment, while teachers should consider project-specific aspects for evaluation. This approach ensures a tailored assessment aligned with the unique requirements of each project.

In practice: if a criterion genuinely does not apply to your project — the threshold bullet for a regression, for example — say so in your README and explain why, rather than leaving a gap for the reader to interpret.
