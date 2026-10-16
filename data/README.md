# The datasets

Three tabular options forming a difficulty ladder, plus an image option. All are licence-clear and all are downloaded by the script at the repo root. Pick one on launch day and stay with it.

```bash
python download_data.py houses     # or: telco, bank, flowers
python download_data.py tabular    # the three tabular ones, if you want to browse before choosing
```

Files land in `data/raw/`, which is gitignored, so someone cloning your repo re-runs the script rather than pulling data out of GitHub.

| | Dataset | Task | Rows | Difficulty |
|---|---|---|---|---|
| **1** | King County house sales | Regression: the sale price | 21,613 | Gentler start |
| **2** | Telco customer churn | Classification: who leaves | 7,043 | Middle |
| **3** | Bank Marketing | Imbalanced classification: who subscribes | 41,188 | Hardest |
| **4** | Flower photos | Image classification: five kinds of flower | 3,670 photos | Image option |

---

## Option 1 — King County house sales · *gentler start · regression*

Every house sold in King County, Washington (Seattle and around) between May 2014 and May 2015.

| File | Rows | What it is |
|---|---|---|
| `kc_house_sales.csv` | 21,613 | One row per sale: `price` (the target), the sale `date`, size (`sqft_living`, `sqft_lot`, `bedrooms`, `bathrooms`, `floors`), quality (`grade`, `condition`, `view`, `waterfront`), age (`yr_built`, `yr_renovated`), and location (`zipcode`, `lat`, `long`). |

**The question it supports.** What is a house worth? A model that a buyer, a seller or an agent could use to check whether an asking price is reasonable, and how far off it is likely to be.

**What makes it the gentlest.** No missing values, almost every column is already a number, and the relationship between size and price is strong from the start. You can spend the week on features and tuning rather than on cleaning.

**The catches.**

- **The target is skewed.** Prices run from 75,000 to 7.7 million with a long right tail (skewness 4.0). An error of 50,000 means something very different on a 300,000 house than on a 3 million one. Look at the log of the price, and think about which metric matches what a user of the model cares about.
- **Some houses sold twice.** 177 `id` values appear more than once. If one sale is in your training set and the other in your test set, your model has partly seen the answer. Decide what to do and say so.
- **Dates and years need turning into features.** `date` is text like `20141013T000000`, and `yr_renovated` is 0 for the 96% of houses never renovated, which is not the year zero.
- **A few rows to check.** One house has 33 bedrooms and 13 have none.

**Licence.** Public domain ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)), published by King County and distributed on [OpenML](https://www.openml.org/d/42092). No attribution is required, but name the source in your README anyway:

> House sales in King County, USA, 2014–2015, via [OpenML dataset 42092](https://www.openml.org/d/42092). Public domain (CC0).

<br>

## Option 2 — Telco customer churn · *middle · classification · the reference solution uses this one*

A sample of 7,043 customers of a fictional telecoms company, published by IBM, with whether they left in the last month.

| File | Rows | What it is |
|---|---|---|
| `telco_churn.csv` | 7,043 | One row per customer: demographics (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), account (`tenure` in months, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`), the services they use (phone, internet, security, streaming and so on), and `Churn`, the target. |

**The question it supports.** Which customers are about to leave, so a retention team can contact them first? With a limited budget for calls or discounts, the model decides who gets one.

**Why it is the middle option.** Most columns are text categories, so the preprocessing pipeline matters. About one customer in four churns (26.5%), which is imbalanced enough that accuracy misleads, but not so much that the rare class is hard to find. The metric and the threshold are real business decisions here.

**The catches.**

- **`TotalCharges` is stored as text.** Eleven customers have a blank value — they are the ones with `tenure` of 0, who have not been billed yet. Find them before you convert the column.
- **`customerID` identifies a row.** It predicts nothing and must not be a feature.
- **Some categories repeat information.** `"No internet service"` appears in six columns and always goes with `InternetService == "No"`. Decide whether that matters for the models you use.
- **`TotalCharges` is close to `tenure × MonthlyCharges`.** Strongly related features are fine for prediction, but they make importance scores harder to read.

**Licence.** [Apache 2.0](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE), from IBM's sample data. Credit it:

> Telco customer churn sample data, IBM, from [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d). Licensed under Apache 2.0.

<br>

## Option 3 — Bank Marketing · *hardest · imbalanced classification*

Phone campaigns run by a Portuguese bank between May 2008 and November 2010: who was called, what they were like, the economy at the time, and whether they subscribed to a term deposit.

| File | Rows | What it is |
|---|---|---|
| `bank_marketing.csv` | 41,188 | One row per call: the client (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`), the call (`contact`, `month`, `day_of_week`, `duration`, `campaign`), previous campaigns (`pdays`, `previous`, `poutcome`), five economic indicators, and `y`, the target. **Separated by semicolons:** `pd.read_csv(path, sep=";")`. |
| `bank_marketing_names.txt` | — | The authors' description of every column. Read it, especially the note on `duration`. |

**The question it supports.** Which clients should the bank call in the next campaign? Every call costs an agent's time; every missed subscriber is lost revenue.

**Why it is the hardest.** Only 11.3% of clients say yes, so a model that always says no is 88.7% accurate and useless. The metric, the threshold and how you compare models all have to be chosen with the imbalance in mind. There are also more traps than in the other two.

**The catches.**

- **`duration` leaks the answer.** It is the length of the call, which is only known after the call has happened — and a long call usually means a yes. A model that uses it scores beautifully and could never be used to decide whom to call. **Drop it**, as the authors' own notes say, and explain why in your README.
- **`"unknown"` is a missing value in disguise.** 12,718 cells hold it, most of them in `default`, where only 3 clients say `"yes"`. Decide whether "unknown" is information (it can be) or a gap.
- **`pdays` uses 999 for "never contacted before",** which is 96% of rows. Treated as a number, 999 days is nonsense.
- **The economic indicators move with time.** The rows are in date order and the five indicators describe the month of the call. A random split is fine for this project, but notice that they partly encode *when* a call happened.
- **12 rows are exact duplicates.**

**Licence.** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite it:

> Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5K306>. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

<br>

## Option 4 — Flower photos · *image option*

3,670 photos of flowers from Flickr, in five folders, one per class. This is the option for the Week 5 Thursday material: a CNN from scratch, then transfer learning.

| Folder | Photos |
|---|---|
| `flower_photos/daisy/` | 633 |
| `flower_photos/dandelion/` | 898 |
| `flower_photos/roses/` | 641 |
| `flower_photos/sunflowers/` | 699 |
| `flower_photos/tulips/` | 799 |

The download is **218 MB** and needs TensorFlow to model: check `python -c "import tensorflow"` before you choose this option.

**The question it supports.** Can an app tell a user which flower they photographed? Frame it as a product: who would use it, and which mistake would annoy them most.

**What makes it different.** There is no table and no `ColumnTransformer`. The preprocessing lives in the model (resizing, rescaling, augmentation layers), the baseline is a small CNN trained from scratch, and the real model is a pretrained network with a new output layer. The rubric is the same; the brief says what each criterion means for images.

**The catches.**

- **The photos are all different sizes** (most around 320 × 240) and you resize them. Bigger images are slower; 160 × 160 is a good balance on a laptop CPU.
- **The classes are slightly unbalanced** (633 daisies against 898 dandelions). Accuracy is still a reasonable metric, but look at the per-class results.
- **Some photos are not clean examples**: several flowers, people, or a flower that is barely visible. Find a few in your EDA; they explain some of your errors.
- **Training time is real.** On a CPU, fine-tuning a whole pretrained network can take an hour. Freeze it first and train only the new layer; that takes minutes.
- **`image_dataset_from_directory` shuffles.** Set a `seed` and use the same one for the training and validation subsets, or the two will overlap.

**Licence.** [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/), one photographer per photo, listed in `flower_photos/LICENSE.txt`. Credit the collection:

> Flower photos from the [TensorFlow example images](https://www.tensorflow.org/tutorials/load_data/images), Flickr photographers listed in `LICENSE.txt`, licensed under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

---

## Bringing your own dataset

You may, and you are graded on exactly the same [rubric](../RUBRIC.md). Three conditions:

- **It has a clear target to predict**, and it is either tabular with **at least a few thousand rows**, or images in labelled folders with **at least a few hundred per class**.
- **Every feature would be known at the moment of prediction.** If you cannot say when each column becomes available, you cannot rule out a leak.
- **Its licence permits educational use.** Anything marked *NonCommercial* is a grey area for a bootcamp; anything with no stated licence at all is a no.

Clear it with your teacher on launch day. Not on Wednesday.

## Attribution, in one place

Whatever you use, your project README must name the source and its licence. For the CC BY datasets that is a requirement of the licence, not a nicety.
