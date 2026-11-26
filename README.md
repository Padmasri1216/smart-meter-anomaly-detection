# Smart Meter Anomaly Detection

This is my Master's internship project. The goal was to build a machine learning model that can spot abnormal electricity readings from smart meters, the kind of readings that might point to a faulty meter, tampering or energy theft.

## The problem

Smart meters send a reading every 30 minutes, which adds up to a lot of data very quickly. Most of it is perfectly normal, so checking it by hand isn't realistic. I wanted a model that could flag the unusual readings automatically.

The tricky part is that anomalies are rare. In this dataset, only **5% of readings are abnormal**. That makes accuracy a misleading measure: a model that simply says "Normal" every time would still be 95% accurate while catching nothing. So throughout this project I focused on three measures for the abnormal class:

- **Recall**: out of all the real anomalies, how many did the model catch?
- **Precision**: out of everything the model flagged, how many were real anomalies?
- **F1 score**: a balance of the two

## The data

- 5,000 readings taken every 30 minutes, from January to April 2024
- 4,750 normal readings and 250 abnormal ones
- Each reading includes electricity consumed, average past consumption, temperature, humidity and wind speed
- From the timestamp, I also created hour, day, month and day-of-week features

When I explored the data, the abnormal readings stood out clearly in electricity consumption and average past consumption: they were either unusually high or unusually low. Weather looked almost the same for normal and abnormal readings.

## What I did

1. **Explored the data** to understand the imbalance and how each feature behaves
2. **Prepared the data**: split it 80/20 into training and test sets, kept the 5% ratio in both, and scaled the features using the training data only
3. **Built a baseline** Random Forest with no special handling of the imbalance
4. **Balanced the training data with SMOTE**, which creates synthetic examples of the rare abnormal class
5. **Tried several models**: Random Forest with SMOTE, a tuned Random Forest (GridSearchCV), and Gradient Boosting with SMOTE
6. **Validated properly**: during cross-validation, SMOTE runs inside a pipeline so it only touches the training folds. This keeps the scores honest.
7. **Tuned the decision threshold**: instead of the default 0.5, I tested different thresholds and chose **0.7** for the best balance between catching anomalies and avoiding false alarms
8. **Evaluated the final model** with a classification report, confusion matrix, ROC curve and feature importance

## Results

Both models were tested on the same 1,000 unseen readings, 50 of which were abnormal.

| Model | Recall | Precision | F1 score | Anomalies caught |
|---|---|---|---|---|
| Baseline Random Forest | 0.72 | 0.92 | 0.81 | 36 of 50 |
| **Gradient Boosting + SMOTE (threshold 0.7)** | **0.94** | **0.80** | **0.86** | **47 of 50** |

**What this means:** the baseline model was cautious. When it flagged something it was usually right, but it **missed 14 real anomalies**. After balancing the data with SMOTE and tuning the threshold, the final model **catches 47 out of 50 anomalies** and misses only 3. The trade-off is 12 false alarms out of 950 normal readings. In a real setting, an engineer double-checking a few extra readings is far better than missing a faulty or tampered meter.

**Overall performance:** the final model has a **ROC AUC of 0.997**, which means it is very good at ranking abnormal readings above normal ones (1.0 would be perfect).

**Which features mattered:** almost all of the model's decisions came from **electricity consumed** and **average past consumption**. Weather and time features added very little, which matches what I saw while exploring the data.

## Tools used

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, seaborn, joblib and Google Colab

## How to run it

1. Open `smart_meter_anomaly_detection.ipynb` in Google Colab
2. Click **Runtime → Run all**
3. Upload `smart_meter_data.csv` when asked

## Files in this repository

| File | What it is |
|---|---|
| `smart_meter_anomaly_detection.ipynb` | The full notebook with code, charts and results |
| `smart_meter_data.csv` | The dataset |
| `smart_meter_model.pkl` | The trained model, saved together with the scaler and the chosen threshold |

## What I'd like to try next

- Adding features based on longer history, such as rolling averages over a day or a week
- Trying unsupervised methods like Isolation Forest, which can find anomalies without needing labels
- Turning the model into a small web app or API so it can check new readings in real time

## About me

**Padmasri**. I completed this project as part of my Master's internship.
GitHub: [Padmasri1216](https://github.com/Padmasri1216)

