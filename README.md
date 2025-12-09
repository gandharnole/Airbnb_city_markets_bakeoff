# Airbnb_city_markets_bakeoff
This project compares XGBoost and Deep Neural Networks (Keras) for predicting Airbnb listing prices across 12 U.S. cities, grouped into three market tiers (big, medium, small).

The analysis replicates a real-world multi-market machine learning pipeline:
Collecting and cleaning heterogeneous datasets
Engineering predictive features
Building and tuning multiple ML models
Evaluating generalization across cities and market tiers
The goal is to determine which model best predicts Airbnb prices under varying market complexity and data conditions.

## 🎯 Project Objectives
Implement neural networks using Keras (Chapter 10 concepts).
Compare XGBoost vs Neural Networks across different city markets.
Explore how model performance varies by market tier (big/medium/small).
Evaluate generalization by testing tier-trained neural networks on other tiers.
Identify which features most strongly affect price prediction.
Produce interpretable visual and quantitative comparisons across markets.

## 📂 Instructions to Run the Code
1. Clone the repository
git clone <your-repo-url>
cd assignment4-bakeoff

2. Open the Google Colab notebook

All code is inside:
Assignment 4.ipynb

3. Upload Inside Airbnb datasets

Upload the 12 files into the Colab file system.
Each file should be named as:

asheville.csv
austin.csv
chicago.csv
columbus.csv
denver.csv
los_angeles.csv
new_york.csv
portland.csv
salem.csv
san_francisco.csv
santa_cruz.csv
seattle.csv

4. Install required libraries (if needed)
!pip install xgboost shap

5. Run the notebook sequentially

The notebook is fully reproducible and includes:

✔ Preprocessing
✔ Feature engineering
✔ Model training
✔ City-level evaluation
✔ Tier-level evaluation
✔ Cross-tier generalization
✔ Visualizations

<img width="1287" height="1041" alt="image" src="https://github.com/user-attachments/assets/eb099e3c-1dbf-4e39-9d33-863917f43d66" />



## 🧪 Major Findings: XGBoost vs Neural Networks
### ⭐ 1. Deep Neural Network (4-layer MLP) is the best model

Achieved extremely low RMSE values (~10–80) across all cities.
Learned complex interactions between features (amenities, review scores, bedrooms, etc.).
Outperformed XGBoost in every city and every tier.

### ⭐ 2. XGBoost performed well but not competitively

RMSE typically between 150–600 depending on city.
Strongest performance in high-volume markets (NYC, LA).
Still significantly worse than the Deep NN.

### ⭐ 3. Small Neural Network performed poorly

RMSE often exceeded 1000–4000.
Demonstrated severe underfitting.
Confirms the necessity of depth in neural networks for tabular Airbnb data.

### ⭐ 4. Tier-Level Analysis
Tier	Best Model	Notes
Big Cities	Deep NN	Handles large, diverse markets well
Medium Cities	Deep NN	Most stable generalization across markets
Small Cities	Deep NN	Still superior even with fewer samples
### ⭐ 5. Cross-Tier Generalization

Big-tier NN → Medium/Small: Excellent generalization
Medium-tier NN → Big: Surprisingly strong
Small-tier NN → Big: Worst generalization (small datasets = less diverse patterns)

This mirrors real-world ML dynamics: larger, diverse datasets produce the best generalization.

## 📈 Visualizations Included

The notebook includes:
RMSE per city (XGBoost vs NN-small vs NN-deep)
RMSE per tier
Cross-tier RMSE heatmap
Cross-tier R² heatmap
Model comparison bar charts

## 🧠 Conclusion

This comparative study demonstrates that:
Deep Neural Networks are the most accurate and robust choice for Airbnb price prediction across markets.
XGBoost remains a strong but inferior alternative for this task.
Shallow neural networks are insufficient, confirming the role of depth in modeling complex tabular interactions.
Market tier matters, and models trained on large cities generalize best.
This assignment showcases end-to-end machine learning skills, including preprocessing, feature engineering, multi-model evaluation, and cross-market insights.
