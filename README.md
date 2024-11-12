# 🔥 Energy Load Prediction Using Hybrid Machine Learning Model

This project leverages a **hybrid machine learning approach** to predict heating and cooling loads in buildings. By combining Multiple Linear Regression (MLR), K-Nearest Neighbors (KNN), and a meta-model based on Gradient Boosting Regressor (GBR), we achieve a robust prediction model. The project includes a Jupyter Notebook (`HYBRIDML.ipynb`) and the dataset (`Heating_coolin_load_dataset.csv`).

## 📋 Table of Contents
- [Introduction](#introduction)
- [Models Employed](#models-employed)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Acknowledgment](#acknowledgment)
- [References](#references)

## 🔍 Introduction
The **heating and cooling loads** of buildings are crucial indicators for efficient energy management. Accurate prediction helps in optimizing HVAC systems, reducing costs, and minimizing environmental impact. This project implements a **stacking ensemble model** that captures both linear and non-linear dependencies in the data.

## ⚙️ Models Employed
The project utilizes a hybrid stacking approach with the following models:

1. **Multiple Linear Regression (MLR)**
   - A simple baseline model that captures the linear relationships between the features and target variables.
   - It provides insights into the linear dependency of building attributes on energy loads.

2. **K-Nearest Neighbors (KNN)**
   - A non-parametric, instance-based model that predicts based on the values of the nearest neighbors.
   - Effective in capturing localized patterns in the data, especially useful for complex, non-linear relationships.

3. **Gradient Boosting Regressor (GBR)**
   - Acts as the meta-model in the stacking ensemble.
   - Corrects errors made by the base models (MLR and KNN) by focusing on the residuals.
   - Iteratively builds weak learners (decision trees) to refine predictions and minimize overall error.

## 📊 Dataset
The dataset used (`energy_load_data.csv`) contains features related to building characteristics and energy loads:

- **Features**: Building orientation, glazing area, wall area, roof area, and more.
- **Target Variables**: Heating load and cooling load.

The dataset provides a variety of building configurations, making it suitable for modeling different energy usage patterns.

## 🛠️ How It Works
The hybrid model follows these steps:

1. **Base Models Training**: 
   - MLR and KNN are trained on the training data.
   - These models make predictions, which are then used to create meta-features.

2. **Meta-Model Training**:
   - The predictions from MLR and KNN serve as input for the Gradient Boosting Regressor (GBR).
   - GBR is trained on these meta-features to correct errors and make the final predictions.

This stacking approach ensures that the model benefits from both linear (MLR) and non-linear (KNN, GBR) predictive capabilities.

## 📈 Evaluation Metrics
The performance of the model is evaluated using the following metrics:

1. **Mean Squared Error (MSE)**:
   \[
   \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \]
   - Measures the average squared difference between predicted and actual values.

2. **Root Mean Squared Error (RMSE)**:
   \[
   \text{RMSE} = \sqrt{\text{MSE}}
   \]
   - Provides the magnitude of prediction errors in the same units as the target variable.

3. **R-Squared (R²) Score**:
   \[
   R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
   \]
   - Indicates the proportion of variance in the target variable explained by the model.

## 📊 Results
The hybrid model demonstrates superior predictive performance compared to individual models. The meta-model effectively combines the strengths of MLR and KNN, achieving a higher R² score and lower RMSE:

| Metric       | MLR   | KNN   | Hybrid Model (GBR) |
|--------------|-------|-------|--------------------|
| MSE (Heating)| 20.15 | 18.09 | 15.02              |
| MSE (Cooling)| 45.00 | 19.88 | 16.31              |
| RMSE         | 4.49  | 4.24  | 3.87               |
| R² Score     | 0.78  | 0.81  | 0.87               |

## 🤝 Acknowledgment
The authors express their gratitude to **Indian Institute of Information Technology, Pune** for providing the necessary resources and support for this research. Special thanks to **Dr. Shrikant Salve Sir** for his invaluable guidance and mentorship throughout the project.

## 📚 References
1. Elhabyb, K., et al. "Machine Learning Algorithms for Predicting Energy Consumption in Educational Buildings," *International Journal of Energy Research*, 2024.
2. Sajjad, M., et al. "Towards Efficient Building Designing: Heating and Cooling Load Prediction via Multi-Output Model," *Sensors*, 2020.
3. Ma, Y., & Qiao, E. "Research on Accurate Prediction of Operating Energy Consumption of Green Buildings," *2021 IEEE International Conference on Industrial Application of Artificial Intelligence (IAAI)*, 2021.
4. Kumar, K. B. A., et al. "Energy Consumption in Smart Buildings using Machine Learning," *2023 7th International Conference on Intelligent Computing and Control Systems (ICICCS)*, 2023.

