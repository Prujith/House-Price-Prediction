# House Price Prediction in Ames, Iowa

This repository contains the code and resources for a machine learning model to predict house prices in Ames, Iowa. The model is based on the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), where participants are challenged to predict house prices based on various features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model](#model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)

## Overview

This project aims to predict house prices in Ames, Iowa using machine learning techniques. The dataset provides various features such as the number of bedrooms, square footage, and neighborhood, among others, to help us make accurate predictions. The goal is to build a model that can generalize well on unseen data and accurately predict house prices.

## Dataset

The dataset used for this project is sourced from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques). It contains both training and testing data. The training set includes 1,460 samples with 79 features each, and the testing set has 1,459 samples. Some of the features include:

- OverallQual: Overall material and finish quality of the house
- GrLivArea: Above ground living area in square feet
- Neighborhood: Physical locations within Ames city limits

For detailed information about all features, please refer to the data description on the Kaggle competition page.

## Requirements

To run the code in this repository, you need the following dependencies:

- Python 3.7 or higher
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using the following command:

## Usage

You can use the provided Jupyter Notebook, `House_Price_Prediction.ipynb`, to train the model and make predictions. The notebook contains detailed explanations and code comments to guide you through the process. Simply open the notebook and follow the instructions provided.

## Model

For this project, we experimented with various regression models, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- KNeighborsRegressor
- Lasso
- Ridge
- AdaBoost Regressor

The best-performing model was selected based on its performance on the validation set.

## Evaluation

The evaluation metric used for this project is the Root Mean Squared Error (RMSE), Mean Absolute Error, and R2 Score. We split the training data into a training set and a validation set to assess the model's performance before making predictions on the test set. Data preprocessing steps included handling missing values, feature engineering, and normalization.

## Results

Our best-performing model achieved an accuracy of 83.28 (R2 score) on the validation set, indicating a good fit to the data. The detailed analysis can be seen in the [House_Price_Prediction.ipynb](House_Price_Prediction.ipynb) notebook.

## Contributing

We welcome contributions to this project. If you find any issues or want to suggest improvements, please feel free to open an issue or submit a pull request.


---
Feel free to customize this template further based on your specific findings and additional details about your project. Happy coding!
