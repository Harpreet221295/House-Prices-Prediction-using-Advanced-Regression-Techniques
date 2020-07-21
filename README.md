# House-Prices-Prediction-using-Advanced-Regression-Techniques

Link to the competition and Dataset - [https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

In this task, we are given a dataset of 79 features describing every aspect of houses in Ames, Iowa with their prices. We need to predict the prices of the house in the test data. The performance metric used is root mean squared error(rmse).

Detailed exploratory data analysis has been carried out. Missing values in both train and test datasets are being filled for each variable. Features were observed by dividing them into categorical and numerical. Correlation among numerical features and numerical features and SalePrice is observed.

Feature Engineering - 
1. New numerical features are created from the existing ones using algebraic operations - House_Age, Renewal_Age, Garage_Age, IsRemodAdd, TotalProchArea, TotalBathrooms, TotalArea etc
2. New indicator features are created from categorical features - hasGarage, hasPool, hasFence, hasMisc, hasFireplace, hasBasement etc.
3. Overfitted categorical variables such as PoolQC, Utilities and Street are removed.
4. Skewness of SalePrice(Target Variable) is corrected using lop1p transformation and skewness of other numerical features are orrected using boxcox1p transformation.
5. Ordinal Features such as Qual(Quality), Cond(Condition) variables, BsmtFinType1 and BsmtFinType2 are label encoded.
6. Year Variables(YrSOld, MoSold) and MSSubClass are converted to categorical since their numerical magnitude do not signify anything and they are just different categories.

Model Fitting

The following models are independently trained on the training data - 
1) Ridge Regression
2) Lasso Regression
3) ElasticNet Regression
4) XGBoost Regressor
5) Light Gradient Boosting Machine(LGBM) Regressor

The hyperparameter tuning is perormed using Randomized Search cross-validation over wide ranges of parameter values.

Blending Model

After hyperparameter tuning, we find the best parameter set for each model and train the corresponding best verios of each model using entire training dataset.

Let s1, s2, s3, s4, s5 corresponds to cross-validation error score of Ridge, Lasso, ElasticNet,XGBoost and LGBM.
Let Total = 1/s1 + 1/s2 + 1/s3 + 1/s4 + 1/s5
Let X_test denotes the test data
Prediction from blend model is Blend(X_test) = (1/s1)/Total * Ridge(X_test) + (1/s2)/Total * Lasso(X_test) + (1/s3)/Total * ElasticNet(X_test) + (1/s4)/Total * XGBoost(X_test) + (1/s5)/Total * LGBM(X_test)


