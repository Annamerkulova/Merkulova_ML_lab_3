**Course:** MOP 3231 — Machine Learning (Python)  
**Student:** Anna Merkulova  
**Student ID last four digits:** 2305

## 1. Dataset

The dataset contains information about 398 cars from model years 1970–1982. The target variable is `fuel_efficiency_km_per_l`.

The dataset includes numerical features such as:

- year
- cylinders
- engine_litres
- power_hp
- mass_kg
- accel_0_100_s

It also contains categorical variables `model` and `region`.

There were 6 missing values in `power_hp`. These rows were removed before fitting the models, leaving 392 observations. This represents approximately 1.51% of the original dataset.

## 2. Step 3 — Simple Linear Regression

The numerical feature most strongly correlated with fuel efficiency was `mass_kg`.

The correlation between `mass_kg` and fuel efficiency was:

**-0.8322**

A simple linear regression model was fitted using `mass_kg` as the predictor.

The resulting equation was:

**fuel efficiency = 19.6470 − 0.0072 × mass**

The intercept was **19.6470**, and the slope was **-0.0072**.

For a car with a mass of 1500 kg, the predicted fuel efficiency was approximately **8.90 km/L**.

The scatter plot and regression line are included in the repository.

## 3. Step 4 — Model Evaluation

The dataset was divided into training and test sets using an 80/20 split with `random_state=2305`.

### Training results

| Metric | Value |
|---|---:|
| MAE | 1.3792 |
| MSE | 3.3581 |
| RMSE | 1.8325 |
| R² | 0.6820 |

### Test results

| Metric | Value |
|---|---:|
| MAE | 1.3949 |
| MSE | 3.5682 |
| RMSE | 1.8890 |
| R² | 0.7022 |

### 5-Fold Cross-Validation

The mean test R² was **0.6782**, with a standard deviation of **0.0175**.

Training performance alone is not sufficient because a model can perform well on the data used for training but perform worse on unseen data. Test evaluation and cross-validation provide a better estimate of how well the model generalizes.

## 4. Step 5 — Multiple Linear Regression

Features were added one group at a time, starting with the feature with the strongest correlation.

| Features | Mean Test R² | Std |
|---|---:|---:|
| mass_kg | 0.6782 | 0.0175 |
| mass_kg, engine_litres | 0.6836 | 0.0181 |
| mass_kg, engine_litres, power_hp | 0.6900 | 0.0131 |
| mass_kg, engine_litres, power_hp, cylinders | 0.6890 | 0.0142 |
| mass_kg, engine_litres, power_hp, cylinders, year | 0.7944 | 0.0380 |
| mass_kg, engine_litres, power_hp, cylinders, year, accel_0_100_s | 0.7927 | 0.0356 |

The R² value did not improve after adding every feature. Adding `cylinders` slightly decreased the mean R² from 0.6900 to 0.6890. Adding `accel_0_100_s` also slightly decreased it from 0.7944 to 0.7927.

This shows that adding more features does not necessarily improve model performance. Some features may provide limited additional information or overlap with information already captured by other variables.

## 5. Step 6 — Polynomial Regression

Polynomial regression models with degrees from 1 to 5 were evaluated.

| Degree | Train R² | Test R² (5-fold CV) |
|---:|---:|---:|
| 1 | 0.6820 | 0.6782 |
| 2 | 0.7037 | 0.7027 |
| 3 | 0.7037 | 0.6997 |
| 4 | 0.7046 | 0.6973 |
| 5 | 0.7071 | 0.6965 |

The highest test R² was obtained with **degree 2**, with a value of **0.7027**.

After degree 2, the training R² continued to increase, while the test R² decreased. This indicates that higher-degree polynomial models begin to overfit the training data. Increasing the model complexity beyond degree 2 therefore did not improve generalization.

## 6. Step 7 — Conclusion

Among the models evaluated in Steps 3–6, I would use the multiple linear regression model with the features `mass_kg`, `engine_litres`, `power_hp`, `cylinders`, and `year` for predicting the fuel efficiency of a new car. This model achieved the highest mean test R² in the experiments, with a 5-fold cross-validation R² of 0.7944. This means that the model explained a larger proportion of the variation in fuel efficiency than the simple linear and polynomial models tested in this laboratory.

The simple linear regression using `mass_kg` was also useful because mass had the strongest correlation with fuel efficiency (-0.8322). However, using additional numerical features improved the cross-validation performance substantially. In contrast, polynomial regression did not provide the same improvement. The degree-2 model achieved a test R² of 0.7027, but higher degrees produced lower test R² values even though their training R² values continued to increase. This suggests that higher-degree models started to overfit the training data.
