import matplotlib
matplotlib.use("Agg")

import pandas as pd

df = pd.read_csv("data/cars_fuel_efficiency.csv")

print("Shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nBasic statistics:")
print(df.describe())

rows_before = len(df)

df = df.dropna(subset=["power_hp"])

rows_after = len(df)

print("\nAfter removing missing power_hp:")
print("Rows before:", rows_before)
print("Rows after:", rows_after)
print("Rows dropped:", rows_before - rows_after)

percentage_dropped = (rows_before - rows_after) / rows_before * 100
print(f"Percentage dropped: {percentage_dropped:.2f}%")

target = "fuel_efficiency_km_per_l"

numeric_df = df.select_dtypes(include="number")

correlations = numeric_df.corr()[target].drop(target)

correlations = correlations.sort_values(key=abs, ascending=False)

print("\nCorrelations with fuel efficiency:")
print(correlations)

import matplotlib.pyplot as plt

best_feature = "mass_kg"
target = "fuel_efficiency_km_per_l"

plt.figure(figsize=(8, 5))

plt.scatter(df[best_feature], df[target])

plt.xlabel("Mass (kg)")
plt.ylabel("Fuel efficiency (km/L)")
plt.title("Fuel Efficiency vs Mass")

plt.savefig("scatter_mass_fuel_efficiency.png", dpi=300)
plt.close()

from sklearn.linear_model import LinearRegression

X = df[["mass_kg"]]
y = df["fuel_efficiency_km_per_l"]

model = LinearRegression()

model.fit(X, y)

theta_0 = model.intercept_
theta_1 = model.coef_[0]

print("\nLinear Regression results:")
print("Intercept (theta_0):", theta_0)
print("Slope (theta_1):", theta_1)

print(f"Equation: y = {theta_0:.4f} + ({theta_1:.4f}) * x")

import matplotlib.pyplot as plt

y_pred = model.predict(X)

plt.figure(figsize=(8, 5))

plt.scatter(X, y, label="Actual data")
plt.plot(X, y_pred, label="Regression line")

plt.xlabel("Mass (kg)")
plt.ylabel("Fuel efficiency (km/L)")
plt.title("Simple Linear Regression")
plt.legend()

plt.savefig("linear_regression_mass.png", dpi=300)
plt.close()

chosen_mass = 1500

prediction = model.predict([[chosen_mass]])

print("\nPrediction:")
print(f"For a car with mass {chosen_mass} kg:")
print(f"Predicted fuel efficiency: {prediction[0]:.2f} km/L")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2305
)

model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, y_test_pred)

print("\nTraining results:")
print(f"MAE:  {train_mae:.4f}")
print(f"MSE:  {train_mse:.4f}")
print(f"RMSE: {train_rmse:.4f}")
print(f"R²:   {train_r2:.4f}")

print("\nTest results:")
print(f"MAE:  {test_mae:.4f}")
print(f"MSE:  {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")

from sklearn.model_selection import KFold, cross_val_score

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=2305
)

cv_scores = cross_val_score(
    LinearRegression(),
    X,
    y,
    cv=kf,
    scoring="r2"
)

print("\n5-Fold Cross-Validation:")
print("R² for each fold:", cv_scores)
print(f"Mean R²: {cv_scores.mean():.4f}")
print(f"Standard deviation: {cv_scores.std():.4f}")

numeric_features = [
    "year",
    "cylinders",
    "engine_litres",
    "power_hp",
    "mass_kg",
    "accel_0_100_s"
]


feature_order = (
    df[numeric_features + [target]]
    .corr()[target]
    .drop(target)
    .abs()
    .sort_values(ascending=False)
    .index
    .tolist()
)

print("\nFeature order:")
print(feature_order)

results = []

for i in range(1, len(feature_order) + 1):

    features = feature_order[:i]

    X_multiple = df[features]

    scores = cross_val_score(
        LinearRegression(),
        X_multiple,
        y,
        cv=kf,
        scoring="r2"
    )

    results.append({
        "Features used": ", ".join(features),
        "Mean test R²": scores.mean(),
        "Std": scores.std()
    })

multiple_results = pd.DataFrame(results)

print(multiple_results.to_string(index=False))

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

polynomial_results = []

for degree in range(1, 6):

    pipe = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(
            degree=degree,
            include_bias=False
        ),
        LinearRegression()
    )

    pipe.fit(X_train, y_train)

    train_pred = pipe.predict(X_train)
    train_r2 = r2_score(y_train, train_pred)

    cv_scores = cross_val_score(
        pipe,
        X,
        y,
        cv=kf,
        scoring="r2"
    )

    test_r2 = cv_scores.mean()

    polynomial_results.append({
        "Degree": degree,
        "Train R²": train_r2,
        "Test R² (5-fold CV)": test_r2
    })

polynomial_results = pd.DataFrame(polynomial_results)

print(polynomial_results.to_string(index=False))