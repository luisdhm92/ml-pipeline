from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV


class LinearRegressionModel:

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, dataset, dataset_labels):
        self.model.fit(dataset, dataset_labels)

    def predict(self, data):
        predictions = self.model.predict(data)
        print(f"Predictions: {predictions}")
        return predictions


class RandomForestRegressorModel:

    def __init__(self):
        self.model = RandomForestRegressor(random_state=42)
        self.param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
        self.best_model = None
        self.best_params = None

    def predict(self, X_test_prepared):
        final_model = self.best_model if self.best_model else self.model
        final_predictions = final_model.predict(X_test_prepared)
        return final_predictions

    def fit(self, housing_prepared, housing_labels):
        self.model.fit(housing_prepared, housing_labels)

    def optimize_hyperparameters(self, housing_prepared, housing_labels):
        rnd_search = RandomizedSearchCV(
            self.model, param_distributions=self.param_distribs, n_iter=10,
            cv=5, scoring='neg_mean_squared_error', random_state=42
        )
        rnd_search.fit(housing_prepared, housing_labels)
        print(rnd_search.best_params_)
        print(rnd_search.best_estimator_)
        self.best_params = rnd_search.best_params_
        self.best_model = rnd_search.best_estimator_
