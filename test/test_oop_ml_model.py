import unittest

from pipeline_core.data_loader import FakeDataLoader
from pipeline_core.data_preparation import DataPreparation
from pipeline_core.dataset_splitter import DatasetSplitter
from pipeline_core.detect_correlation import CorrelationDetection
from pipeline_core.ml_metric import MeanAbsoluteError
from pipeline_core.ml_model import RandomForestRegressorModel
from pipeline_core.model_serializer import ModelSerializer


class TestOopMLModel(unittest.TestCase):

    def test_oop_ml_model(self):
        dataset = FakeDataLoader.load_dataset_data(base_path="../resources/datasets/", dataset="housing")
        print(dataset.head())

        train_set, test_set = DatasetSplitter.train_test_split(dataset)
        dataset = train_set.copy()
        correlation = CorrelationDetection.correlation(dataset, "median_house_value")
        print(f"Correlation \n{correlation}")

        dataset = train_set.drop("median_house_value", axis=1)  # drop labels for training set
        housing_labels = train_set["median_house_value"].copy()

        dataset_prepared = DataPreparation.full_preparation_pipeline(dataset, cat_attribs=["ocean_proximity"])
        print(dataset_prepared.shape)

        model = RandomForestRegressorModel()
        model.optimize_hyperparameters(dataset_prepared, housing_labels)

        print(model.best_params)
        print(model.best_model)

        X_test = test_set.drop("median_house_value", axis=1)
        y_test = test_set["median_house_value"].copy()

        X_test_prepared = DataPreparation.full_preparation_pipeline(X_test, cat_attribs=["ocean_proximity"])
        final_predictions = model.predict(X_test_prepared)

        print(f"Final predictions: {final_predictions}")

        mae = MeanAbsoluteError.metric(y_test, final_predictions)
        print(f"MAE: {mae}")

        ModelSerializer.dump_model(model.best_model, "best_random_forest")

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
