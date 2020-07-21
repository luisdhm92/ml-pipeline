from pipeline_core.data_loader import FakeDataLoader
from pipeline_core.data_preparation import DataPreparation
from pipeline_core.dataset_splitter import DatasetSplitter
from pipeline_core.detect_correlation import CorrelationDetection
from pipeline_core.ml_metric import MeanAbsoluteError
from pipeline_core.ml_model import RandomForestRegressorModel
from pipeline_core.model_serializer import ModelSerializer


class MLPipeline:

    def __init__(self):
        self.data_loader = FakeDataLoader()
        self.data_preparation = DataPreparation()
        self.dataset_splitter = DatasetSplitter()
        self.correlation_detection = CorrelationDetection()
        self.ml_model = RandomForestRegressorModel()
        self.ml_metric = MeanAbsoluteError()

    def pipeline(self, target_label="median_house_value", cat_attribs=["ocean_proximity"]):
        dataset = self.data_loader.load_dataset_data()
        print(dataset.head())

        train_set, test_set = self.dataset_splitter.train_test_split(dataset)
        dataset = train_set.copy()
        correlation = self.correlation_detection.correlation(dataset, target_label)
        print(f"Correlation \n{correlation}")

        dataset = train_set.drop(target_label, axis=1)  # drop labels for training set
        dataset_labels = train_set[target_label].copy()

        dataset_prepared = self.data_preparation.full_preparation_pipeline(dataset, cat_attribs=cat_attribs)
        print(dataset_prepared.shape)

        self.ml_model.optimize_hyperparameters(dataset_prepared, dataset_labels)

        print(self.ml_model.best_params)
        print(self.ml_model.best_model)

        X_test = test_set.drop(target_label, axis=1)
        y_test = test_set[target_label].copy()

        X_test_prepared = DataPreparation.full_preparation_pipeline(X_test, cat_attribs=cat_attribs)
        final_predictions = self.ml_model.predict(X_test_prepared)

        print(f"Final predictions: {final_predictions}")

        mae = MeanAbsoluteError.metric(y_test, final_predictions)
        print(f"MAE: {mae}")

        ModelSerializer.dump_model(self.ml_model.best_model, "best_random_forest")


if __name__ == '__main__':
    pipeline = MLPipeline()
    pipeline.pipeline()

