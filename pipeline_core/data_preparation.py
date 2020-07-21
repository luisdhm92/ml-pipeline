from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np


class DataPreparation:
    """
    Pipeline for data preparation (numerical and categorical)
    """

    @staticmethod
    def numeric_pipeline():
        return Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    @staticmethod
    def prepare_numeric_data(dataset):
        """

        :param dataset: data to be preprocessed
        :return: TBD
        """
        # Select numeric data
        dataset_num = dataset.select_dtypes(include=[np.number])

        num_pipeline = DataPreparation.numeric_pipeline()

        datset_num_tr = num_pipeline.fit_transform(dataset_num)
        return datset_num_tr

    @staticmethod
    def full_preparation_pipeline(dataset, cat_attribs=["ocean_proximity"]):
        """

        :param dataset: data to be preprocessed
        :param cat_attribs:
        :return: Prepared data
        """
        dataset_num = dataset.select_dtypes(include=[np.number])
        num_attribs = list(dataset_num)
        num_pipeline = DataPreparation.numeric_pipeline()

        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])

        dataset_prepared = full_pipeline.fit_transform(dataset)
        return dataset_prepared

