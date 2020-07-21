import pandas as pd
import os


class DataLoader:

    @staticmethod
    def load_dataset_data(base_path: str, dataset: str):
        """
        Load data set from a file
        :param dataset: Name of the dataset
        :return: Pandas dataframe with the loaded data
        """
        pass

    @staticmethod
    def describe_dataset(dataset):
        """
        Describe the Pandas dataframe
        :param dataset: Pandas dataframe
        :return: None
        """
        if dataset:
            dataset.info()
            dataset.describe()
        else:
            raise Exception('You should provide a dataset')


class CSVDataLoader(DataLoader):

    @staticmethod
    def load_dataset_data(base_path: str, dataset: str):
        # todo: add support for more data source if it's necessary
        csv_path = os.path.join(base_path, f"{dataset}.csv")
        return pd.read_csv(csv_path)


class FakeDataLoader(DataLoader):

    @staticmethod
    def load_dataset_data(base_path="", dataset=""):
        csv_path = os.path.join(base_path, f"{dataset}.csv")
        return pd.read_csv(csv_path)
