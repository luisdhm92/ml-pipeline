from sklearn.model_selection import train_test_split


class DatasetSplitter:

    @staticmethod
    def train_test_split(dataset, test_size=0.2, random_state=42):
        """
        Split the dataset for training and test
        :param dataset: Target dataset
        :param test_size: portion of the dataset to be used for testing
        :param random_state: needed parameter for the sklearn method (define formally the parameter)
        :return:
        """
        train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
        return train_set, test_set
