import joblib


class ModelSerializer:
    """
    Support serialize and deserialize the model
    """
    @staticmethod
    def dump_model(model, name: str, path="../resources/models"):
        """
        Persist the model to be loaded later
        :param model: Machine Learning model
        :param name: Name to save the model
        :return: None
        """
        if model:
            joblib.dump(model, f"{path}/{name}.pkl")

    @staticmethod
    def load_model(name: str):
        """
        Load a previously persisted model
        :param name: Name to retrieve the model
        :return:
        """
        model = joblib.load(f"{name}.pkl")
        if model:
            return model
        else:
            raise Exception('There is no model stored with this name')
