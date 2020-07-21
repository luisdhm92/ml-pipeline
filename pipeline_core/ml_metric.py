from sklearn.metrics import mean_absolute_error


class MeanAbsoluteError:

    @staticmethod
    def metric(real_labels, predicted_labels):
        """
        Calculate MAE
        :param real_labels:
        :param predicted_labels:
        :return:
        """
        mae = mean_absolute_error(real_labels, predicted_labels)
        return mae

