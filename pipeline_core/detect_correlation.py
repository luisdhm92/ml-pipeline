class CorrelationDetection:

    @staticmethod
    def correlation(dataset, field):
        """
        Calculate correlation related to field parameter
        :param dataset: Target Pandas dataset
        :param field: Target field
        :return:
        """
        corr_matrix = dataset.corr()
        return corr_matrix[field].sort_values(ascending=False)
