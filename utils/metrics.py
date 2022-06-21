import numpy as np
from sklearn.metrics import recall_score, mean_squared_error, mean_absolute_error


class EvalMetrics:
    def CCC(y_true, y_pred):
        x_mean = np.nanmean(y_true, dtype="float32")
        y_mean = np.nanmean(y_pred, dtype="float32")
        x_var = 1.0 / (len(y_true) - 1) * np.nansum((y_true - x_mean) ** 2)
        y_var = 1.0 / (len(y_pred) - 1) * np.nansum((y_pred - y_mean) ** 2)
        # x_var = np.nansum((y_true - x_mean) ** 2)
        # y_var = np.nansum((y_pred - y_mean) ** 2)
        cov = np.nanmean((y_true - x_mean) * (y_pred - y_mean))
        return round((2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)

    def MAE(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)

    def MSE(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def UAR(y_true, y_pred):
        return recall_score(y_true, y_pred, average="macro")
