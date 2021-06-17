import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class Beta:
    def __init__(self, asset, index, constant=False):
        """
        constant: whether to include constant in market model equation
        """
        self.asset = asset
        self.index = index
        self.constant = constant
        self.alpha = None
        self.beta = None
        self.coef = None
        self.fitted_values = None

    def fit(self):

        X = self.index.values.reshape(-1, 1)
        # если хотим добавить константу в уравнение, добавим единичный столбец к X
        if self.constant:
            X = np.column_stack([np.ones(shape=(self.index.shape[0], 1)),
                                 self.index.values.reshape(-1, 1)])

        model = LinearRegression()
        model.fit(X, self.asset.values)
        
        # сохраним метрики качества и коэффиценты
        self.coef = model.coef_
        self.R2 = r2_score(self.asset.values, model.predict(X))
        self.RMSE = mean_squared_error(self.asset.values, model.predict(X))**(1/2)

        # сохраним в отдельные атрибуты коэффициенты
        if self.constant:
            self.alpha = model.coef_.ravel()[0]
            self.beta = model.coef_.ravel()[1]
        self.beta = model.coef_.ravel()[0]

        # сохраним оценки y_hat
        self.fitted_values = pd.Series(data=model.predict(X),
                                       index=self.asset.index,
                                       name=self.asset.name)

    # на всякий случай, вдруг понадобится именно прогноз
    def predict(self, X):
        return (testmodel.coef_@X.T).T.rename(self.asset.name)

        
def ContainerFun(asset, index, constant):
    """Wrapper for parallel fitting of Beta Class"""
    obj = Beta(asset, index, constant)
    obj.fit()
    return obj