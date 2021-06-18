class BetaCov:
    """
    Returns covariance matrix of returns fitted with market model
    and characteristics of market model regressions
    """

    def __init__(self, data: pd.DataFrame, index):
        """
        data: asset returns
        index: index to estimate market model
        constant: whether to include constant in market model equation
        """
        self.data = data
        self.index = index
        # сохраним оценки доходностей, полученные из market model
        self.fitted_data = pd.DataFrame(data=None,
                                        columns=self.data.columns)

        self.betas = pd.Series(data=None,
                               index=self.data.columns)
        self.R2 = self.betas.copy()
        self.RMSE = self.betas.copy()
        self.vcov = None

    def fit(self, agg_period=None, window=None, method='mean', ar_method='mean',
            parallel=False, constant=False, **ar_kwargs):
        """
        Fits betas for each asset and averages them or builds autoregressive model for them
        method: how to work with betas
              - mean: use mean of estimated betas
              - median: use median of estimated betas
              - ar: use autoregression to predict beta for each asset
              - ar_method:
                  * mean to calculate mean beta from betas fitted with AR model
                  * forecast to forecast beta one period ahead using AR model
        """

        self.parallel = parallel
        self.constant = constant

        # если мы усредняем бету
        if agg_period:

            # изменим тип атррибута на DataFrame
            self.betas = pd.DataFrame(data=None,
                                      columns=self.data.columns)
            self.R2 = self.betas.copy()
            self.RMSE = self.betas.copy()
            # сохраним параметры усреднения
            self.agg_period = agg_period
            self.window = window
            self.method = method
            self.average_betas, self.median_betas = dict(), dict()

            # оценим среднюю бету для каждого актива
            for ticker in self.data.columns:

                beta = AverageBeta(asset=self.data[ticker],
                                   index=self.index,
                                   constant=self.constant,
                                   agg_period=self.agg_period,
                                   window=self.window,
                                   parallel=self.parallel)
                beta.fit()
                self.betas[ticker] = pd.Series(beta.betas)
                self.R2[ticker] = pd.Series(beta.R2s)
                self.RMSE[ticker] = pd.Series(beta.errors)
                self.average_betas[ticker], self.median_betas[ticker] = beta.average_beta, beta.median_beta

            # оценим ковариационную матрицу
            # Оценим авторегрессию по оценкам бет и сделаем прогноз
            if self.method = 'ar':

                # будем сохранять прогнозы и оценки бет в новый атрибут класса
                self.ar_forecasts = pd.Series(data=None,
                                              index=self.data.columns)
                self.ar_params = self.ar_forecasts.copy()
                self.fitted_betas =  self.ar_forecasts.copy()

                for ticker in self.data.columns:
                    train = self.betas[ticker]
                    ts_model = AutoReg(train, **ar_kwargs)
                    res = ts_model.fit()
                    print(res.summary())

                    # сохраним разные параметры и метрики модели
                    self.fitted_betas[ticker] = res.fittedvalues
                    params = ['aic', 'bic', 'llf',
                              'constant', 'AR_coef',
                              'constant_pvalue', 'AR_pvalue']
                    self.ar_params[ticker] = pd.Series(data=[res.aic, res.bic, res.llf,
                                                             res.params[0], res.params[1],
                                                             res.pvalues[0], res.pvalues[1]],
                                                       index=params)

                # оценим ковариационную матрицу
                if ar_method == 'forecast':
                    # прогноз на один шаг вперёд
                    forecast= res.predict(start=train.shape[0],
                                          end=train.shape[0]).values[0]
                    print(forecast)
                    self.ar_forecasts[ticker]= forecast
                    self.fitted_data = self.index * self.forecast
                    self.vcov = self.fitted_data.cov()

                
                elif ar_method == 'mean':
                    # усредним оценки бет из AR-модели и по ним оценим доходности 
                    self.fitted_data = self.index * self.fitted_betas.mean()
                    self.vcov = self.fitted_data.cov()





            # берём среднюю бету из всех оценённых бет для каждого актива
            elif self.method == 'mean':
                self.fitted_data = self.index * pd.Series(self.average_betas)
                self.vcov = self.fitted_data.cov()
            # берём медианную бету из всех оценённых бет для каждого актива
            elif self.method == 'median':
                self.fitted_data = self.index * pd.Series(self.median_betas)
                self.vcov = self.fitted_data.cov()

        else:
            # если мы не усредняем бету для каждого актива
            # то тогда просто считаем одну бету по всей выборке для каждого актива
            # и строим ковариационную матрицу по оценкам доходности с помощью этих бет
            for ticker in self.data.columns:
                beta = Beta(asset=self.data[ticker],
                            index=self.index,
                            constant=self.constant)
                beta.fit()
                self.fitted_data[ticker] = beta.fitted_values
                # сохраним характеристики уравнения
                self.betas[ticker], self.R2[ticker], self.RMSE[ticker] = beta.beta, beta.R2, beta.RMSE

            # оценим ковариационную матрицу
            self.vcov = self.fitted_data.cov()