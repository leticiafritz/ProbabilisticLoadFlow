import statsmodels.api as sm
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

from pandas.tseries.holiday import AbstractHolidayCalendar as calendar
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error


def season_calc(month):
    if month in [4, 5, 6]:
        return "fall"
    elif month in [7, 8, 9]:
        return "winter"
    elif month in [10, 11, 12]:
        return "spring"
    else:
        return "summer"


def create_features(df):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    mapped = {True: 1, False: 0}
    df['Date'] = pd.to_datetime(df.date.dt.date)
    df['year'] = df.date.dt.year
    df['month'] = df.date.dt.month
    df['day'] = df.date.dt.dayofyear
    df['hour'] = df.date.dt.hour
    df['weekday'] = df.date.dt.weekday.map(weekdays)
    df['season'] = df.date.dt.month.apply(season_calc)
    cal = calendar()
    holidays = cal.holidays(start=df['Date'].min(), end=df['Date'].max())
    df['holiday'] = df['Date'].isin(holidays)
    df.holiday = df.holiday.map(mapped)

    return df


class Pldpredict:
    def __init__(self, pld):
        self.pld = pd.DataFrame(pld)

    def get_first_data_analise(self):
        res = sm.tsa.seasonal_decompose(self.pld['tariff'], freq=365)
        fig = res.plot()
        fig.set_figheight(8)
        fig.set_figwidth(15)
        plt.show()

    def get_p_value(self):
        res = sm.tsa.adfuller(self.pld['tariff'].dropna(), regression='ct')
        print('p-value:{}'.format(res[1]))
        res = sm.tsa.adfuller(self.pld['tariff'].diff().dropna(), regression='c')
        print('p-value:{}'.format(res[1]))

    def get_data_organize(self):
        self.pld = self.pld.reset_index()
        self.pld.index = pd.period_range('2019-01-01', periods=len(self.pld), freq='H')
        self.pld = create_features(self.pld)
        tariff_data = self.pld['tariff']
        tariff_exog = self.pld[['month', 'day', 'hour', 'weekday']]
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Monday", 2)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Tuesday", 3)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Wednesday", 4)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Thursday", 5)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Friday", 6)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Saturday", 7)
        tariff_exog['weekday'] = tariff_exog['weekday'].replace("Sunday", 1)

        return tariff_data, tariff_exog

    def get_day_ahead(self):
        # Primeiras análises
        self.get_first_data_analise()
        self.get_p_value()

        # Organização dos dados
        tariff_data, tariff_exog = self.get_data_organize()

        # Tirando o ultimo dia para previsão
        tariff_test = pd.DataFrame()
        tariff_train = pd.DataFrame()
        tariff_test['origial'] = tariff_data.iloc[(tariff_data.size - 24):, ]
        tariff_exog_test = tariff_exog.iloc[(tariff_data.size - 24):, ]
        tariff_train['tariff'] = tariff_data.iloc[:(tariff_data.size - 24), ]
        tariff_exog_train = tariff_exog.iloc[:(tariff_data.size - 24), ]

        # SARIMAX
        # order=(2, 0, 1), seasonal_order=(1, 0, 1, 24)
        model_opt = SARIMAX(tariff_train, order=(2, 0, 2), seasonal_order=(1, 0, 1, 24), trend='c')
        results = model_opt.fit()
        print(results.summary())
        residuals = results.resid
        residuals.plot()
        plt.show()

        # Previsão
        yhat = results.predict(start=len(tariff_train) - 1, end=len(tariff_train) + 23)[1:]
        tariff_predict = pd.DataFrame(yhat, columns=["predicted"])
        tariff_test['origial'].plot(x=tariff_test.index, y=tariff_test['origial'], legend='original')
        tariff_predict['predicted'].plot(x=tariff_predict.index, y=tariff_predict['predicted'],
                                         legend='predict', color='red')
        plt.xlabel('Date')
        plt.ylabel(r'PLD Sombra [R$/kWh]')
        plt.show()

        # Calculo do MAE
        print('SARIMA model MSE:{}'.format(mean_squared_error(tariff_test['origial'],
                                                              tariff_predict['predicted'])))

        tariff_pred_array = tariff_predict['predicted'].to_numpy()

        return tariff_pred_array

    def get_pld_dummy(self):
        tariff_pred_array = np.array([0.33384969, 0.33047564, 0.32759049, 0.32557059, 0.32449571,
                                      0.31842698, 0.31681974, 0.32364556, 0.33306611, 0.33919796,
                                      0.34341575, 0.34361678, 0.34174386, 0.3442499,  0.34619512,
                                      0.34597138, 0.34392547, 0.34005339, 0.3388137,  0.34243473,
                                      0.34289874, 0.342966,   0.34112383, 0.33683447])
        return tariff_pred_array