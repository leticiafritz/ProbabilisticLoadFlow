import warnings

warnings.filterwarnings('ignore')

from pandas.tseries.holiday import AbstractHolidayCalendar as calendar
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import statsmodels.api as sm
import pmdarima as pm


def season_calc(month):
    """adding season based on the data on SDGE's site -> https://www.sdge.com/whenmatters#how-it-works;
       months from June to October are denoted as 'summer' and months from November to May as 'winter'. """
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


# Dados de entrada
data = pd.read_excel('pld_sombra_2020.xlsx', sheet_name='train', index_col='date', parse_dates=['date'])

# Analisando os dados
##res = sm.tsa.seasonal_decompose(data['tariff'], freq=365)
##fig = res.plot()
##fig.set_figheight(8)
##fig.set_figwidth(15)
##plt.show()
res = sm.tsa.adfuller(data['tariff'].dropna(), regression='ct')
print('p-value:{}'.format(res[1]))
res = sm.tsa.adfuller(data['tariff'].diff().dropna(), regression='c')
print('p-value:{}'.format(res[1]))

# Organizando os dados
data = data.reset_index()
data.index = pd.period_range('2019-01-01', periods=len(data), freq='H')
data = create_features(data)
tariff_data = data['tariff']
tariff_exog = data[['month', 'day', 'hour', 'weekday']]
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Monday", 2)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Tuesday", 3)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Wednesday", 4)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Thursday", 5)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Friday", 6)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Saturday", 7)
tariff_exog['weekday'] = tariff_exog['weekday'].replace("Sunday", 1)

# Tirando o ultimo dia para previsão
tariff_test = pd.DataFrame()
tariff_train = pd.DataFrame()
tariff_test['tariff'] = tariff_data.iloc[(tariff_data.size-24):, ]
tariff_exog_test = tariff_exog.iloc[(tariff_data.size-24):, ]
tariff_train['tariff'] = tariff_data.iloc[:(tariff_data.size-24), ]
tariff_exog_train = tariff_exog.iloc[:(tariff_data.size-24), ]

'''
results = pm.auto_arima(tariff_train,
                        d=1,  # non-seasonal difference order
                        start_p=0,  # initial guess for p
                        start_q=0,  # initial guess for q
                        max_p=2,  # max value of p to test
                        max_q=2,  # max value of q to test
                        exogenous=tariff_exog_train,  # including the exogenous variables
                        seasonal=True,  # is the time series seasonal? YES
                        m=24,  # the seasonal period
                        # D=1, # seasonal difference order
                        start_P=1,  # initial guess for P
                        start_Q=1,  # initial guess for Q
                        max_P=1,  # max value of P to test
                        max_Q=1,  # max value of Q to test
                        information_criterion='aic',  # used to select best model
                        trace=True,  # print results whilst training
                        error_action='ignore',  # ignore orders that don't work
                        stepwise=True,  # apply intelligent order search
                        )
'''
#model_opt = SARIMAX(tariff_train, order=(2, 1, 1),
#                    seasonal_order=(1, 0, 1, 24), exog=tariff_exog_train, trend='c')
model_opt = SARIMAX(tariff_train, order=(2, 1, 1), seasonal_order=(1, 0, 1, 24), exog=tariff_exog_train, trend='c')
results = model_opt.fit()
print(results.summary())
residuals = results.resid
residuals.plot()
plt.show()
yhat = results.predict(start=len(tariff_train), end=len(tariff_train) + 24)
print(yhat)
tariff_test['tariff'].plot(x=tariff_test.index, y=tariff_test['tariff'], legend='original')
yhat.plot(color='red', legend='previsao')
plt.show()
'''
# Choosing a subset of the above dataframe; removing the lags and the hour bins
tariff_lin = pd.get_dummies(data, drop_first=True)
# Creating the lag variables
for i in range(24):
    tariff_lin['lag' + str(i + 1)] = tariff_lin['tariff'].shift(i + 1)
lag_tariff = tariff_lin.dropna()
add_fourier_terms(lag_tariff, year_k=2, week_k=2, day_k=2)

tariffcyc = lag_tariff.drop([col for col in lag_tariff if
                             col.startswith('time') or col.startswith('season') or col.startswith('lag')], axis=1)
X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test(tariffcyc, test_size=24)
scaler1 = StandardScaler()
y_train_lag = pd.DataFrame(scaler1.fit_transform(y_train_lag.values.reshape(-1, 1)), index=y_train_lag.index,
                           columns=['tariff'])
y_train_lag = y_train_lag.reset_index()
y_train_lag.index = pd.period_range('2018-01-01', periods=len(y_train_lag), freq='H')
tariff_train = y_train_lag['tariff']
model_opt = SARIMAX(tariff_train, order=(2, 1, 1), seasonal_order=(1, 0, 1, 24), exog=X_train_lag, trend='c')
#ts = pd.DatetimeIndex(y_train_lag.index).to_period('H')
#y_train_lag['ts'], X_train_lag['ts'] = ts, ts
#y_train_lag, X_train_lag = y_train_lag.reset_index().set_index('ts'), X_train_lag.reset_index().set_index('ts')
#y_train_lag, X_train_lag = y_train_lag.drop('date', axis=1), X_train_lag.drop('date', axis=1)
#mod = SARIMAX(endog, order=(1, 0, 0), trend='c')
#res = mod.fit()
#model_opt = SARIMAX(y_train_lag, order=(2, 1, 1), seasonal_order=(1, 0, 1, 24), exog=X_train_lag, trend='c')
#results = model_opt.fit()
#print(results.summary())
'''
