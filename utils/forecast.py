"""
Tonin Davide VR437255

Utils Functions to analyze and forecast time series

=========================================================================================================

Python       Documentation: https://docs.python.org/3/
Statsmodels  Documentation: https://www.statsmodels.org/stable/index.html
Scikit-Learn Documentation: https://scikit-learn.org/stable/
Pandas       Documentation: https://pandas.pydata.org/docs/
Matplotlib   Documentation: https://matplotlib.org/

=========================================================================================================

Other Resources:
https://otexts.com/fpp2/
"""

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import meanabs
from math import sqrt

from pmdarima.arima import auto_arima
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

plt.style.use('seaborn')

TRAIN_COLOR = "black"
TEST_COLOR = "dodgerblue"
FITTED_COLOR = "lime"
FORECAST_COLOR = "red"
CONF_INT_COLOR = "orange"
SIMULATIONS_COLOR = "orange"

TEST_LS = ":"  # tipo di linea

def load_csv_timestamp(fname):
    """
    Load pandas DataFrame from csv, with timestamp index

    Params:
    fname: filename of the dataset to load
    """
    ts = pd.read_csv(fname)
    index = pd.to_datetime(ts.iloc[:, 0])
    ts = ts.drop(ts.columns[0], axis=1)
    return ts.set_index(index).asfreq('W')


def ts_diagnostic_plot(ts, lags=None, figsize=(12, 6), show_zero=False):
    """
    Helper function to plot time series, acf and pacf

    Params:
    ts          : time series
    lags        : number of lags (default automatic)
    figsize     : figure size
    show_zero   : show zero lag => True | False

    ==============================
    Return:
    (ts_ax, acf_ax, pacf_ax)
    """
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    ts.plot(ax=ts_ax)
    smt.graphics.plot_acf(ts, lags=lags, ax=acf_ax, zero=show_zero)
    smt.graphics.plot_pacf(ts, lags=lags, ax=pacf_ax, zero=show_zero)
    plt.tight_layout()

    return ts_ax, acf_ax, pacf_ax


def decompose_ts(ts, show_plot=False, period=None, type="STL", model='additive'):
    """
    Get ts decomposition components: Trend, Seasonal, Residual

    Parameters
    ----------------
    ts:     Time series
    period: Mandatory if there isn't a frequency in the time series
    type:   {'STL', 'classic'}, default = STL
    model: {'additive', 'multiplicative'}, default = additive, used only if type = classic

    ================
    Return: (trend, seasonal, resid)
    """
    if type == "STL":
        decomposition = STL(ts, period=period).fit()
    else:
        decomposition = seasonal_decompose(ts, model=model, period=period)

    trend = pd.DataFrame(columns=['trend'], data=decomposition.trend.astype('float'))
    seasonal = pd.DataFrame(columns=['season' if type == 'STL' else 'seasonal'], data=decomposition.seasonal.astype('float'))
    resid = pd.DataFrame(columns=['resid'], data=decomposition.resid.astype('float'))

    if show_plot:
        plt.subplot(411)
        plt.plot(ts, label='Original')
        plt.legend(loc='best')

        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')

        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal')
        plt.legend(loc='best')

        plt.subplot(414)
        plt.plot(resid, label='Residual')
        plt.legend(loc='best')

        plt.tight_layout()

    return trend, seasonal, resid


def get_accuracy(true, predict):
    errors_df = pd.DataFrame()
    ex = False
    try:
        ms_error = np.round(mean_squared_error(true, predict), 5)
        rms_error = np.round(sqrt(mean_squared_error(true, predict)), 5)
        meanabs_error = np.round(np.mean(meanabs(true, predict)), 5)
    except:
        ex = True

    if ex:
        errors_df = errors_df.append(
            {"MSE": "exception", "RMSE": "exception", "MAE": "exception"}, ignore_index=True)
    else:
        errors_df = errors_df.append(
            {"MSE": ms_error, "RMSE": rms_error, "MAE": meanabs_error}, ignore_index=True)

    return errors_df


def get_prediction_ts(ts, h, freq="W"):
    """
    Params:
    ts  :   Time Series di TRAIN (NOTA). Passare la serie di train, perchè potrebbe essere che la parte di forecast sia completamente out-of-sample (quindi serie di test vuota)
    freq:   Frequenza della serie
    h   :   Step di forecast

    ===========
    Return:

    pd.Dataframe contenente l'indice per la serie di forecast
    """
    if freq == "W":
        min_date = max(ts.index) + pd.DateOffset(weeks=+1)
        max_date = datetime.date(
            min_date.year, min_date.month, min_date.day) + pd.DateOffset(weeks=+h)
    else:
        min_date = max(ts.index) + pd.DateOffset(months=+1)
        max_date = datetime.date(
            min_date.year, min_date.month, min_date.day) + pd.DateOffset(months=+h)
    date_range = pd.date_range(min_date, max_date, freq=freq)
    res = pd.DataFrame(index=date_range, columns=ts.columns)

    return res


def naive(train, h, freq="W", show_plot=False):
    """
    Naive forecast

    Params:
    train       : train series
    h           : steps of forecast
    show_plot   : {True, False} show the plot of ts, rolling mean and rolling std

    =====================
    Return:
    forecast result => pd.Dataframe()
    """
    res = get_prediction_ts(ts=train, freq=freq, h=h)[:h]

    for col in train.columns:
        res[col] = float(train[col].iloc[-1])
    return res


def generate_auto_arima_model(train, m, seasonal=False):
    model = auto_arima(train, min_p=0, min_q=0, min_P=0, min_Q=0, max_D=1, m=m)
    model = model.to_dict()
    order = model['order']
    seasonal_order = model['seasonal_order']

    return {'order': order, 'seasonal_order': seasonal_order}


def arima_model(train, order, seasonal_order=(0, 0, 0, 0)):
    """
    ARIMA model, uses ARIMA from statsmodels (SARIMAX model)

    Params:
    seasonal_order  : optional, only to fit a Seasonal ARIMA
    =============
    Return: ARIMAResults
    """
    model = ARIMA(train, order=order, seasonal_order=seasonal_order)
    model = model.fit()
    model.fittedvalues = model.fittedvalues.iloc[order[1]:]

    return model


def stl_arima_model(train, order):
    """
    Uses STL from statsmodels (STL model) with ARIMA
    =============
    Return: STLForecastResults
    """
    model = STLForecast(
        train, ARIMA, model_kwargs=dict(order=order, trend="t"))
    model = model.fit()

    return model


def forecast_plot(train, test, fitted_values, forecast_values, new_fig=False, plot_title="Forecast"):
    """
    Plot train data, test data, fitted values of the model and the predicted/forecast values

    Params:
    train           : the train dataframe
    test            : the test dataframe
    fitted_values   : fitted values of the model
    forecast_values : predicted/forecast values
    plot_title
    """
    test_window = 0
    if test is not None:
        test_window = len(test) if len(forecast_values) >= len(
            test) else len(forecast_values)

    if new_fig:
        plt.figure(figsize=(12, 6))
    plt.title(plot_title)

    plt.plot(train, color=TRAIN_COLOR, label='Train')
    if test_window > 0:
        plt.plot(test[:test_window], color=TEST_COLOR,
                 ls=TEST_LS, label='Test')

    if fitted_values is not None:
        plt.plot(fitted_values, color=FITTED_COLOR, label="Fitted Values")
    plt.plot(forecast_values, color=FORECAST_COLOR, label="Predicted Values")

    plt.legend(loc="best")


def compare_forecast_plot(train, test, ts_forecast=None, fig=None, plot_title="Compare Forecast"):
    """
    Plot train, test time time series
    Plot multiple forecast to compare
    """
    if fig is None:
        plt.figure(figsize=(12, 6))
    plt.title(plot_title)

    plt.plot(train, color=TRAIN_COLOR, label='Train')
    if test is not None and len(test) > 0:
        plt.plot(test, color=TEST_COLOR, ls=TEST_LS, label='Test')

    ax = plt.gca()
    if ts_forecast is not None:
        ts_forecast.plot(ax=ax)

    plt.legend(loc="best")


def save_input_plots(train, test, folder):
    fig = plt.figure()

    for key in tqdm(train.keys()):
        tmp_train = train[key]
        tmp_test = test[key]
        plt.title(key)
        plt.plot(tmp_train, color=TRAIN_COLOR, label="Train")
        plt.plot(tmp_test, color=TEST_COLOR, ls=TEST_LS, label="Test")
        plt.legend(loc="best")
        plt.savefig(os.path.join(folder, key+".png"))
        fig.clear()


def save_input_diagnostic_plots(ts, folder, lags=None):
    fig = plt.figure()

    for key in tqdm(ts.keys()):
        plt.title(f"DIAGNOSTIC: {key}")
        ts_diagnostic_plot(ts[key], lags=lags)
        plt.savefig(os.path.join(folder, key+".png"))
        fig.clear()


def save_input_decomposition_plots(ts, folder):
    fig = plt.figure()

    for key in tqdm(ts.keys()):
        plt.title(f"STL DECOMPOSITION: {key}")
        decompose_ts(ts[key], show_plot=True)
        plt.savefig(os.path.join(folder, key+".png"))
        fig.clear()