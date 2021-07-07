"""
Tonin Davide VR437255
"""
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
from tqdm import tqdm
from utils.forecast import *

warnings.filterwarnings("ignore")

FREQ = "W"
SEASONAL = False
SEASONAL_PERIOD = {
    'W': 52,
    'M': 12
}

# EXECUTION SETTINGS
EXECUTE_NAIVE = True
EXECUTE_ARIMA = True
EXECUTE_STLARIMA = True
SAVE_BEST_MAE = True
SAVE_BEST_RMSE = True

SAVE_PLOT = {
    'NAIVE'     : True,
    'ARIMA'     : True,
    'STLARIMA'  : True
}

SAVE_FORECAST_RESULTS = {
    'NAIVE'     : True,
    'ARIMA'     : True,
    'STLARIMA'  : True
}

SAVE_ERRORS = True
SAVE_ERRORS_STATISTICS = True

# INPUT PLOT SETTINGS
SAVE_INPUT_PLOT = False
SAVE_INPUT_DECOMPOSITION_PLOT = False
SAVE_INPUT_DIAGNOSTIC_PLOT = False

# ARIMA SETTINGS
GENERATE_AUTO_ARIMA = False
LOAD_ARIMA_FROM_FILE = not GENERATE_AUTO_ARIMA
SAVE_SUMMARY = False
SAVE_ORDERS = False

METHODS = ['naive', 'stlarima', 'arima']
ACCURACY = ['MSE', 'RMSE', 'MAE']

# ===================================
# Considerando un segnale di 2 anni
TRAIN_SIZE = 20 if FREQ == 'M' else 80
PRED_STEPS = 10 if FREQ == 'M' else 30

# FOLDER & FILE
FREQ_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), "monthly_data") if FREQ == 'M' else os.path.join(os.path.dirname(os.path.realpath(__file__)), "weekly_data")

INPUT_FOLDER = os.path.join(FREQ_FOLDER, "input")
OUTPUT_FOLDER = os.path.join(FREQ_FOLDER, "output_seasonal") if SEASONAL else os.path.join(FREQ_FOLDER, "output")
OUTPUT_ERRORS_FOLDER = os.path.join(OUTPUT_FOLDER, "errors")

DATASET_FNAME = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "dataset/input_completo.csv")

ARIMA_MODEL_FNAME = os.path.join(FREQ_FOLDER, "arima_model_seasonal.csv") if SEASONAL else os.path.join(FREQ_FOLDER, "arima_model.csv")

INPUT_PLOT_FOLDERS = {
    'INPUT'         : os.path.join(INPUT_FOLDER, 'input_plot'),
    'DIAGNOSTIC'    : os.path.join(INPUT_FOLDER, 'input_diagnostic_plot'),
    'DECOMPOSITION' : os.path.join(INPUT_FOLDER, 'input_decomposition_plot')
}

OUTPUT_FORECAST_FOLDERS = {
    'OUTPUT'    : os.path.join(OUTPUT_FOLDER, 'forecast_results'), # base folder
    'ARIMA'     : os.path.join(OUTPUT_FOLDER, 'forecast_results/arima'),
    'STLARIMA'  : os.path.join(OUTPUT_FOLDER, 'forecast_results/stl_arima'),
    'NAIVE'     : os.path.join(OUTPUT_FOLDER, 'forecast_results/naive')
}

OUTPUT_PLOT_FOLDERS = {
    'OUTPUT'    : os.path.join(OUTPUT_FOLDER, 'output_plot'),  # base folder
    'ARIMA'     : os.path.join(OUTPUT_FOLDER, 'output_plot/arima'),
    'STLARIMA'  : os.path.join(OUTPUT_FOLDER, 'output_plot/stl_arima'),
    'NAIVE'     : os.path.join(OUTPUT_FOLDER, 'output_plot/naive'),
    'BEST'      : os.path.join(OUTPUT_FOLDER, 'output_plot/best'),
}

OUTPUT_SUMMARY_FOLDERS = {
    'OUTPUT': os.path.join(OUTPUT_FOLDER, 'output_summary'),  # base folder
    'ARIMA' : os.path.join(OUTPUT_FOLDER, 'output_summary/arima')
}

# check if all folders exist
# create directories if they don't exist

if not os.path.isdir(FREQ_FOLDER):
    os.mkdir(FREQ_FOLDER)
if not os.path.isdir(INPUT_FOLDER):
    os.mkdir(INPUT_FOLDER)
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
if not os.path.isdir(OUTPUT_ERRORS_FOLDER):
    os.mkdir(OUTPUT_ERRORS_FOLDER)

for key, directory in INPUT_PLOT_FOLDERS.items():
    if not os.path.isdir(directory):
        os.mkdir(directory)
for key, directory in OUTPUT_FORECAST_FOLDERS.items():
    if not os.path.isdir(directory):
        os.mkdir(directory)
for key, directory in OUTPUT_PLOT_FOLDERS.items():
    if not os.path.isdir(directory):
        os.mkdir(directory)
for key, directory in OUTPUT_SUMMARY_FOLDERS.items():
    if not os.path.isdir(directory):
        os.mkdir(directory)

# ===================================

ts = load_csv_timestamp(DATASET_FNAME)
if FREQ == 'M':
    ts = ts.resample('M').mean()

train = ts[:TRAIN_SIZE]
test = ts[TRAIN_SIZE:]

if SAVE_INPUT_PLOT:
    print("Saving input plots")
    save_input_plots(train, test, INPUT_PLOT_FOLDERS['INPUT'])
if SAVE_INPUT_DIAGNOSTIC_PLOT:
    print("Saving input diagnostic plots")
    save_input_diagnostic_plots(train, INPUT_PLOT_FOLDERS['DIAGNOSTIC'], lags=9)
if SAVE_INPUT_DECOMPOSITION_PLOT:
    print("Saving input decomposition plots")
    save_input_decomposition_plots(train, INPUT_PLOT_FOLDERS['DECOMPOSITION'])

# INIT ERRORS Dataframe
errors = {
    'MSE': pd.DataFrame(index=pd.Index(ts.keys()), columns=METHODS),
    'RMSE': pd.DataFrame(index=pd.Index(ts.keys()), columns=METHODS),
    'MAE': pd.DataFrame(index=pd.Index(ts.keys()), columns=METHODS)
}

fig = plt.figure()

if EXECUTE_NAIVE:
    for key in tqdm(ts.keys()):
        tmp_train = pd.DataFrame(train[key])
        tmp_test = pd.DataFrame(test[key])

        forecast = naive(train=tmp_train, h=PRED_STEPS, freq=FREQ)

        if SAVE_PLOT['NAIVE']:
            forecast_plot(train=tmp_train, test=tmp_test, fitted_values=None, forecast_values=forecast, plot_title=f'Naive {key}')
            plt.savefig(os.path.join(OUTPUT_PLOT_FOLDERS['NAIVE'], key+".png"))
            fig.clear()

        accuracy = get_accuracy(tmp_test, forecast[:len(tmp_test)])
        for a in ACCURACY:
            errors[a].loc[key, 'naive'] = accuracy[a][0]

        if SAVE_FORECAST_RESULTS:
            index_label = "month" if FREQ == 'M' else 'week'
            forecast.to_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['NAIVE'], key+'.csv'), index_label=index_label, header=['value'])


if EXECUTE_ARIMA:
    print("\n")
    print("="*50)
    print("\nARIMA MODELS\n")

    if SAVE_ORDERS:
        arima_model_out = open(ARIMA_MODEL_FNAME, "w")
        arima_model_out.write("key;order;seasonal_order\n")
    if LOAD_ARIMA_FROM_FILE:
        arima_model_orders = pd.read_csv(ARIMA_MODEL_FNAME, delimiter=";", index_col="key")

    for key in tqdm(ts.keys()):
        tmp_train = train[key]
        tmp_test = test[key]

        if GENERATE_AUTO_ARIMA:
            if SEASONAL:
                arima_model_orders = generate_auto_arima_model(train=tmp_train, seasonal=True, m=SEASONAL_PERIOD[FREQ])
            else:
                arima_model_orders = generate_auto_arima_model(train=tmp_train, seasonal=False, m=1)

            order = arima_model_orders['order']
            seasonal_order = arima_model_orders['seasonal_order']
        elif LOAD_ARIMA_FROM_FILE:
            order = arima_model_orders['order'][key].strip("(").strip(")").split(", ")
            order = (int(order[0]), int(order[1]), int(order[2]))

            seasonal_order = arima_model_orders['seasonal_order'][key].strip("(").strip(")").split(", ")
            seasonal_order = (int(seasonal_order[0]), int(seasonal_order[1]), int(seasonal_order[2]), int(seasonal_order[3]))

        model = arima_model(train=tmp_train, order=order, seasonal_order=seasonal_order)
        if SAVE_SUMMARY:
            summary_out = open(os.path.join(OUTPUT_SUMMARY_FOLDERS['ARIMA'], key+"_summary.txt"), "w")
            summary_out.write(str(model.summary()))
            summary_out.close()

        if SAVE_ORDERS:
            arima_model_out.write(f"{key};{str(order)};{str(seasonal_order)}\n")
            arima_model_out.flush()

        forecast = model.get_forecast(PRED_STEPS)
        prediction = forecast.predicted_mean
        conf_int_95 = forecast.conf_int(alpha=0.5)
        fitted = model.fittedvalues

        if SAVE_PLOT['ARIMA']:
            plot_title = f"ARIMA{order}{seasonal_order} {key}" if SEASONAL else f"ARIMA{order} {key}"
            forecast_plot(train=tmp_train, test=tmp_test, fitted_values=fitted,
                            forecast_values=prediction, plot_title=plot_title, new_fig=False)
            plt.fill_between(
                x=conf_int_95.index, y1=conf_int_95[f'lower {key}'], y2=conf_int_95[f'upper {key}'], alpha=0.3, color=CONF_INT_COLOR)

            plt.savefig(os.path.join(OUTPUT_PLOT_FOLDERS['ARIMA'], key+".png"))
            fig.clear()

        accuracy = get_accuracy(tmp_test, prediction[:len(tmp_test)])
        for a in ACCURACY:
            errors[a].loc[key, 'arima'] = accuracy[a][0]

        if SAVE_FORECAST_RESULTS:
            index_label = "month" if FREQ == 'M' else 'week'
            prediction.to_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['ARIMA'], key+'.csv'), index_label=index_label, header=['value'])

    if SAVE_ORDERS:
        arima_model_out.close()


if EXECUTE_STLARIMA:
    print("\n")
    print("="*50)
    print("\nSTL FORECAST with ARIMA\n")
    # LOAD ARIMA ORDERS FROM FILE
    arima_model_orders = pd.read_csv(ARIMA_MODEL_FNAME, delimiter=";", index_col="key")

    for key in tqdm(ts.keys()):
        tmp_train = train[key]
        tmp_test = test[key]

        order = arima_model_orders['order'][key].strip("(").strip(")").split(", ")
        order = (int(order[0]), int(order[1]), int(order[2]))

        # STL ARIMA
        model = stl_arima_model(tmp_train, order)
        forecast = model.forecast(PRED_STEPS)

        if SAVE_PLOT['STLARIMA']:
            forecast_plot(train=tmp_train, test=tmp_test, fitted_values=None,
                            forecast_values=forecast, plot_title=f"STL FORECAST with ARIMA{order}")
            plt.savefig(os.path.join(OUTPUT_PLOT_FOLDERS['STLARIMA'], key+".png"))
            fig.clear()

        accuracy = get_accuracy(tmp_test, forecast[:len(tmp_test)])
        for a in ACCURACY:
            errors[a].loc[key, 'stlarima'] = accuracy[a][0]
        if SAVE_FORECAST_RESULTS:
            index_label = "month" if FREQ == 'M' else 'week'
            forecast.to_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['STLARIMA'], key+'.csv'), index_label=index_label, header=['value'])


if SAVE_ERRORS:
    for key, element in errors.items():
        element.sort_values(by=METHODS).to_csv(os.path.join(OUTPUT_ERRORS_FOLDER, key + ".csv"), index_label="key")

# LOAD ERRORS FROM FILE
mae = pd.read_csv(os.path.join(OUTPUT_ERRORS_FOLDER, 'MAE.csv'), index_col='key')
rmse = pd.read_csv(os.path.join(OUTPUT_ERRORS_FOLDER, 'RMSE.csv'), index_col='key')

best_mae = {}
best_rmse = {}

for method in METHODS:
    mae = mae.sort_values(by=method)
    best_mae[method] = mae.iloc[0:5].index.to_list()

    rmse = rmse.sort_values(by=method)
    best_rmse[method] = rmse.iloc[0:5].index.to_list()

ts_forecast_index = get_prediction_ts(train, freq=FREQ, h=PRED_STEPS).index[:PRED_STEPS]

if SAVE_BEST_MAE:
    print("\n")
    print("="*50)
    print("\nSAVE BEST RESULTS (MAE)\n")
    for m, keys in tqdm(best_mae.items()):
        for key in keys:
            forecast_naive = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['NAIVE'], key+'.csv'), index_col="week")
            forecast_arima = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['ARIMA'], key+'.csv'), index_col="week")
            forecast_stlarima = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['STLARIMA'], key+'.csv'), index_col="week")

            ts_forecast = pd.DataFrame(index=ts_forecast_index, columns=['naive', 'arima', 'stlarima'])
            ts_forecast['naive'] = forecast_naive.values
            ts_forecast['arima'] = forecast_arima.values
            ts_forecast['stlarima'] = forecast_stlarima.values

            tmp_train = train[key]
            tmp_test = test[key]
            compare_forecast_plot(train=tmp_train, test=tmp_test, ts_forecast=ts_forecast, fig=fig, plot_title=f"{key} - Compare Plot")
            plt.savefig(os.path.join(OUTPUT_PLOT_FOLDERS['BEST'], key+".png"))
            fig.clear()

if SAVE_BEST_RMSE:
    print("\n")
    print("="*50)
    print("\nSAVE BEST RESULTS (RMSE)\n")
    for m, keys in tqdm(best_rmse.items()):
        for key in keys:
            forecast_naive = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['NAIVE'], key+'.csv'), index_col="week")
            forecast_arima = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['ARIMA'], key+'.csv'), index_col="week")
            forecast_stlarima = pd.read_csv(os.path.join(OUTPUT_FORECAST_FOLDERS['STLARIMA'], key+'.csv'), index_col="week")

            ts_forecast = pd.DataFrame(index=ts_forecast_index, columns=['naive', 'arima', 'stlarima'])
            ts_forecast['naive'] = forecast_naive.values
            ts_forecast['arima'] = forecast_arima.values
            ts_forecast['stlarima'] = forecast_stlarima.values

            tmp_train = train[key]
            tmp_test = test[key]
            compare_forecast_plot(train=tmp_train, test=tmp_test, ts_forecast=ts_forecast, fig=fig, plot_title=f"{key} - Compare Plot")
            plt.savefig(os.path.join(OUTPUT_PLOT_FOLDERS['BEST'], key+".png"))
            fig.clear()

if SAVE_ERRORS_STATISTICS:
    errors_statistics = pd.DataFrame(index=pd.Index(['MAE', 'RMSE']), columns=METHODS)

    print("MAE STATISTICS")
    errors_statistics.loc["MAE", "naive"] = len(mae.loc[(mae["naive"] < mae["arima"]) & (mae["naive"] < mae["stlarima"])])
    errors_statistics.loc["MAE", "arima"] = len(mae.loc[(mae["arima"] < mae["naive"]) & (mae["arima"] < mae["stlarima"])])
    errors_statistics.loc["MAE", "stlarima"] = len(mae.loc[(mae["stlarima"] < mae["naive"]) & (mae["stlarima"] < mae["arima"])])

    print("RMSE STATISTICS")
    errors_statistics.loc["RMSE", "naive"] = len(rmse.loc[(rmse["naive"] < rmse["arima"]) & (rmse["naive"] < rmse["stlarima"])])
    errors_statistics.loc["RMSE", "arima"] = len(rmse.loc[(rmse["arima"] < rmse["naive"]) & (rmse["arima"] < rmse["stlarima"])])
    errors_statistics.loc["RMSE", "stlarima"] = len(rmse.loc[(rmse["stlarima"] < rmse["naive"]) & (rmse["stlarima"] < rmse["arima"])])

    errors_statistics.to_csv(os.path.join(OUTPUT_ERRORS_FOLDER, 'statistics.csv'), index_label="index")