import pandas as pd
import os
import sys
from tqdm import tqdm


FOLDER = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "dataset/moscot")
END_DATE = pd.to_datetime('2021-01-01')
START_DATE = pd.to_datetime('2021-01-01') - pd.DateOffset(weeks=104)


def load_csv_timestamp(fname, start_date=None, end_date=None):
    """
    Load pandas DataFrame from csv, with timestamp index

    Params:
    fname: filename of the dataset to load
    """
    ts = pd.read_csv(fname)
    ts = ts.iloc[0]

    index = pd.date_range(start_date, end_date, freq="W")
    ts = pd.DataFrame(ts.values[1:], index=index, columns=[ts.values[0:1]])
    return ts


def load_dataset(dataset_path, freq='W', start_date=None, end_date=None):
    ts = None

    if os.path.isdir(dataset_path):
        for fname in tqdm(os.listdir(dataset_path)):
            if not ".csv" in fname:
                continue
            df = pd.DataFrame(load_csv_timestamp(fname=os.path.join(dataset_path, fname), start_date=start_date, end_date=end_date))
            if ts is None:
                ts = df
            else:
                ts[df.keys()[0]] = df.values

    # if ts is not None:
    #     ts = ts.resample(freq).mean().interpolate(method='time').dropna()
    return ts


if __name__ == "__main__":
    ts = load_dataset(dataset_path=FOLDER, freq='W', start_date=START_DATE, end_date=END_DATE)
    ts.to_csv("./dataset/input.csv", header=True, index=True, index_label="week")
