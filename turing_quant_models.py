import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Turing_quant_models:

    def __init__(self, df):
        
        close_columns = []
        high_columns = []
        low_columns = []
        open_columns = []
        volume_columns = []
        open_int_columns = []

        for i in df.columns:
            if "close" in i:
                close_columns.append(i)
            elif "high" in i:
                high_columns.append(i)
            elif "low" in i:
                low_columns.append(i)
            elif "open_int" in i:
                open_int_columns.append(i)
            elif "open" in i:
                open_columns.append(i)
            elif "volume" in i:
                volume_columns.append(i)

        self.close_df = df[close_columns]
        self.high_df = df[high_columns]
        self.low_df = df[low_columns]
        self.open_df = df[open_columns]
        self.volume_df = df[volume_columns]
        self.open_int_df = df[open_int_columns]

        self.returns_daily = self.close_df.pct_change()
        self.returns_monthly = self.close_df.pct_change(
            20).dropna().resample('BM').last().ffill()

        self.vol_daily = self.returns_daily.ewm(
            adjust=True, com=60, min_periods=0).std().dropna()
        self.vol_monthly = (
            np.sqrt(261)*self.vol_daily).resample('BM').last().ffill()

        del(close_columns)
        del(high_columns)
        del(low_columns)
        del(open_columns)
        del(volume_columns)
        del(open_int_columns)

    def prepare_yahoo_df(df):
        
        close_df = df["Close"]
        close_df.columns = df["Close"].columns + "_close"

        high_df = df["High"]
        high_df.columns = df["High"].columns + "_high"

        low_df = df["Low"]
        low_df.columns = df["Low"].columns + "_low"

        open_df = df["Open"]
        open_df.columns = df["Open"].columns + "_open"

        volume_df = df["Volume"]
        volume_df.columns = df["Volume"].columns + "_volume"

        open_int_df = df["Adj Close"]
        open_int_df.columns = df["Adj Close"].columns + "_open_int"

        df2 = pd.concat([close_df, high_df, low_df, open_df, volume_df, open_int_df], axis=1)

        return df2

    def parkinson_vol(high_df, low_df, period=60):
        
        x = np.log(np.divide(high_df, low_df)) ** 2
        x.columns = [x[0:3] + "pv" for x in x.columns]

        pv = x.copy()

        const = 1 / (4 * period * np.log(2))

        pv.iloc[:period, :] = np.nan

        for row in range(period, len(high_df)):
            pv.iloc[row] = np.sqrt(const * np.sum(x.iloc[row-period:row, :]))

        return pv

    def garman_klass_vol(high_df, low_df, close_df, open_df, period=60):

        x_hl = (1/2)*(np.log(np.divide(high_df, low_df))) ** 2
        x_co = - (2 * np.log(2) - 1) * \
            (np.log(np.divide(close_df, open_df))**2)

        x = x_hl + x_co.values

        x.columns = [x[0:3] + "gk" for x in x.columns]

        gk = x.copy()

        const = 1/period

        gk.iloc[:period, :] = np.nan

        for row in range(period, len(high_df)):
            gk.iloc[row] = np.sqrt(const * np.sum(x.iloc[row-period:row, :]))

        return gk
    