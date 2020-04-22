import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

from turing_quant_models import Turing_quant_models


class tsmom (Turing_quant_models):

    def __init__(self, df):

        Turing_quant_models.__init__(self, df)

    def signal(self, df, date, passive, method):

        num_assets = len(df.iloc[-1])
        signal = []

        if method == "momentum":

            returns = df.pct_change(
                20 * 12).resample('BM').last().ffill()[:date]

            if passive:
                signal = np.ones(num_assets)
            else:
                signal = np.where(returns.iloc[-1] > 0, 1, -1)

        elif method == "momentum_lagged":

            returns_12 = df.pct_change(
                21 * 12).resample('BM').last().ffill()[:date]

            returns_6 = df.pct_change(
                21 * 6).resample('BM').last().ffill()[:date]

            returns_3 = df.pct_change(
                21 * 3).resample('BM').last().ffill()[:date]

            momentum_mean = (
                returns_12.iloc[-1] + returns_6.iloc[-1] + returns_3.iloc[-1]) / 3

            if passive:
                signal = np.ones(num_assets)
            else:
                signal = np.where(momentum_mean > 0, 1, -1)

        return signal

    def tsmom(self, df, returns_monthly, vol_monthly, date, method='momentum', risk=0.4, passive=False, momentum_window=12):

        position = self.signal(df, date, passive, method)

        weights = (risk / vol_monthly.iloc[date-1])

        weights /= len(weights)

        portfolio = position * weights

        return (1+np.dot(portfolio, returns_monthly.iloc[date]))

    def backtesting(self, start_date, years, vol, method, plot=True):

        returns_model = []  # retorno do TSMOM
        returns_baseline = []  # retorno passivo
        start = start_date
        years = years
        end = 12*(int(start/12) + years)

        for i in range(start, end):

            returns_model.append(self.tsmom(self.close_df, self.returns_monthly,
                                            self.vol_monthly, i))

            returns_baseline.append(self.tsmom(self.close_df, self.returns_monthly,
                                               self.vol_monthly, i, passive=True))

        returns_model = pd.DataFrame(returns_model)
        returns_baseline = pd.DataFrame(returns_baseline)

        returns_model.index = self.returns_monthly.iloc[start:end].index
        returns_baseline.index = self.returns_monthly.iloc[start:end].index

        if plot:
            self.plot_backtesting(
                returns_model, returns_baseline, "TSMOM", "Long only", "Cumulative returns")

