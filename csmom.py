import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from turing_quant_models import Turing_quant_models


class csmom (Turing_quant_models):

    def __init__(self, df):

        Turing_quant_models.__init__(self, df)

    def signal(df, date, passive, method):

        num_assets = len(df.iloc[-1])
        signal = []

        returns = df.pct_change(20 * 12).resample('BM').last().ffill()[:date]

        returns_rank = rankdata(returns.iloc[-1])

        if passive:
            signal = np.ones(num_assets)
        else:
            signal = np.where(returns_rank > int(
                num_assets * 0.7), 1, np.where(returns_rank < int(num_assets * 0.3), -1, 0))

        return signal

    def csmom(self, df, returns_monthly, vol_monthly, date, method='momentum', risk=0.4, passive=False, momentum_window=12):

        position = signal(df, date)

        num_assets = len(df.iloc[-1])

        weights = 1 / (int(num_assets - num_assets * 0.8) +
                       int(num_assets * 0.2))

        portfolio = position * weights

        return (1+np.dot(portfolio, returns_monthly.iloc[date]))

    def backtesting(self, start_date, years, vol, method, plot=True):

        returns_model = []
        start = start_date
        years = years
        end = 12*(int(start/12) + years)

        for i in range(start, end):

            returns_model.append(self.csmom(self.close_df, self.returns_monthly,
                                            self.vol_monthly, i))

        returns_model = pd.DataFrame(returns_model)

        returns_model.index = self.returns_monthly.iloc[start:end].index

        return returns_model
