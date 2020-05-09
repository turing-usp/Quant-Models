import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from turing_quant_models import Turing_quant_models


class markowitz (Turing_quant_models):

    def __init__(self, df):

        Turing_quant_models.__init__(self, df)

    def markowitz(self, returns_monthly, date, min_weight=0.1, max_weight=0.7, method="sharpe",
                  num_portfolios=5000, risk_free=0, plotProgress=False, plotEfficient=False):

        exp_return = returns_monthly.iloc[-12:-1].mean()

        # vetores de dados
        portfolio_weights = []
        portfolio_exp_returns = []
        portfolio_vol = []
        portfolio_sharpe = []

        # simulando diversos portfolios
        for i in range(num_portfolios):

            if (plotProgress):
                self.printProgressBar(i, num_portfolios)

            # construindo o vetor de pesos
            weights = np.array(np.random.uniform(
                low=min_weight, high=max_weight, size=len(returns_monthly.columns)))
            weights = weights/np.sum(weights)

            # calculando o retorno esperado da carteira
            returns = np.dot(weights, exp_return)

            # matriz de covariancia dos ativos
            covariance = np.cov(returns_monthly.iloc[-1])

            # calculando a variancia do portfolio
            vol = np.sqrt(np.dot(np.dot(weights, covariance), weights.T))

            # calculando o sharpe esperado
            sharpe = (returns - risk_free) / vol

            portfolio_weights.append(weights)
            portfolio_exp_returns.append(returns)
            portfolio_vol.append(vol)
            portfolio_sharpe.append(sharpe)

        if(plotEfficient):
            self.plot_efficient_frontier(portfolio_exp_returns, portfolio_vol)

        # portfolio = {
        #    "Expected Returns": pd.DataFrame(portfolio_exp_returns),
        #    "Weights": pd.DataFrame(portfolio_pesos),
        #    "Volatility": pd.DataFrame(portfolio_vol),
        #    "Sharpe": pd.DataFrame(portfolio_sharpe)
        # }

        index = pd.DataFrame(portfolio_sharpe).idxmax()

        w = pd.DataFrame(portfolio_weights).iloc[index]

        portfolio = np.dot(w, returns_monthly.iloc[date])

        return float(1 + portfolio)

    def plot_efficient_frontier (self, returns, risk):
        plt.scatter(risk, returns)
        plt.title("Efficient Frontier")
        plt.xlabel("Volatily")
        plt.ylabel("Expected return")
        plt.show()

    def backtesting(self, start_date, years, plot=True):

        returns_model = []  # retorno do modelo
        returns_baseline = []

        start = start_date
        years = years
        end = 12*(int(start/12) + years)

        for i in range(start, end):

            self.printProgressBar(i-start , end-start )

            returns_model.append(self.markowitz(self.returns_monthly, i))

            returns_baseline.append(1 + self.returns_monthly.iloc[i].mean())

        returns_model = pd.DataFrame(returns_model)
        returns_baseline = pd.DataFrame(returns_baseline)

        returns_model.index = self.returns_monthly.iloc[start:end].index
        returns_baseline.index = self.returns_monthly.iloc[start:end].index

        if plot:
            self.plot_backtesting(returns_model, returns_baseline, "Efficient Frontier",
                                  "Equally weighted portfolio", "Cumulative returns")