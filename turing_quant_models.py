import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Turing_quant_models:

    def __init__(self, df, isDataReader=True):

        if (isDataReader):
            self.close_df = df["Close"]
            self.high_df = df["High"]
            self.low_df = df["Low"]
            self.open_df = df["Open"]
            self.volume_df = df["Volume"]
            self.open_int_df = df["Adj Close"]

        else:
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
        self.returns_monthly = self.close_df.pct_change(20).dropna().resample('BM').last().ffill()

        # EWMA por padrão
        self.vol_daily = self.returns_daily.ewm(adjust=True, com=60, min_periods=0).std().dropna()
        self.vol_monthly = (np.sqrt(261)*self.vol_daily).resample('BM').last().ffill()

    def prepare_yahoo_df(self, df):

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

        df2 = pd.concat([close_df, high_df, low_df, open_df,
                         volume_df, open_int_df], axis=1)

        return df2

    def parkinson_vol(self, high_df, low_df, period=60):
        """
        Estimando a volatilidade a partir dos preço de Alta e de Baixa
        """
        
        # Calculando parcela interna da somatoria
        x = np.log(np.divide(high_df, low_df)) ** 2
        x.columns = [x[0:3] + "pv" for x in x.columns]
        
        # Criando dataframe para atribuir as volatilidades
        pv = x.copy()
        
        # Termo constante fora da somatoria (Considerando vol diaria)
        const = 1 / (4 * period * np.log(2))
        
        # Atribuindo not a number, para os valores iniciais
        pv.iloc[:period,:] = np.nan
            
        # iteração do centro de massa da vol
        for row in range(period, len(high_df)):
            pv.iloc[row] = np.sqrt(const * np.sum(x.iloc[row-period:row,:]))
            
        return pv

    def garman_klass_vol(self, high_df, low_df, close_df, open_df, period=60):
        """
        Estima a volatilidade a partir dos seguintes preços: alta, baixa, abertura e fechamento
        """
        # Calculando parcelas internas da somatoria
        x_hl = (1/2)*(np.log(np.divide(high_df, low_df))) ** 2
        x_co = - (2 * np.log(2) - 1)* (np.log(np.divide(close_df, open_df))**2)
        
        # Somando parcelas calculadas
        x = x_hl + x_co.values
        
        x.columns = [x[0:3] + "gk" for x in x.columns]
        
        # Criando dataframe para atribuir as volatilidades
        gk = x.copy()
        
        # Termo constante fora da somatoria (Considerando vol diaria)
        const = 1/period
        
        # Atribuindo not a number, para os valores iniciais
        gk.iloc[:period,:] = np.nan
        
        # iteração do centro de massa da vol
        for row in range(period, len(high_df)):
            gk.iloc[row] = np.sqrt(const * np.sum(x.iloc[row-period:row,:]))
            
        return gk

    def plot_backtesting(self, returns_model, returns_baseline, label_model="Model", label_baseline="Baseline", title="Cumulative returns"):

        plt.figure(figsize=(16, 9))

        plt.plot(100*returns_model.cumprod(), label=label_model, color='blue')
        plt.plot(100*returns_baseline.cumprod(),
                 label=label_baseline, color='red')

        plt.yscale('log')
        plt.legend()
        plt.title(title)
        plt.show()

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()