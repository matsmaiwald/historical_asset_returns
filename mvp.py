import pandas as pd
import numpy as np

df = pd.read_csv('./data_input/histretSP.csv')

for col in ['snp500', '3mon', '10yr']:
    df[col] = df[col].str.rstrip('%').astype('float') / 100.0
df = df.set_index('year')

class asset_markets:
    def __init__(self, df_asset_returns):
        self.df_asset_returns = df_asset_returns

    def yield_returns(self, year):
        return self.df_asset_returns.loc[year, :]

class portfolio:
    def __init__(self, wealth, loadings):
        self.wealth = wealth
        self.loadings = loadings

    def update_wealth(self, asset_returns):
        portfolio_return = (
            np.matmul(self.loadings, asset_returns)
            )
        self.wealth = self.wealth * (1 + portfolio_return)

my_portfolio = portfolio(1000,  np.array([0.25, 0.25, 0.5]))
my_asset_markets = asset_markets(df_asset_returns=df) 

# total_wealth = 1000
# loadings = np.array([0.25, 0.25, 0.5])
returns_1928 = df.to_numpy()[3,:].T
# net_return = np.matmul(loadings, df.to_numpy()[3,:].T)
# total_wealth_new = total_wealth * (1 + net_return)
