import pandas as pd
import numpy as np

df = pd.read_csv('./data_input/histretSP.csv')

for col in ['snp500', '3mon', '10yr']:
    df[col] = df[col].str.rstrip('%').astype('float') / 100.0
df = df.set_index('year')

def get_asset_loadings():
    question = "What percentage of your portfolio do you want to invest in?"
    print("SnP500 loading")
    loading_snp = input("Please enter a decimal fraction.")
    print("3mon loading")
    loading_3mon = input("Please enter a decimal fraction.")
    print("10yr loading")
    loading_10yr = input("Please enter a decimal fraction.")
    loadings = np.array(
        [
        loading_snp,
        loading_3mon,
        loading_10yr
        ]
        )
    return loadings

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
            np.matmul(self.loadings, asset_returns.to_numpy())
            )
        self.wealth = self.wealth * (1 + portfolio_return)

my_portfolio = portfolio(1000,  np.array([0.25, 0.25, 0.5]))
my_asset_markets = asset_markets(df_asset_returns=df) 

for year in range(df.index.min(), df.index.max()):
    print("It is the end of the year {}.".format(year))
    current_returns = my_asset_markets.yield_returns(year=year)
    my_portfolio.update_wealth(current_returns)
    print("The portfolio has a value of {}."
    .format(round(my_portfolio.wealth, 0)))
    change_loadings = input("Would you like to change your asset loadings? (y/n)")
    if change_loadings == "y":
        my_portfolio.loadings = get_asset_loadings()
