import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data_input/histretSP.csv')

for col in ['snp500', '3mon', '10yr']:
    df[col] = df[col].str.rstrip('%').astype('float') / 100.0
df = df.set_index('year')


def get_portfolio_returns(asset_returns: np.array, 
                          asset_loadings: np.array,
                          n_assets: int,
                          n_time_periods: int):
    portfolio_returns = (
        np.matmul(
            asset_returns.reshape(n_time_periods, n_assets),
            asset_loadings.reshape(n_assets, 1)
            )
    )
    
    return portfolio_returns
                                                    


def get_cum_net_returns(net_returns: np.array):
    """
    Get array of cumulative net returns.
    
    net_returns:
        Contains net return figures for different periods (row-wise) and 
        assets (column-wise). 
    """
    cum_gross_returns = np.ones(net_returns.shape)
    for row_idx in range(net_returns.shape[0]):
        if row_idx == 1:
            cum_gross_returns[row_idx, :] = 1 + net_returns[row_idx, :]
        cum_gross_returns[row_idx, :] = (
            cum_gross_returns[row_idx - 1, :] * (
                1 + net_returns[row_idx, :]
                ) 
            )

    return cum_gross_returns - 1

def make_plot_df(df: pd.DataFrame):
    """
    Make dataframe for plotting of cumulative gross returns.

    Add the year previous to the first entry as base with value 1.
    Rescale everything to 100.
    
    df:
        contains net returns over time (row-wise) and by asset (column-wise).
    """
    index_old = df_cum_returns.index        
    index_new = range(index_old.min() - 1, index_old.max())
    df_plot = df.reindex(index_new).fillna(0) 
    df_plot = (df_plot + 1) * 100
    return df_plot

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

df_all_returns = df.copy()
df_all_returns["portfolio"] = (
    get_portfolio_returns(
        df.to_numpy(), 
        np.array([0.25, 0.25, 0.5]),
        n_assets=3, 
        n_time_periods=df_all_returns.shape[0]
        )
        )

year_start = df.index.min()
year_end = df.index.max()
step_size = 5
year_steps = range(year_start, year_end - step_size, step_size)
year_steps_shifted = range(year_start + step_size, year_end, step_size)

dfs = {}

for beginning, end in zip(year_steps, year_steps_shifted):
    print(beginning, end-1)
    df_returns = df_all_returns.loc[beginning:end, :].head()
    # print(str(df_returns))
    df_cum_returns = (
        pd.DataFrame(data=get_cum_net_returns(df_returns.to_numpy()),
                    index=range(beginning, end))
    )
    df_cum_returns.columns = df_returns.columns
    # print(str(df_cum_returns.head()))
    # make_plot_df(df_cum_returns).plot()
    # plt.show()
    dfs[str(beginning)] = make_plot_df(df_cum_returns)
    

#for year in range(df.index.min(), df.index.max()):
#    print("It is the end of the year {}.".format(year))
#    current_returns = my_asset_markets.yield_returns(year=year)
#    my_portfolio.update_wealth(current_returns)
#    print("The portfolio has a value of {}."
#    .format(round(my_portfolio.wealth, 0)))
#    change_loadings = input("Would you like to change your asset loadings? (y/n)")
#    if change_loadings == "y":
#        my_portfolio.loadings = get_asset_loadings()
