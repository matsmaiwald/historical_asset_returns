import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go 
from datetime import datetime
from parameters import params

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
    index_old = df.index        
    index_new = range(index_old.min() - 1, index_old.max() + 1)
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

def create_scatter(df: pd.DataFrame, y_col_name: str, y_display_name: str):
    """Create a scatter plot plotly object."""
    graph = (
        go.Scatter(
            x=df.index,
            y=df[y_col_name],
            name=y_display_name,
            mode="lines",
            visible=False
        )
    )
    return graph


def create_booleans(loc_of_true: int, total_length: int):
    """
    Create a list of booleans with only one True entry and False entries otherwise.
    
    loc_of_true: index position of the True entry.
    
    total_length: length of boolean list to be created. 
    """
    return [True if ind == loc_of_true else False for ind, x in enumerate(range(total_length + 1))]

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

if __name__ == "__main__":
    df = pd.read_csv('./data_input/histretSP.csv')

    for col in ['snp500', '3mon', '10yr']:
        df[col] = df[col].str.rstrip('%').astype('float') / 100.0
    df = df.set_index('year')
    portfolio_allocation = np.array(
        [
        params["share_snp500"], 
        params["share_3mon"],
        params["share_10yr"], 
        ]
        )

    my_portfolio = portfolio(1000,  portfolio_allocation)
    my_asset_markets = asset_markets(df_asset_returns=df) 

    df_all_returns = df.copy()
    df_all_returns["portfolio"] = (
        get_portfolio_returns(
            df.to_numpy(), 
            portfolio_allocation,
            n_assets=3, 
            n_time_periods=df_all_returns.shape[0]
            )
            )

    year_start = params["year_start"]
    year_end = params["year_end"]
    step_size = params["year_step_size"]
    year_steps = range(year_start, year_end, step_size)
    year_steps_shifted = range(year_start + step_size, year_end + step_size, step_size)

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
    
	
    # create start points in time for each different plot
    start_points = list(year_steps)

    # create a list of scatter plot objects
    data = [
        create_scatter(
            dfs[str(year)], 
            y_col_name = "portfolio", 
            y_display_name="portfolio") for year in start_points
    ]
    
    # create a list of dictionaries which will be passed in as options for plotly's buttons
    button_list = [dict(
        label = year,
        method = 'update',
        args = [
            {'visible': create_booleans(loc_of_true=ind, total_length=len(start_points))},
            {'title': year}
        ]
    ) for ind, year in enumerate(start_points)]

    # create the update menus which allow to switch between different plots
    updatemenus = list(
        [
        dict(active=-1,
             buttons=list(button_list),
        )
    ]
    )

    # create the layout which uses the update menus
    layout = dict(title="Cumulative Returns", 
                  showlegend=True,
                  updatemenus=updatemenus
                 )

    # create the final figure
    fig = dict(data=data, layout=layout)


#for year in range(df.index.min(), df.index.max()):
#    print("It is the end of the year {}.".format(year))
#    current_returns = my_asset_markets.yield_returns(year=year)
#    my_portfolio.update_wealth(current_returns)
#    print("The portfolio has a value of {}."
#    .format(round(my_portfolio.wealth, 0)))
#    change_loadings = input("Would you like to change your asset loadings? (y/n)")
#    if change_loadings == "y":
#        my_portfolio.loadings = get_asset_loadings()
