import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def get_cumulative_returns_matrix(df_returns: pd.DataFrame, 
                                  year_start: int, 
                                  year_end: int, 
                                  year_step_size: int,
                                  asset_of_interest: str
                                 ) -> pd.DataFrame:
    """
    Get a matrix of historical k-year returns for each year in a date range.
    
    df_returns: contains returns by year (rows) and asset (columns).
    year_start: first year to consider data for.
    year_end: last year to consider.
    year_step_size: number of years to calculate cumulative returns for at a time.
    asset_of_interest: column name of asset of interest in df_returns.
    
    """
    # initialise matrix
    return_mat = np.zeros((year_end - year_start + 1 - year_step_size + 1, year_step_size + 1))
    
    for first_year in range(year_start, year_end + 1 - year_step_size + 1):
        
        end = first_year + year_step_size - 1
        df_temp = df_returns.loc[first_year:end, :]

        # get cumulative returns
        df_cum_returns = (pd.DataFrame(data=get_cum_net_returns(df_temp.to_numpy()),
                                       index=range(first_year, end + 1))
                         )
        df_cum_returns.columns = df_temp.columns
        
        # normalise to start at 100
        df_cum_returns_normalised = make_plot_df(df_cum_returns)

        # assign to row in final matrix
        row_idx = first_year - year_start
        return_mat[row_idx,:] = df_cum_returns_normalised[asset_of_interest].to_numpy().round(2)
    
    return return_mat

def make_quantile_mat(return_mat: np.array, quantiles: list):
    """Make a matrix containing different quantiles of the cumulative return matrix."""
    n_periods = return_mat.shape[1]
    n_quantiles = len(quantiles)
    quantile_mat = np.zeros([n_quantiles, n_periods])
    
    for col_idx in range(quantile_mat.shape[1]): # iterate over columns 
        for counter, quantile in enumerate(quantiles):
            quantile_mat[counter, col_idx] = np.quantile(return_mat[:,col_idx], quantile)
    
    return quantile_mat

def make_confidence_interval_graph(df: pd.DataFrame, color: str="green") -> plt.figure:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10,10))
    cmap = plt.cm.RdYlGn
    colour_mapping = {
        "top_10": cmap(1.0),
        "50-90": cmap(0.66),
        "10-50": cmap(0.33),
        "bottom_10": cmap(0.1), 
                         }

    #df.iloc[2].plot(ax=ax, color="green")
    ax.fill_between(df.columns, df.iloc[3], df.iloc[4], alpha=0.8, color=colour_mapping["top_10"])
    ax.fill_between(df.columns, df.iloc[2], df.iloc[3], alpha=0.8, color=colour_mapping["50-90"])
    ax.fill_between(df.columns, df.iloc[1], df.iloc[2], alpha=0.8, color=colour_mapping["10-50"])
    ax.fill_between(df.columns, df.iloc[0], df.iloc[1], alpha=0.8, color=colour_mapping["bottom_10"])
    
    
    
    
    ax.grid(True)
    
    ax.set_xlabel('Years since inception of portfolio')
    ax.set_ylabel('Porfolio value')
    legend_elements = [
        Patch(color=colour_mapping["top_10"], label="top 10 percent"),               
        Patch(color=colour_mapping["50-90"], label="50th to 90th percentile"),
        Patch(color=colour_mapping["10-50"], label="10th to 50th percentil"),
        Patch(color=colour_mapping["bottom_10"], label="bottom 10 percent"),
                      ]


    ax.legend(handles=legend_elements, loc='upper left')
    plt.close(fig)
    return fig
