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

def make_confidence_interval_graph(df: pd.DataFrame, color: str="green") -> plt.figure:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10,10))

    df.iloc[2].plot(ax=ax, color="green")
    ax.fill_between(df.columns, df.iloc[0], df.iloc[1], alpha=0.3, color=color)
    ax.fill_between(df.columns, df.iloc[1], df.iloc[2], alpha=0.6, color=color)
    ax.fill_between(df.columns, df.iloc[2], df.iloc[3], alpha=0.6, color=color)
    ax.fill_between(df.columns, df.iloc[3], df.iloc[4], alpha=0.3, color=color)
    plt.close(fig)
    return fig
