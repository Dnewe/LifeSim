import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from typing import List

CMAP = plt.get_cmap("Set2")


def clear_subaxs(axs):
    for subax in axs.ravel():
        subax.clear()


def line_plot(ax, x, y, title:str, ylim=None):
    ax.clear()
    ax.set_title(title)
    ax.plot(x, y)
    ax.set_ylim(ylim)
    

def multiline_plot(ax, x, ys:List, labels:List, title:str):
    ax.clear()
    ax.set_title(title)
    for y, l in zip(ys, labels):
        ax.plot(x, y, label=l)
    ax.legend()


    
def line_multiplot(axs, x, ys, titles:List[str]):
    for i, y in enumerate(ys):
        axs[i].clear()
        axs[i].set_title(titles[i])
        axs[i].plot(x, y) 

        
def scatter_plot(ax, points_list:List, labels_list:List, markers:List, title:str):
    ax.clear()
    ax.set_title(title)
    for points, labels, marker in zip(points_list, labels_list, markers):
        colors = CMAP(labels % CMAP.N)
        ax.scatter(points[:,0], points[:,1], color=colors, marker=marker)


def bar_plot(ax, x, ys, labels, xticks, title:str):
    ax.clear()
    n = len(ys)
    width = 0.75 / (n)
    multiplier = 0
    for l, y in zip(labels, ys):
        color = CMAP(l % CMAP.N)
        offset = width * multiplier
        rects = ax.bar(x + offset, y, width, label=l, color=color)
        ax.bar_label(rects, padding=3)
        multiplier += 1
    ax.set_title(title)
    ax.set_xticks(x + width/2, xticks)
    ax.set_ylabel('values')
    ax.legend(loc='upper left', ncols=n) 


        
def density_plot(ax, df: pd.DataFrame, x:str, y:str, hue:str, title:str):
    ax.clear()
    df["proportion"] = df.groupby(x)[y].transform(lambda x: x / x.sum())
    df_pivot = df.pivot(index=x, columns= hue, values="proportion").fillna(0)
    # Compute cumulative sum for stacking
    df_cumsum = df_pivot.cumsum(axis=1)
    palette = sns.color_palette("Set2", n_colors=df_pivot.shape[1])
    cols = df_pivot.columns
    bottom = np.zeros(len(df_pivot))
    for i, c in enumerate(cols):
        ax.fill_between(
            df_pivot.index,
            bottom,
            df_cumsum[c],
            step="mid",
            color=palette[i],
            alpha=0.7,
            label=f"{c}"
        )
        bottom = df_cumsum[c].values
    ax.set_xlabel(x)
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc='upper left')