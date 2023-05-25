import seaborn as sns
import pandas as pd


def plot_boxplot(array_list, name_list, title="", save_path=None):
    # create df
    df = pd.DataFrame({name_list[i]: array_list[i] for i in range(len(array_list))})
    # plot
    ax = sns.boxplot(data=df)
    ax.set_title(title)
    if save_path is not None:
        fig = ax.get_figure()
        fig.savefig(save_path)
        fig.clf()
    return ax
