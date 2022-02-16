import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Load the data
df = pd.read_csv("./data.csv")


def assess_correlation(dataframe, column1, column2):
    dataframe.plot(column1, column2, style='o')
    title = column1 + ' vs ' + column2
    filename = column1 + '_vs_' + column2 + '.pdf'
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(title)
    plt.savefig('images/' + filename)
 

sub_df = df[["Car", "Origin"]]
sns_plot = sns.catplot(x="Car", y="Origin", data=sub_df)
sns_plot.fig.suptitle("Sex vs survived.pdf")