import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
import csv

os.getcwd()

# Load the data
df = pd.read_csv("./data.csv")
print(df)

print(df.corr())

def assess_correlation(dataframe, column1, column2):
    dataframe.plot(column1, column2, style='o')
    title = column1 + ' vs ' + column2
    filename = column1 + '_vs_' + column2 + '.pdf'
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(title)
    plt.savefig('./images/' + filename)
 
print(df.head(3))
print(df.columns)
assess_correlation(df, "MPG", "Weight")
sub_df = df[["MPG", "Weight"]]
sns_plot = sns.catplot(x="MPG", y="Weight", data=df)
sns_plot.fig.suptitle("MPG vs Weight.pdf")


def make_scatter_plot(dataframe, column1, column2, column3):
    dataframe.plot.scatter(column1, column2, c=column3, cmap='Wistia')
    title = column1 + " vs " + column2 + " vs " + column3
    filename = f"Scatter_{column1}_vs_{column2}_vs_{column3}.pdf"
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(title)
    plt.show()
    plt.savefig('./images/' + filename)

make_scatter_plot(df, "Model", "MPG", "Weight")