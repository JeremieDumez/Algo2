import matplotlib.pyplot as plt
import numpy as np
import math
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pandas as pd
from IPython import display
from graphviz import Graph

dataframe = pd.read_csv("data.csv")

nb_cars = 15
threshold = 500

print(nb_cars)

print(dataframe.info())
print(dataframe.columns)
print(dataframe.head(10))

car_id = 1
print(f"---\nall information car {car_id}\n---")
print(dataframe.loc[car_id])


def compute_dissimilarity(car_1_id, car_2_id):
    car_1_mpg = dataframe.loc[car_1_id][1]
    car_2_mpg = dataframe.loc[car_2_id][1]

    car_1_weight = dataframe.loc[car_1_id][5]
    car_2_weight = dataframe.loc[car_2_id][5]

    car_1_origin = dataframe.loc[car_1_id][8]
    car_2_origin = dataframe.loc[car_2_id][8]

    if car_1_origin == car_2_origin:
        dissimilarity_origin = 0
    else:
        dissimilarity_origin = 1

    dissimilarity = math.sqrt((car_1_mpg-car_2_mpg)**2+(car_1_weight-car_2_weight)**2+dissimilarity_origin)

    print("----")
    car_1_name = dataframe.loc[car_1_id]["Car"]
    car_2_name = dataframe.loc[car_2_id]["Car"]
    print(f"plyr 1 {car_1_name}, plyr 2 {car_2_name}, dissimilarity: {dissimilarity}")
    return dissimilarity

dissimilarity_matrix = np.zeros((nb_cars, nb_cars))
print("compute dissimilarities")
for car_1_id in range(nb_cars):
    for car_2_id in range(nb_cars):
        dissimilarity = compute_dissimilarity(car_1_id, car_2_id)
        dissimilarity_matrix[car_1_id, car_2_id] = dissimilarity

print("dissimilarity matrix")
print(dissimilarity_matrix)

dot = Graph(comment='Graph of cars',
            strict=True)
for car_id in range(nb_cars):
    car_name = dataframe.loc[car_id][0]
    dot.node(car_name)

for car_1_id in range(nb_cars):
    for car_2_id in range(nb_cars):
        if not car_1_id == car_2_id:
            car_1_name = dataframe.loc[car_1_id][0]
            car_2_name = dataframe.loc[car_2_id][0]
            if dissimilarity_matrix[car_1_id, car_2_id] > threshold:
                dot.edge(car_1_name,
                         car_2_name,
                         color='darkolivegreen4',
                         penwidth='1.1')

dot.attr(label=f"threshold {threshold}", fontsize='20')
graph_name = f"images/complex_data_threshold_{threshold}"
dot.render(graph_name)

dot