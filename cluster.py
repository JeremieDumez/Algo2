import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("./data.csv")

data.head()

data = data.loc[:, ['Cylinders', 'Horsepower']]
data.head(2)

X = data.values

#sns.scatterplot(X[:,0], X[:, 1])
#filename = f"Test.pdf"
#plt.xlabel('Income')
#plt.ylabel('Loan')
#plt.show()
#plt.savefig('./images/' + filename)

def calculate_cost(X, centroids, cluster):
  sum = 0
  for i, val in enumerate(X):
    sum += np.sqrt((centroids[int(cluster[i]), 0]-val[0])**2 +(centroids[int(cluster[i]), 1]-val[1])**2)
  return sum

def kmeans(X, k):
    diff = 1
    cluster = np.zeros(X.shape[0])
    centroids = data.sample(n=k).values
    while diff:
        for i, row in enumerate(X):
            mn_dist = float('inf')
        # dist of the point from all centroids
        for idx, centroid in enumerate(centroids):
            d = np.sqrt((centroid[0]-row[0])**2 + (centroid[1]-row[1])**2)
            # store closest centroid
            if mn_dist > d:
               mn_dist = d
               cluster[i] = idx
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        # if centroids are same then leave
        #print("-------------")
        #print("centroid")
        #print(centroids)
        #print("New centroid - ")
        #print(new_centroids)
        if np.count_nonzero(centroids-new_centroids) == 0:
            print("if equal")
            diff = 0
        else:
            print("leave")
            centroids = new_centroids
            
    return centroids, cluster

print("aegrg")

cost_list = []
for k in range(1, 10):
    centroids, cluster = kmeans(X, k)
    # WCSS (Within cluster sum of square)
    cost = calculate_cost(X, centroids, cluster)
    cost_list.append(cost)

#print("h")
#sns.lineplot(x=range(1,10), y=cost_list, marker='o')
#filename2 = f"Test2.pdf"
#plt.xlabel('k')
#plt.ylabel('WCSS')
#plt.savefig('./images/' + filename2)
#plt.show()

k = 4
print("aegrg")
centroids, cluster = kmeans(X, k)

filename = f"Test.pdf"
sns.scatterplot(X[:,0], X[:, 1], hue=cluster)
sns.scatterplot(centroids[:,0], centroids[:, 1], s=100, color='y')
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()
plt.savefig('./images/' + filename)