import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random

data = pd.read_csv("./data.csv")

data = data.loc[:, ['MPG', 'Weight']]

X = data.values

m=X.shape[0]
n=X.shape[1] 
n_iter=15

K=3

centroids=np.array([]).reshape(n,0) 

for k in range(K):
    centroids=np.c_[centroids,X[random.randint(0,m-1)]]
    
output={}

euclid=np.array([]).reshape(m,0)

for k in range(K):
       dist=np.sum((X-centroids[:,k])**2,axis=1)
       euclid=np.c_[euclid,dist]

minimum=np.argmin(euclid,axis=1)+1

cent={}
for k in range(K):
    cent[k+1]=np.array([]).reshape(2,0)

for k in range(m):
    cent[minimum[k]]=np.c_[cent[minimum[k]],X[k]]
for k in range(K):
    cent[k+1]=cent[k+1].T

for k in range(K):
    centroids[:,k]=np.mean(cent[k+1],axis=0)

for i in range(n_iter):
    euclid=np.array([]).reshape(m,0)
    for k in range(K):
        dist=np.sum((X-centroids[:,k])**2,axis=1)
        euclid=np.c_[euclid,dist]
    C=np.argmin(euclid,axis=1)+1
    cent={}
    for k in range(K):
        cent[k+1]=np.array([]).reshape(2,0)
    for k in range(m):
        cent[C[k]]=np.c_[cent[C[k]],X[k]]
    for k in range(K):
        cent[k+1]=cent[k+1].T
    for k in range(K):
        centroids[:,k]=np.mean(cent[k+1],axis=0)
    final=cent
    plt.scatter(X[:,0],X[:,1])
    plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
    plt.title('Data.csv')
    
    for k in range(K):
        plt.scatter(final[k+1][:,0],final[k+1][:,1])
    filename = f"Cluster.pdf"
    plt.scatter(centroids[0,:],centroids[1,:],s=300,c='yellow')
    plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
    plt.xlabel('MPG')
    plt.ylabel('Weight')
    plt.savefig('./images/' + filename)
    plt.show()