import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv("boston_data.csv")
#print(dataset)

# : means select all rows, price clomun
#array saves all rows and 2 features from dataset
x = dataset.iloc[:, [12,13]].values


# list to store error rate
Error = []
for i in range(1, 13):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

plt.plot(range(1,13),Error,marker='o')
plt.xlabel("Number Of clusters")
plt.ylabel("Error")
plt.show()

kmeans3 = KMeans(n_clusters=3)
y_kmeans=kmeans3.fit_predict(x)
print(y_kmeans)
print(kmeans3.cluster_centers_)

plt.scatter(x[:, 0], x[:,1] , c=y_kmeans, cmap='rainbow')
plt.show()


