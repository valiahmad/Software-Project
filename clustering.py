import numpy as np
import json
from sklearn.cluster import KMeans
from main import Labels



def Cluster(arr: np.array, col: str):

    kmeans = KMeans(n_clusters=Labels, init='k-means++', n_init=15, max_iter=400, tol=0.0001, verbose=0,
                    random_state=0, copy_x=True, algorithm='elkan')
    kmeans.fit(arr)
    with open('centers_clusters.json', 'a') as file:
        json.dump({col:kmeans.cluster_centers_},file)
    file.close()
    return kmeans.predict(arr)