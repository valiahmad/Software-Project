import numpy as np
from sklearn.cluster import KMeans




def Cluster(arr: np.array):

    kmeans = KMeans(n_clusters=250, init='k-means++', n_init=15, max_iter=400, tol=0.0001, verbose=0,
                    random_state=0, copy_x=True, algorithm='elkan')

    return kmeans.fit_predict(arr)
