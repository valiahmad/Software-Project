import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import pickle
from parameters import Labels
from parameters import date_string


def Cluster(arr: np.array, col: str):

    kmeans = KMeans(n_clusters=Labels, init='k-means++', n_init=15, max_iter=400, tol=0.0001, verbose=0,
                    random_state=0, copy_x=True, algorithm='elkan')
    kmeans.fit(arr)

    df = pd.read_csv('centers_clusters.csv')
    df[col] = kmeans.cluster_centers_.tolist()
    df.to_csv('centers_clusters.csv', index=False)

    pickle.dump(kmeans, open('./' + date_string + '/Models/kmeans-'+ col +'.pickle', "wb"))


    return kmeans.predict(arr)