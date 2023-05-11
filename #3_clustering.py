from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=250, init='k-means++', n_init=15, max_iter=400, tol=0.0001, verbose=0,
            random_state=0, copy_x=True, algorithm='elkan')

kmeans.fit_predict(df)
