import numpy as np

df = np.random.rand(20, 300)



'''
Self-Organized Map Dimensionality Reduction
'''
from susi import SOMClustering

som = SOMClustering(n_rows=80, n_columns=80, init_mode_unsupervised='random', n_iter_unsupervised=50000,
                    train_mode_unsupervised='batch', neighborhood_mode_unsupervised='linear',
                    learn_mode_unsupervised='min', distance_metric='euclidean', learning_rate_start=0.5,
                    learning_rate_end=0.05, nbh_dist_weight_mode='pseudo-gaussian', n_jobs=3, verbose=0)

som.fit_transform(df)



'''
tDistributed Stochastic Neighbor Embedding Dimensionality Reduction
'''
from sklearn.manifold import TSNE
tSNE = TSNE(n_components=2, perplexity=40, learning_rate=300, early_exaggeration=24, n_iter=2500,
          n_iter_without_progress=500, min_grad_norm=1e-7, metric='euclidean', init='pca',
          random_state=25, square_distances='legacy')

tSNE.fit_transform(df)