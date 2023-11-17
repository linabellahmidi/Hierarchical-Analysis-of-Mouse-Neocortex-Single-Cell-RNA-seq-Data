import numpy as np
from math import log2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

x = np.load('X.npy')
max = 0

xl = np.zeros((511,45768))
for i in range(511):
    for j in range(45768):
        xl[i][j] = log2(x[i][j]+1)

#for i in range(511):
#    element = xl[i][0]
#    if (element>max):
#        max = element

#pca_x = PCA().fit(x)
pca_xl = PCA().fit_transform(xl)
#mds = MDS(verbose=1,eps=1e-5)
#mds.fit(xl)
#tsne = TSNE(verbose = 1, perplexity=40)
#z_tsne = tsne.fit_transform(xl)
#kmeans = KMeans(n_clusters=4,n_init=5)
#y = kmeans.fit_predict(pca_xl[:,0:50])
#all_kmeans=[KMeans(n_clusters=i+1,n_init=100) for i in range(8)]
#for i in range(8):
#    all_kmeans[i].fit(xl)

#inertias = [all_kmeans[i].inertia_ for i in range(8)]
hierarchical_cluster = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='single')
labels = hierarchical_cluster.fit_predict(xl)

#pc = np.linspace(1,511,511)
#p_xl = np.cumsum(pca_xl.explained_variance_ratio_)
#print(p_xl)
#plt.scatter(xl[:,0],xl[:,1])
#plt.scatter(pca_xl[:,0],pca_xl[:,1])
#plt.plot(pc,p_xl)
#plt.scatter(mds.embedding_[:,0],mds.embedding_[:,1])
#plt.scatter(z_tsne[:,0],z_tsne[:,1])
#plt.scatter(mds.embedding_[:,0],mds.embedding_[:,1],c=y)
#plt.scatter(pca_xl[:,0],pca_xl[:,1],c=y)
#plt.scatter(z_tsne[:,0],z_tsne[:,1],c=y)
#plt.plot(np.arange(2,9),inertias[1:])
plt.scatter(pca_xl[:,0], pca_xl[:,1], c=labels)

plt.show()

