# Implementation of Emulated SOM for large datasets

from sklearn.cluster import KMeans
from sklearn.manifold import MDS
import math
import numpy as np
import pandas as pd

def eSOM(dset,k):
    # Compute SOM grid size
    dlen = dset.shape[0]
    munits = math.ceil(5*dlen**0.54321)

    # 1st level k-means (big-K)
    K = munits#
    kmeans_1 = KMeans(n_clusters=K, random_state=0, n_init='auto').fit(dset)
    proto = kmeans_1.cluster_centers_

    # 2nd level k-means (small-k)
    k = k# adjust to the number of species in Pedro's table
    kmeans_2 = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(proto)
    labels = kmeans_2.labels_
    centroids = kmeans_2.cluster_centers_   
    print(len(labels))
    
    # MDS mapping
    proto_cent = np.vstack((proto, centroids))
    mds = MDS(n_components=2, random_state=0, normalized_stress='auto')
    mapping = mds.fit_transform(proto_cent)
    # print(proto_cent.shape)
    # print(mapping.shape)
    
    mapped_proto = mapping[:len(labels),:]
    mapped_centroids = mapping[len(labels):,:]
    
    df = pd.DataFrame(mapped_proto, columns=['x','y'])
    print(df.shape)
    df['cluster'] = labels
    
    return df, mapped_centroids