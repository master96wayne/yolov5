## Script to plot fish species cluster in training set of LifeCLEF2015

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import seaborn as sns
from emulated_som import eSOM as es
from random import randint, sample
from sklearn.mixture import GaussianMixture
from sklearn.cluster import Birch, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score, adjusted_rand_score, homogeneity_completeness_v_measure
from scipy.spatial import procrustes
from scipy.stats import mode
# import image_padding

##%=====================================================================
#%% Get the images
##%=====================================================================

# img_root = 'C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\masters\\Data\\fc_2015_species_samples_padded'
img_root = "C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\data\\images\\fc_2015_species_samples_resized"

sub_folders = os.walk(img_root)

img_list = [os.path.join(root,name)
            for root, dirs, files in sub_folders
            for name in files]

sub_folders = os.listdir(img_root)

num_imgs = 20 # number of images per species (15x20=300 total)

imgs2go_all = []
for folder in sub_folders:
    images = [os.path.join('data\\images\\fc_2015_species_samples_resized', folder, f) for f in os.listdir(os.path.join(img_root, folder))]
    imgs2go = sample(images, num_imgs)
    imgs2go_all += imgs2go
                        
# Saving image list to txt file so I can externally extract features with yolov5            
if not os.path.isfile('C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\species_img_list_resized_300.txt'):
    print('Creating txt with image list...')
    with open('C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\species_img_list_resized_300.txt', 'w') as outfile:
      outfile.write('\n'.join(str(i) for i in imgs2go_all))

# txt = 'C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\species_img_list_resized_300.txt'

# with open(txt, 'r') as f:
    # img_list = f.readlines()

# img_list = [line[:-len('\n')] for line in img_list]

gt = [img.split('\\')[-2] for img in img_list]  

le = LabelEncoder()
le.fit(gt)
gt_enc = le.transform(gt)
# print(gt_enc)

##%=====================================================================      
#%% Perform vectorization and normalization to get 1D embedding of each image. Store in a matrix
##%=====================================================================

# runs_root = 'C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\runs\\detect'
# exps = ['exp17'] # I ran these two experiences to extract the features (exp15 - padding, exp17 - resizing)
runs_root = 'C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\feat_extract'
# runs_root = 'C:\\Users\\leofi\\OneDrive - Universidade de Lisboa\\Documents\\GitHub\\yolov5\\yolov5\\feat_extract\\resized_300'
exps = ['exp', 'exp2']

exps_roots = [os.path.join(runs_root,exp) for exp in exps]
sample_folders = [os.walk(exps_root) for exps_root in exps_roots]
feature_type = 'stage9_SPPF_features.npy'
# feature_type = 'stage'

# Get all dirs with SPPF features for the two exps
features_path_list_all = []
for sample_folder in sample_folders:
    # get the directories for every sample of the output of SPPF layer
    features_path_list = [os.path.join(root,name)
                    for root, dirs, files in sample_folder
                    for name in files if feature_type in name]
    features_path_list_all += features_path_list

features_path_list_all = sorted(features_path_list_all, key=os.path.getmtime)

# Initialize an empty array
embeddings_matrix = np.empty((0, 0), int)

Test = input('Run test version (reduced feature matrix)? (y, n) \n')
if Test == 'y':
    idxs = np.random.randint(len(features_path_list_all), size=np.floor(.1*len(features_path_list_all)).astype(int))
    # print(idxs)
    features_path_list_all = [features_path_list_all[idx] for idx in idxs]
    gt_enc = [gt_enc[idx] for idx in idxs]
    # print(gt_enc)

print('Gathering features...')
for features_path in tqdm.tqdm(features_path_list_all):
    # print(features_path)
    features = np.load(features_path).flatten().reshape(-1,1) # vectorize the feature maps
    # print(features_path)
    # print(features.shape)
    # features /= np.linalg.norm(features) # normalize features
    
    # print(features.shape)
    # Check if the result array is empty
    if embeddings_matrix.size == 0:
        embeddings_matrix = features
    else:
        # Add the column array to the result array
        embeddings_matrix = np.hstack((embeddings_matrix, features))
        

# scaler = MinMaxScaler()
scaler = StandardScaler()
embeddings_matrix = scaler.fit_transform(embeddings_matrix)

##%=====================================================================      
#%% Dimensionality reduction
##%=====================================================================
print('Reducing dimensionality...')

#PCA
#--------------
embeddings_matrix = np.transpose(embeddings_matrix)

n_components = 2

pca = PCA(n_components=n_components)
pca.fit(embeddings_matrix)

# print(pca.explained_variance_ratio_)

X_pca = pca.transform(embeddings_matrix)

# print(embeddings_matrix.shape)
# print(X_pca.shape)



#t-SNE
#--------------
tsne = TSNE(n_components=n_components, perplexity=10, random_state=42)
X_tsne = tsne.fit_transform(embeddings_matrix)

# #NMF
# #--------------
# nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=500)
# X_nmf = nmf.fit_transform(embeddings_matrix)

#========================================================================
# EVALUATION


#1. visualization
#-------------------------------------------------------- 

# print(le.classes_[:len(np.unique(gt_enc[:embeddings_matrix.shape[0]]))])

if n_components == 2:
    # fig, axs = plt.subplots(1,3)
    # # print(axs.shape)
    # # print(X_pca.shape)
    # axs[0].scatter(X_pca[:, 0], X_pca[:,1], c=gt_enc[:embeddings_matrix.shape[0]])
    # axs[0].set_title("PCA")
    # axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
    # axs[1].set_title("t-SNE")
    # axs[2].scatter(X_nmf[:, 0], X_nmf[:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
    # axs[2].set_title("NMF")
    # plt.show(block=False)
    
    fig, axs = plt.subplots(1,2)
    fig.suptitle('Dimesionality reduction')
    # print(axs.shape)
    # print(X_pca.shape)
    axs[0].scatter(X_pca[:, 0], X_pca[:,1], c=gt_enc[:embeddings_matrix.shape[0]])
    axs[0].set_title("PCA")
    scatter = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
    axs[1].set_title("t-SNE")
    plt.legend(handles=scatter.legend_elements()[0], labels=['Abudefduf vaigiensis', 'Amphiprion clarkii'])
    # plt.legend(['Abudefduf vaigiensis', 'Amphiprion clarkii'], loc='upper right')
    plt.show(block=False)
    
elif n_components == 3:
    # fig = plt.figure()
    # gs = fig.add_gridspec(1,3)
    # ax1 = fig.add_subplot(gs[0], projection='3d')
    # ax2 = fig.add_subplot(gs[1], projection='3d')
    # ax3 = fig.add_subplot(gs[2], projection='3d')
    # # print(axs.shape)
    # # print(X_pca.shape)
    # ax1.scatter(X_pca[:, 0], X_pca[:,1], X_pca[:,2], c=gt_enc[:embeddings_matrix.shape[0]])
    # ax1.set_title("PCA")
    # ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:,2], c=gt_enc[:embeddings_matrix.shape[0]])
    # ax2.set_title("t-SNE")
    # ax3.scatter(X_nmf[:, 0], X_nmf[:, 1], X_nmf[:,2], c=gt_enc[:embeddings_matrix.shape[0]])
    # ax3.set_title("NMF")
    # plt.show(block=False) 
    fig = plt.figure()
    gs = fig.add_gridspec(1,2)
    ax1 = fig.add_subplot(gs[0], projection='3d')
    ax2 = fig.add_subplot(gs[1], projection='3d')
    # print(axs.shape)
    # print(X_pca.shape)
    ax1.scatter(X_pca[:, 0], X_pca[:,1], X_pca[:,2], c=gt_enc[:embeddings_matrix.shape[0]])
    ax1.set_title("PCA")
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:,2], c=gt_enc[:embeddings_matrix.shape[0]])
    ax2.set_title("t-SNE")
    plt.show(block=False) 

#PCA
#--------------------------------------------------------

# Perform PCA
n_comp_pca = 100
pca = PCA(n_components=n_comp_pca)
X_pca = pca.fit_transform(embeddings_matrix)

# Compute explained variance ratio
explained_var_ratio = pca.explained_variance_ratio_

# Plot cumulative explained variance ratio
cumulative_var_ratio = np.cumsum(explained_var_ratio)
plt.figure()
plt.plot(range(1, len(cumulative_var_ratio)+1), cumulative_var_ratio, marker='o')
plt.axhline(y=0.95, color='grey', linestyle='--')
plt.xticks(np.arange(2, n_comp_pca+1, step=2))
plt.grid(color='0.95')
plt.text(1.1, 0.9, '95% cut-off threshold', color = 'black', fontsize=12)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Cumulative Explained Variance')
plt.tight_layout()
plt.show(block=False)

# Plot scree plot
plt.figure()
plt.plot(range(1, len(explained_var_ratio)+1), explained_var_ratio, marker='o')
plt.xticks(np.arange(2, n_comp_pca+1, step=2))
plt.grid(color='0.95')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA: Scree Plot')
plt.tight_layout()
plt.show(block=False)

# Function to perform PCA with n-Components
def transform_pca(X, n):

    pca = PCA(n_components=n)
    pca.fit(X)
    X_new = pca.inverse_transform(pca.transform(X))

    return X_new

# Plot the components
rows = 4
cols = 4
comps = 1

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

fig, axes = plt.subplots(rows, 
                         cols, 
                         figsize=(12,12), 
                         sharex=True, 
                         sharey=True)

for row in range(rows):
    for col in range(cols):
        
        X_new = transform_pca(embeddings_matrix, comps)
        ax = sns.scatterplot(x=embeddings_matrix[:, 0], 
                             y=embeddings_matrix[:, 1], 
                             ax=axes[row, col], 
                             color='grey', 
                             alpha=.3)
        ax = sns.scatterplot(x=X_new[:, 0], 
                             y=X_new[:, 1], 
                             ax=axes[row, col], 
                             color='black')
        ax.set_title(f'PCA Components: {comps}');

        comps += np.ceil(n_comp_pca/16).astype(int)
        if comps > n_comp_pca:
            comps = n_comp_pca
        
plt.tight_layout()
plt.show(block=False)
# plt.savefig('pcavisualize_2.png', dpi=300)

# Reconstruct the data using a reduced number of principal components
X_reconstructed = pca.inverse_transform(X_pca)

# Compute reconstruction error
reconstruction_error = np.mean(np.square(embeddings_matrix - X_reconstructed))

print("Reconstruction Error:", reconstruction_error)


#t-SNE
#--------------------------------------------------------  

#2. preservation of global structure (Procrustes distance)
#--------------------------------------------------------
# Calculate pairwise distances in the original and reduced spaces
dist_original = pairwise_distances(embeddings_matrix, metric='euclidean')
dist_reduced_pca = pairwise_distances(X_pca, metric='euclidean')
dist_reduced_tsne = pairwise_distances(X_tsne, metric='euclidean')

# Measure Procrustes distance between the two distance matrices
_,_,procrustes_dist_pca = procrustes(dist_original, dist_reduced_pca)
_,_,procrustes_dist_tsne = procrustes(dist_original, dist_reduced_tsne)
print("Procrustes distance:")
print(f"  - PCA:   {procrustes_dist_pca}")
print(f"  - t-SNE: {procrustes_dist_tsne}")

#3. stability analysis
#--------------------------------------------------------
Join = input('Perform t-SNE stability analysis?(y, n)\n')

if Join == 'y':
    # Perform multiple runs of t-SNE with different parameters
    perplexities = range(5,55,5)
    tsne_results = []
    
    # print(len(perplexities))

    print('t-SNE stability analysis...')
    print('Running t-SNE for multiple perplexities')
    for perplexity in tqdm.tqdm(perplexities):
        tsne_result = TSNE(n_components=n_components, perplexity=perplexity, random_state=42).fit_transform(embeddings_matrix)
        # print(tsne_result.shape)
        tsne_results.append(tsne_result)
    
    # print(tsne_results)

    procrustes_dists = np.empty((len(perplexities), len(perplexities)))

    print('Computing procrustes distances...')
    for i in range(len(perplexities)):
        print(f'Run {i}/{len(perplexities)}')
        for j in tqdm.tqdm(range(len(perplexities))):
            # Measure stability using Procrustes distance
            _,_,procrustes_dist = procrustes(tsne_results[i], tsne_results[j])
            procrustes_dists[i,j] = procrustes_dist
    
    # Compute the mean of each row
    pa_mean = np.mean(procrustes_dists, axis=1)

    # Compute the standard deviation of each row
    pa_std = np.std(procrustes_dists, axis=1)

            
    print(perplexities)
    print(procrustes_dists.shape)
    df = pd.DataFrame(columns=['perplexity', 'pa_mean', 'pa_std'])
    df.pa_mean = pa_mean
    df.pa_std = pa_std
    df.perplexity = perplexities
    
    plt.figure()
    ax = sns.heatmap(procrustes_dists, linewidth=0.5, xticklabels=perplexities, yticklabels=perplexities)
    plt.xlabel('perplexity')
    plt.ylabel('perplexity')
    plt.title('Disparity between runs of t-SNE for different perplexities')
    plt.show(block=False)
    
    print(df)
    # print(df.to_latex())
    
    n_plots = input('How many plots? (3/ a (all))\n')

    if n_plots == '3':
        fig, axs = plt.subplots(3,1,sharex=True, sharey=True)
        axs[0].scatter(tsne_results[0][:, 0], tsne_results[0][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
        axs[0].set_title(f"perplexity = {perplexities[0]}")
        axs[1].scatter(tsne_results[4][:, 0], tsne_results[4][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
        axs[1].set_title(f"perplexity = {perplexities[4]}")
        axs[2].scatter(tsne_results[9][:, 0], tsne_results[9][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
        axs[2].set_title(f"perplexity = {perplexities[9]}")
        plt.legend(handles=scatter.legend_elements()[0], labels=['Abudefduf vaigiensis', 'Amphiprion clarkii'])
        plt.show(block=False) 

    elif n_plots == 'a':
        if n_components == 2:
            fig, axs = plt.subplots(5,2,sharex=True, sharey=True)
            axs[0,0].scatter(tsne_results[0][:, 0], tsne_results[0][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[0,0].set_title(f"perplexity = {perplexities[0]}")
            axs[0,1].scatter(tsne_results[1][:, 0], tsne_results[1][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[0,1].set_title(f"perplexity = {perplexities[1]}")
            axs[1,0].scatter(tsne_results[2][:, 0], tsne_results[2][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[1,0].set_title(f"perplexity = {perplexities[2]}")
            axs[1,1].scatter(tsne_results[3][:, 0], tsne_results[3][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[1,1].set_title(f"perplexity = {perplexities[3]}")
            axs[2,0].scatter(tsne_results[4][:, 0], tsne_results[4][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[2,0].set_title(f"perplexity = {perplexities[4]}")
            axs[2,1].scatter(tsne_results[5][:, 0], tsne_results[5][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[2,1].set_title(f"perplexity = {perplexities[5]}")
            axs[3,0].scatter(tsne_results[6][:, 0], tsne_results[6][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[3,0].set_title(f"perplexity = {perplexities[6]}")
            axs[3,1].scatter(tsne_results[7][:, 0], tsne_results[7][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[3,1].set_title(f"perplexity = {perplexities[7]}")
            axs[4,0].scatter(tsne_results[8][:, 0], tsne_results[8][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[4,0].set_title(f"perplexity = {perplexities[8]}")
            axs[4,1].scatter(tsne_results[9][:, 0], tsne_results[9][:, 1], c=gt_enc[:embeddings_matrix.shape[0]])
            axs[4,1].set_title(f"perplexity = {perplexities[9]}")
            plt.show(block=False)
            
        elif n_components == 3:
            fig = plt.figure()
            gs = fig.add_gridspec(5,2)
            ax1 = fig.add_subplot(gs[0,0], projection='3d')
            ax2 = fig.add_subplot(gs[0,1], projection='3d')
            ax3 = fig.add_subplot(gs[1,0], projection='3d')
            ax4 = fig.add_subplot(gs[1,1], projection='3d')
            ax5 = fig.add_subplot(gs[2,0], projection='3d')
            ax6 = fig.add_subplot(gs[2,1], projection='3d')
            ax7 = fig.add_subplot(gs[3,0], projection='3d')
            ax8 = fig.add_subplot(gs[3,1], projection='3d')
            ax9 = fig.add_subplot(gs[4,0], projection='3d')
            ax10 = fig.add_subplot(gs[4,1], projection='3d')
            ax1.scatter(tsne_results[0][:, 0], tsne_results[0][:, 1], tsne_results[0][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax1.set_title(f"perplexity = {perplexities[0]}")
            ax2.scatter(tsne_results[1][:, 0], tsne_results[1][:, 1], tsne_results[1][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax2.set_title(f"perplexity = {perplexities[1]}")
            ax3.scatter(tsne_results[2][:, 0], tsne_results[2][:, 1], tsne_results[2][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax3.set_title(f"perplexity = {perplexities[2]}")
            ax4.scatter(tsne_results[3][:, 0], tsne_results[3][:, 1], tsne_results[3][:, 2],  c=gt_enc[:embeddings_matrix.shape[0]])
            ax4.set_title(f"perplexity = {perplexities[3]}")
            ax5.scatter(tsne_results[4][:, 0], tsne_results[4][:, 1], tsne_results[4][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax5.set_title(f"perplexity = {perplexities[4]}")
            ax6.scatter(tsne_results[5][:, 0], tsne_results[5][:, 1], tsne_results[5][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax6.set_title(f"perplexity = {perplexities[5]}")
            ax7.scatter(tsne_results[6][:, 0], tsne_results[6][:, 1], tsne_results[6][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax7.set_title(f"perplexity = {perplexities[6]}")
            ax8.scatter(tsne_results[7][:, 0], tsne_results[7][:, 1], tsne_results[7][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax8.set_title(f"perplexity = {perplexities[7]}")
            ax9.scatter(tsne_results[8][:, 0], tsne_results[8][:, 1], tsne_results[8][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax9.set_title(f"perplexity = {perplexities[8]}")
            ax10.scatter(tsne_results[9][:, 0], tsne_results[9][:, 1], tsne_results[9][:, 2], c=gt_enc[:embeddings_matrix.shape[0]])
            ax10.set_title(f"perplexity = {perplexities[9]}")
            plt.show(block=False)
            
    
##%=====================================================================      
#%% Clustering methods
##%=====================================================================
print('Clustering without dimensionality reduction...')
n_clusters = len(np.unique(gt_enc[:embeddings_matrix.shape[0]]))

# # Emulated SOM
# df, centroids = es(embeddings_matrix,n_clusters)
# # df['true'] = gt[:embeddings_matrix.shape[0]]

# print(df.head())
# print(df.shape)

colors = []

for i in range(n_clusters):
    colors.append('#%06X' % randint(0, 0xFFFFFF))

# for label in range(n_clusters):
    # df_cluster_filter = df[df['cluster'] == label]
    # plt.scatter(df_cluster_filter.x, df_cluster_filter.y, color=colors[label])
# plt.show(block=False)

# Expectation Maximization (Gaussian mixture)
# gm = GaussianMixture(n_components=n_clusters, random_state=0).fit(embeddings_matrix)
# classes = gm.predict(embeddings_matrix)

# BIRCH 
print('Applying BIRCH clustering...')
# tic
brc = Birch(n_clusters=n_clusters, threshold=0.2).fit(embeddings_matrix)
brc_classes = brc.predict(embeddings_matrix)
# print(toc)

#MiniBatchKMeans
print('Applying MiniBatchKMeans...')
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6, n_init='auto').fit(embeddings_matrix)
kmeans_classes = kmeans.predict(embeddings_matrix)

#DBSCAN
print('Applying DBSCAN...')
dbscan = DBSCAN(eps=3, min_samples=2).fit(embeddings_matrix)
dbscan_classes = dbscan.labels_

#Evaluation

#0. visualization
fig, axs = plt.subplots(2, 2, sharey=True)
fig.suptitle('Clusterings')
fig.legend(le.classes_)
# ax1.scatter(embeddings_matrix[:,0], embeddings_matrix[:,1], c=gt_enc[:embeddings_matrix.shape[0]])
# ax2.scatter(embeddings_matrix[:,0], embeddings_matrix[:,1], c=classes)
axs[0,0].scatter(X_pca[:,0], X_pca[:,1], c=gt_enc[:embeddings_matrix.shape[0]])
axs[0,0].set_title('Ground Truth')
axs[0,1].scatter(X_pca[:,0], X_pca[:,1], c=brc_classes)
axs[0,1].set_title('BIRCH')
axs[1,0].scatter(X_pca[:,0], X_pca[:,1], c=kmeans_classes)
axs[1,0].set_title('MiniBatchKMeans')
axs[1,1].scatter(X_pca[:,0], X_pca[:,1], c=dbscan_classes)
axs[1,1].set_title('DBSCAN')
plt.show(block=False)

#1. silhouette_score
#--------------------------------------------------------
try:
    s_birch = silhouette_score(embeddings_matrix, brc_classes)
except:
    print('not able to compute')
    s_tsne_birch = 0
    
s_kmeans = silhouette_score(embeddings_matrix, kmeans_classes)

#2. ARI score   (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------
ari_birch = adjusted_rand_score(gt_enc[:embeddings_matrix.shape[0]], brc_classes)
ari_kmeans = adjusted_rand_score(gt_enc[:embeddings_matrix.shape[0]], kmeans_classes)

#3. Homogeneity, Completeness, and V-measure  (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------
homo_birch, comp_birch, v_measure_birch = homogeneity_completeness_v_measure(gt_enc[:embeddings_matrix.shape[0]], brc_classes)
homo_kmeans, comp_kmeans, v_measure_kmeans = homogeneity_completeness_v_measure(gt_enc[:embeddings_matrix.shape[0]], kmeans_classes)

#4. Cluster purity    (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------
# Compute cluster purity
def cluster_purity(y_true, labels):
    clusters = np.unique(labels)
    total_samples = len(y_true)
    purity = 0

    for cluster in clusters:
        cluster_indices = np.where(labels == cluster)[0]
        cluster_labels = [y_true[idx] for idx in cluster_indices]
        majority_label = mode(cluster_labels, keepdims=True)[0][0]
        cluster_size = len(cluster_indices)
        purity += cluster_size * np.sum(cluster_labels == majority_label)

    purity /= total_samples

    return purity

# Compute cluster purity
purity_brc = cluster_purity(gt_enc[:embeddings_matrix.shape[0]], brc_classes)
purity_kmeans = cluster_purity(gt_enc[:embeddings_matrix.shape[0]], kmeans_classes)


#######################
#Clusters after Dim Red
# BIRCH 
threshold = 0.2
print('Applying BIRCH clustering...')
brc = Birch(n_clusters=n_clusters, threshold=threshold).fit(X_tsne)
brc_tsne_classes = brc.predict(X_tsne)
# print(toc)

#MiniBatchKMeans
print('Applying MiniBatchKMeans...')
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6, n_init='auto').fit(X_tsne)
kmeans_tsne_classes = kmeans.predict(X_tsne)
#_____________________________________________
# BIRCH 
print('Applying BIRCH clustering...')
# tic
brc = Birch(n_clusters=n_clusters, threshold=threshold).fit(X_pca)
brc_pca_classes = brc.predict(X_pca)
# print(toc)

#MiniBatchKMeans
print('Applying MiniBatchKMeans...')
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=6, n_init='auto').fit(X_pca)
kmeans_pca_classes = kmeans.predict(X_pca)

#========================================================================
# EVALUATION

#1. silhouette_score
#--------------------------------------------------------
try:
    s_tsne_birch = silhouette_score(X_tsne, brc_tsne_classes)
except:
    print('not able to compute')
    s_tsne_birch = 0
    
s_tsne_kmeans = silhouette_score(X_tsne, kmeans_tsne_classes)

try:
    s_pca_birch = silhouette_score(X_pca, brc_pca_classes)
except:
    print('not able to compute')
    s_pca_birch = 0
    
s_pca_kmeans = silhouette_score(X_pca, kmeans_pca_classes)

#2. ARI score   (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------
ari_tsne_birch = adjusted_rand_score(gt_enc[:X_tsne.shape[0]], brc_tsne_classes)
ari_tsne_kmeans = adjusted_rand_score(gt_enc[:X_tsne.shape[0]], kmeans_tsne_classes)
ari_pca_birch = adjusted_rand_score(gt_enc[:X_tsne.shape[0]], brc_pca_classes)
ari_pca_kmeans = adjusted_rand_score(gt_enc[:X_tsne.shape[0]], kmeans_pca_classes)

#3. Homogeneity, Completeness, and V-measure  (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------
homo_tsne_birch, comp_tsne_birch, v_measure_tsne_birch = homogeneity_completeness_v_measure(gt_enc[:X_tsne.shape[0]], brc_tsne_classes)
homo_tsne_kmeans, comp_tsne_kmeans, v_measure_tsne_kmeans = homogeneity_completeness_v_measure(gt_enc[:X_tsne.shape[0]], kmeans_tsne_classes)
homo_pca_birch, comp_pca_birch, v_measure_pca_birch = homogeneity_completeness_v_measure(gt_enc[:X_tsne.shape[0]], brc_pca_classes)
homo_pca_kmeans, comp_pca_kmeans, v_measure_pca_kmeans = homogeneity_completeness_v_measure(gt_enc[:X_tsne.shape[0]], kmeans_pca_classes)

#4. Cluster purity    (REVER PORQUE AS LABELS PODEM VIR TROCADAS!!)
#--------------------------------------------------------

# Compute cluster purity
purity_tsne_brc = cluster_purity(gt_enc[:X_tsne.shape[0]], brc_tsne_classes)
purity_tsne_kmeans = cluster_purity(gt_enc[:X_tsne.shape[0]], kmeans_tsne_classes)
purity_pca_brc = cluster_purity(gt_enc[:X_tsne.shape[0]], brc_pca_classes)
purity_pca_kmeans = cluster_purity(gt_enc[:X_tsne.shape[0]], kmeans_pca_classes)

# Dataframe with scores
df = pd.DataFrame(columns=['method','sc','ari','homo','comp','v_measure','purity'])
df.method = ['BIRCH', 'MiniBatchKMeans', 't-SNE+BIRCH', 't-SNE+MiniBatchKMeans', 'PCA+BIRCH', 'PCA+MiniBatchKMeans']
df.sc = [s_birch, s_kmeans, s_tsne_birch, s_tsne_kmeans, s_pca_birch, s_pca_kmeans]
df.ari = [ari_birch, ari_kmeans, ari_tsne_birch, ari_tsne_kmeans, ari_pca_birch, ari_pca_kmeans]
df.homo = [homo_birch, homo_kmeans, homo_tsne_birch, homo_tsne_kmeans, homo_pca_birch, homo_pca_kmeans]
df.comp = [comp_birch, comp_kmeans, comp_tsne_birch, comp_tsne_kmeans, comp_pca_birch, comp_pca_kmeans]
df.v_measure = [v_measure_birch, v_measure_kmeans, v_measure_tsne_birch, v_measure_tsne_kmeans, v_measure_pca_birch, v_measure_pca_kmeans]
df.purity = [purity_brc, purity_kmeans, purity_tsne_brc, purity_tsne_kmeans, purity_pca_brc, purity_pca_kmeans]

print('Clustering evaluation metrics:')
print(df)
# print(df.to_latex())

# visualization
#--------------------------------------------------------
fig, axs = plt.subplots(2, 2, sharey=True)
fig.suptitle('Clusterings after t-SNE')
fig.legend(le.classes_)
# ax1.scatter(X_tsne[:,0], X_tsne[:,1], c=gt_enc[:X_tsne.shape[0]])
# ax2.scatter(X_tsne[:,0], X_tsne[:,1], c=classes)
axs[0,0].scatter(X_tsne[:,0], X_tsne[:,1], c=gt_enc[:X_tsne.shape[0]])
axs[0,0].set_title('Ground Truth')
axs[0,1].scatter(X_tsne[:,0], X_tsne[:,1], c=brc_tsne_classes)
axs[0,1].set_title(f'BIRCH \n (sil={s_tsne_birch:.2f}|ARI={ari_tsne_birch:.2f}|pur={purity_tsne_brc:.2f})')
axs[1,1].scatter(X_tsne[:,0], X_tsne[:,1], c=kmeans_tsne_classes)
axs[1,1].set_title(f'KMeans \n (sil={s_tsne_kmeans:.2f}|ARI={ari_tsne_kmeans:.2f}|pur={purity_tsne_kmeans:.2f})')
plt.show(block=False)

fig, axs = plt.subplots(2, 2, sharey=True)
fig.suptitle('Clusterings after PCA')
fig.legend(le.classes_)
# ax1.scatter(X_pca[:,0], X_pca[:,1], c=gt_enc[:X_pca.shape[0]])
# ax2.scatter(X_pca[:,0], X_pca[:,1], c=classes)
axs[0,0].scatter(X_pca[:,0], X_pca[:,1], c=gt_enc[:X_pca.shape[0]])
axs[0,0].set_title('Ground Truth')
axs[0,1].scatter(X_pca[:,0], X_pca[:,1], c=brc_pca_classes)
axs[0,1].set_title(f'BIRCH \n (sil={s_pca_birch:.2f}|ARI={ari_pca_birch:.2f}|pur={purity_pca_brc:.2f})')
axs[1,1].scatter(X_pca[:,0], X_pca[:,1], c=kmeans_pca_classes)
axs[1,1].set_title(f'KMeans \n (sil={s_pca_kmeans:.2f}|ARI={ari_pca_kmeans:.2f}|pur={purity_pca_kmeans:.2f})')
plt.show(block=False)

plt.show()