# Python Libraies
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans , AgglomerativeClustering
from collections import Counter
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
import pydotplus
from sklearn import tree
from sklearn.metrics import precision_score , recall_score

### PART 1 ###
print ("PART 1\n")
#Loading the data
sample_docs = []
with open("top1000_movie_summaries.tsv") as fi:
    tsvReader = csv.reader(fi, delimiter='\t')
    for i, (title, plot) in enumerate(tsvReader):
        sample_docs.append(plot)

#Tfidf Vectorizer
vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = vectorizer.fit_transform(sample_docs)
tfidf_array = tfidf_matrix.toarray()


#Displaying Top 10 Terms and their Tfidf Scores
terms_list = [(score, term)  for score, term in zip(tfidf_array[0], vectorizer.get_feature_names()) if score > 0]
terms_list = sorted(terms_list, key=lambda x: x[0],reverse = True)
i = 0
print ("Displaying Top 10 Terms and their Tfidf Scores")
for tuples in terms_list:
    i+=1
    print (tuples)
    if i>=10:
        break

#Using K-Means Algorithm to compute clusters
kmeans = KMeans(n_clusters=20, random_state=0)
kmeans.fit(tfidf_matrix)

#Exploring Cluster Sizes
cluster_labels =  kmeans.labels_
abc =  Counter(cluster_labels)
print ("\nSize of K-Means Clusters :")
for key,value in abc.items():
    print (str(key) + " : " + str(value))

#Using Hierarchical clustering algorithm with cosine similarity
hier_clus = AgglomerativeClustering(n_clusters=20, affinity='cosine', linkage='average')
hier_clus.fit(tfidf_array)

hier_cluster_labels = hier_clus.labels_
pqr = Counter(hier_cluster_labels)
print ("\nSize of Hierarchical clustering clusters :")
for key,value in pqr.items():
    print (str(key) + " : " + str(value))

#Comparision of K-Means Clustering and Hierarchical clustering
'''
#1) In Hierarchical clustering one cluster contains almost 90% of the records while this is not the case of K- Means
#2) Except from the Cluster '0' all other clusters have almost same number of records while records distribution in K-Means is quite uneven
'''

#Top Terms per Cluster
print("\nTop terms per cluster:")
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(20):
    term_list = []
    for ind in order_centroids[i, :5]:
        term_list.append(terms[ind])
    print  ("Cluster "+str(i)+ " : " +str(term_list))




### PART 2 ###
print ("\nPART 2")
#Finding 10 Nearest Neighbours
neigh = NearestNeighbors(n_neighbors=10)
neigh.fit(tfidf_array)


docs = {}
with open("top1000_movie_summaries.tsv") as fi:
    tsvReader = csv.reader(fi, delimiter='\t')
    for i, (title, plot) in enumerate(tsvReader):
        docs[i] = {'Title':title,'Plot':plot}

print ("\nTop 10 Similar Movies.")
def get_nn_movie_names(movie_index):
    tfidf_array[movie_index]
    print("{}\n===".format(docs[movie_index]['Title']))
    for idx, dist in zip(np.nditer(neigh.kneighbors(tfidf_array[movie_index])[1]), 
                         np.nditer(neigh.kneighbors(tfidf_array[movie_index])[0])):
        if idx != movie_index:
            print(u'{}'.format(docs[int(idx)]['Title']))

get_nn_movie_names(34)
print 
get_nn_movie_names(26)
print 
get_nn_movie_names(199)

#Performance Analysis
'''
Performance of this KNN Classifier is not good.
Its performance can be improved by increasing the data and by removing stopwords and by using techniques like lemmatization and stemming.
Tuning better features will also improve the performance.
'''




### PART 3 ###
print ("\nPART 3")

#Reading the data
df = pd.read_csv('Consumer_complaints.csv')
df1 = df[['Product', 'Sub-product', 'Issue', 'Sub-issue']]
train_idxs  = np.random.choice(27000, 3000, replace=False)
df_train = df1.ix[train_idxs]
df_train.fillna( 'NA', inplace = True )
x_df_train = df_train.T.to_dict().values()

#DictVectorizer
DV = DictVectorizer(sparse = False)
vec_x_df_train = DV.fit_transform(x_df_train)
target = df[['Company response to consumer']].ix[train_idxs]
target_names = target.iloc[:,0].unique()


#Building a decision tree
dtc  = DecisionTreeClassifier(random_state=0)
dtc.fit(vec_x_df_train,target)

#Drawing the Decision Tree
dot_data = tree.export_graphviz(dtc, out_file="tree.dot", 
                         feature_names=DV.feature_names_,  
                         class_names=target_names,  
                         filled=True, rounded=True,  
                         special_characters=True) 

graph = pydotplus.graph_from_dot_file("tree.dot")  
graph.write_png("tree.png")

#Precision and recall scores
train_idxs  = np.random.choice(27000, 6000, replace=False)
df_train = df1.ix[train_idxs]
df_train.fillna( 'NA', inplace = True )
target = df[['Company response to consumer']].ix[train_idxs]

X_test = df_train.tail(3000)
Y_test = target.tail(3000)

X_train = df_train.head(3000)
Y_train  = target.head(3000)
X_train_todict = X_train.T.to_dict().values()
X_test_todict = X_test.T.to_dict().values()
X_train_vec = DV.fit_transform(X_train_todict)
X_test_vec = DV.fit_transform(X_test_todict)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_vec,Y_train)
predict = clf.predict(X_test_vec)

print ("\nPrecision Score : " +str(precision_score(Y_test.as_matrix(), predict, average='micro')  ))
print ("Recall Score : " + str(recall_score(Y_test.as_matrix(), predict, average='micro')  ))

