import turicreate as graphlab
import matplotlib.pyplot as plt
graphlab.config.set_runtime_config('TURI_NUM_GPUS', 4)
graphlab.visualization.set_target('gui')

# DATA_DIR='dev/ml-washington-univ/course1/Week 3/amazon_baby.gl/'
DATA_DIR='people_wiki.gl/'
people = graphlab.SFrame(DATA_DIR)
people.head(3)
people.shape
# Explore Dataset
obama = people[people['name'] == "Barack Obama"]
obama['text']

obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])
obama['word_count']
# Sort word count of Obama
obama_word_count = obama[['word_count']].stack('word_count', new_column_name=['word','count'])
obama_word_count.sort('count', ascending=False)
people['word_count']= graphlab.text_analytics.count_words(people['text'])
people.head(3)
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
tfidf
len(tfidf)
people['tfidf'] = tfidf
people.head(3)
obama = people[people['name'] == "Barack Obama"]
owc = obama[['tfidf']].stack('tfidf', new_column_name=['word','tfidf'])
owc
owc.sort('tfidf', ascending=False)
clinton = people[people['name'] == 'Bill Clinton']
clinton
beckham = people[people['name'] == 'David Beckham']
beckham
# Obama Closer to Clinton or to beckham?
graphlab.distances.cosine(obama['tfidf'][0], clinton['tfidf'][0])
graphlab.distances.cosine(obama['tfidf'][0], beckham['tfidf'][0])
# Build nearest neighbor model for doc retrieval
knn = graphlab.nearest_neighbors.create(people, features=['tfidf'], label='name')
# Apply nn model for retrieval
# Who is closest to obama
knn.query(obama)
knn.query(beckham)
tswift = people[people['name'] == "Taylor Swift"]
knn.query(tswift)
ajolie = people[people['name'] == "Angelina Jolie"]
knn.query(ajolie)
# Top word count words for Elton John
#(the, john, singer)
#(england, awards, musician)
#(the, in, and) V
#(his, the, since)
#(rock, artists, best)
elton = people[people['name'] == "Elton John"]
elton['word_count'] = graphlab.text_analytics.count_words(elton['text'])
elton['word_count']
elton_word_count = elton[['word_count']].stack('word_count', new_column_name=['word','count'])
obama_word_count.sort('count', ascending=False)
# Top TF-IDF words for Elton John
# (furnish,elton,billboard) V
# (john,elton,fivedecade)
# (the,of,has)
# (awards,rock,john)
# (elton,john,singer)
elton
eltonwc = elton[['tfidf']].stack('tfidf', new_column_name=['word','tfidf'])
eltonwc.sort('tfidf', ascending=False)
# The cosine distance between 'Elton John's and 'Victoria Beckham's articles
# (represented with TF-IDF) falls within which range?
# 0.1 to 0.29
# 0.3 to 0.49
# 0.5 to 0.69
# 0.7 to 0.89
# 0.9 to 1.0 V
vbeckham = people[people['name'] == 'Victoria Beckham']
vbeckham
vbeckhamwc = vbeckham[['tfidf']].stack('tfidf', new_column_name=['word','count'])
graphlab.distances.cosine(elton['tfidf'][0], vbeckham['tfidf'][0])
# The cosine distance between 'Elton John's and 'Paul McCartney's articles
# (represented with TF-IDF) falls within which range?
# 0.1 to 0.29
# 0.3 to 0.49
# 0.5 to 0.69
# 0.7 to 0.89 V
# 0.9 to 1.0
mac = people[people['name'] == "Paul McCartney"]
macwc = mac[['tfidf']].stack('tfidf', new_column_name=['word','count'])
graphlab.distances.cosine(elton['tfidf'][0], mac['tfidf'][0])

# Who is closer to 'Elton John', 'Victoria Beckham' or 'Paul McCartney'?
# Victoria Beckham
# Paul McCartney V

# Who is the nearest neighbor to 'Elton John' using raw word counts?
# Billy Joel
# Cliff Richard V
# Roger Daltrey
# George Bush
people.head()
knn = graphlab.nearest_neighbors.create(people, distance='cosine', features=['word_count'], label='name')
elton = people[people['name'] == "Elton John"]
#elton['word_count'] = graphlab.text_analytics.count_words(elton['text'])
eq = knn.query(elton, k=None)

poi = ['Billy Joel', 'Cliff Richard', 'Roger Daltrey', 'George Bush']
eq['reference_label', 'distance'].filter_by(poi, 'reference_label')

# Who is the nearest neighbor to 'Elton John' using TF-IDF?
# Roger Daltrey
# Rod Stewart V
# Tommy Haas
# Elvis Presley
people
tfidf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tfidf'] = tfidf
people
knn = graphlab.nearest_neighbors.create(people, distance='cosine', features=['tfidf'], label='name')
elton
eq = knn.query(elton, k=None)
poi = ['Roger Daltrey', 'Rod Stewart', 'Tommy Haas', 'Elvis Presley']
eq['reference_label', 'distance'].filter_by(poi, 'reference_label')
#Who is the nearest neighbor to 'Victoria Beckham' using raw word counts?
#Stephen Dow Beckham
#Louis Molloy V
#Adrienne Corri
#Mary Fitzgerald (artist)
knn = graphlab.nearest_neighbors.create(people, distance='cosine', features=['word_count'], label='name')
vbeckham = people[people['name'] == 'Victoria Beckham']
vbeckham
vbr = knn.query(vbeckham, k=None)
poi = ['Stephen Dow Beckham', 'Louis Molloy', 'Adrienne Corri', 'Mary Fitzgerald (artist)']
vbr['reference_label', 'distance'].filter_by(poi, 'reference_label')
# Who is the nearest neighbor to 'Victoria Beckham' using TF-IDF?
# Mel B V
# Caroline Rush
# David Beckham
# Carrie Reichardt
knn = graphlab.nearest_neighbors.create(people, distance='cosine', features=['tfidf'], label='name')
vbr = knn.query(vbeckham, k=None)
poi = ['Mel B', 'Caroline Rush', 'David Beckham', 'Carrie Reichardt']
vbr['reference_label', 'distance'].filter_by(poi, 'reference_label')
#########################################################################################################
# Scikit learn
#########################################################################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from itertools import islice
import numpy as np
import pandas as pd

DATA_DIR='people_wiki.gl/'
people = graphlab.SFrame(DATA_DIR)
people.head(3)
people = people.to_dataframe()
people.head(3)
cvec = CountVectorizer(stop_words='english', min_df=0.0025, max_df=.1, ngram_range=(1,1))
cvec.fit(people['text'])
list(islice(cvec.vocabulary_.items(), 20))
len(cvec.vocabulary_)
cvec_counts = cvec.transform(people['text'])
cvec_counts.shape
cvec_counts.nnz
cvec_counts.shape[0] * cvec_counts.shape[1]
print('sparsity: %.2f%%' % (100.0 * cvec_counts.nnz / (cvec_counts.shape[0] * cvec_counts.shape[1])))
people['word_count'] = cvec_counts
people.head(3)

occ = np.asarray(cvec_counts.sum(axis=0)).ravel().tolist()
counts_df = pd.DataFrame({'term': cvec.get_feature_names(), 'occurrences': occ})
counts_df.sort_values(by='occurrences', ascending=False).head(20)
transformer = TfidfTransformer()
transformed_weights = transformer.fit_transform(cvec_counts)
transformed_weights
weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
weights_df.sort_values(by='weight', ascending=False).head(20)
tvec = TfidfVectorizer(min_df=.0002, stop_words='english', ngram_range=(1,1))
tvec_weights = tvec.fit_transform(people['text'].dropna())
tvec_weights.shape
labels = np.asarray(people['name'])
labels.shape

labels[35817]
inverted_vocab = dict([[v,k] for k,v in cvec.vocabulary_.items()])

# Words related to obama
cvec_counts[35817]

obama_words = np.squeeze(np.asarray(cvec_counts[35817].todense()))
j=0
for i in (-obama_words).argsort():
    print(i, inverted_vocab[i], obama_words[i])
    j += 1
    if j == 10:
        break


count_neigh = NearestNeighbors(n_neighbors=5, n_jobs=-1, metric='cosine')
count_neigh.fit(cvec_counts, labels)
distances, indices = count_neigh.kneighbors(cvec_counts[35817])
indices.ravel().shape
for i,distance in zip(indices.ravel(), distances.ravel()):
    print(labels[i], distance)



neigh = NearestNeighbors(algorithm='auto', n_neighbors=10, n_jobs=-1, metric='cosine')
neigh.fit(tvec_weights, labels)





people.index[people['name'] == 'Barack Obama'].tolist()

distances, indices = neigh.kneighbors(tvec_weights[35817])
for i,distance in zip(indices.ravel(), distances.ravel()):
    print(labels[i], distance)
people.index[people['name'] == 'Victoria Beckham'].tolist()
distances, indices = neigh.kneighbors(tvec_weights[50411])
for i,distance in zip(indices.ravel(), distances.ravel()):
    print(i, labels[i], distance)
from sklearn.metrics.pairwise import cosine_distances
people.index[people['name'] == 'Mary Fitzgerald (artist)'].tolist()


cosine_distances(tvec_weights[50411], tvec_weights[669])
