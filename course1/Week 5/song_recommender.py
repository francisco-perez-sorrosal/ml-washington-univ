import turicreate
turicreate.config.set_runtime_config("TURI_NUM_GPUS", 0)

songs = turicreate.SFrame('song_data.gl/')
songs.shape
songs.head()
turicreate.visualization.set_target('gui')
songs['song'].show()
users = songs['user_id'].unique()
len(users)

n_of_songs = len(songs['song_id'].unique())
n_of_songs

train_data, test_data = songs.random_split(.8, seed=0)
# Recommender
popularity_model = turicreate.popularity_recommender.create(train_data, user_id='user_id', item_id='song')
users[0]
popularity_model.recommend(users=[users[0]])
popularity_model.recommend(users=[users[1]]) # Same stuff is recommended
# Recommend with personalization
personalize_model = turicreate.item_similarity_recommender.create(train_data, user_id='user_id', item_id='song')
personalize_model.recommend(users=[users[0]])
personalize_model.recommend(users=[users[1]])
personalize_model.get_similar_items(['With Or Without You - U2'])
import matplotlib.pyplot as plt
%matplotlib inline

model_performance = turicreate.recommender.util.compare_models(test_data,
                                                               [popularity_model, personalize_model],
                                                               user_sample=0.05)
# Counting unique users
# Question 1: Which of the artists below have had the most unique users listening to their songs?
# Answer: Taylor Swift
kw_uu = songs[songs['artist'] == 'Kanye West']['user_id'].unique()
ff_uu = songs[songs['artist'] == 'Foo Fighters']['user_id'].unique()
ts_uu = songs[songs['artist'] == 'Taylor Swift']['user_id'].unique()
lgg_uu = songs[songs['artist'] == 'Lady GaGa']['user_id'].unique()
len(kw_uu), len(ff_uu), len(ts_uu), len(lgg_uu)

# Question 2: Which of the artists below is the most popular artist, the one with highest total listen_count, in the data set?
# Answer: Kings of Leon
popular_artists = songs.groupby(key_column_names='artist', operations={'total_count': turicreate.aggregate.SUM('listen_count')})
popular_artists.sort('total_count', ascending=False)
# Question 3: Which of the artists below is the least popular artist, the one with smallest total listen_count, in the data set?
# Answer: William Tabbert

popular_artists.sort('total_count', ascending=True)
## Other stuff
len(users)
users_test=test_data['user_id'].unique()
len(users_test)
subset_test_users = users_test[0:10000]
len(subset_test_users)
recommended_songs=personalize_model.recommend(subset_test_users, k=1)
most_rec_song = recommended_songs.groupby(key_column_names='song', operations={'total_count': turicreate.aggregate.COUNT()})
most_rec_song.sort('total_count', ascending=False)
# Scikit learn!
# https://github.com/llSourcell/recommender_live

import turicreate
from sklearn.model_selection import train_test_split
songs = turicreate.SFrame('song_data.gl/')
songs.head(1)
songs_df = songs.to_dataframe()
songs_df.head(1)
songs_df.groupby(['song']).agg({'listen_count': 'sum'}).sort_values(by=["listen_count"], ascending=False).head(3) #.plot(kind='bar')


train_df, test_df = train_test_split(songs_df, test_size=0.2, random_state=0)


songs_df['song_id'].unique()

class PopularityRecommender(object):
    def __init__(self, train, target, based_on):
        self.train_data = train
        self.target = target
        self.based_on = based_on

        self.train_data['score'] = self.train_data['user_id']
        print(self.train_data.head(1))
        print(self.train_data.shape)
        #Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = self.train_data.groupby(self.based_on).agg({self.target: 'count'}).reset_index()
        print(train_data_grouped.head(1))
        print(train_data_grouped.shape)
        train_data_grouped.rename(columns = {self.target: 'score'},inplace=True)
        # train_data_grouped['song'] = train_data_grouped.apply(lambda r : self.train_data['song'] if self.train_data['song_id'].unique() == r['song_id'] else "N/A title")
        #Sort the songs based upon recommendation score
        train_data_sort = train_data_grouped.sort_values(['score'] + self.based_on, ascending = [0] + range(len(self.based_on)))

        #Generate a recommendation rank based upon score
        train_data_sort['rank'] = train_data_sort['score'].rank(ascending=False, method='first')

        #Get the top 10 recommendations
        self.recommendations = train_data_sort.head(10)

    #Use the popularity based recommender system model to
    #make recommendations
    def recommend(self, user):
        user_recommendations = self.recommendations

        #Add user_id column for which the recommendations are being generated
        user_recommendations['user_id'] = user

        #Bring user_id column to the front
        cols = user_recommendations.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        user_recommendations = user_recommendations[cols]

        return user_recommendations

popularity_recommender = PopularityRecommender(train_df, "user_id", ["song_id", "song"])


popularity_recommender.recommendations


users_df[0]
popularity_recommender.recommend(users_df[0])

## Popularity
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
