import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the anime dataset
anime_df = pd.read_csv('C:/Users/avery/Desktop/Projects/Anime Recommender/anime_data.csv')

# fill missing values with empty strings
anime_df = anime_df.fillna('')

# select the features to include in the vectorizer
features = ['Title', 'Genre', 'Description', 'Studio', 'Year', 'Rating']

# create a combined feature column for the vectorizer
anime_df['combined_features'] = anime_df.apply(lambda x: ' '.join(str(x[feature]) for feature in features), axis=1)


# create the vectorizer
vectorizer = CountVectorizer()

# create the vectorized features
vectorized_features = vectorizer.fit_transform(anime_df['combined_features'])

# compute the cosine similarity matrix
cosine_sim_matrix = cosine_similarity(vectorized_features)

# define the function to get recommendations
def get_recommendations(anime_title, cosine_sim_matrix=cosine_sim_matrix, anime_df=anime_df):
    # get the index of the anime
    idx = anime_df[anime_df['Title'] == anime_title].index[0]
    
    # get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # sort the anime based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # select the top 10 most similar anime
    sim_scores = sim_scores[1:11]
    
    # get the anime indices
    anime_indices = [i[0] for i in sim_scores]
    
    # return the top 10 most similar anime
    return anime_df[['Title', 'Genre', 'Description', 'Studio', 'Year', 'Rating']].iloc[anime_indices]

# prompt the user for an anime title
anime_title = input("Enter an anime title: ")

# get the recommendations
recommendations = get_recommendations(anime_title)

# print the recommendations
print("Here are some anime similar to '{}':".format(anime_title))
for index, anime in recommendations.iterrows():
    print("Title: {}".format(anime['Title']))
    print("Genre: {}".format(anime['Genre']))
    print("Description: {}".format(anime['Description']))
    print("Studio: {}".format(anime['Studio']))
    print("Year: {}".format(anime['Year']))
    print("Rating: {}".format(anime['Rating']))
    print()
