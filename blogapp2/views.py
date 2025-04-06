import numpy as np
import pandas as pd
import ast
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from django.shortcuts import render

# Initialize the Porter Stemmer (do this once)
ps = PorterStemmer()

# Load the datasets (do this once when the app starts)
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = movies.merge(credits, on='title')
    movies.dropna(inplace=True)

    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    def convert3(obj):
        return [i['name'] for i, _ in zip(ast.literal_eval(obj), range(3))]

    def fetchDirector(obj):
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []

    def stem1(text):
        if isinstance(text, list):
            text = " ".join(text)
        if not text:
            return ""
        return " ".join([ps.stem(word) for word in text.split()])

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetchDirector)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tag'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tag']].copy()
    new_df.loc[:, 'tag'] = new_df['tag'].apply(lambda x: " ".join(x))
    new_df.loc[:, 'tag'] = new_df['tag'].apply(lambda x: x.lower())

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tag']).toarray()

    similarity = cosine_similarity(vectors)

    # Store new_df and similarity for later use
    global movie_df
    global similarity_matrix
    movie_df = new_df
    similarity_matrix = similarity

except FileNotFoundError:
    print("Error: Make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in the project root.")
    movie_df = None
    similarity_matrix = None

def recommend_movies(movie_name):
    if movie_df is None or similarity_matrix is None:
        return "Error: Movie data not loaded.", []
    try:
        movie_index = movie_df[movie_df['title'] == movie_name.title()].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        recommended_movies = [movie_df.iloc[i[0]].title for i in movies_list]
        return None, recommended_movies
    except IndexError:
        return "Movie not found in the database.", []

def recommend_view(request):
    if request.method == 'POST':
        movie_name = request.POST.get('movie_name')
        error, recommendations = recommend_movies(movie_name)
        return render(request, 'blogapp2/index.html', {'recommendations': recommendations, 'error': error, 'searched_movie': movie_name})
    return render(request, 'blogapp2/index.html')
