import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

# Function to load data safely
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("movies.csv")
        ratings = pd.read_csv("ratings.csv")
        return movies, ratings
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please upload 'movies.csv' and 'ratings.csv' to your GitHub repository.")
        return None, None

movies, ratings = load_data()

if movies is not None:
    # Clean titles for search
    def clean_title(title):
        return re.sub("[^a-zA-Z0-0 ]", "", title)

    movies["clean_title"] = movies["title"].apply(clean_title)

    # Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies["clean_title"])

    # Search function
    def search(title):
        title = clean_title(title)
        query_vec = vectorizer.transform([title])
        similarity = cosine_similarity(query_vec, tfidf).flatten()
        indices = similarity.argsort()[-5:][::-1]
        return movies.iloc[indices]

    # Recommendation Logic
    def find_similar_movies(movie_id):
        similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
        similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
        
        similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
        similar_user_recs = similar_user_recs[similar_user_recs > .10]
        
        all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
        all_user_recs = all_users["movieId"].value_counts() / len(ratings["userId"].unique())
        
        rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
        rec_percentages.columns = ["similar", "all"]
        
        rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
        rec_percentages = rec_percentages.sort_values("score", ascending=False)
        
        return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

    # UI
    movie_input = st.text_input("Search for a movie you like:", "Toy Story")

    if len(movie_input) > 2:
        results = search(movie_input)
        movie_id = results.iloc[0]["movieId"]
        
        st.subheader(f"Recommendations for: {results.iloc[0]['title']}")
        recommendations = find_similar_movies(movie_id)
        st.table(recommendations)
