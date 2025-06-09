import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df_rec = pd.read_csv('netflix_titles.csv') 

df_rec['combined'] = df_rec['description'] + ' ' + df_rec['listed_in']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_rec['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_rec.index, index=df_rec['title']).drop_duplicates()

def recommend(title, n=5):
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return df_rec['title'].iloc[movie_indices].tolist()

st.title("Netflix Content Recommendation System")

title_input = st.selectbox(
    "Select a Netflix show/movie title:",
    df_rec['title'].sort_values().unique()
)

if st.button("Recommend"):
    if title_input:
        recommendations = recommend(title_input)
        if recommendations:
            st.write(f"Top recommendations similar to '{title_input}':")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.write(f"Sorry, '{title_input}' not found in dataset.")
    else:
        st.write("Please select a title.")
