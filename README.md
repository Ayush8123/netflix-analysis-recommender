# 🎬 Netflix Data Analysis & Recommendation System

This project explores and analyzes a Netflix dataset using Python libraries like **Pandas**, **Matplotlib**, and **Seaborn**. It also includes a **content-based recommendation system** built using **TF-IDF Vectorizer** and **cosine similarity**, and deployed as a simple web app using **Streamlit**.

---

## 📊 Features

- Clean and preprocess Netflix dataset
- Visualize:
  - Top content-producing countries
  - Content released per year
  - Most common ratings
  - Duration and type of content (Movies/TV Shows)
  - Top directors and genres
- Build a **content-based recommendation system** using:
  - `description` + `genre` text fields
  - TF-IDF Vectorizer
  - Cosine similarity
- **Interactive Streamlit app** to get top 5 similar titles based on user-selected input

---

# 🧠 Recommendation Logic
I used the description and listed_in fields of Netflix titles, combined them into a single string, vectorized using TF-IDF, and calculated cosine similarity to recommend shows/movies similar to the user’s selected title.

# 📌 Dataset
This dataset was sourced from publicly available Netflix content data (kaggle). It contains metadata such as title, type, genre, director, cast, country, date added, rating, and description.

