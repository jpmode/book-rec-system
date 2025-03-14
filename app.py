
import os 
import kagglehub
import streamlit as st
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

middlelight_goodreadsbookswithgenres_path = kagglehub.dataset_download('middlelight/goodreadsbookswithgenres')

print('Data source import complete.')

print(os.listdir(middlelight_goodreadsbookswithgenres_path))

csv_filename = "Goodreads_books_with_genres.csv"  # Replace this with the actual filename
csv_path = os.path.join(middlelight_goodreadsbookswithgenres_path, csv_filename)

df = pd.read_csv(csv_path)

# Preprocess data
df.dropna(subset=['Title', 'Author', 'genres', 'average_rating', 'publication_date'], inplace=True)
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')

# Streamlit app layout
st.title("Book Rating Dashboard")

# 1. Display Top Rated Books
st.header("Top Rated Books")
top_rated_books = df[['Title', 'Author', 'average_rating']].sort_values(by='average_rating', ascending=False).head(10)
st.write(top_rated_books)

# 2. Genre Distribution
st.header("Genre Distribution")
# Split genres into individual rows
df_genres = df['genres'].str.split(';', expand=True).stack().reset_index(drop=True)
genre_counts = df_genres.value_counts()

# Display genre distribution as a bar chart
fig, ax = plt.subplots()
genre_counts.plot(kind='bar', ax=ax)
ax.set_title("Genre Distribution")
ax.set_xlabel("Genre")
ax.set_ylabel("Count")
st.pyplot(fig)

# 3. Most Popular Books by Year
st.header("Most Popular Books by Year")
# Convert publication_date to datetime
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df['year'] = df['publication_date'].dt.year

# Group by year and find the top book (highest average rating)
top_books_by_year = df.groupby('year').apply(lambda x: x.loc[x['average_rating'].idxmax()])
top_books_by_year = top_books_by_year[['Title', 'Author', 'average_rating', 'year']]

# Display as a table
st.write(top_books_by_year)

# 4. Book Rating Distribution (optional)
st.header("Book Rating Distribution")
fig, ax = plt.subplots()
df['average_rating'].plot(kind='hist', bins=20, ax=ax)
ax.set_title("Rating Distribution")
ax.set_xlabel("Rating")
ax.set_ylabel("Frequency")
st.pyplot(fig)



