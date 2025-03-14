# Book Recommender System
This project implements a book recommendation system that suggests books to users based on their past ratings and reading preferences. The system utilizes a dataset of books with genres and applies a combination of user-item matrix construction, genre-based filtering, and cosine similarity to generate recommendations.

## Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, Scikit-Learn, FuzzyWuzzy, streamlit, kagglehub, matplotlib, seaborn

- **Data Source:** Kaggle [GoodReads Books with Genres](https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres) 
- **Data Downloaded:** March 2024

## Features
### Main Branch (Recommendation System)
- Loads and preprocess dataset 
- Collects user input for book ratings
- Handles misspelled or similar book titles using fuzzy matching
- Creates a user-item matrix based on user ratings
- Computes a weighted genre preference vector
- Applies cosine similarity to recommend books based on the user's reading history
- Stores the user’s past reads and generated recommendations in a text file (user_reads_and_recommendations.txt)
### Development Branch (Streamlit Dashboard)
- Streamlit Dashboard (In Progress):
- Top Rated Books: Displays the highest-rated books from the dataset.
- Genre Distribution: Visualizes the distribution of book genres.
- Most Popular Books by Year: Displays the most popular books, grouped by publication year.
- Rating Distribution: Shows the distribution of ratings for all books.

## Installation & Setup
### Requirements
Python 3.x
pip install pandas streamlit kagglehub matplotlib seaborn

### 1. Clone the Repository
```bash
git clone https://github.com/jpmode/book-rec-system.git
cd book-rec-system