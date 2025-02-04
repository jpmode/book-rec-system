# Book Recommender System
This project is a book recommendation system that suggests books to users based on their past ratings and reading preferences. It utilizes a dataset of books with genres and applies a combination of user-item matrix construction, genre-based filtering, and cosine similarity to generate recommendations.

Libraries: Pandas, Scikit-learn, FuzzyWuzzy 

Data from Kaggle: https://www.kaggle.com/datasets/middlelight/goodreadsbookswithgenres, last downloaded March 2024

## Features
- Loads and preprocess dataset 
- Collects user input for book ratings
- Handles misspelled or similar book titles using fuzzy matching
- Creates a user-item matrix based on user ratings
- Computes a weighted genre preference vector
- Applies cosine similarity to recommend books based on the user's reading history
- Stores the user’s past reads and generated recommendations in a text file (user_reads_and_recommendations.txt)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jpmode/book-rec-system.git
cd book-rec-system