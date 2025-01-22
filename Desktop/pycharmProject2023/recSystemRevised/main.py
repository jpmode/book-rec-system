import pandas as pd
'''
BOOK RECOMMENDER SYSTEM
Mar 10 2024
'''


# Load dataset
books_df = pd.read_csv("/Users/supahmo/Desktop/Goodreads_books_with_genres.csv")

# Preprocess data
books_df.dropna(how='any', inplace=True)
books_df = books_df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
books_df.drop_duplicates(subset=['isbn13'], inplace=True)
books_df.drop_duplicates(subset=['Title', 'Author'], inplace=True)

# One-hot encode genres
genres_df = books_df['genres'].str.get_dummies(';')

# Concatenate the dataframes and drop genres column
booksEncoded_df = pd.concat([books_df, genres_df], axis=1)
booksEncoded_df.drop('genres', axis=1, inplace=True)

# To find similar book titles
from fuzzywuzzy import process

def find_similar_books(user_input, dataset_titles, threshold=80):
    matches = process.extract(user_input, dataset_titles, limit=10)
    similar_books = [title for title, score, _ in matches if score >= threshold]
    return similar_books

# Collect user input (title and rating) and store it
def collect_user_ratings(dataset):
    ratings = {}
    while True:
        book_title = input("Enter the title of the book you've read (or type 'done' to finish): ").strip()
        if book_title.lower() == 'done':
            break
        elif book_title.lower() not in dataset['Title'].str.lower().values:
            similar_books = find_similar_books(book_title, dataset['Title'].str.lower())
            if similar_books:
                print("Did you mean one of these titles?")
                for i, similar_book in enumerate(similar_books, start=1):
                    print(f"{i}. {similar_book}")
                choice = input("Enter the number corresponding to the suggested title or enter 'skip' to continue: ")
                if choice.isdigit() and 1 <= int(choice) <= len(similar_books):
                    book_title = similar_books[int(choice) - 1]
                elif choice.lower() == 'skip':
                    continue
                else:
                    print("Invalid input. Skipping this book.")
                    continue
            else:
                print("No similar titles found. Please enter a valid book title.")
                continue
        while True: # loop until a valid rating is entered
            rating = input("Enter your rating for this book (between 1 and 5): ").strip()
            if rating == '0':
                break # exit the loop and return the collected ratings
            try:
                rating = int(rating)
                if 1 <= rating <= 5:
                    ratings[book_title] = rating
                    break
                else:
                    print("Invalid rating. Please enter a number between 1 and 5 (or 0 to exit).")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    return ratings

# Collect user ratings
user_ratings = collect_user_ratings(booksEncoded_df)

# Convert user ratings to DataFrame
user_ratings_df = pd.DataFrame(list(user_ratings.items()), columns=['Book', 'Rating'])

# Create user-item matrix
def create_user_item_matrix(user_ratings_df, booksEncoded_df):
    user_ratings_merged = pd.merge(user_ratings_df, booksEncoded_df[['Title']], left_on='Book', right_on='Title', how='inner')
    user_ratings_merged['UserID'] = 1
    user_ratings_merged = user_ratings_merged.drop_duplicates()
    user_item_matrix = user_ratings_merged.pivot(index='UserID', columns='Title', values='Rating').fillna(0)
    return user_item_matrix

user_item_matrix = create_user_item_matrix(user_ratings_df, booksEncoded_df)

# Get list of book titles from user matrix
user_titles_list = user_item_matrix.columns.tolist()

# Filter genres_df to include only genres corresponding to the user's books
genres_df.index = booksEncoded_df['Title']
filtered_genres_df = genres_df.loc[user_titles_list]
filtered_genres_df.reset_index(inplace=True)
filtered_genres_df = filtered_genres_df.drop_duplicates(subset=['Title'])
filtered_genres_df.reset_index(drop=True, inplace=True)

# Transpose the filtered genres df
filtered_genres_df_transposed = filtered_genres_df.T.reset_index(drop=True)
filtered_genres_df_transposed.columns = filtered_genres_df_transposed.iloc[0]
filtered_genres_df_transposed = filtered_genres_df_transposed[1:].reset_index(drop=True)

# Calculate the weighted genre vector
filtered_genres_df_transposed_numeric = filtered_genres_df_transposed.apply(pd.to_numeric, errors='coerce')
filtered_genres_df_transposed_filled = filtered_genres_df_transposed_numeric.fillna(0)

user_item_matrix.reset_index(drop=True, inplace=True)
user_item_matrix.rename_axis(None, axis=1, inplace=True)
user_item_matrix = user_item_matrix.apply(pd.to_numeric, errors='coerce').fillna(0)

# Calculate the weighted genre vector
weighted_genre_vector = filtered_genres_df_transposed_filled.mul(user_item_matrix.values.flatten(), axis=1).sum(axis=1)

# Calculate cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
weighted_genre_vector = weighted_genre_vector.values.reshape(1, -1)
cosine_sim = cosine_similarity(weighted_genre_vector, genres_df.values)

# Get recommendations
sorted_indices = cosine_sim.argsort(axis=1)[0][::-1]
recommended_books_indices = [idx for idx in sorted_indices if idx not in user_item_matrix.columns[user_item_matrix.loc[0].notna()].values]
recommended_books_indices = recommended_books_indices[:10]  # Top 10 recommendations

# Reset the index of the books_df DataFrame
books_df.reset_index(inplace=True, drop=True)

# Get recommended books
recommended_books = books_df.loc[recommended_books_indices, ['Title', 'Author', 'publication_date', 'genres']]
recommended_books['genres'] = recommended_books['genres'].str.split(';').str[:3].str.join(';')
recommended_books.reset_index(drop=True, inplace=True)

# Store user's past reads and program's recommendations into a file
with open("user_reads_and_recommendations.txt", "w") as f:
    f.write("User's Past Reads:\n")
    for title, rating in user_ratings.items():
        f.write(f"Title: {title}\n")
        f.write(f"Rating: {rating}\n\n")

    f.write("Program Recommendations:\n")
    for idx in recommended_books_indices:
        book_info = books_df.iloc[idx]
        genres = book_info['genres'].split(';')[:3]
        f.write(f"Title: {book_info['Title']}\n")
        f.write(f"Author: {book_info['Author']}\n")
        f.write(f"Publication Date: {book_info['publication_date']}\n")
        f.write(f"Genres: {';'.join(genres)}\n\n")

