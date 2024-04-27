import shutil
import tempfile
from enum import Enum

import numpy as np
import pandas as pd

# Load the datasets
ratings_df = pd.read_csv("Ratings.csv")
books_df = pd.read_csv("Books.csv")

# Convert 'Year-Of-Publication' to numeric type
books_df["Year-Of-Publication"] = pd.to_numeric(
    books_df["Year-Of-Publication"], errors="coerce"
)

# Remove any columns with the word "URL" in the title
books_df = books_df[books_df.columns.drop(list(books_df.filter(regex="URL")))]


# Enum class for column types
class ColumnType(Enum):
    SPARSE = 1
    NOISY = 2
    CORRELATED = 3
    NONLINEAR = 4
    CATEGORICAL = 5
    BINARY = 6


# Function to add a column based on the chosen type
def add_custom_column(df, column_type, column_name):
    if column_type == ColumnType.SPARSE:
        # Add a sparse column with mostly zeros, some ones, and some missing values
        # p=[0.8, 0.1, 0.1] specifies the probabilities of choosing 0, 1, and np.nan respectively
        df[column_name] = np.random.choice(
            [0, 1, np.nan], size=len(df), p=[0.8, 0.1, 0.1]
        )
    elif column_type == ColumnType.NOISY:
        # Add a noisy column with values drawn from a normal distribution
        # Normal distribution: $X \sim \mathcal{N}(\mu, \sigma^2)$
        # where $\mu=0$ is the mean and $\sigma=1$ is the standard deviation
        df[column_name] = np.random.normal(0, 1, size=len(df))
    elif column_type == ColumnType.CORRELATED:
        # Add a column correlated with the 'Year-Of-Publication' column
        # New column = 'Year-Of-Publication' + Normal noise
        # Correlated column: $Y = X + \epsilon$, where $X$
        # is 'Year-Of-Publication' and $\epsilon \sim \mathcal{N}(0, 1)$
        df[column_name] = df["Year-Of-Publication"] + np.random.normal(
            0, 1, size=len(df)
        )
    elif column_type == ColumnType.NONLINEAR:
        # Add a nonlinear column based on the sine of 'Year-Of-Publication'
        # Nonlinear transformation: $Y = \sin(X)$, where $X$ is 'Year-Of-Publication'
        df[column_name] = np.sin(df["Year-Of-Publication"])
    elif column_type == ColumnType.CATEGORICAL:
        # Add a categorical column with random categories
        categories = ["Category_A", "Category_B", "Category_C"]
        df[column_name] = np.random.choice(categories, size=len(df))
    elif column_type == ColumnType.BINARY:
        # Add a binary column with random 0s and 1s
        df[column_name] = np.random.choice([0, 1], size=len(df))
    return df


# Invented column names
column_names = [
    "User_Engagement_Score",
    "Book_Popularity_Index",
    "Author_Diversity_Metric",
    "Publication_Trend_Indicator",
    "Genre_Preference_Score",
    "User_Loyalty_Measure",
    "Book_Complexity_Rating",
    "Author_Influence_Factor",
    "Publication_Quality_Metric",
    "Genre_Diversity_Index",
]

# Add custom columns to the books dataset
books_df = add_custom_column(books_df, ColumnType.SPARSE, column_names[0])
books_df = add_custom_column(books_df, ColumnType.NOISY, column_names[1])
books_df = add_custom_column(books_df, ColumnType.CORRELATED, column_names[2])
books_df = add_custom_column(books_df, ColumnType.NONLINEAR, column_names[3])
books_df = add_custom_column(books_df, ColumnType.CATEGORICAL, column_names[4])
books_df = add_custom_column(books_df, ColumnType.BINARY, column_names[5])


# reduce the number of rows by 1 / 4 in both files
frac = 1/20
ratings_df = ratings_df.sample(frac=frac)
books_df = books_df.sample(frac=frac)

# Save the updated ratings dataset
ratings_df.to_csv("RatingsV2.csv", index=False)

# Save the updated books dataset
books_df.to_csv("BooksV2.csv", index=False)

# Package BooksV2.csv and Ratings.csv into a zip file called
# the-booksv2-dataset.zip


# Create a temporary directory
with tempfile.TemporaryDirectory() as temp_dir:
    # Copy the CSV files to the temporary directory
    shutil.copy("BooksV2.csv", temp_dir)
    shutil.copy("RatingsV2.csv", temp_dir)

    # Create the ZIP archive using the temporary directory as base_dir
    shutil.make_archive("the-booksv2-dataset", "zip", temp_dir)
