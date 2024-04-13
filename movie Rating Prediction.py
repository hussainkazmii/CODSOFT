import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Read the movie dataset from the CSV file
movie_data = pd.read_csv('/Users/LENOVO/Downloads/IMDb Movies India.csv', encoding='latin1')

# Drop rows with missing Rating values
movie_data = movie_data.dropna(subset=['Rating'])

print(movie_data.head())
print(movie_data.info())
print(movie_data.shape)
print(movie_data.isnull().sum())

# Replace unknown classes in categorical columns with a default value
default_value = 'Unknown'
movie_data.fillna({col: default_value for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']}, inplace=True)

# Split the data into features (actors and director) and target variable (rating)
X = movie_data[['Director', 'Actor 1', 'Actor 2', 'Actor 3']]
y = movie_data['Rating']

# One-hot encode categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model within a pipeline
model = Pipeline([
    ('regressor', LinearRegression())
])
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Example of predicting the rating for a new movie based on director and actors
new_movie = {
    'Director': ['J.S. Randhawa'],  # Example new movie director
    'Actor 1': ['Manmauji'],  # Example new movie actor 1
    'Actor 2': ['Birbal'],  # Example new movie actor 2
    'Actor 3': ['Rajendra Bhatia']  # Example new movie actor 3
}
new_movie_encoded = encoder.transform(pd.DataFrame(new_movie))

# Make predictions for the new movie using the trained model
rating_prediction = model.predict(new_movie_encoded)

print(f'Predicted Rating for New Movie: {rating_prediction[0]}')
