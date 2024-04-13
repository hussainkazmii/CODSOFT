import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def train_and_evaluate_model():
    # Read the credit card dataset from the CSV file
    creditcard_data = pd.read_csv('/Users/LENOVO/Downloads/creditcard.csv.zip', encoding='latin1')

    # visualizing the data in terminal and looking for missing values
    print(creditcard_data)
    print(creditcard_data.notnull().sum())

    # hndling the missing values (if there)
    creditcard_data = creditcard_data.dropna()

    # visualizing data after handling the missing data 
    print(creditcard_data)

    # Handle class imbalance using Random Under-sampling
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(creditcard_data.drop('Class', axis=1), creditcard_data['Class'])

    # Normalize numerical features
    scaler = StandardScaler()
    X_resampled[['Amount', 'Time']] = scaler.fit_transform(X_resampled[['Amount', 'Time']])

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions on the testing set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    confusion_mat = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    # Display evaluation metrics in GUI
    metrics_label.config(text="Evaluation Metrics")
    confusion_mat_label.config(text="Confusion Matrix:\n" + str(confusion_mat))
    classification_rep_label.config(text="Classification Report:\n" + str(classification_rep))
    accuracy_label.config(text="Accuracy Score: " + str(accuracy))

# Create GUI
root = tk.Tk()
root.title("Credit Card Fraud Detection")

# Button to train and evaluate model
train_button = ttk.Button(root, text="Train and Evaluate Model", command=train_and_evaluate_model)
train_button.pack()

# Labels to display evaluation metrics
metrics_label = tk.Label(root, text="Evaluation Metrics", font=("Arial", 14, "bold"))
metrics_label.pack(pady=10)

confusion_mat_label = tk.Label(root, text="", font=("Arial", 12))
confusion_mat_label.pack()

classification_rep_label = tk.Label(root, text="", font=("Arial", 12))
classification_rep_label.pack()

accuracy_label = tk.Label(root, text="", font=("Arial", 12))
accuracy_label.pack()

# Start GUI main loop
root.mainloop()
