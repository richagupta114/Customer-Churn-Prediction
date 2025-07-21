import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import json

def train_and_save_model():
    # Create a directory for saving models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv('churn.csv')
    
    # Print dataset info
    print("\nDataset Info:")
    print(df.info())
    print("\nClass Distribution:")
    print(df['Exited'].value_counts(normalize=True))
    
    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Prepare features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Split the data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create copies for training
    X_train_processed = X_train.copy()
    X_test_processed = X_test.copy()
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    label_encoders = {}
    categorical_columns = ['Geography', 'Gender']
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        X_train_processed[column] = label_encoders[column].fit_transform(X_train_processed[column])
        X_test_processed[column] = label_encoders[column].transform(X_test_processed[column])
        print(f"\n{column} encoding:")
        print(f"Classes: {label_encoders[column].classes_}")
        print(f"Encoded values: {label_encoders[column].transform(label_encoders[column].classes_)}")
    
    # Scale numerical features
    print("\nScaling numerical features...")
    scaler = StandardScaler()
    X_train_processed = scaler.fit_transform(X_train_processed)
    X_test_processed = scaler.transform(X_test_processed)
    
    # Train the model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_processed, y_train)
    
    # Make predictions and calculate accuracy
    y_pred = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print model performance
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(cm)
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model and preprocessing objects
    print("\nSaving model and preprocessing objects...")
    joblib.dump(model, 'models/model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(label_encoders, 'models/label_encoders.joblib')
    
    # Save the accuracy to a file
    with open('models/accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    
    # Save confusion matrix
    cm_data = {
        'matrix': cm.tolist(),
        'labels': ['No Churn', 'Churn']
    }
    with open('models/confusion_matrix.json', 'w') as f:
        json.dump(cm_data, f)
    
    print("\nModel and preprocessing objects saved successfully!")
    return accuracy

if __name__ == "__main__":
    train_and_save_model() 