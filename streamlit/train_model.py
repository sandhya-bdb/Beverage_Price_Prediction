# train_local_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier # You can choose any model
import joblib
import numpy as np
import os

# --- Configuration ---
DATA_FILE = 'survey_results.csv'
TARGET_COLUMN = 'price_range'
RESPONDENT_ID_COLUMN = 'respondent_id'
# We will save the entire pipeline that includes preprocessing and the model
BEST_MODEL_FILENAME = 'best_price_range_pipeline.pkl'

# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(data_file, target_column, respondent_id_column):
    print(f"Loading data from {data_file}...")
    try:
        df = pd.read_csv(data_file)
        print("Data loaded successfully.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        
        # Store original column names for Streamlit app display and DataFrame creation
        original_feature_names = df.columns.drop(target_column, errors='ignore').tolist()
        if respondent_id_column in original_feature_names:
            original_feature_names.remove(respondent_id_column)

        # Separate target
        y = df[target_column]
        # Ensure we drop respondent_id_column if it exists, and also the target column
        X = df.drop(columns=[target_column, respondent_id_column], errors='ignore')

        # Identify categorical and numerical features based on the DataFrame X
        categorical_features = X.select_dtypes(include=['object', 'category']).columns
        numerical_features = X.select_dtypes(include=np.number).columns

        # Create preprocessing steps using ColumnTransformer
        # OneHotEncoder for categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # StandardScaler for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep any other columns (shouldn't be any if X is clean)
        )

        # Handle target encoding if it's categorical
        target_encoder = None
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)
            print(f"Encoded target column '{target_column}'. Classes: {target_encoder.classes_}")
        else:
            y_encoded = y

        print("Preprocessing setup complete.")
        # Return original feature names for Streamlit
        return X, y_encoded, preprocessor, target_encoder, original_feature_names

    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error during data loading/preprocessing: {e}")
        return None, None, None, None, None

# --- Model Training and Saving ---
def train_and_save_model(X_train, y_train, preprocessor, model_name, model_params):
    """Trains a model, wraps it in a pipeline with the preprocessor, and saves it."""
    
    # Choose the model
    model = None
    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**model_params)
    # Add other models here if you want to experiment
    # elif model_name == "LogisticRegression":
    #     model = LogisticRegression(**model_params)
    else:
        print(f"Model '{model_name}' not supported in this example. Using RandomForestClassifier.")
        model = RandomForestClassifier(**model_params) # Default to RandomForest

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    print(f"Training pipeline with {model_name}...")
    pipeline.fit(X_train, y_train)
    print(f"Pipeline with {model_name} trained successfully.")
    
    # Save the entire pipeline locally
    pipeline_filename = BEST_MODEL_FILENAME
    joblib.dump(pipeline, pipeline_filename)
    print(f"Pipeline saved locally as '{pipeline_filename}'")

    return pipeline, pipeline_filename

# --- Main Execution Block ---
if __name__ == "__main__":
    # Load and preprocess data
    X, y, preprocessor, target_encoder, original_feature_names = load_and_preprocess_data(
        DATA_FILE, TARGET_COLUMN, RESPONDENT_ID_COLUMN
    )

    if X is not None and y is not None and preprocessor is not None and original_feature_names is not None:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")

        # Define the model and its parameters
        # We'll train just one model for simplicity in this example.
        MODEL_NAME = "RandomForestClassifier"
        MODEL_PARAMS = {"n_estimators": 100, "random_state": 42}

        # Train and save the pipeline
        pipeline, pipeline_filename = train_and_save_model(
            X_train, y_train, preprocessor, MODEL_NAME, MODEL_PARAMS
        )

        # Save the target encoder and original feature names for the Streamlit app
        if target_encoder:
            joblib.dump(target_encoder, 'target_encoder.pkl')
            print("'target_encoder.pkl' saved.")
        
        joblib.dump(original_feature_names, 'original_feature_names.pkl')
        print("'original_feature_names.pkl' saved.")
        
        print("\nModel training and saving complete. You can now build the Streamlit app.")

    else:
        print("\nExiting script due to data loading or preprocessing errors.")

