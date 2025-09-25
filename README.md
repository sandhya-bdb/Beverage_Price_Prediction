
# Beverage Price Prediction 

This Streamlit web app predicts the preferred price range for a new beverage product based on consumer demographic and behavioral inputs. It uses a machine learning classification model trained on customer survey data to segment users into one of four price brackets.



## Features

### User Interface
Clean, responsive UI built with Streamlit for seamless user experience

Intuitive form-based input system with real-time validation
### User Input Fields

The application collects comprehensive consumer data including:

####  Demographic Information
Age and Gender
Geographic Zone (regional classification)
Occupation category
#### Economic & Behavioral Data
Income level classification
Consumption habits and frequency
Brand preferences and awareness levels
#### Personal Preferences
Health concerns and dietary considerations
Flavor preferences and taste profiles
Packaging choices and format preferences

## Price Range Prediction
The model predicts one of four distinct price segments:

$50–100 (Economy range)
$100–150 (Value range)
$150–200 (Premium range)
$200–250 (Super premium range)
## Engineered Features
### CF/AB Score
Consumption Frequency vs. Brand Awareness metric
Measures alignment between usage patterns and brand recognition
Formula: CF/AB = f(consumption_frequency, brand_awareness)
### ZAS Score
Zone × Affluence composite indicator
Combines geographic and economic factors
Formula: ZAS = zone_factor × affluence_level
### BSI (Brand Switch Index)
Brand Loyalty Quantification
Captures likelihood of switching from established brands
Higher values indicate greater propensity for brand switching
Formula: BSI = g(brand_preferences, awareness_levels)
### Key Capabilities
Real-time price prediction based on consumer profiles
Interactive data visualization of feature importance
Export functionality for analysis results
Mobile-responsive design for cross-device compatibility
## Model
Pretrained ML model loaded using pickle

Preprocessing includes label encoding, one-hot encoding, and feature engineering

Prediction logic modularized in prediction_helper.py


## File Structure

*   `artifacts/`
    *   `├── model.pkl` ← Pretrained model file
*   `main.py` ← Streamlit frontend
*   `prediction_helper.py` ← Input preprocessing and prediction logic



streamlit run main.py
