# app.py
import streamlit as st
import pandas as pd
import joblib
import os

# --- Configuration ---
BEST_PIPELINE_FILENAME = 'best_price_range_pipeline.pkl'
TARGET_ENCODER_FILENAME = 'target_encoder.pkl'
ORIGINAL_FEATURE_NAMES_FILENAME = 'original_feature_names.pkl'

# --- Helper Functions ---
@st.cache_resource
def load_pipeline_and_artifacts():
    """Loads the trained pipeline, target encoder, and feature names."""
    pipeline = None
    target_encoder = None
    original_feature_names_from_pkl = None # Stores the names as loaded from the pickle file
    load_errors = []

    # --- Load Pipeline ---
    if not os.path.exists(BEST_PIPELINE_FILENAME):
        load_errors.append(f"Prediction pipeline file not found at: `{BEST_PIPELINE_FILENAME}`.")
        load_errors.append("Please ensure `train_local_model.py` has been run successfully and the file exists in the same directory.")
    else:
        try:
            pipeline = joblib.load(BEST_PIPELINE_FILENAME)
            # Removed: st.success("Prediction pipeline loaded successfully!")
        except Exception as e:
            load_errors.append(f"Error loading pipeline `{BEST_PIPELINE_FILENAME}`: {e}")

    # --- Load Target Encoder (if it exists) ---
    if os.path.exists(TARGET_ENCODER_FILENAME):
        try:
            target_encoder = joblib.load(TARGET_ENCODER_FILENAME)
        except Exception as e:
            load_errors.append(f"Error loading target encoder `{TARGET_ENCODER_FILENAME}`: {e}")
    else:
        load_errors.append(f"Target encoder file `{TARGET_ENCODER_FILENAME}` not found. Predictions might not be decoded correctly.")

    # --- Load Original Feature Names ---
    if os.path.exists(ORIGINAL_FEATURE_NAMES_FILENAME):
        try:
            original_feature_names_from_pkl = joblib.load(ORIGINAL_FEATURE_NAMES_FILENAME)
            # Ensure feature names are strings
            original_feature_names_from_pkl = [str(name) for name in original_feature_names_from_pkl]
            # Removed: st.success("Original feature names loaded successfully!")
        except Exception as e:
            load_errors.append(f"Error loading original feature names `{ORIGINAL_FEATURE_NAMES_FILENAME}`: {e}")
    else:
        load_errors.append(f"Original feature names file `{ORIGINAL_FEATURE_NAMES_FILENAME}` not found. Input fields might not be generated correctly.")

    if load_errors:
        st.error("Errors occurred during loading:")
        for error in load_errors:
            st.error(f"- {error}")
        return None, None, None

    return pipeline, target_encoder, original_feature_names_from_pkl

def predict_price_range(pipeline, target_encoder, input_data, pipeline_feature_names):
    """Makes a prediction using the loaded pipeline."""
    try:
        processed_input_dict = {}
        # Ensure the input data keys exactly match the feature names the pipeline expects
        for feature_name in pipeline_feature_names:
            value = input_data.get(feature_name) # Get value using the pipeline's expected name
            
            if isinstance(value, str) and value == "--- Select ---":
                processed_input_dict[feature_name] = None # Treat placeholder as missing value
            else:
                processed_input_dict[feature_name] = value
        
        df_input = pd.DataFrame([processed_input_dict])
        
        prediction = pipeline.predict(df_input)
        
        if target_encoder:
            try:
                decoded_prediction = target_encoder.inverse_transform(prediction)
                return decoded_prediction[0]
            except (IndexError, ValueError, TypeError) as e:
                st.warning(f"Could not decode prediction using target encoder. Raw prediction: {prediction[0]}. Error: {e}")
                return f"Prediction: {prediction[0]} (decoding failed)"
        else:
            return f"Prediction: {prediction[0]}"
            
    except KeyError as e:
        st.error(f"Missing expected feature in input data: {e}. Please ensure all fields are filled correctly.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None

# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Beverage Price Predictor", layout="wide")
    st.title("Codex Beverage: Price Prediction")
    # Removed: st.markdown("Enter the details below to predict the beverage price range. The model was trained on your `survey_results.csv` dataset.")

    # Load the pipeline and artifacts
    pipeline, target_encoder, pipeline_feature_names = load_pipeline_and_artifacts()

    if pipeline is None or pipeline_feature_names is None:
        st.stop()

    # --- Define mapping from pipeline feature names to display labels and options ---
    # Use the EXACT names from pipeline_feature_names as keys here.
    feature_mapping = {
        'age': {
            'display_label': 'Age',
            'widget': 'number',
            'help': "Enter the age of the respondent.",
            'min_value': 0, 'max_value': 120, 'value': 25, 'step': 1
        },
        'gender': {
            'display_label': 'Gender',
            'widget': 'select',
            'options': ["M", "F", "Other"],
            'help': "Select the gender of the respondent."
        },
        'zone': {
            'display_label': 'Zone',
            'widget': 'select',
            'options': ["Urban", "Rural", "Semi-Urban"],
            'help': "Select the geographical zone."
        },
        'occupation': {
            'display_label': 'Occupation',
            'widget': 'select',
            'options': ["Working Professional", "Student", "Business", "Homemaker", "Retired", "Unemployed", "Other"],
            'help': "Select the respondent's occupation."
        },
        # --- CORRECTED KEYS FOR INCOME AND CONSUMPTION FREQUENCY ---
        'income_levels': { # Key matches the image: 'income_levels'
            'display_label': 'Income Level (In L)', # This is what the user sees
            'widget': 'select',
            'options': ["<10L", ">35L", "16L - 25L", "None", "10L - 15L", "26L - 35L"], # Options from your image
            'help': "Select the respondent's income bracket."
        },
        'consume_frequency(weekly)': { # Key matches the image: 'consume_frequency(weekly)'
            'display_label': 'Consume Frequency (weekly)', # This is what the user sees
            'widget': 'select',
            'options': ["3-4 times", "5-7 times", "0-2 times"], # Options from your image
            'help': "How often the respondent consumes beverages weekly."
        },
        # --- ADDING OTHER CATEGORICAL FEATURES WITH CORRECT DISPLAY LABELS ---
        'current_brand': {
            'display_label': 'Current Brand',
            'widget': 'select',
            'options': ["Newcomer", "Established", "Popular", "Competitor", "Unknown"],
            'help': "The respondent's current preferred brand."
        },
        'preferable_consumption_size': {
            'display_label': 'Preferable Consumption Size',
            'widget': 'select',
            'options': ["Small (250 ml)", "Medium (500 ml)", "Large (1L)", "Extra Large (2L+)"],
            'help': "Preferred serving size."
        },
        'awareness_of_other_brands': {
            'display_label': 'Awareness of other brands',
            'widget': 'select',
            'options': ["0 to 1", "1 to 2", "2 to 3", "3+"],
            'help': "Number of other beverage brands the respondent is aware of."
        },
        'reasons_for_choosing_brands': {
            'display_label': 'Reasons for choosing brands',
            'widget': 'select',
            'options': ["Price", "Quality", "Brand Name", "Availability", "Taste", "Health", "Convenience", "Other"],
            'help': "Primary reasons for brand choice."
        },
        'flavor_preference': {
            'display_label': 'Flavor Preference',
            'widget': 'select',
            'options': ["Traditional", "Modern", "Fruity", "Spicy", "Herbal", "None"],
            'help': "Preferred type of flavor."
        },
        'purchase_channel': {
            'display_label': 'Purchase Channel',
            'widget': 'select',
            'options': ["Online", "Offline", "Both"],
            'help': "Where the respondent typically purchases beverages."
        },
        'packaging_preference': {
            'display_label': 'Packaging Preference',
            'widget': 'select',
            'options': ["Simple", "Attractive", "Eco-friendly", "Functional", "None"],
            'help': "Preferred packaging style."
        },
        'health_concerns': {
            'display_label': 'Health Concerns',
            'widget': 'select',
            'options': ["Low (Not very concerned)", "Medium", "High (Very concerned)"],
            'help': "Level of concern about health aspects."
        },
        'typical_consumption_situations': {
            'display_label': 'Typical Consumption Situations',
            'widget': 'select',
            'options': ["Active (eg. Sports, gym)", "Relaxed (eg. Home)", "Social (eg. Parties)", "Work", "Travel", "Other"],
            'help': "Situations where beverages are consumed."
        }
    }
    placeholder_option = "--- Select ---"

    # Layout with columns
    num_columns = 3
    if len(pipeline_feature_names) > 6:
        num_columns = 3
    elif len(pipeline_feature_names) > 3:
        num_columns = 2
    else:
        num_columns = 1
        
    cols = st.columns(num_columns)
    col_index = 0
    
    input_data = {} # Dictionary to store user inputs. Keys MUST match pipeline_feature_names.

    with st.form("prediction_form"):
        for feature_name_pipeline in pipeline_feature_names: # Iterate using pipeline's feature names
            col = cols[col_index % num_columns]
            widget_info = feature_mapping.get(feature_name_pipeline) # Get config using pipeline's name

            if widget_info:
                widget_type = widget_info['widget']
                display_label = widget_info.get('display_label', feature_name_pipeline) # Use display label
                help_text = widget_info.get('help', f"Enter value for {display_label}")

                if widget_type == 'number':
                    input_data[feature_name_pipeline] = col.number_input(
                        display_label,
                        min_value=widget_info.get('min_value', 0),
                        max_value=widget_info.get('max_value', 1000000), 
                        value=widget_info.get('value', 0),
                        step=widget_info.get('step', 1),
                        help=help_text,
                        key=f"input_{feature_name_pipeline}"
                    )
                elif widget_type == 'select':
                    # Add placeholder to the options for selectboxes
                    options_with_placeholder = [placeholder_option] + widget_info.get('options', [])
                    input_data[feature_name_pipeline] = col.selectbox(
                        display_label,
                        options=options_with_placeholder,
                        help=help_text,
                        key=f"input_{feature_name_pipeline}"
                    )
                else: # Fallback for unhandled widget types
                    st.warning(f"Unsupported widget type '{widget_type}' for feature '{display_label}'. Using text input.")
                    input_data[feature_name_pipeline] = col.text_input(
                        display_label,
                        value="",
                        help=help_text,
                        key=f"input_{feature_name_pipeline}"
                    )
            else:
                # Fallback for features not found in feature_mapping
                st.warning(f"Configuration for feature '{feature_name_pipeline}' not found. Using text input.")
                input_data[feature_name_pipeline] = col.text_input(
                    feature_name_pipeline, # Use the pipeline name if no mapping
                    value="",
                    help=f"Enter value for {feature_name_pipeline}",
                    key=f"input_{feature_name_pipeline}"
                )
            
            col_index += 1

        # --- Submit Button ---
        calculate_button = st.form_submit_button("Calculate Price Range")

    # --- Prediction Logic ---
    if calculate_button:
        has_meaningful_input = False
        for key, value in input_data.items():
            if value is not None and value != placeholder_option and (not isinstance(value, str) or value.strip()):
                has_meaningful_input = True
                break
        
        if not has_meaningful_input:
            st.warning("Please provide some input details to make a prediction.")
        else:
            # Pass the exact pipeline feature names and the input data
            prediction = predict_price_range(pipeline, target_encoder, input_data, pipeline_feature_names)

            if prediction:
                st.subheader("Prediction Result")
                st.success(f"The predicted price range is: **{prediction}**")
            else:
                st.error("Could not generate a prediction. Please review your inputs and try again.")

# --- Run the App ---
if __name__ == "__main__":
    main()
