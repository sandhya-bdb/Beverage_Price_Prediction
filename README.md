# Beverage Price Prediction

A Streamlit web application that predicts the preferred price range for a **new beverage product**, based on consumer demographic, economic, behavioral and preference-data inputs. The model segments consumers into one of four price-brackets based on a trained classification model built from customer survey data.

---

## 🚀 Features

### User Interface  
- Clean, responsive UI built with [Streamlit](https://streamlit.io) for seamless user experience.  
- Intuitive form-based input system with real-time validation of user data.

### User Input Fields  
The application collects the following categories of consumer information:  
#### Demographic Information  
- Age  
- Gender  
- Geographic Zone (regional classification)  
- Occupation category  

#### Economic & Behavioral Data 
- Income level classification  
- Consumption habits & frequency  
- Brand preferences and awareness levels  

#### Personal Preferences 
- Health concerns and dietary considerations  
- Flavor preferences and taste profiles  
- Packaging choices and format preferences  

### Price Range Prediction  
The model outputs one of four distinct price segments:  
- **$50–100** (Economy range)  
- **$100–150** (Value range)  
- **$150–200** (Premium range)  
- **$200–250** (Super-premium range)

### Engineered Features  
- **CF/AB Score** — Consumption Frequency vs. Brand Awareness metric; measures alignment between usage patterns and brand recognition.  
  Formula: `CF/AB = f(consumption_frequency, brand_awareness)`.  
- **ZAS Score** — Zone × Affluence composite indicator; combines geographic zone and economic affluence.  
  Formula: `ZAS = zone_factor × affluence_level`.  
- **BSI (Brand Switch Index)** — Quantifies brand-loyalty vs. brand-switching behaviour; higher values indicate greater propensity to switch.  
  Formula: `BSI = g(brand_preferences, awareness_levels)`.

### Key Capabilities  
- Real-time price-segment prediction based on consumer profiles.  
- Interactive data-visualization of feature importance and model insights.  
- Export functionality for analysis results (e.g., CSV download).  
- Mobile-responsive design enabling cross-device compatibility.

---

## 🧠 Model & Pipeline  
- Pre-trained machine-learning classification model loaded using **pickle**.  
- Pre-processing pipeline includes label-encoding, one-hot encoding, and engineered features (CF/AB, ZAS, BSI).  
- Prediction logic modularised in `prediction_helper.py`.

---

## 📂 File Structure

---

## 🛠️ Getting Started  
1. Clone the repository:  
   ```bash
   git clone <repo-URL>
   cd beverage-price-prediction
2. pip install -r requirements.txt
3. streamlit run main.py
   
## ✅ Usage Example

- User opens the form UI and enters: Age = 28, Gender = Male, Zone = Urban East, Income level = High, Consumption frequency = Weekly, etc.

- The model computes CF/AB, ZAS, BSI features, feeds them (along with standard inputs) into the classifier.

- The result: the predicted bracket is $150–200 (Premium range).

- The user can view the reasoning/feature importances and export the result for further analysis.

## 🔮 Future Improvements

- Expand dataset with more beverage categories (carbonated, functional, alcoholic, etc.) and market regions.

- Experiment with advanced model architectures (e.g., ensemble methods, neural networks) and hyper-parameter tuning.

- Deploy as a web-service (Docker container or cloud) with REST API for integration into product-management dashboards.

- Add active-learning loop: collect user feedback to continuously refine the model and expand segment definitions.

- Improve explainability: provide more detailed insights on “why” a given price bracket was assigned (e.g., SHAP values).


- streamlit run main.py
