# YouTube Video Performance Predictor

## Overview
This project is a **Decision Support System** that predicts whether a YouTube video is likely to go *viral* based on its metadata and title characteristics. It combines **data analysis and machine learning** (developed in a Jupyter Notebook) with an **interactive Streamlit web application** for real-time predictions.

The workflow is split into two main components:
- **Model development & training** (`Untitled.ipynb`)
- **User-facing prediction app** (`app.py`)

---

## Project Structure
```
├── app.py                Streamlit web application
├── Untitled.ipynb         Data analysis, feature engineering, and model training
├── model.pkl              Trained machine learning model 
├── youtube_clean_final.csv   Cleaned dataset 
└── README.md              Project documentation
```

---

##  Untitled.ipynb Model Development
The Jupyter Notebook handles the **end-to-end machine learning pipeline**:

### 1. Data Loading & Exploration
- Loads a cleaned YouTube dataset (`youtube_clean_final.csv`)
- Performs exploratory data analysis (EDA):
  - Dataset structure and statistics
  - Missing value checks
  - Visualizations such as:
    - Views distribution
    - Views vs Likes scatter plots
    - Average views by publish year

### 2. Feature Engineering
Key engineered features used for prediction:
- `log_likes` – Log-transformed likes
- `log_comments` – Log-transformed comments
- `duration_min` – Video duration in minutes
- `word_count` – Number of words in the video title

A binary target variable (`viral`) is used to classify videos as viral or not.

### 3. Model Training
- Data is split into training and testing sets
- Machine learning models explored include:
  - Linear Regression (baseline analysis)
  - **Random Forest Classifier** (final model)

### 4. Model Evaluation & Saving
- Model performance evaluated using accuracy and classification metrics
- Final trained classifier is serialized and saved as:
  ```
  model.pkl
  ```

This saved model is later used by the Streamlit app.

---

## app.py – Streamlit Web Application
The Streamlit application provides an **interactive interface** for users to predict video virality.

### Features
- Loads the pre-trained model (`model.pkl`)
- User input fields for:
  - Number of likes
  - Number of comments
  - Video duration (seconds)
  - Video title

### Real-Time Prediction
- Automatically applies the same feature engineering steps used during training
- Generates:
  - Binary prediction (Viral / Not Viral)
  - Probability score of virality

### Output
- Success message if the video is likely to go viral
-  Warning message if the video is unlikely to go viral

---

##  How to Run the Project

### 1. Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 2. Train the Model (Optional)
Run `Untitled.ipynb` to:
- Explore the data
- Train the model
- Generate `model.pkl`

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

---

## Use Case
This project is useful for:
- Content creators optimizing video titles and metadata
- Marketing teams estimating video performance
- Data science learners building end-to-end ML applications

---

## Future Improvements
- Add more NLP-based title features
- Support multi-class virality levels
- Deploy the app to cloud platforms (Streamlit Cloud, Heroku)
- Improve model explainability

---



