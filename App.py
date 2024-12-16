import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor


# Page configuration
st.set_page_config(page_title='Apartment Price Prediction: Daegu', layout='wide')

# Sidebar for page navigation
page = st.sidebar.selectbox("Select Page", ['Data Visualization', 'Prediction'])

# Load dataset
@st.cache_data
def load_data():
    filepath = r'C:\Users\Asus\Desktop\DS Purwa\Module_3\Capstone Module 3\Daegu_Cleaned.csv'
    df = pd.read_csv(filepath, index_col=0)
    return df

# Train model 
def train_model(df):
    numerical_col_standard = ['N_FacilitiesNearBy(ETC)', 'N_FacilitiesNearBy(PublicOffice)', 'N_SchoolNearBy(University)', 'YearBuilt', 'Size(sqf)']
    numerical_col_robust = ['N_Parkinglot(Basement)']
    categorical_col = ['HallwayType', 'TimeToSubway', 'SubwayStation']

    X = df.drop(columns='SalePrice', axis=1)
    y = df['SalePrice']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing
    ct = ColumnTransformer(transformers=[
        ('Standard', StandardScaler(), numerical_col_standard),
        ('Robust', RobustScaler(), numerical_col_robust),
        ('OneHot', OneHotEncoder(handle_unknown='ignore'), categorical_col)
    ]
    )

    # Create pipeline
    model = TransformedTargetRegressor(regressor=XGBRegressor(
        colsample_bytree = 0.8,
        gamma = 0,
        learning_rate = 0.2,
        max_depth = 3,
        n_estimators = 100,
        subsample = 0.8,
        random_state = 42
    ), func = np.log1p, 
    inverse_func = np.expm1
    )

    pipeline = Pipeline(steps=[('preprocessor', ct), ('model', model)])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipeline, rmse, mae, r2

# Save the model
def save_model(model, filename='Model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Load the model
def load_model(filename='Model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Data Visualization
if page == 'Data Visualization':
    st.title('Apartment Daegu Data Visualization')

    # Load dataset
    data = load_data()

    # Data overview: Show the first 10 rows of the dataset
    st.subheader("Data Overview")
    st.write("Here is the overview of the dataset:")
    st.dataframe(data.head(10))

    # Data summary: Show the summary statistics of the dataset
    st.subheader("Data summary")
    st.write("Here is the summary statistics of the dataset:")
    st.write(data.describe(include='all'))

    # Feature distribution: Show the distribution of a selected feature
    st.subheader("Feature Distribution")
    st.write("Here is the distribution of a selected feature:")
    selected_feature = st.selectbox("Select Feature", data.columns)
    plt.figure(figsize=(10, 5))
    sns.histplot(data[selected_feature], kde=True, color='blue')
    plt.title(f"Distribution of {selected_feature}")
    st.pyplot(plt)

    # Show relationship between Target features and selected features
    st.subheader("Target vs Features")
    feature1 = data['SalePrice']
    feature2 = st.selectbox("Select Features", data.columns)
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=feature1, y=feature2, data=data, color='blue')
    plt.title('Relationship between SalePrice and Selected Feature')
    st.pyplot(plt)

# Predictions
elif page == 'Prediction':

    st.title('Apartment Price Prediction')
    st.write("Input Apartment Details: ")

    input_data = {
        'Size(sqf)': st.number_input("Size (sqft)", min_value=0.0, step=1.0),
        'YearBuilt': st.number_input("Year Built", min_value=1900, max_value=2024, step=1),
        'N_Parkinglot(Basement)': st.number_input("Number of Parking Lots (Basement)", min_value=0, step=1),
        'HallwayType': st.selectbox("Hallway Type", options=['Mixed', 'Corridor', 'Terraced']),
        'TimeToSubway': st.selectbox("Time to Subway", options=['0-5min', '5-10min', '10-15min', '15-20min', 'no_bus_stop_nearby']),
        'SubwayStation': st.selectbox("Subway Station", options=['Bangoge', 'Kyungbuk_uni_hospital', 'Chil-sung-market', 'Daegu', 'Banwoldang', 'Sin-nam', 'Myung-duk', 'no_subway_nearby']),
        'N_FacilitiesNearBy(ETC)': st.number_input("Number of Nearby Facilities (ETC)", min_value=0, step=1),
        'N_FacilitiesNearBy(PublicOffice)': st.number_input("Number of Nearby Public Office Facilities", min_value=0, step=1),
        'N_SchoolNearBy(University)': st.number_input("Number of Nearby Universities", min_value=0, step=1),
    }

    if st.button('Predict'):
        with st.spinner('Predicting...'):
            data = load_data()
            model, rmse, mae, r2 = train_model(data)
            input_df = pd.DataFrame([input_data])
            predictions = model.predict(input_df)
            st.success('Prediction Complete!')
            st.metric(label=f"The predicted price is:", value=f"₩{predictions[0]:,.2f}")
            st.write("Model Performance:")
            st.metric(label="RMSE", value=f"{rmse:.2f}")
            st.metric(label="MAE", value=f"{mae:.2f}")
            st.metric(label="R2 Score", value=f"{r2:.2f}")

