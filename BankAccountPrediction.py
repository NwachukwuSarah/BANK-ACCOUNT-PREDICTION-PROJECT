# Importing the neccessary Libraries
import pandas as pd
import streamlit as st 
import pickle

# Load the model and encoder
with open("FImodel.pkl", "rb") as f:
    bundle = pickle.load(f)

# Creating a function that the predict button will execute to predict if a person has a bank account or not
def predictHasAccount():
    input_data = {
        "country": country_selection,
        "year": year_selection,
        "location_type": location_type_selection,
        "cellphone_access": cellphone_access_selection,
        "household_size": household_size_input,
        "age_of_respondent": age_of_respondent_input,
        "gender_of_respondent": gender_of_respondent_selection,
        "relationship_with_head": relationship_with_head_selection,
        "marital_status": marital_status_selection,
        "education_level": education_level_selection,
        "job_type": job_type_selection
    }

    data = pd.DataFrame([input_data]) 

    for col in data.columns:
        if col not in ['household_size', 'age_of_respondent']:
            data[col] = data[col].astype(str)

    data['year'] = data['year'].astype('object')

    # Transform the input data using the pre-trained encoder
    data_encoded = bundle["encoder"].transform(data[bundle["columns"]])
    
    # Make prediction
    pred_prob = bundle["FImodel"].predict(data_encoded)
    pred_value = (pred_prob > 0.5).astype(int)[0][0] 

    if pred_value == 1:
        st.write("This Person has a Bank Account")
    else:
        st.write("This Person Doesn't have a Bank Account")


# Creating the Streamlit user interface
st.title("BANK ACCOUNT PREDICTOR")
st.write("Select the correct options that matches the person you want to predict and click predict to see whether they have a bank account or not")

country_selection = st.selectbox('Country', ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'])
year_selection = st.selectbox('Year', [2016, 2017, 2018])
location_type_selection = st.selectbox('Location Type', ['Rural', 'Urban'])
cellphone_access_selection = st.selectbox('Cellphone Access', ['Yes', 'No'])
household_size_input = st.number_input('Household Size', min_value=1, max_value=21, value=3) 
age_of_respondent_input = st.number_input('Age of Respondent', min_value=16, max_value=100, value=35) 
gender_of_respondent_selection = st.selectbox('Gender of Respondent', ['Female', 'Male'])
relationship_with_head_selection = st.selectbox('Relationship with Head', ['Child', 'Head of Household', 'Other non-relatives', 'Other relative', 'Parent', 'Spouse'])
marital_status_selection = st.selectbox('Marital Status', ['Divorced/Seperated', 'Dont know', 'Married/Living together', 'Single/Never Married', 'Widowed'])
education_level_selection = st.selectbox('Education Level', ['No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training'])
job_type_selection = st.selectbox('Job Type', ['Dont Know/Refuse to answer', 'Farming and Fishing', 'Formally employed Government', 'Formally employed Private', 'Government Dependent', 'Informally employed', 'No Income', 'Other Income', 'Remittance Dependent', 'Self employed'])


# Prediction button
if st.button('Predict'):

    predictHasAccount()
