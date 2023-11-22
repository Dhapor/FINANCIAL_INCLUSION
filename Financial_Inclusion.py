import pandas as pd
import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


data = pd.read_csv('Financial_inclusion_dataset.csv')
ds = data.copy()



# # Standard scaling and label encoding
# def transformer(dataframe):
#     from sklearn.preprocessing import StandardScaler, LabelEncoder
#     scaler = StandardScaler()
#     encoder = LabelEncoder()

#     for i in dataframe.columns:
#         if dataframe[i].dtypes != 'O':
#             dataframe[i] = scaler.fit_transform(dataframe[[i]])
#         else:
#             dataframe[i] = encoder.fit_transform(dataframe[i])
#     return dataframe



# ds = ds.drop(['age_of_respondent', 'uniqueid'], axis = 1)


# feature selection
sel_col = ['household_size', 'job_type', 'education_level', 'country', 'gender_of_respondent', 'location_type', 'marital_status', 'relationship_with_head']
new_ds = data[sel_col]
# new_ds.head()
# new_ds = transformer(new_ds)

# # MODELLING
# x = ds.drop('bank_account', axis = 1)
# y = ds.bank_account
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# x_train, x_test, y_train, y_test = train_test_split(ds, y, test_size = 0.25, random_state = 47, stratify = y)



# # Modelling
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# model = RandomForestClassifier() 
# model.fit(x_train, y_train) 
# cross_validation = model.predict(x_train)
# pred = model.predict(x_test)


import streamlit as st
print('done')

st.markdown("<h1 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>FINANCIAL INCLUSION PROJECT</h1>",unsafe_allow_html=True)

st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: center; font-family:montserrat'>Financial inclusion Built By Datapsalm</h3>",unsafe_allow_html=True)

st.markdown("<br></br>", unsafe_allow_html=True)
st.image('pngwing.com (6).png',  width = 650)


st.markdown("<br></br>", unsafe_allow_html=True)

st.markdown("<h3 style = 'margin: -15px; color: #2B2A4C; text-align: center; font-family:montserrat'>Background to the story</h3>",unsafe_allow_html=True)

st.markdown("<p>The dataset contains demographic information and what financial services are used by approximately 33,600 individuals across East Africa. The ML model role is to predict which individuals are most likely to have or use a bank account. The term financial inclusion means:  individuals and businesses have access to useful and affordable financial products and services that meet their needs – transactions, payments, savings, credit and insurance – delivered in a responsible and sustainable way.</p>", unsafe_allow_html = True)


st.write(new_ds.head())
st.sidebar.image('profile image.jpg')
st.sidebar.markdown('<br>', unsafe_allow_html= True)


import pickle
model_u= pickle.load(open('Financial_Inclusion.pkl','rb'))


Household_size = st.sidebar.number_input("Household size", new_ds['household_size'].min(), new_ds['household_size'].max())
Job_type = st.sidebar.selectbox("Job Type", new_ds['job_type'].unique())
Edu_Lev = st.sidebar.selectbox("Education level", new_ds['education_level'].unique())
Marital_status = st.sidebar.selectbox("marital_status", new_ds['marital_status'].unique())
Country = st.sidebar.selectbox('country', new_ds['country'].unique())
Location_Type = st.sidebar.selectbox("location type", new_ds['location_type'].unique())      
Gender = st.sidebar.selectbox("Gender of respondent", new_ds['gender_of_respondent'].unique())
Relationship_with_head = st.sidebar.selectbox("relationship_with_head", new_ds['relationship_with_head'].unique())



st.header('Inputed Values')
input_variables = pd.DataFrame([{
    'household_size':Household_size,
    'job_type': Job_type,
    'education_level': Edu_Lev,
    'country': Country, 
    'gender_of_respondent': Gender,
    'location_type': Location_Type, 
    'marital_status': Marital_status,
    'relationship_with_head':Relationship_with_head 

}])


st.write(input_variables)
cat = input_variables.select_dtypes(include = ['object', 'category'])
num = input_variables.select_dtypes(include = 'number')

# Standard Scale the Input Variable.
from sklearn.preprocessing import StandardScaler, LabelEncoder
for i in input_variables.columns:
    if i in num.columns:
        input_variables[i] = StandardScaler().fit_transform(input_variables[[i]])
for i in input_variables.columns:
    if i in cat.columns: 
        input_variables[i] = LabelEncoder().fit_transform(input_variables[i])

st.markdown('<hr>', unsafe_allow_html=True)

if st.button('Press To Predict'):
    st.markdown("<h4 style = 'color: #2B2A4C; text-align: left; font-family: montserrat '>Model Report</h4>", unsafe_allow_html = True)
    predicted = model_u.predict(input_variables)
    st.toast('bank_account Predicted')
    # st.image('image.jpg', width = 100)
    st.success(f'Model Predicted {predicted}')
    if predicted == 0:
        st.success('The person does not have an account')
    else:
        st.success('the person has an account')


st.markdown('<hr>', unsafe_allow_html=True)

st.markdown("<h8 style = 'color: #2B2A4C; text-align: center; font-family:montserrat'>FINANCIAL INCLUSION BUILT BY DATAPSALM</h8>",unsafe_allow_html=True)


    