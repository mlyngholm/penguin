import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
         # Penguin prediction app
         This app predicts the Palmer Penguin species!
         * Adeile, Chinstrap or Gentoo
         """)

st.sidebar.header('User input')
st.sidebar.markdown("""
                    [Example CSV Input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
                    """)


# Collect user input features
upload_file = st.sidebar.file_uploader('Upload your CSV file ', type=['csv'])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4200.0)

        data = {'island': island,
                'sex': sex,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g}

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()


# Combine user input features with entire dataset
penguins_raw = pd.read_csv('cleaned_dataset.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)


# Encode ordinal features
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]

st.subheader('User input features')

if upload_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters')
    st.write(df)


# Read in saved classification model
load_model = pickle.load(open('penguins_model.pkl', 'rb'))


# Apply model to make predicitions
prediction = load_model.predict(df)
predict_proba = load_model.predict_proba(df)


st.subheader('Prediction of species')
penguin_species = np.array(['Adeile', 'Chinstrap', 'Gentoo'])
st.write(penguin_species[prediction][0])

st.subheader('Prediction probability')

col1, col2, col3 = st.columns(3)

col1.write('Probability of Adeile:')
col1.write(predict_proba[0][0])

col2.write('Probability of Chinstrap:')
col2.write(predict_proba[0][1])

col3.write('Probability of Gentoo:')
col3.write(predict_proba[0][2])

st.image('image.png')
