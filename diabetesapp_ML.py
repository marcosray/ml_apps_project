import streamlit as st
import pandas as pd
import joblib
import pickle
import seaborn as sns
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('diabetes.csv')
df.head()
print(df['Outcome'].value_counts())
sns.countplot(x='Outcome', data=df)
plt.show()
# Detecting outliers using box plots
plt.figure(figsize=(12, 8))
df.boxplot()
plt.title("Boxplot for each input feature")
plt.show()
# Generating a correlation matrix
correlation_matrix = df.corr()

# Visualizing the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()

# Data preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# Use the specified hyperparameters
best_params = {'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

# Create a RandomForestRegressor with the specified parameters
grid_search = RandomForestRegressor(
    max_depth=best_params['max_depth'],
    max_features=best_params['max_features'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators']
)

# Fit the model
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluating the model
y_pred = grid_search.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Saving the model as a joblib file
import pickle
with open("diabetes_test.pkl","wb") as d:
  pickle.dump(grid_search,d)




# Load the trained model
#model = joblib.load('diabetes_model.joblib')
model = joblib.load('diabetes_test.pkl')

# Streamlit app description
st.write("""
# Diabetes Prediction App
This app predicts whether a person has diabetes based on some features. Please input the following values to assess the diabetes status.
""")

# Input features with explanations
st.header('User Input Features')

# User inputs with explanations
def user_input_features():
    st.write('Number of times pregnant')
    pregnancies = st.slider('Pregnancies', 0, 17, 3)

    st.write('Plasma glucose concentration')
    glucose = st.slider('Glucose', 0, 200, 100)

    st.write('Diastolic blood pressure (mm Hg)')
    blood_pressure = st.slider('Blood Pressure', 0, 122, 70)

    st.write('Triceps skin fold thickness (mm)')
    skin_thickness = st.slider('Skin Thickness', 0, 99, 20)

    st.write('2-Hour serum insulin (mu U/ml)')
    insulin = st.slider('Insulin', 0, 846, 79)

    st.write('Body mass index (weight in kg/(height in m)^2)')
    bmi = st.slider('BMI', 0.0, 67.1, 31.4)

    st.write('Diabetes pedigree function')
    diabetes_pedigree_function = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)

    st.write('Age (years)')
    age = st.slider('Age', 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree_function,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction')
    if prediction[0] == 0:
        st.success('Great news! The assessment suggests that you are unlikely to have diabetes. Keep up the healthy lifestyle!')
    else:
        st.error('Hey there! The assessment suggests that you might have diabetes. We recommend consulting a healthcare professional for further guidance.')
