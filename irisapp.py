import streamlit as st
import pickle

# Load the model
with open('iris.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
def main():
    st.title('Iris Plant Classification App')
    
    st.sidebar.title('About')
    st.sidebar.info(
        "This is a simple Iris plant classification app built with Streamlit."
        " Enter the values for the Sepal Length, Sepal Width, Petal Length, and Petal Width, then click 'Predict' to see the predicted species."
    )

    st.write("Please enter the following information for the prediction:")

    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.1, max_value=10.0, value=5.4, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.1, max_value=10.0, value=3.4, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.1, max_value=10.0, value=1.3, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=10.0, value=0.2, step=0.1)

    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    if st.button('Predict'):
        prediction = model.predict(features)
        #species = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        #predicted_species = species[prediction[0]]
        st.write(f'The predicted species is {prediction}')

if __name__ == '__main__':
    main()
