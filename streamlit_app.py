import streamlit as st
import joblib

# Load the Multinomial Random Forest model 
filename = 'rf_classifier.joblib'
classifier = joblib.load(open(filename, 'rb'))
cv = joblib.load(open('tfidf_vectorizer.joblib', 'rb'))

def predict_spam(message):
    data = [message]
    vect = cv.transform(data).toarray()
    prediction = classifier.predict(vect)
    return prediction

def main():
    st.title('Email Spam Detection')

    message = st.text_area('Enter email message here:', height=200)
    if st.button('Predict'):
        prediction = predict_spam(message)
        if prediction == 1:
            st.write('This email is classified as spam.')
        else:
            st.write('This email is not spam.')

if __name__ == '__main__':
    main()
