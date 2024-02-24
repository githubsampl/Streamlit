from flask import Flask, render_template, redirect, url_for, request
import joblib

# Load the Multinomial Random Forest model 
filename = 'rf_classifier.joblib'
classifier = joblib.load(open(filename, 'rb'))
cv = joblib.load(open('tfidf_vectorizer.joblib', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        return redirect(url_for('predict'))

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
