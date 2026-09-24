from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    transformed_message = cv.transform([message])

    prediction = model.predict(transformed_message)[0]

    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return render_template(
        'index.html',
        prediction=result
    )


if __name__ == '__main__':
    app.run(debug=True)