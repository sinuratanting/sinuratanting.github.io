from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('iris_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define home page
@app.route('/')
def home():
    return render_template('index.html')

# Define prediction
@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Scale the input data
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get the predicted class
    predicted_class = model.predict(input_data)[0]
    
    return render_template('index.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
