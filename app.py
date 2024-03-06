from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Loading a model from a .pkl file
loaded_model = joblib.load('diamond_model.pkl')

# Home Page
@app.route('/')
def home():
	return render_template('index.html')

# Processing data and predictions
@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
			# Retrieving data from forms on a web page
			carat = float(request.form['carat'])
			cut = int(request.form['cut'])
			color = int(request.form['color'])
			clarity = int(request.form['clarity'])
			depth = float(request.form['depth'])
			table = float(request.form['table'])
			volume = float(request.form['volume'])

			# Create a DataFrame from the received data
			new_diamond = pd.DataFrame([[carat, cut, color, clarity, depth, table, volume]],
													columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'volume'])

			# Price prediction using the loaded model
			predicted_price = loaded_model.predict(new_diamond)
			
			# Submit predicted price on website
			return render_template('index.html', price=predicted_price[0])

if __name__ == '__main__':
    app.run(debug=True)
