from flask import Flask, request
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask_cors import CORS

df = pd.read_csv("jobs_in_data.csv")
df_clean = df.loc[:, ["work_year", "job_category", "experience_level", "work_setting", "salary_in_usd"]]
X = df_clean.copy().dropna()
scalery = StandardScaler().fit(X["salary_in_usd"].to_frame())

model = load_model('./myModel.keras')

input = []

app = Flask(__name__)
CORS(app)

@app.route('/api/', methods=['GET'])
def get():
	year = request.args.get("year", type=int)
	job = request.args.getlist("job", type= int)
	exp = request.args.getlist("exp", type=int)
	mode = request.args.getlist("mode", type=int)
	
	input = []
	newInput = []
	input.append(year)
	for jobs in job:
		input.append(jobs)
	for exps in exp:
		input.append(exps)
	for modes in mode:
		input.append(modes)		
	newInput.append(input)

	newInput =  np.array(newInput)

	prediction = model.predict(newInput)
	output = scalery.inverse_transform(prediction)

	newOutput = output.tolist()

	return {
		"salario": newOutput[0][0]
	}

if __name__ == '__main__':
	app.run(port=5000)
