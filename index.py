from flask import Flask, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask_cors import CORS
import onnxruntime

df = pd.read_csv("jobs_in_data.csv")
df_clean = df.loc[:, ["work_year", "job_category", "experience_level", "work_setting", "salary_in_usd"]]
X = df_clean.copy().dropna()
scalery = StandardScaler().fit(X["salary_in_usd"].to_frame())

#model = load_model('api/myModel.keras')
session = onnxruntime.InferenceSession('myModel.onnx')

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
	input.extend(job)
	input.extend(exp)
	input.extend(mode)
	newInput.append(input)

	newInput =  np.array(newInput, dtype='f')

	prediction = session.run(None, {'x': newInput})
	output = scalery.inverse_transform(prediction[0])

	newOutput = output.tolist()

	return {
		"salario": newOutput[0][0]
	}

if __name__ == '__main__':
	app.run(port=5000)
