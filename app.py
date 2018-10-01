
import flask
from flask import Flask,render_template,request
from sklearn.externals import joblib
import pandas as pd
from flask import jsonify 
import json
app=Flask(__name__)

@app.route('/')
@app.route("/index")
def index():
	return flask.render_template('home.html')

@app.route("/input")
def predict():
	return flask.render_template('predict.html')

@app.route('/predict',methods=['POST'])
def make_prediction():
	if request.method=='POST':
		sample=[]
		data = {}
		age=request.form.get('age')
		sample.append(int(age))
		sex=request.form.get('sex')
		sample.append(int(sex))
		cpt=request.form.get('cpt')
		sample.append(int(cpt))
		rbp=request.form.get('rbp')
		sample.append(int(rbp))
		chol=request.form.get('chol')
		sample.append(int(chol))
		fbs=request.form.get('fbs')
		sample.append(int(fbs))
		ecg=request.form.get('ecg')
		sample.append(int(ecg))
		hr=request.form.get('hr')
		sample.append(int(hr))
		eia=request.form.get('eia')
		sample.append(int(eia))
		oldpeak=request.form.get('oldpeak')
		sample.append(float(oldpeak))
		slope=request.form.get('slope')
		sample.append(float(slope))
		nov=request.form.get('nov')
		sample.append(int(nov))
		thal=request.form.get('thal')
		sample.append(int(thal))
		rafg1C=request.form.get('rafg1C')
		sample.append(float(rafg1C))
		rafg1M=request.form.get('rafg1M')
		sample.append(float(rafg1M))
		rafg1F=request.form.get('rafg1F')
		sample.append(float(rafg1F))
		rafg2C=request.form.get('rafg2C')
		sample.append(float(rafg2C))
		rafg2M=request.form.get('rafg2M')
		sample.append(float(rafg2M))
		rafg2F=request.form.get('rafg2F')
		sample.append(float(rafg2F))
		sampleDf= pd.DataFrame(sample)
		if len(sample)<19: 
			return render_template('prediction.html', label1="Missing data or incorrect data entered")
		# make prediction
	   
		prediction = model.predict(sampleDf.T)
		confidence = model.predict_proba(sampleDf.T)
		data["Prediction"]=prediction[0]

		json_data = json.dumps(data)
        jsonify(data=data)
        return render_template('result.html', data=data)
		
if (__name__ == '__main__'):

	 model = joblib.load('source.pkl')
	 app.run(port = 5000,debug = True) 
	 
		