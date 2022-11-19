from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import pickle
model=pickle.load(open("xgb.pkl","rb"))

app=Flask(__name__)
@app.route("/")
def hellow():
    return render_template("index.html")

@app.route("/predict",methods=["Post","Get"])
def predict():
    input_features=[float(x) for x in request.form.values()]
    feature_values=[np.array(input_features)]
    features_name=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    df=pd.DataFrame(feature_values,columns=features_name)
    output=model.predict(df)
    if output==0:
        output="Breast cancer"
    else:
        output="No breast cancer"
    return output

if __name__ == "__main__":
    app.run()
