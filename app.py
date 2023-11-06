from flask import Flask,render_template,request,app,jsonify,url_for
import pickle 
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

insurance_model = pickle.load(open("./insuranceModel.pkl","rb"))
scaler_model = pickle.load(open("./scalerModel.pkl","rb"))


@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/predict_model",methods=["POST"])
def predict_model():
    age = float(request.form["age"])
    bmi = float(request.form["bmi"])
    smoker = int(request.form.get("smoker",0))
    
    data = [age, bmi, smoker]
    
    print(data)
    final_input = scaler_model.transform(np.array(data).reshape(1, -1))
    output = insurance_model.predict(final_input)

    return render_template("index2.html", prediction_text=" {}".format(output[0]))

if __name__=="__main__":
    app.run(debug=True)