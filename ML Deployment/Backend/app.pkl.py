import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#create flask app
app = Flask(__name__)

#load the pickle model
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def values():
    return render_template("values.html")



@app.route("/templates/values", methods = ["POST","GET"])
def predict(value):
     if PPE <=0.82273 && DFA <= 0.65177 || RPDE <= 0.24071 || numpulses >= 446 && numPeriodPulses >= 445:
        return render_template('values.html', Prediction_text='THE PATIENT DOESNOT HAVE A PARKINSONS DISEASE')
    else:
        return render_template('values.html', Prediction_text='THE PATIENT HAVE A PARKINSONS DISEASE')

@app.route('/predict_api',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
return jsonify(output)
if __name__ == "__main__":
   app.run(debug=False)
