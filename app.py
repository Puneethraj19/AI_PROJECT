from flask import Flask, render_template, request
import numpy as np
import pickle
import os


app = Flask(__name__)

# Load model, scaler, and encoders
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
except Exception as e:
    raise RuntimeError(f"Failed to load model or preprocessing files: {e}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get inputs from form
        amount = float(request.form['amount'])
        transaction_type_raw = request.form['transaction_type'].strip()
        status_raw = request.form['status'].strip()
        device_raw = request.form['device'].strip()
        slice_id_raw = request.form['slice'].strip()
        latency = float(request.form['latency'])
        bandwidth = float(request.form['bandwidth'])
        pin = int(request.form['pin'])

        # Debug: print raw values
        print("Raw form data:", request.form)

        # Encode categorical values using the loaded encoders
        transaction_type = encoders['type'].transform([transaction_type_raw])[0]
        status = encoders['status'].transform([status_raw])[0]
        device = encoders['device'].transform([device_raw])[0]
        slice_id = encoders['slice'].transform([slice_id_raw])[0]

        # Prepare and scale input
        input_data = np.array([[amount, transaction_type, status, device, slice_id, latency, bandwidth, pin]])
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        print("Scaled input:", input_scaled)
        print("Prediction result:", prediction)

        # Output result
        result = "⚠️ Fraudulent Transaction Detected!" if prediction == 1 else "✅ Legitimate Transaction"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
   port=int(os.environ.get("PORT",5000))
   app.run(host="0.0.0.0",port=port,debug=True)


    
