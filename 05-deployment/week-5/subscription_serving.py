from flask import Flask, request, jsonify
import pickle

app = Flask('subscription-prediction')

# Load the DictVectorizer and model
with open('dv.bin', 'rb') as f_dv:
    dv = pickle.load(f_dv)

with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:, 1] 
    return y_pred[0]

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict', methods=['GET','POST'])
def predict():
    customer = request.get_json()
    
    # Ensure customer data is present
    if not customer:
        return jsonify({'error': 'Customer data is missing'}), 400

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'subscription_probability': float(prediction),
        'subscription': bool(churn),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
