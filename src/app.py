from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        text = request.form["news"]
        text_vec = vectorizer.transform([text])
        prediction = model.predict(text_vec)[0]
        result = "Fake News ❌" if prediction == 1 else "Real News ✅"
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
