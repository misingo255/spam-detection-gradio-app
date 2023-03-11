import joblib
import gradio as gr

def classify(contents):

    model = joblib.load("./models/spam-detection-model.pkl")

    vectorizer = joblib.load("./preprocessing/count-vectorizer.pkl")

    results = model.predict(vectorizer.transform([contents])[0])

    prediction = str(results[0])

    if prediction == "0":
        return "The given message is not a spam"
    else:
        return "The given message is a spam"
    


app = gr.Interface(fn=classify, inputs="text", outputs="text")

app.launch()