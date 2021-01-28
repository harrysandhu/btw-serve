from flask import Flask, request, jsonify, json, make_response
app = Flask(__name__)
from transformers import pipeline, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModel.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

valid_api_keys = ['10321032']

def summary(text, ln):
    return summarizer(str(text), max_length=int(ln))



@app.route("/")
def hello():
    return 'Hello, world'


@app.route("/summary", methods=['POST'])
def handle():
    data = request.get_json()
    print()
    params = ['api_key', 'text', 'length']
    for p in params:
        if p not in data:
            return make_response(jsonify(error="Invalid request"), 400)
    if data['api_key'] not in valid_api_keys:
        return  make_response(jsonify(error="Invalid api_key"), 400)

    s = summary(data['text'], data['length'])
    return make_response(jsonify(summary=s), 200)


if __name__ == '__main__':
    app.run('0.0.0.0', 4069, debug=True)
