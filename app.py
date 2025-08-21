import os
from flask import Flask, request, jsonify, g
import tensorflow as tf
import numpy as np
import sklearn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import GPT2Tokenizer

from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.models import load_model
from joblib import dump, load
import re
from html import unescape
import gdown
from dotenv import load_dotenv
import os
import gdown
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification



app = Flask(__name__)
CORS(app)

#with app.app_context():
    #g.bias_bert_model = TFBertForSequenceClassification.from_pretrained("models/BIAS_BERT")

# generating relative path for reading in best model
base_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(base_dir, 'models','best_lstm_model_11_21.h5')


# load models from google drive if no such local folder
def load_model_from_drive(model_name, file_id, num_labels):
    output_path = f"models/{model_name}/tf_model.h5"

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Downloading {model_name} from Google Drive...")
        gdown.download(url, output_path, quiet=False)

    # initialize the model
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
    # load weights
    model.load_weights(output_path)
    return model


# Flask app initialization
with app.app_context():
    # Tokenizer 公用一个
    app.config["BERT_TOKENIZER"] = BertTokenizer.from_pretrained("bert-base-uncased")

    # load Bias BERT
    app.config["BIAS_BERT"] = load_model_from_drive(
        "BIAS_BERT",
        os.getenv("BIAS_BERT_FILE_ID"),
        num_labels=2
    )

    # load Message BERT
    app.config["MESSAGE_BERT"] = load_model_from_drive(
        "MESSAGE_BERT",
        os.getenv("MESSAGE_BERT_FILE_ID"),
        num_labels=8
    )


def bias_bert(input_data):
    tokenizer = app.config["BERT_TOKENIZER"]
    input_encoded = tokenizer(
        [input_data],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="tf"
    )

    model = app.config["BIAS_BERT"]
    outputs = model(input_encoded)
    pred = int(np.argmax(outputs.logits.numpy(), axis=1)[0])

    mapping = {0: "Neutral", 1: "Partisan"}
    return mapping[pred]


def message_bert(input_data):
    tokenizer = app.config["BERT_TOKENIZER"]
    input_encoded = tokenizer(
        [input_data],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="tf"
    )

    model = app.config["MESSAGE_BERT"]
    outputs = model(input_encoded)
    pred = int(np.argmax(outputs.logits.numpy(), axis=1)[0])

    mapping = {
        0: 'Attack', 1: 'Constituency', 2: 'Information',
        3: 'Media', 4: 'Mobilization', 5: 'Personal',
        6: 'Policy', 7: 'Support'
    }
    return mapping[pred]


@app.route('/generate_results', methods=['POST'])
def generate_results():
    #bias_bert_model = init_bias_bert()
    print('HELLO!!')
    data = request.get_json()

    
    input_data = data['userInput']
    print('input_data=', input_data)

    
    clean = re.sub(r'\\x[0-9A-Fa-f]{2}', '', input_data)
    clean = re.sub(r'[^\x00-\x7F]+', '', clean)
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    clean = re.sub(pattern, '', clean)
    clean = unescape(clean)

    input_data = clean

    bias_bert_result = bias_bert(input_data)
    message_bert_result = message_bert(input_data)
    #bias_gpt_result = bias_gpt(input_data)
    #bias_nb_result = bias_nb(input_data)


    # vec = vectorize_input(input_data)
    # Use the loaded NLP model to generate results
    # vec3 = [vec, vec, vec]
    # y_pred = loaded_model.predict([vec, vec, vec])
 
    # print(y_pred)
    # result = "Positive" if y_pred[0][0] >= y_pred[0][1] else "Negative"
    result = [0,0,0,0]

    result[0] = bias_bert_result
    result[1] = message_bert_result

    # return the result as a dictionary
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(port=5000)

