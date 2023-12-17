import nltk
from nltk.stem.lancaster import LancasterStemmer
import tflearn
import tensorflow as tf
import numpy as np
import pickle
import random
import re
from flask import Flask, request, jsonify, render_template
import json

app = Flask(__name__)
stemmer = LancasterStemmer()

def bag_of_words(s, words):
    bag = [0 for x in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for si in s_words:
        for i, w in enumerate(words):
            if w == si:
                bag[i] = 1
    return np.array(bag)

def load_or_train_model(training, output):
    tf.compat.v1.reset_default_graph()
    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load('model.tflearn')
    except:
        model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
        model.save('model.tflearn')
    return model

def process_math_question(question):
    match = re.match(r'(\d+)\s*([*/+-])\s*(\d+)', question)
    if match:
        num1, operator, num2 = match.groups()
        num1, num2 = int(num1), int(num2)
        if operator == '+':
            result = num1 + num2
        elif operator == '-':
            result = num1 - num2
        elif operator == '*':
            result = num1 * num2
        elif operator == '/':
            if num2 == 0:
                return 'Invalid divide math operation'
            result = num1 / num2
        return result
    return None
with open('intents.json') as js_file:
    data = json.load(js_file)
with open('data.pickle', 'rb') as pick_file:
    words, labels, training, output = pickle.load(pick_file)
model = load_or_train_model(training, output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response',methods=['POST'])
def get_response():
    user_input = request.json.get('user_input', '')
    if not user_input:
        return jsonify({'bot_response': "I bring my appologies, but I don't rcognise your request."})
    try:
        math_result = process_math_question(user_input)
        if math_result is not None:
            return jsonify({'bot_response': f"Reuslt: {math_result}"})
        results = model.predict([bag_of_words(user_input, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        if len(responses) < 3:
            responses.append("I bring my appologies, but my data collection is not full")
        bot_response = random.choice(responses)
        if 'name' in user_input.lower():
            bot_response = f"Hello, {random.choice(['friend', 'stranger', 'man'])}!{bot_response}"
        return jsonify({'bot_response': bot_response})
    except Exception as e:
        return jsonify({'bot_response': f"Error: {str(e)}"})
    
if __name__ == '__main__':
    app.run(debug=True)