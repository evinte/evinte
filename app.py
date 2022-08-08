from flask import Flask, render_template, request
from keras.models import load_model
import math
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize

path = 'evaluator_qna.csv'
df = pd.read_csv(path)
df.drop('Unnamed: 0', axis=1, inplace=True)

stopwords =['i', 'me', 'my', 'myself', 'we', 'our','ours', 'ourselves','you', "you're", "you've", "you'll", "you'd", 'your','yours', 'yourself','yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which','who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an','the', 'and', 'but', 'if', 'or','because','as', 'until', 'while','of','at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below','to', 'from','up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',"wouldn't"]
vectorizer = CountVectorizer()
vectorizer.fit_transform(df['question'])

def remove_punctuation(text):
    text = str(text)
    text = text.strip('.,><?/:;[]{}\|=+_-!@#$%^&*()')
    return text

def preprocessing(text):
    text = text.lower()
    text = remove_punctuation(text)
    text = word_tokenize(text)
    #lemm = nltk.stem.WordNetLemmatizer()
    #text = [lemm.lemmatize(item) for item in text]
    cleaned_text = [item for item in text if item not in stopwords]
    text = ' '.join(text)
    return text

q_clean = [preprocessing(item) for item in df['question'].values]

evaluator_model = load_model('model.h5')
app = Flask(__name__)
def countParameters(netsales, totalasset, equity, netdebt, freecashflow, liability, longtermdebt, netincome, tax, inventory, revenue):
    canot_zero = [totalasset, freecashflow, liability, inventory, equity, revenue, totalasset]
    for items in canot_zero:
        if items == 0:
            return "sorry invalid input"
    assetturnover = float(netsales/totalasset)
    currentevfcf = float((equity+netdebt)/freecashflow)
    currentratio = float(totalasset/liability)
    inventoryturnover = float(netsales/inventory)
    ltdebtequity = float(longtermdebt/equity)
    pretaxmargin = float((netincome+tax)/revenue)
    quickratio = float((totalasset-inventory)/liability)
    roa = float(netincome/totalasset)
    roe = float(netincome/equity)
    der = float(netdebt/equity)
    parameter = [assetturnover, currentevfcf, currentratio, inventoryturnover, ltdebtequity, pretaxmargin, quickratio, roa,roe, der]
    
    input = [parameter]
    score = float(evaluator_model.predict(input))
    if score > 10:
        score = 10
    elif score < 0:
        score = 0
    score = 10-score
    return round(score*10,2)

def chatbot(text):
    if text == 'thank you' or text == 'bye':
        return "thank you for using factiva consultant! Is there any other thing that i can help for you?"
    elif text == 'hi':
        return "hi, I am factiva consultant, what can i help for you?"
    else :
        q = preprocessing(text)
        vec_question_list = vectorizer.transform(df['question'])
        veq_q = vectorizer.transform([q])
        sim_list = cosine_similarity(veq_q, vec_question_list)
        sim_max = np.max(sim_list)
        if sim_max > 0.5:
            question_idx = np.argmax(sim_list)
            ans = df['answer'].iloc[question_idx]
            return ans
        else :
            return "Sorry currently I cannot answer your question, you might try another question or use different keywords."

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/evaluator')
def man():
    return render_template('evaluator.html')

@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    netsales = float(request.form['revenue'])
    equity = float(request.form['equity'])
    totalasset = float(request.form['totalasset'])
    freecashflow = float(request.form['fcf'])
    liability = float(request.form['liability'])
    netdebt = liability
    longtermdebt = liability
    netincome = float(request.form['netincome'])
    inventory = float(request.form['inventory'])
    revenue = float(request.form['revenue'])
    tax = revenue - netincome
    assetturnover = float(netsales/totalasset)
    currentevfcf = float((equity+netdebt)/freecashflow)
    currentratio = float(totalasset/liability)
    inventoryturnover = float(netsales/inventory)
    ltdebtequity = float(longtermdebt/equity)
    pretaxmargin = float((netincome+tax)/revenue)
    quickratio = float((totalasset-inventory)/liability)
    roa = float(netincome/totalasset)
    roe = float(netincome/equity)
    der = float(netdebt/equity)
    evaluation = float(countParameters(netsales, totalasset, equity, netdebt, freecashflow, liability, longtermdebt, netincome, tax, inventory, revenue))
    #print("The healthiness score of your business financial is", float(p))
    parameter = [assetturnover, currentevfcf, currentratio, inventoryturnover, ltdebtequity, pretaxmargin, quickratio, roa,roe, der]
    parameter.append(evaluation)
    for items in parameter:
        items = round(items,2)
    return render_template('result.html', data=parameter)

@app.route('/chatbot')
def render():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot(userText)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
