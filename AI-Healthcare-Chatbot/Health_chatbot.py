import re
import random
import pandas as pd
import numpy as np  
import csv 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from difflib import get_close_matches
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- loading data --- #
traning = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

# clean duplicate column names
traning.columns = traning.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)

traning = traning.loc[:, ~traning.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# features and labels
cols = traning.columns[:-1]
x = traning[cols]
y = traning['prognosis']

# encode target
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

# train_test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# dictionaries
severityDictioary = {}
description_list = {}
precausionDictionary = {}
symptoms_dict = {symptom: idx for idx, symptom in enumerate(x.columns)}

# load description
def getDescription():
    with open('MasterData/symptom_Description.csv') as csv_file:
        for row in csv.reader(csv_file):
            description_list[row[0]] = row[1]

# load severity
def getSeverityDict():
    with open('MasterData/symptom_severity.csv') as csv_file:
        for row in csv.reader(csv_file):
            try:
                severityDictioary[row[0]] = int(row[1])
            except:
                pass

# load precautions
def getpercautionDict():
    with open('MasterData/symptom_precaution.csv') as csv_file:
        for row in csv.reader(csv_file):
            precausionDictionary[row[0]] = [row[1], row[2], row[3], row[4]]

# predefined synonyms
symptoms_synonyms = {
    "stomach ache": "stomach_pain",
    "belly pain": "stomach_pain",
    "tummy pain": "stomach_pain",
    "loose motion": "diarrhea",
    "motions": "diarrhea",
    "high temperature": "fever",
    "feaver": "fever",
    "coughing": "cough",
    "throat pain": "sore_throat",
    "cold": "chills",
    "breathing issue": "breathlessness",
    "shortness of breath": "breathlessness",
    "body ache": "muscle_pain",
}

# extract symptoms
def extract_symptoms(user_input, all_symptoms):
    extracted = []
    text = user_input.lower().replace("-", " ")

    # synonym mapping
    for phrase, mapped in symptoms_synonyms.items():
        if phrase in text:
            extracted.append(mapped)

    # direct match
    for symptom in all_symptoms:
        if symptom.replace("_", " ") in text:
            extracted.append(symptom)

    # fuzzy match (typo handling)
    words = re.findall(r"\w+", text)
    for word in words:
        close = get_close_matches(word, [s.replace("_", " ") for s in all_symptoms], n=1, cutoff=0.8)
        if close:
            for sym in all_symptoms:
                if sym.replace("_", " ") == close[0]:
                    extracted.append(sym)

    return list(set(extracted))

# prediction
def predict_disease(symptoms_list):
    input_vector = np.zeros(len(symptoms_dict))

    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    pred_proba = model.predict_proba([input_vector])[0]
    pred_class = np.argmax(pred_proba)
    disease = le.inverse_transform([pred_class])[0]
    confidence = round(pred_proba[pred_class] * 100, 2)

    return disease, confidence, pred_proba

# empathy quotes
quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# chatbot
def chatbot():
    getSeverityDict()
    getDescription()
    getpercautionDict()

    print("Welcome to Healthcare ChatBot 🤖")
    print("Answer a few questions so I can help you.\n")

    # basic info
    name = input("What is your name? ")
    age = input("Enter your age: ")
    gender = input("Enter your gender: ")

    # symptoms
    symptoms_input = input("\nDescribe your symptoms (e.g., 'I have fever and stomach pain'): ")
    symptoms_list = extract_symptoms(symptoms_input, cols)

    if not symptoms_list:
        print("Sorry, I could not detect valid symptoms. Please try again.")
        return

    print("\nDetected symptoms:", ", ".join(symptoms_list))

    # additional inputs
    num_days = int(input("For how many days have you had these symptoms? "))
    severity_scale = int(input("On a scale of 1-10, how severe is it? "))
    pre_exist = input("Any pre-existing conditions? ")
    lifestyle = input("Do you smoke/drink or have poor sleep? ")
    family = input("Family history of illness? ")

    # initial prediction
    disease, confidence, _ = predict_disease(symptoms_list)

    print("\nLet me ask a few more questions related to", disease)

    disease_symptoms = list(traning[traning['prognosis'] == disease].iloc[0][:-1].index[
        traning[traning['prognosis'] == disease].iloc[0][:-1] == 1
    ])

    asked = 0
    for sym in disease_symptoms:
        if sym not in symptoms_list and asked < 8:
            ans = input(f"Do you also have {sym.replace('_',' ')}? (yes/no): ").strip().lower()
            if ans == "yes":
                symptoms_list.append(sym)
            asked += 1

    # final prediction
    disease, confidence, _ = predict_disease(symptoms_list)

    print("\n------------- RESULT ------------")
    print(f"You may have: {disease}")
    print(f"Confidence: {confidence}%")
    print(f"About: {description_list.get(disease, 'No description available.')}")

    # precautions
    if disease in precausionDictionary:
        print("\nSuggested Precautions:")
        for i, prec in enumerate(precausionDictionary[disease], 1):
            print(f"{i}. {prec}")

    # empathy quote
    print("\n" + random.choice(quotes))
    print(f"\nThank you for using this ChatBot, take care {name}! ❤️")

# run
if __name__ == "__main__":
    chatbot()