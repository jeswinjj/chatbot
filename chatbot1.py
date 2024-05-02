import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
from tkinter import *
from tkinter import ttk 

warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)
print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def calc_condition(exp, days):
    sum_severity = sum(severityDictionary[symptom] for symptom in exp)
    if (sum_severity * days) / (len(exp) + 1) > 13:
        print("You should take the consultation from a doctor.")
    else:
        print("It might not be that bad but you should take precautions.")

def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)

def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:

        print("\nEnter the symptom you are experiencing  \t\t", end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ")", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            print("Are you experiencing any ")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if(inp == "yes" or inp == "no"):
                        break
                    else:
                        print("provide proper answers i.e. (yes/no) : ", end="")
                if(inp == "yes"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if(present_disease[0] == second_prediction[0]):
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])

            else:
                print("You may have ", present_disease[0], "or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            print("Take following measures : ")
            for i, j in enumerate(precution_list):
                print(i+1, ")", j)

    recurse(0, 1)

def predict_disease(symptoms_exp, num_days):
    symptoms_dict = {symptom: index for index, symptom in enumerate(cols)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    predicted = clf.predict([input_vector])[0]
    predicted_disease = le.inverse_transform([predicted])[0]

    sum_severity = sum(severityDictionary[symptom] for symptom in symptoms_exp)
    avg_severity = sum_severity / (len(symptoms_exp) + 1)
    
    if avg_severity > 13:
        print("You should take the consultation from doctor. ")
    else:
        print("It might not be that bad but you should take precautions.")

    return predicted_disease

getDescription()
getSeverityDict()
getprecautionDict()

root = Tk()
root.title("Healthcare Chatbot")
root.geometry("800x600")

def predict_disease_gui():
    name = name_entry.get()
    greeting_message = f"Hello, {name}!"
    chat_display.insert(END, greeting_message + "\n\n")
    symptoms_exp = [symptom1_var.get(), symptom2_var.get(), symptom3_var.get(), symptom4_var.get(), symptom5_var.get()]
    num_days = int(num_days_entry.get())

    # Placeholder logic for predicting disease based on selected symptoms
    predicted_disease = predict_disease(symptoms_exp, num_days)

    # Display the predicted disease and its description
    prediction_text = f"Predicted Disease: {predicted_disease}\n\nDescription: {description_list.get(predicted_disease, 'No description available')}"
    precaution_text = f"\n\n Precautions : {precautionDictionary.get(predicted_disease, 'No precaution available')}"
    chat_display.insert(END, prediction_text + precaution_text +"\n\n")

def clear_chat():
    chat_display.delete("1.0", END)

name_label = Label(root, text="Enter your name:")
name_label.pack()

name_entry = Entry(root, width=30)
name_entry.pack()


def load_symptom_options():
    options = ['acidity','abdominal_pain', 'bloody_stool', 'blurred_and_distorted_vision', 'brittle_nails', 'increased_appetite', 'abnormal_menstruation', 
'back_pain', 'blackheads',  'continuous_feel_of_urine', 'blurred_and_distorted_vision', 'chest_pain', 'enlarged_thyroid', 'diarrhoea', 'irritability', 'depression', 'muscle_pain',
'bladder_discomfort', 'bladder_discomfort', 'breathlessness', 'cough', 'chest_pain', 'fast_heart_rate', 'excessive_hunger', 'excessive_hunger', 'mild_fever', 'polyuria',
'breathlessness', 'blister', 'chest_pain', 'dark_urine', 'cough', 'loss_of_appetite', 'muscle_pain', 'increased_appetite', 'polyuria', 'yellowing_of_eyes',
'burning_micturition', 'bruising', 'dischromic _patches', 'diarrhoea', 'nausea',  'muscle_weakness', 'swollen_extremeties',
'chest_pain', 'chest_pain',  'obesity',
'chills',  'diarrhoea', 'extra_marital_contacts', 'family_history', 'prominent_veins_on_calf',
'constipation', 'cold_hands_and_feets', 'dischromic _patches', 'family_history', 'headache', 'puffy_face_and_eyes',
'continuous_sneezing', 'dehydration',  'irregular_sugar_level', 'internal_itching', 'yellowing_of_eyes',
'cough', 'dizziness', 'extra_marital_contacts', 'irritation_in_anus', 'irregular_sugar_level',
'cramps', 'foul_smell_of urine', 'foul_smell_of urine', 'lack_of_concentration', 'irritation_in_anus',
'fatigue',  'lethargy', 'lack_of_concentration',
'headache', 'knee_pain', 'hip_joint_pain', 'loss_of_appetite', 'loss_of_balance',
'high_fever', 'loss_of_appetite', 'lethargy', 'loss_of_balance', 'mucoid_sputum',
'indigestion', 'nausea', 'loss_of_appetite', 'movement_stiffness', 
'joint_pain', 'neck_pain', 'loss_of_balance', 'painful_walking', 'painful_walking',
'mood_swings', 'nodal_skin_eruptions', 'mood_swings', 'passage_of_gases', 'swollen_blood_vessels',
'muscle_wasting', 'pain_during_bowel_movements', 'movement_stiffness', 
'muscle_weakness',   
'neck_pain', 'patches_in_throat', 'neck_pain', 'swollen_legs',
'pain_during_bowel_movements', 'restlessness', 'nodal_skin_eruptions', 
'patches_in_throat', 'shivering', 'obesity', 
'pus_filled_pimples', 'pain_in_anal_region', 'yellow_crust_ooze',
'shivering', 'stiff_neck', 'red_sore_around_nose', 'yellowing_of_eyes',
'skin_rash', 'sweating', 'restlessness',
'stiff_neck', 'swelling_joints', 'scurring',
'stomach_pain',  
'sunken_eyes',  'ulcers_on_tongue',
'vomiting', 'weakness_in_limbs', 'watering_from_eyes',
'weight_gain', 'yellowish_skin',
'weight_loss',

]  # List to store options for each symptom
    with open('symptoms.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            options.append(row[''])  
    return options

def update_dropdown_options1(event):
    entered_text = event.widget.get().lower()  # Get the entered text from the event object and convert to lowercase
    filtered_options = [options for options in symptom1_options if entered_text in options.lower()]  # Filter options based on entered text
    Symptom1_dropdown['values'] = filtered_options  # Update dropdown options

def update_dropdown_options2(event):
    entered_text = event.widget.get().lower()  
    filtered_options = [options for options in symptom2_options if entered_text in options.lower()]  
    Symptom2_dropdown['values'] = filtered_options  

def update_dropdown_options3(event):
    entered_text = event.widget.get().lower()  
    filtered_options = [options for options in symptom3_options if entered_text in options.lower()]  
    Symptom3_dropdown['values'] = filtered_options  

def update_dropdown_options4(event):
    entered_text = event.widget.get().lower()  
    filtered_options = [options for options in symptom4_options if entered_text in options.lower()]  
    Symptom4_dropdown['values'] = filtered_options  

def update_dropdown_options5(event):
    entered_text = event.widget.get().lower()  
    filtered_options = [options for options in symptom5_options if entered_text in options.lower()]  
    Symptom5_dropdown['values'] = filtered_options  

# Load options for each symptom
symptom1_options = load_symptom_options()
symptom2_options = load_symptom_options()
symptom3_options = load_symptom_options()
symptom4_options = load_symptom_options()
symptom5_options = load_symptom_options()

# Create StringVar variables to store selected options
symptom1_var = StringVar(root)
symptom2_var = StringVar(root)
symptom3_var = StringVar(root)
symptom4_var = StringVar(root)
symptom5_var = StringVar(root)

# Set default value for dropdown menus
# symptom1_var.set(symptom1_options[0])  # Set default value to the first option
# symptom2_var.set(symptom2_options[0])
# symptom3_var.set(symptom3_options[0])
# symptom4_var.set(symptom4_options[0])
# symptom5_var.set(symptom5_options[0])


Symptom1_label = Label(root, text="Select Symptom 1:")
Symptom1_label.pack()
Symptom1_dropdown = ttk.Combobox(root, width=30,textvariable=symptom1_var, values=symptom1_options)
Symptom1_dropdown.pack()

# Bind KeyRelease event to the Symptom1 textbox to update dropdown options
Symptom1_dropdown.bind('<KeyRelease>', update_dropdown_options1)



Symptom2_label = Label(root, text="Select Symptom 2:")
Symptom2_label.pack()
Symptom2_dropdown = ttk.Combobox(root, width=30, textvariable=symptom2_var, values=symptom2_options)
Symptom2_dropdown.pack()

Symptom2_dropdown.bind('<KeyRelease>', update_dropdown_options2)

Symptom3_label = Label(root, text="Select Symptom 3:")
Symptom3_label.pack()
Symptom3_dropdown = ttk.Combobox(root, width=30, textvariable=symptom3_var, values=symptom3_options)
Symptom3_dropdown.pack()

Symptom3_dropdown.bind('<KeyRelease>', update_dropdown_options3)

Symptom4_label = Label(root, text="Select Symptom 4:")
Symptom4_label.pack()
Symptom4_dropdown = ttk.Combobox(root, width=30, textvariable=symptom4_var, values=symptom4_options)
Symptom4_dropdown.pack()

Symptom4_dropdown.bind('<KeyRelease>', update_dropdown_options4)

Symptom5_label = Label(root, text="Select Symptom 5:")
Symptom5_label.pack()
Symptom5_dropdown = ttk.Combobox(root, width=30, textvariable=symptom5_var, values=symptom5_options)
Symptom5_dropdown.pack()

Symptom5_dropdown.bind('<KeyRelease>', update_dropdown_options5)

num_days_label = Label(root, text="Enter the number of days you've had symptoms:")
num_days_label.pack()

num_days_entry = Entry(root, width=30)
num_days_entry.pack()

predict_button = Button(root, text="Predict Disease", command=predict_disease_gui)
predict_button.pack()

clear_button = Button(root, text="Clear Chat", command=clear_chat)
clear_button.pack()

chat_display = Text(root, width=60, height=20)
chat_display.pack()

root.mainloop()
