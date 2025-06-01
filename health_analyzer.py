


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_and_process():
    df = pd.read_csv(r'F:\code in place\Project\students_mental_health_survey.csv')  # Dataset: http://kaggle.com/datasets/sonia22222/students-mental-health-assessments
    df = df.dropna()
    #If the person has depression score more than 2 and anxiety score more than 2.then the person is at risk
    df['has_issue']=((df['Depression_Score']>2) | (df['Anxiety_Score']>2)).astype(int)
    features= ['Gender','Course','CGPA','Stress_Level','Sleep_Quality','Physical_Activity',
        'Diet_Quality','Social_Support','Relationship_Status','Substance_Use',
        'Counseling_Service_Use','Family_History','Chronic_Illness',
        'Financial_Stress','Extracurricular_Involvement']
    label_encoders={}
    for column in features:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le
    return df, features, label_encoders

def train_model(df,features):
    X = df[features]
    y = df['has_issue']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    print("\nModel Evaluation")
    print("Accuracy:",accuracy_score(y_test,predictions))
    print("\n",classification_report(y_test,predictions))
    return model

def predict_user(model,label_encoders):
    print("\nEnter Student Info for Risk Prediction:")
    user_input = {}
    user_input['Gender']=input("Gender (Male/Female): ").strip().capitalize()
    user_input['Course']=input("Course (Business,Engineering,Law,Medical etc): ").strip().title()
    user_input['CGPA']=float(input("CGPA(0.0-4.0): "))
    user_input['Stress_Level']=int(input("Stress Level(0-4): "))

    
    user_input['Sleep_Quality']=input("Sleep Quality(Poor/Average/Good): ").strip().capitalize()
    user_input['Physical_Activity']=input("Physical Activity(Low/Moderate/High): ").strip().capitalize()
    user_input['Diet_Quality']=input("Diet Quality(Poor/Average/Good): ").strip().capitalize()
    user_input['Social_Support']=input("Social Support(Low/Moderate/High): ").strip().capitalize()
    user_input['Relationship_Status'] =input("Relationship Status(Single/Married/In a Relationship): ").strip().title()
    user_input['Substance_Use']=input("Substance Use(Never/Occasionally/Frequently): ").strip().capitalize()
    user_input['Counseling_Service_Use']=input("Counseling Service Use (Never/Occasionally/Frequently): ").strip().capitalize()
    user_input['Family_History']=input("Family History of Mental Illness (Yes/No): ").strip().capitalize()
    user_input['Chronic_Illness']=input("Chronic Illness (Yes/No): ").strip().capitalize()
    user_input['Financial_Stress']=int(input("Financial Stress(0-5): "))
    user_input['Extracurricular_Involvement']=input("Extracurricular Involvement(Low/Moderate/High): ").strip().capitalize()



    input_encoded=[]
    for feature, value in user_input.items():
        if feature in label_encoders:
            le=label_encoders[feature]
            try:
                encoded_val=le.transform([value])[0]
            except ValueError:
                print(f"Invalid input for {feature}.Please restart and try again.")
                return
            input_encoded.append(encoded_val)
        else:
            input_encoded.append(value)
    prediction=model.predict([input_encoded])[0]
    print("\nPrediction Result:")
    if prediction==1:
        print("High Risk of Mental Health Issues.Recommend counseling.")
    else:
        print("No high risk detected.Keep maintaining your mental wellness!")




def show_data_insights(df):
    print("\nMental Health Issue Distribution:")
    print(df['has_issue'].value_counts())
    sns.countplot(x='has_issue',data=df)
    plt.title("Mental Health Risk Distribution")
    plt.xlabel("Has Mental Health Issue")
    plt.ylabel("Number of Students")
    plt.xticks([0, 1],['No Issue','Has Issue'])
    plt.show()



def main():
    print("Student Mental Health Risk Analyzer")
    df,features,encoders=load_and_process()
    model=train_model(df,features)
    while True:
        print("\n options:")
        print("1.Predict Mental Health Risk")
        print("2.Show Data Insights")
        print("3.Exit")
        choice=input("Choose an option: ")
        if choice== "1":
            predict_user(model,encoders)
        elif choice=="2":
            show_data_insights(df)
        elif choice=="3":
            print("Goodbye!Stay mentally strong and healthy!")
            break
        else:
            print("Invalid choice.Please try again.")
if __name__ =="__main__":
    main()
