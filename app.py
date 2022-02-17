import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import time
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_data():
    return pd.read_csv('diabetes_data_upload.csv')

def train_model():
    x = 0.2
    data = get_data()

    from sklearn.preprocessing import LabelEncoder
    objectList = data.select_dtypes(include = 'object').columns
    le = LabelEncoder()
    for i in objectList:
        data[i] = le.fit_transform(data[i])

    X = data.drop(["class"],axis=1)
    y = data["class"]
    
    from sklearn.preprocessing import MinMaxScaler
    mm = MinMaxScaler()
    X[['Age']] = mm.fit_transform(X[['Age']])
  
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=x)

    modelos = ['CatBoost', 'Random Forest']

    column_names = ["Model","Accuracy","Precision","Recall","F1","Classifier"]
    results = pd.DataFrame(columns = column_names)

    for i in range(0,len(modelos)):

        if i == 0:
            from catboost import CatBoostClassifier
            classifier = CatBoostClassifier(iterations=2,
                                            learning_rate=1,
                                            depth=2)

        elif i == 1:
            from sklearn.ensemble import RandomForestClassifier
            classifier = RandomForestClassifier(n_estimators=100)

        start_time = time.time()
        classifier.fit(X_train,y_train)
        time_train = time.time() - start_time
        
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        time_test = time.time() - start_time
    
        from sklearn import metrics
        acc = metrics.accuracy_score(y_test, y_pred)*100
        prc = metrics.precision_score(y_test, y_pred)*100
        rec = metrics.recall_score(y_test, y_pred)*100
        f1 = metrics.f1_score(y_test, y_pred)*100

        
        data = [[modelos[i],acc, prc, rec, f1,classifier,time_train,time_test]]
        column_names = ["Model","Accuracy","Precision","Recall","F1",
                        "classifier", "time_train","time_test"]
        model_results = pd.DataFrame(data = data, columns = column_names)
        results = results.append(model_results, ignore_index = True)

    return results

data = get_data()
html_temp = """
  <div style="background-color:blue;padding:10px">
  <h2 style="color:white;text-align:center;"> Early Stage Diabetes Risk Prediction app </h2>
  </div>

  """
st.markdown(html_temp, unsafe_allow_html=True)

image = Image.open('Logo.jpeg')
st.sidebar.image(image, width=100, height=50)

st.sidebar.subheader("Health informations")
In1 =  st.sidebar.number_input("Age", min_value=20,max_value=65,step=1)
In2 =  st.sidebar.selectbox("Gender:", ["Man","Women"])
In3 =  st.sidebar.selectbox("Polyuria:",["No","Yes"])
In4 =  st.sidebar.selectbox("Polydipsia:",["No","Yes"])
In5 =  st.sidebar.selectbox("sudden weight loss:",["No","Yes"])
In6 =  st.sidebar.selectbox("weakness:",["No","Yes"])
In7 =  st.sidebar.selectbox("Polyphagia:",["No","Yes"])
In8 =  st.sidebar.selectbox("Genital thrush :",["No","Yes"])
In9 =  st.sidebar.selectbox("visual blurring :",["No","Yes"])
In10 = st.sidebar.selectbox("Itching:",["No","Yes"])
In11 = st.sidebar.selectbox("Irritability:",["No","Yes"])
In12 = st.sidebar.selectbox("delayed healing:",["No","Yes"])
In13 = st.sidebar.selectbox("partial paresis:",["No","Yes"])
In14 = st.sidebar.selectbox("muscle stiffness:",["No","Yes"])
In15 = st.sidebar.selectbox("Alopecia:",["No","Yes"])
In16 = st.sidebar.selectbox("Obesity:",["No","Yes"])

results = train_model()

btn_predict = st.sidebar.button("SUBMIT")



st.subheader("")

if btn_predict:
    values = [In1,In2,In3,In4,In5,In6,In7,In8,In9,In10,In11,In12,In13,In14,In15,In16]
    column_names = ["Age","Gender","Polyuria","Polydipsia","sudden weight loss","weakness","Polyphagia","Genital thrush	",\
                    "visual blurring","Itching","Irritability",\
                    "delayed healing","partial paresis","muscle stiffness","Alopecia","Obesity"]
    df = pd.DataFrame(values, column_names)

    st.write(df)

    if df[0][1] == 'Man':
        df[0][1] = 1
    elif df[0][1] == 'Women':
        df[0][1] = 0

    for x in range(2, 16):
        if df[0][x] == 'Yes':
            df[0][x] = 1
        elif df[0][x] == 'No':
            df[0][x] = 0

    df[0][0] = (df[0][0] - 16) / 74
        

    pred = [list(df[0])]

    classifier_best = results['classifier'][results['Recall'] == results['Recall'].max()].values
    classifier = classifier_best[0]

    model_best = results['Model'][results['Recall'] == results['Recall'].max()].values
    model = model_best[0]

    result = classifier.predict(pred)

    result = result[0]

    if result == 0: st.error('your result: **NEGATIVE**.  \n Diagnosis suggests that patient have Diabetes Risk.  \n Please get checked soon')
    if result == 1: st.success("your result: **POSITIVE**.  \n  "
                               + "Diagnosis suggests that patient does not have Diabetes Risk.")


