# Mini Project: Titanic Survival Prediction (Assignments 1-8 Synthesis)
# Deployable via Streamlit/Docker [Krish Naik Roadmap insp.][youtube]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib  # Production save

# 1. Load & Transform (A1/A7)
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Features from EDA [file:92]
df['Fare_log'] = np.log1p(df['Fare'])  # Skew fix
Q1,Q3 = df['Fare'].quantile([0.25,0.75]); IQR=Q3-Q1
df['Fare_cap'] = df['Fare'].clip(Q1-1.5*IQR, Q3+1.5*IQR)

# NOTE: Select/encoding, splitting and model training are handled later inside
# the robust pipeline block to ensure consistent preprocessing and a saved
# artifact that the Streamlit app can load.

# 4. Save Production Model (use a robust pipeline)
# We'll build a pipeline that scales numeric features and saves metadata required
import os

def prepare_data(df):
    df = df.copy()
    df['Age'] = df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    Q1,Q3 = df['Fare'].quantile([0.25,0.75]); IQR = Q3-Q1
    df['Fare_cap'] = df['Fare'].clip(Q1-1.5*IQR, Q3+1.5*IQR)
    le = LabelEncoder()
    df['Sex_le'] = le.fit_transform(df['Sex'])
    X = pd.get_dummies(df[['Pclass','Sex_le','Age','Fare_cap','Embarked']], columns=['Embarked'], drop_first=True)
    y = df['Survived']
    feature_cols = list(X.columns)
    return X, y, feature_cols, (Q1,Q3), le

artifact_file = 'titanic_pipeline.pkl'
if not os.path.exists(artifact_file):
    X_full, y_full, feature_cols, (FQ1,FQ3), sex_le = prepare_data(df)
    # Train/test split and evaluate
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:,1])
    print(f"Trained model — Accuracy: {acc:.3f}, AUC: {auc:.3f}")
    # Save pipeline and metadata
    joblib.dump({'pipeline': pipeline,
                 'feature_cols': feature_cols,
                 'fare_q1': FQ1,
                 'fare_q3': FQ3,
                 'sex_classes': sex_le.classes_}, artifact_file)

# 5. Streamlit App
import streamlit as st
import joblib
import numpy as np

st.title('Titanic Survival Predictor')

artifact = joblib.load(artifact_file)
pipeline = artifact['pipeline']
feature_cols = artifact['feature_cols']
FQ1 = artifact['fare_q1']
FQ3 = artifact['fare_q3']
sex_classes = list(artifact['sex_classes'])
# Map sex input to training encoding
sex_map = {cls: i for i, cls in enumerate(sex_classes)}

Pclass = st.selectbox('Pclass', [1,2,3])
Sex = st.selectbox('Sex', sex_classes)
Age = st.number_input('Age', min_value=0.0, max_value=120.0, value=30.0)
Fare = st.number_input('Fare', min_value=0.0, value=20.0)
Embarked = st.selectbox('Embarked', ['C','Q','S'])

if st.button('Predict'):
    # Preprocess single input to match training features
    Sex_le = sex_map[Sex]
    # Fare cap using training quantiles
    IQR = FQ3 - FQ1
    Fare_cap = np.clip(Fare, FQ1 - 1.5*IQR, FQ3 + 1.5*IQR)
    # Build input row with all feature cols present
    row = dict.fromkeys(feature_cols, 0.0)
    row['Pclass'] = Pclass
    row['Sex_le'] = Sex_le
    row['Age'] = Age
    row['Fare_cap'] = Fare_cap
    # Embarked dummies (drop_first=True during training removed one level)
    if f'Embarked_Q' in feature_cols:
        row['Embarked_Q'] = 1.0 if Embarked == 'Q' else 0.0
    if f'Embarked_S' in feature_cols:
        row['Embarked_S'] = 1.0 if Embarked == 'S' else 0.0
    # Build a DataFrame so sklearn sees feature names (avoids warnings)
    import pandas as pd
    inp = pd.DataFrame([row], columns=feature_cols).astype(float)
    prob = pipeline.predict_proba(inp)[0,1]
    pred = pipeline.predict(inp)[0]
    st.write(f"Survival probability: {prob:.3f}")
    st.write('Prediction:', 'Survived' if pred==1 else 'Not Survived')

