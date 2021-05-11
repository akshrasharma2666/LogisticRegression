import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/drive/My Drive/ML_sem6/logisticmodel.pkl', 'rb')) 
# Feature Scaling
dataset = pd.read_csv('Classification Dataset2.csv')
# Extracting independent variable:
X = dataset.iloc[:, [1,2,3]].values
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(UserID, Gender,Age,EstimatedSalary):
  output= model.predict(sc.transform([[Gender,Age,EstimatedSalary]]))
  print("Output:", output)
  if output==[1]:
    prediction="Patient have no disease"
  else:
    prediction="Patient has disease"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:Brown;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:black;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:black;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:black;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Classifier to predict whether new patient will have that disease or not")
    
    st.text_input("Insulin","")
    
    #Gender1 = st.select_slider('Select a Gender Male:1 Female:0',options=['1', '0'])
    Gender1 = st.number_input('Insert Gender Male:1 Female:0')
    Age = st.number_input('Insert a Age',18,60)
   
    EstimatedSalary = st.number_input("Insert Estimated Salary",15000,150000)
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(UserID, Gender1,Age,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
      
    if st.button("About"):
      st.subheader("Developed ")
      st.subheader("Head , Department of Computer Engineering")

if __name__=='__main__':
  main()