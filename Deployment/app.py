from flask import Flask, render_template, request
import requests
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
import spacy


app = Flask(__name__)

mms = pickle.load(open('mms.pkl','rb'))
ohe = pickle.load(open('ohe.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html.txt')


@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
        
        div_name = request.form['Division']
        
        dept_name = request.form['Dept_Name']
        
        class_name = request.form['Class_Name']
        
        age = request.form['Age']
        age = int(age)
        
        pfc = request.form['pfc']
        pfc = int(pfc)
        
        review = str(request.form['Review'])
        
        cols = ['Age','Positive Feedback Count','Division Name','Department Name','Class Name','Review']
        #user values to dataframe
        test_df = pd.DataFrame([[age,pfc,div_name,dept_name,class_name, review]], columns=cols)
        #applied sqrt twice
        for i in ['Age','Positive Feedback Count']:
            test_df[i] = test_df[i].apply(lambda x : np.sqrt(x))
            test_df[i] = test_df[i].apply(lambda x : np.sqrt(x))

        
### OHE

        ohe_test = test_df[['Division Name','Department Name','Class Name']]
        
        ohe_df = pd.DataFrame(ohe.transform(ohe_test))
        
        
   
   
   

        ohe_col = ['Division Name_Initmates',
 'Department Name_Dresses',
 'Department Name_Intimate',
 'Department Name_Jackets',
 'Department Name_Tops',
 'Department Name_Trend',
 'Class Name_Dresses',
 'Class Name_Fine gauge',
 'Class Name_Intimates',
 'Class Name_Jackets',
 'Class Name_Jeans',
 'Class Name_Knits',
 'Class Name_Layering',
 'Class Name_Legwear',
 'Class Name_Lounge',
 'Class Name_Pants',
 'Class Name_Shorts',
 'Class Name_Skirts',
 'Class Name_Sleep',
 'Class Name_Sweaters',
 'Class Name_Swim',
 'Class Name_Trend']
        
        
            

 
        ohe_df.columns = ohe_col

        ohe_test = ohe_test.reset_index(drop = True)
       
        
        
   
        ohe_df['Age'] = test_df['Age'].copy()
        ohe_df['Positive Feedback Count'] = test_df['Positive Feedback Count'].copy()
        
        print(ohe_df.shape)
        df1 = ohe_df.copy()

        

### MMS


        df1.iloc[:,[-2,-1]] = mms.transform(df1.iloc[:,[-2,-1]])
        
        
        
### Re
    
        def regex(string):
            s1 = re.sub(r'[^a-zA-Z\s]','',string)
            s2 = re.sub(r'\s{2,}','', s1)
            s3 = re.sub(r'[\d_]+','',s2)
            words = [w for w in s3.split(' ') if len(w) > 1]
            return ' '.join(words)
        

        def reduce_length(string):
    
            word = re.compile(r"(.)\1{2,}")
            return word.sub(r'\1\1', string)


        sp = spacy.load('en_core_web_sm')
        
        def s_lemmatizer(string):
            text = sp(string)
            return ' '.join(word.lemma_ for word in text)
        
        x1 = test_df['Review'].copy()
        x2 = df1.copy()

        x1 = x1.apply(regex)    
        x1 = x1.apply(reduce_length)
        x1 = x1.apply(s_lemmatizer)
        
        
        x1 = tfidf.transform(x1)
        x1 = pd.DataFrame(x1.toarray())
        
        
        x1 = x1.reset_index(drop=True)
        x2 = x2.reset_index(drop=True)
        
        df2 = pd.concat([x1,x2],axis=1)
        
        
        label = model.predict(df2)
        label_prob = model.predict_proba(df2)[:,1]
        
        if label==1:
            output= 'The probability of customer recommending the product is:' + str(round(label_prob[0]*100, 2)) + '%'
        
        elif label==0:
            output = 'The probability of customer not recommending the product is:' + str(round((1-label_prob[0])*100, 2)) + '%'
    

        return render_template('result.html.txt',prediction_text=output)





if __name__=="__main__":
    app.run(debug=True)