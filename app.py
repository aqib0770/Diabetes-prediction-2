from flask import Flask,render_template,app,request
import pickle

app=Flask(__name__)


scaler=pickle.load(open('models/scaler.pkl','rb'))
model=pickle.load(open('models/svc_classifier.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""
    if request.method=='POST':
        Pregnancies=int(request.form.get('Pregnancies'))
        Glucose=float(request.form.get('Glucose'))
        BloodPressure=float(request.form.get("BloodPressure"))
        SkinThickness=float(request.form.get("SkinThickness"))
        Insulin=float(request.form.get('Insulin'))
        Bmi=float(request.form.get('Bmi'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age=int(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,
                                    Insulin,Bmi,DiabetesPedigreeFunction,Age]])
        prediction = model.predict(new_data)

        if prediction[0]==1:
            result='Diabetic'
        else:
            result='Non-Diabetic'
        
        return render_template('home.html',results=result)
    else:
        return render_template('home.html')
    
if __name__=='__main__':
    app.run(host="0.0.0.0")