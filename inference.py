import pickle
from flask import Flask,request,jsonify
import pandas as pd


app = Flask(__name__)

X_train = pd.read_csv('X_train.csv')
with open('ridge.pkl','rb') as file:
    loaded_model = pickle.load(file)


def minmaxscaler_xtest(X_test):
    with open('min_max_scaler.pkl','rb') as file:
        minmaxscaler = pickle.load(file)
        X_test_scaled = minmaxscaler.transform(X_test.select_dtypes('number'))
        X_test_num = pd.DataFrame(X_test_scaled, columns=X_test.select_dtypes('number').columns, index=X_test.index)
        return X_test_num



def onehot(X_test):
    with open('ohe.pkl','rb') as file:
        ohe = pickle.load(file)
    encoded_data = ohe.transform(X_test[['Mjob','Fjob','reason','guardian','famsize','Pstatus','paid','activities','nursery','address','sex']])
    encoded_df = pd.DataFrame(encoded_data.toarray(),
                                  columns=ohe.get_feature_names_out(['Mjob','Fjob','reason','guardian','famsize','Pstatus','paid','activities','nursery','address','sex']), index = X_test.index)
    return encoded_df

def get_xtest(X_test,X_train = X_train):
    X_test_nom = onehot(X_test)
    X_test_num = minmaxscaler_xtest(X_test)
    X_test = pd.concat([X_test_num, X_test_nom], axis=1)
    print('X_test columns : ', X_test.columns)
    print('X_train columns :', X_train.columns)
    X_test.columns = X_train.columns
    return X_test




@app.route('/process_test_data', methods = ['POST'])

def process_test_data():
    data = request.get_json()
    medu = data['Medu']
    fedu = data['Fedu']
    traveltime = data['traveltime']
    studytime = data['studytime']
    freetime = data['freetime']
    health = data['health']
    sex = data['sex']
    address = data['address']
    famsize = data['famsize']
    pstatus = data['Pstatus']
    paid = data['paid']
    activities = data['activities']
    nursery = data['nursery']
    internet = data['internet']
    mjob = data['mjob']
    fjob = data['fjob']
    reason = data['reason']
    guardian = data['guardian']
    X_test = pd.DataFrame({'Medu':[medu],'Fedu':[fedu],'traveltime':[traveltime],'studytime':[studytime],'freetime':[freetime],
                           'health':[health],'sex':[sex],'address':[address],'famsize':[famsize],'Pstatus':[pstatus],
                           'paid':[paid],'activities':[activities],'nursery':[nursery],'internet':[internet],'Mjob':[mjob],
                           'Fjob':[fjob],'reason':[reason],'guardian':[guardian]})
    X_test = get_xtest(X_test)
    prediction = loaded_model.predict(X_test)
    return str(prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

















