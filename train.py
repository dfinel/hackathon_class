import pickle

from sklearn.linear_model import Ridge
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def get_df():
    df = pd.read_csv('/Users/danfinel/Downloads/student-mat (1).csv')
    return df


def remove_feat_xtrain(df):
    df.drop(columns=['school', 'age', 'romantic', 'failures', 'G1', 'G2','absences','Dalc','Walc','goout','schoolsup','higher','famsup','famrel'], inplace=True)
    df = df[df.G3!=0]
    X_train = df.drop(columns='G3')
    y_train = df.G3
    return X_train,y_train



def minmaxscaler_xtrain(X_train):
    mmscaler = MinMaxScaler()
    mmscaler.fit(X_train.select_dtypes('number'))
    with open('min_max_scaler.pkl', 'wb') as file:
        pickle.dump(mmscaler, file)
    X_train_scaled = mmscaler.transform(X_train.select_dtypes('number'))
    X_train_num = pd.DataFrame(X_train_scaled, columns=X_train.select_dtypes('number').columns, index=X_train.index)
    return X_train_num




def onehot(X_train):
    ohe = OneHotEncoder()
    ohe.fit(X_train[['Mjob','Fjob','reason','guardian','famsize','Pstatus','paid','activities','nursery','address','sex']])
    with open('ohe.pkl', 'wb') as file:
        pickle.dump(ohe, file)
    encoded_data = ohe.transform(X_train[['Mjob','Fjob','reason','guardian','famsize','Pstatus','paid','activities','nursery','address','sex']])
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=ohe.get_feature_names_out(['Mjob','Fjob','reason','guardian','famsize','Pstatus','paid','activities','nursery','address','sex']), index = X_train.index)
    return encoded_df


def get_xtrain(X_train):
    X_train_nom = onehot(X_train)
    X_train_num = minmaxscaler_xtrain(X_train)
    X_train =  pd.concat([X_train_num,X_train_nom], axis = 1)

    return X_train



def get_model(X_train,y_train):
    ridge = Ridge(alpha = 2.5)
    ridge.fit(X_train, y_train)
    with open('ridge.pkl', 'wb') as file:
        pickle.dump(ridge,file)


def main_func():
    df = get_df()
    X_train,y_train = remove_feat_xtrain(df)
    X_train = get_xtrain(X_train)
    return X_train

df = get_df()
X_train = main_func()
y_train = remove_feat_xtrain(df)[1]
get_model(X_train,y_train)
X_train.to_csv('X_train.csv', index=False)







