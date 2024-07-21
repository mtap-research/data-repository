import os
import csv
import time
import joblib
import random
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler


def calculate_descriptors(smile, descriptor_name):
    molecule = Chem.MolFromSmiles(smile)
    descriptor_func = getattr(Descriptors, descriptor_name)
    data = descriptor_func(molecule)
    return data

def data_prepare(data_csv,mm = "GBR", tar = "LF",verbos=False):
    df = pd.read_csv(data_csv)
    smiles = pd.read_csv(data_csv)["SMILE"]
    features_name = pd.read_csv("./features_name.csv")[mm + "_" + tar]
    features = []
    for smile in smiles:
        data_s = []
        data_s.extend([smile.replace('\xa0', '')])
        for feature in features_name:
            try:
                dft_fea = df[df['SMILE']==smile][feature].values[0].tolist()
                data_s.extend([dft_fea])
            except:
                rdkit_fea = calculate_descriptors(smile,feature)
                data_s.extend([rdkit_fea])
        target = df[df.SMILE==smile][tar].values[0].tolist()
        data_s.extend([target])
        features.append(data_s)
    columns = ['SMILE']
    for feature in features_name:
        columns.extend([feature])
    columns.extend([tar])
    df_features = pd.DataFrame(features, columns=columns)
   
    if verbos:
        print(df_features)
        
    df_features.to_csv("dataset4" + mm + "_" + tar + ".csv", index=True, index_label="index")
    return df_features

def data_split(mm = "GBR", tar = "LF",ratio = 0.8,verbos = False):
    df = pd.read_csv("dataset4" + mm + "_" + tar + ".csv")
    N_materials = df.shape[0]
    N_features = df.shape[1] - 3
    diverse_ratio = ratio
    data_file_name = "dataset4" + mm + "_" + tar + ".csv"
    if os.path.exists("split" + "_" + mm + "_" + tar + ".txt"):
        diverse_set=[]
        remaining_set=[]
        txt = open("split" + "_" + mm + "_" + tar + ".txt",'r').read()
        s1=txt.find("[",0)
        s2=txt.find("]",s1)
        diverse_set=txt[s1+1:s2].split(", ")
        diverse_set=[int(i) for i in diverse_set]
        s3=txt.find("[",s2)
        s4=txt.find("]",s3)
        remaining_set=txt[s3+1:s4].split(", ")
        remaining_set=[int(i) for i in remaining_set]
    else:
        with open(data_file_name) as f:
            data_file = csv.reader(f)
            next(data_file)
            data = np.empty((N_materials, N_features))
            for i, d in enumerate(data_file):
                data[i] = np.asarray(d[2:2+N_features], dtype=np.float64)
        feature_0 = data.T[0]
        feature_1 = data.T[1]
        feature_2 = data.T[2]
        feature_3 = data.T[3]
        feature_4 = data.T[4]
        feature_5 = data.T[5]
        feature_6 = data.T[6]
        feature_7 = data.T[7]
        feature_8 = data.T[8]
        feature_9 = data.T[9]
        feature_0 = (feature_0 - np.min(feature_0))/(np.max(feature_0) - np.min(feature_0))
        feature_1 = (feature_1 - np.min(feature_1))/(np.max(feature_1) - np.min(feature_1))
        feature_2 = (feature_2 - np.min(feature_2))/(np.max(feature_2) - np.min(feature_2))
        feature_3 = (feature_3 - np.min(feature_3))/(np.max(feature_3) - np.min(feature_3))
        feature_4 = (feature_4 - np.min(feature_4))/(np.max(feature_4) - np.min(feature_4))
        feature_5 = (feature_5 - np.min(feature_5))/(np.max(feature_5) - np.min(feature_5))
        feature_6 = (feature_6 - np.min(feature_6))/(np.max(feature_6) - np.min(feature_6))
        feature_7 = (feature_7 - np.min(feature_7))/(np.max(feature_7) - np.min(feature_7))
        feature_8 = (feature_8 - np.min(feature_8))/(np.max(feature_8) - np.min(feature_8))
        feature_9 = (feature_9 - np.min(feature_9))/(np.max(feature_9) - np.min(feature_9))
        if N_features == N_features:
            x = np.concatenate((feature_0.reshape(1,N_materials),feature_1.reshape(1,N_materials),feature_2.reshape(1,N_materials),
                                feature_3.reshape(1,N_materials),feature_4.reshape(1,N_materials),feature_5.reshape(1,N_materials),
                                feature_6.reshape(1,N_materials),feature_7.reshape(1,N_materials),feature_8.reshape(1,N_materials),
                                feature_9.reshape(1,N_materials)))
        N_sample = int(N_materials * diverse_ratio)-1
        time.sleep(1)
        diverse_set = []
        remaining_set = list(range(N_materials))
        idx_init = random.sample(list(np.arange(N_materials)),1)[0]
        diverse_set.append(idx_init)
        remaining_set.remove(idx_init)
        N_diverse = 1
        while N_diverse <= N_sample:
            print("Selecting point ", N_diverse)
            min_d_to_diverse_set = np.zeros((N_materials-N_diverse,))
            for i in range(N_materials - N_diverse):
                d_from_each_diverse_pt = np.linalg.norm(x[:,diverse_set] - x[:,remaining_set[i]].reshape(N_features,1),axis=0)
                min_d_to_diverse_set[i] = np.min(d_from_each_diverse_pt)
            idx_select = remaining_set[np.argmax(min_d_to_diverse_set)]
            assert (len(remaining_set) == np.size(min_d_to_diverse_set))
            diverse_set.append(idx_select)
            remaining_set.remove(idx_select)
            N_diverse += 1
        if verbos:
            print("Total number of materials : ", data.shape[0])
            print("Number of features: ", N_features)
        with open("split" + "_" + mm + "_" + tar + ".txt", "w") as f:
            f.write(str(diverse_set)+" "+str(remaining_set))
    N_targets = 1
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        next(data_file)
        number = np.empty((N_materials,))
        structure = np.empty((N_materials,))
        data = np.empty((N_materials, N_features))
        target = np.empty((N_materials,N_targets))
        structure = []
        for i, d in enumerate(data_file):
            number[i] = np.asarray(d[0],dtype=int)
            structure.append(d[1])
            data[i] = np.asarray(d[2:2+N_features], dtype=np.float64)
            target[i] = np.asarray(d[-N_targets:], dtype=np.float64)
    diverse_set_total=[]
    remaining_set_total=[]
    for i,diverse in enumerate(diverse_set):
        arridx = np.where(number == diverse)
        for _,element_div in enumerate(arridx[0]):
            diverse_set_total.append(element_div)
    for i,remaining in enumerate(remaining_set):
        arridx = np.where(number == remaining)
        for _,element_rem in enumerate(arridx[0]):
            remaining_set_total.append(element_rem)        
    X_train = data[diverse_set_total]
    y_train = target[diverse_set_total]
    X_test = data[remaining_set_total]
    y_test = target[remaining_set_total]
    columns = df.columns.tolist()[2:-1]
    df_Xtrain = pd.DataFrame(X_train,columns=columns)
    df_Ytrain = pd.DataFrame(y_train,columns=[tar])
    df_Xtest = pd.DataFrame(X_test,columns=columns)
    df_Ytest = pd.DataFrame(y_test,columns=[tar])
    return df_Xtrain, df_Ytrain, df_Xtest, df_Ytest

def normal(df_Xtrain, df_Ytrain, df_Xtest, df_Ytest, mm='GBR',tar='LF'):
    scaler = StandardScaler()
    if os.path.exists("scaler" + "_" + mm + "_" + tar + ".gz"):
        scaler = joblib.load("scaler" + "_" + mm + "_" + tar + ".gz")
        Xtrain = scaler.transform(df_Xtrain)
        Xtest = scaler.transform(df_Xtest)
    else:
        Xtrain = scaler.fit_transform(df_Xtrain)
        joblib.dump(scaler, "scaler" + "_" + mm + "_" + tar + ".gz")
        scaler = joblib.load("scaler" + "_" + mm + "_" + tar + ".gz")
        Xtest = scaler.transform(df_Xtest)
    Ytrain = df_Ytrain[tar]
    Ytest = df_Ytest[tar]
    df_Xtrain = pd.DataFrame(Xtrain,columns=df_Xtrain.columns.tolist())
    df_Ytrain = pd.DataFrame(Ytrain,columns=[tar])
    df_Xtest = pd.DataFrame(Xtest,columns=df_Xtrain.columns.tolist())
    df_Ytest = pd.DataFrame(Ytest,columns=[tar])
    return df_Xtrain, df_Ytrain, df_Xtest, df_Ytest

