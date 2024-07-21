import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(smile, descriptor_name):
    molecule = Chem.MolFromSmiles(smile)
    descriptor_func = getattr(Descriptors, descriptor_name)
    data = descriptor_func(molecule)
    return data

def predict(data_csv):
    mm = ['mlp','rf']
    for m in mm:
        df = pd.read_csv(data_csv)
        smiles = pd.read_csv(data_csv)["SMILE"]
        features_name = pd.read_csv("./features_name.csv")[m]
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
            features.append(data_s)
        columns = ['SMILE']
        for feature in features_name:
            columns.extend([feature])
        if m == 'mlp':
            scaler = joblib.load("scaler_MLP_LF.gz")
            features_pre = scaler.transform(pd.DataFrame(features,columns=columns).iloc[:,1:])
            model = joblib.load('mlp_LF.pkl')
            lf = model.predict(features_pre)
            result = []
            for i in range(len(lf)):
                result.append([smiles[i],lf[i]])
                df_result = pd.DataFrame(result, columns=["smile","lf"])
                df_result.to_csv("lf.csv", index=False)
        else:
            features_pre = pd.DataFrame(features,columns=columns).iloc[:,1:]
            model = joblib.load('rf_GWP.pkl')
            gwp = model.predict(features_pre)
            result = []
            for i in range(len(gwp)):
                result.append([smiles[i],gwp[i]])
                df_result = pd.DataFrame(result, columns=["smile","gwp"])
                df_result.to_csv("gwp.csv", index=False)
        df_features = pd.DataFrame(features, columns=columns)
        df_features.to_csv("feature_" + m + ".csv", index=False)