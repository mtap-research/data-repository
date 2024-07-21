import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

np.int = int

def data_prepare(data_csv,verbos=False, saveto: str="dataset4pre.csv")-> pd.DataFrame:
    df = pd.read_csv(data_csv)
    smiles = pd.read_csv(data_csv)["SMILE"]
    features = []
    for smile in smiles:
        dft_features = df[df.SMILE==smile][["SMILE","IP","EA","HomoLumoGap"]].values[0].tolist()
        mol = Chem.MolFromSmiles(smile)
        MolWt = Chem.Descriptors.MolWt(mol)
        MolLogP = Chem.Descriptors.MolLogP(mol)
        MolMR = Chem.Descriptors.MolMR(mol)
        HeavyAtomCount = Chem.Descriptors.HeavyAtomCount(mol)
        LabuteASA = Chem.Descriptors.LabuteASA(mol)
        BalabanJ = Chem.Descriptors.BalabanJ(mol)
        BertzCT = Chem.Descriptors.BertzCT(mol)
        rdkit_features = [MolWt,MolLogP,MolMR,HeavyAtomCount,LabuteASA,BalabanJ,BertzCT]
        all_features = dft_features + rdkit_features
        features.append(all_features)
    df_all_features = pd.DataFrame(features, columns=["SMILE",'IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT'])
    if verbos:
        print(df_all_features)
    if saveto:
        df_all_features.to_csv(saveto, index=True, index_label='Number')
    return df_all_features

def normal(data_csv):
    X_ori = pd.read_csv(data_csv)[['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']]
    scaler = joblib.load("scaler.gz")
    X = scaler.transform(X_ori)
    SMILE = pd.read_csv(data_csv)["SMILE"]
    df_SMILE = pd.DataFrame(SMILE,columns=["SMILE"])
    df_X = pd.DataFrame(X,columns=['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT'])
    df_all = pd.concat([df_SMILE,df_X], axis=1)
    return df_all

def normal_tree(data_csv):
    SMILE = pd.read_csv(data_csv)["SMILE"]
    df_SMILE = pd.DataFrame(SMILE,columns=["SMILE"])
    X_ori = pd.read_csv(data_csv)[['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']]
    df_X = pd.DataFrame(X_ori,columns=['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT'])
    df_all = pd.concat([df_SMILE,df_X], axis=1)
    return df_all

def show(data_csv):
    smiles = pd.read_csv(data_csv)["SMILE"]
    mols = []
    names = smiles.tolist()
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mols.append(mol)
    vis = Draw.MolsToGridImage(mols,molsPerRow = 5,subImgSize = (200,200),legends=names)
    return vis

def gbr(X, tar, save = True):
    model = joblib.load('gbr_'+tar+'.pkl')
    X4pre = X[['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']]
    output = model.predict(X4pre)
    if save:
        df_Y = pd.DataFrame(output,columns=[tar])
        df_all = pd.concat([X,df_Y], axis=1)
        df_all.to_csv("predict_gbr_" + tar + ".csv")

def mlp(X, tar, save = True):
    model = joblib.load('mlp_'+tar+'.pkl')
    X4pre = X[['IP','EA','HomoLumoGap','MolWt','MolLogP','MolMR','HeavyAtomCount','LabuteASA','BalabanJ','BertzCT']]
    output = model.predict(X4pre)
    if save:
        df_Y = pd.DataFrame(output,columns=[tar])
        df_all = pd.concat([X,df_Y], axis=1)
        df_all.to_csv("predict_mlp_" + tar + ".csv")
