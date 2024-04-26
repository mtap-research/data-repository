import warnings
import os
import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.int = int
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def split(csv, tar='LF'):
    data = pd.read_csv(csv)
    Y = data[tar]
    X = data.iloc[:, :-2]
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=1220)

    return Xtrain, Ytrain, Xtest, Ytest

def normal(csv, Xtrain, Ytrain, Xtest, Ytest):
    scaler = StandardScaler()
    # if os.path.exists("scaler-mlp.gz"):
    #     scaler = joblib.load("scaler-mlp.gz")
    if os.path.exists("scaler.gz"):
        scaler = joblib.load("scaler.gz")
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
    else:
        Xtrain = scaler.fit_transform(Xtrain)
        # joblib.dump(scaler, "scaler-mlp.gz")
        # scaler = joblib.load("scaler-mlp.gz")
        joblib.dump(scaler, "scaler.gz")
        scaler = joblib.load("scaler.gz")
        Xtest = scaler.transform(Xtest)
    df = pd.read_csv(csv, nrows=0)
    column_names = df.columns.tolist()[:-2]
    Xtrain = pd.DataFrame(Xtrain,columns=column_names)
    Xtest = pd.DataFrame(Xtest,columns=column_names)
    return Xtrain, Ytrain, Xtest, Ytest

def gbr(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
    reg = XGBRegressor()
    space  = [Integer(1,200, name='n_estimators'),
            Integer(1, 10, name='max_depth'),
            Integer(1, 10, name='num_parallel_tree'),
            Integer(1, 10, name='min_child_weight'),
            Real(0.001,1,"log-uniform",name='learning_rate'),
            Real(0.01,1,name='subsample'),
            Real(0.001,10,"log-uniform",name='gamma'),
            Real(0, 1, name='alpha'),
            Real(2, 10, name='reg_alpha'),
            Real(10, 50, name='reg_lambda')
         ]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    print("Best score=%.4f" % res_gp.fun)
    reg_opt = XGBRegressor(n_estimators=res_gp.x[0],
                            max_depth=res_gp.x[1],
                            num_parallel_tree=res_gp.x[2],
                            min_child_weight=res_gp.x[3],
                            learning_rate=res_gp.x[4],
                            subsample=res_gp.x[5],
                            gamma=res_gp.x[6],
                            alpha=res_gp.x[7],
                            reg_alpha=res_gp.x[8],
                            reg_lambda=res_gp.x[9]
                            )
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain, reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns
    df_feature_importance = pd.DataFrame(feature_importance, columns=['feature_importance'])
    df_feature_importance['feature_names'] = feature_names
    sorted_df = df_feature_importance.sort_values(by='feature_importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)
    
def rf(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    reg = RandomForestRegressor()
    space  = [Integer(1, 200, name='n_estimators'),
              Integer(1, 30, name='max_depth'),
              Integer(2, 30, name='min_samples_split'),
              Integer(1, 30, name='min_samples_leaf'),
              Integer(1, 300, name='random_state')
              ]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain.values.ravel(), cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
  
    reg_opt = RandomForestRegressor(n_estimators=res_gp.x[0],
                                max_depth=res_gp.x[1],
                                min_samples_split=res_gp.x[2],
                                min_samples_leaf=res_gp.x[3],                            
                                random_state=res_gp.x[4])
    reg_opt.fit(Xtrain, Ytrain.values.ravel())
    r2_train = r2_score(Ytrain.values.ravel(), reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest.values.ravel(),reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain.values.ravel(),reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest.values.ravel(),reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain.values.ravel(),reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest.values.ravel(),reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns
    df_feature_importance = pd.DataFrame(feature_importance, columns=['feature_importance'])
    df_feature_importance['feature_names'] = feature_names
    sorted_df = df_feature_importance.sort_values(by='feature_importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)

def dt(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    reg = DecisionTreeRegressor()
    space = [Integer(1, 20, name='max_depth'),
             Integer(2, 30, name='min_samples_split'),
             Integer(1, 30, name='min_samples_leaf')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                           scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    reg_opt = DecisionTreeRegressor(max_depth=res_gp.x[0],
                                    min_samples_split=res_gp.x[1],
                                    min_samples_leaf=res_gp.x[2])
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns
    df_feature_importance = pd.DataFrame(feature_importance, columns=['feature_importance'])
    df_feature_importance['feature_names'] = feature_names
    sorted_df = df_feature_importance.sort_values(by='feature_importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)
      

def lasso(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    reg = Lasso()
    space = [Real(1e-8, 1, prior='log-uniform', name='alpha')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    reg_opt = Lasso(alpha=res_gp.x[0],max_iter=10000)
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
    feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
    feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                        index=list(feature_importances.keys()), 
                                        columns=['Importance'])
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)


def kr(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    reg = KernelRidge()
    space = [
        Real(1e-6, 1, prior='log-uniform', name='alpha'),
        Real(1e-6, 1, prior='log-uniform', name='gamma')
    ]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result=-np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                            scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    reg_opt = KernelRidge(alpha=res_gp.x[0], kernel='rbf', gamma=res_gp.x[1])
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)

    importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
    feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
    feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                        index=list(feature_importances.keys()), 
                                        columns=['Importance'])
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)

def ada(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    Ytrain = Ytrain.values.ravel()
    Ytest = Ytest.values.ravel()
    reg = AdaBoostRegressor()
    space = [
        Real(1e-6, 1, prior='log-uniform', name='learning_rate'),
        Integer(1, 200, name='n_estimators')
    ]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                          scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    print("Best score=%.4f" % res_gp.fun)
    reg_opt = AdaBoostRegressor(learning_rate=res_gp.x[0], n_estimators=int(res_gp.x[1]))
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    feature_importance = reg_opt.feature_importances_
    feature_names = Xtrain.columns
    df_feature_importance = pd.DataFrame(feature_importance, columns=['feature_importance'])
    df_feature_importance['feature_names'] = feature_names
    sorted_df = df_feature_importance.sort_values(by='feature_importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)

def svr(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    Ytrain = Ytrain.values.ravel()
    Ytest = Ytest.values.ravel()
    reg = SVR()
    space = [Real(1e-6, 1, prior='log-uniform', name='C'),
             Real(1e-6, 1, prior='log-uniform', name='epsilon')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                           scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    print("Best score=%.4f" % res_gp.fun)
    reg_opt = SVR(C=res_gp.x[0], epsilon=res_gp.x[1])
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    
    importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
    feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
    feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                        index=list(feature_importances.keys()), 
                                        columns=['Importance'])
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)
      

def knn(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    reg = KNeighborsRegressor()
    space = [Integer(1, 20, name='n_neighbors'),
             Categorical(('uniform', 'distance'), name='weights')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                           scoring="neg_mean_squared_error"))
        print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call)
    print("Best score=%.4f" % res_gp.fun)
    reg_opt = KNeighborsRegressor(n_neighbors=res_gp.x[0], weights=res_gp.x[1])
    reg_opt.fit(Xtrain, Ytrain)
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    
    importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
    feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
    feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                        index=list(feature_importances.keys()), 
                                        columns=['Importance'])
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)

def mlp(Xtrain, Ytrain, Xtest, Ytest, n_job, call):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    Ytrain = Ytrain.values.ravel()
    Ytest = Ytest.values.ravel()
    reg = MLPRegressor(random_state=1129) 
    space = [Categorical(['sgd','adam', 'lbfgs'], name='solver')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error", error_score='raise'))
        # print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
    if res_gp.x[0]== 'sgd':
        print("use sgd!!!")
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                    Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                    Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
                    Real(1e-6, 1, prior='log-uniform', name='learning_rate_init'),
                    Integer(10, 10000, name='max_iter'),
                    Categorical([True, False], name='warm_start'),
                    Real(0.1, 1, prior='uniform', name='momentum')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error", error_score='raise'))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "sgd", alpha=res_gp.x[2], learning_rate = "constant",
                            learning_rate_init = res_gp.x[3], max_iter = res_gp.x[4], warm_start=res_gp.x[5], momentum=res_gp.x[6],random_state=66)
        reg_opt.fit(Xtrain, Ytrain)
        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    elif res_gp.x[0]== 'adam':
        print("use adam!!!")
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
                Real(1e-6, 1, prior='log-uniform', name='learning_rate_init'),
                Integer(100, 10000, name='max_iter'),
                Categorical([True, False], name='warm_start'),
                Real(0.001, 0.999, prior='uniform', name='beta_1'),
                Real(0.001, 0.999, prior='uniform', name='beta_2'),
                Real(1e-12, 1e-1, prior='log-uniform', name='epsilon')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error", error_score='raise'))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "adam", alpha=res_gp.x[2],learning_rate_init=res_gp.x[3],
                                max_iter = res_gp.x[4], warm_start=res_gp.x[5],beta_1=res_gp.x[6],beta_2=res_gp.x[7], epsilon=res_gp.x[8],random_state=88)
        reg_opt.fit(Xtrain, Ytrain)
        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    else:
        print("use lbfgs!!!")
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                Real(1e-5, 1e-2, prior='log-uniform', name='alpha')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error", error_score='raise'))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        # print("Best parameters:")
        # print("- hidden_layer_sizes=%d" % res_gp.x[0])
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), max_iter=500, activation=res_gp.x[1], solver = "lbfgs",alpha=res_gp.x[2],random_state=44)
        reg_opt.fit(Xtrain, Ytrain)
        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    
    importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
    feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
    feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                        index=list(feature_importances.keys()), 
                                        columns=['Importance'])
    sorted_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_10_features = sorted_df.head(10)
    print(top_10_features)
        
