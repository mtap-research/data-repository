import warnings
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
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve 

np.int = int
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=ConvergenceWarning)


def gbr(Xtrain, Ytrain, Xtest, Ytest, tar, n_job, call, save = True):
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
        print(result)
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
    if save:
        joblib.dump(reg_opt,"gbr_" + tar + ".pkl")
        result = pd.ExcelWriter("gbr_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        feature_importance = reg_opt.feature_importances_
        feature_names = Xtrain.columns
        print("impact: ", feature_names, feature_importance)
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "score": feature_importance})
        feature_importance_df.to_excel(result, index=False, sheet_name = "importance")
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()


def rf(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"rf_" + tar + ".pkl")
        result = pd.ExcelWriter("rf_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        feature_importance = reg_opt.feature_importances_
        feature_names = Xtrain.columns
        print("impact: ", feature_names, feature_importance)
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "score": feature_importance})
        feature_importance_df.to_excel(result, index=False, sheet_name = "importance")
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def dt(Xtrain, Ytrain, Xtest, Ytest, tar, n_job, call, save = False):
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
    if save:
        joblib.dump(reg_opt,"dt_" + tar + ".pkl")
        result = pd.ExcelWriter("dt_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        feature_importance = reg_opt.feature_importances_
        feature_names = Xtrain.columns
        print("impact: ", feature_names, feature_importance)
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "score": feature_importance})
        feature_importance_df.to_excel(result, index=False, sheet_name = "importance")
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def lasso(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"lasso_" + tar + ".pkl")
        result = pd.ExcelWriter("lasso_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        print(feature_importance_df)
        feature_importance_df.to_excel(result, sheet_name="importance")
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def kr(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"kr_" + tar + ".pkl")
        result = pd.ExcelWriter("kr_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        feature_importance_df.to_excel(result, sheet_name="importance")
        print(feature_importance_df)
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def ada(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"ada_" + tar + ".pkl")
        result = pd.ExcelWriter("ada_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        feature_importance = reg_opt.feature_importances_
        feature_names = Xtrain.columns
        print("impact: ", feature_names, feature_importance)
        feature_importance_df = pd.DataFrame({"Feature": feature_names, "score": feature_importance})
        feature_importance_df.to_excel(result, index=False, sheet_name = "importance")
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def svr(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"svr_" + tar + ".pkl")
        result = pd.ExcelWriter("svr_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        feature_importance_df.to_excel(result, sheet_name="importance")
        print(feature_importance_df)
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def knn(Xtrain, Ytrain, Xtest, Ytest,tar, n_job, call, save = True):
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
    if save:
        joblib.dump(reg_opt,"knn_" + tar + ".pkl")
        result = pd.ExcelWriter("knn_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.values.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.values.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        feature_importance_df.to_excel(result, sheet_name="importance")
        print(feature_importance_df)
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

def mlp(Xtrain, Ytrain, Xtest, Ytest, tar, n_job, call, save = True):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    Ytrain = Ytrain.values.ravel()
    Ytest = Ytest.values.ravel()

    reg = MLPRegressor(random_state=1129) 
    space = [Categorical(['sgd','adam', 'lbfgs'], name='solver')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
        # print(result)
        return result
    res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
    if res_gp.x[0]== 'sgd':
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                    Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                    Real(1e-5, 1e-2, prior='log-uniform', name='alpha'),
                    Real(1e-6, 1, prior='log-uniform', name='learning_rate_init'),
                    Integer(100, 10000, name='max_iter'),
                    Categorical([True, False], name='warm_start'),
                    Real(0.1, 1, prior='uniform', name='momentum')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "sgd", alpha=res_gp.x[2], learning_rate = "constant",
                            learning_rate_init = res_gp.x[3], max_iter = res_gp.x[4], warm_start=res_gp.x[5], momentum=res_gp.x[6],random_state=66)
        reg_opt.fit(Xtrain, Ytrain)
        
        train_sizes, train_scores, validation_scores = learning_curve(estimator=reg_opt,X=Xtrain,y=Ytrain,train_sizes=np.linspace(0.1, 1.0, 5),
                                                                        cv=5,scoring='neg_mean_squared_error')
        
        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    elif res_gp.x[0]== 'adam':
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
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), activation=res_gp.x[1], solver = "adam", alpha=res_gp.x[2],learning_rate_init=res_gp.x[3],
                                max_iter = res_gp.x[4], warm_start=res_gp.x[5],beta_1=res_gp.x[6],beta_2=res_gp.x[7], epsilon=res_gp.x[8],random_state=88)
        reg_opt.fit(Xtrain, Ytrain)
        train_sizes, train_scores, validation_scores = learning_curve(estimator=reg_opt,X=Xtrain,y=Ytrain,train_sizes=np.linspace(0.1, 1.0, 5),
                                                                        cv=5,scoring='neg_mean_squared_error')

        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    else:
        space = [Integer(10, 400, name='hidden_layer_sizes'),
                Categorical(['identity','relu', 'tanh', 'logistic'], name='activation'),
                Real(1e-5, 1e-2, prior='log-uniform', name='alpha')]
        @use_named_args(space)
        def objective(**params):
            reg.set_params(**params)
            result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job, scoring="neg_mean_squared_error"))
            # print(result)
            return result
        res_gp = gp_minimize(objective, space, n_calls=call,random_state=1129)
        # print("Best score=%.4f" % res_gp.fun)
        # print("Best parameters:")
        # print("- hidden_layer_sizes=%d" % res_gp.x[0])
        reg_opt = MLPRegressor(hidden_layer_sizes=(res_gp.x[0],), max_iter=500, activation=res_gp.x[1], solver = "lbfgs",alpha=res_gp.x[2],random_state=44)
        reg_opt.fit(Xtrain, Ytrain)

        train_sizes, train_scores, validation_scores = learning_curve(estimator=reg_opt,X=Xtrain,y=Ytrain,train_sizes=np.linspace(0.1, 1.0, 5),
                                                                        cv=5,scoring='neg_mean_squared_error')
        r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
        r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
        mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
        mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
        rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
        rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
        print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
        print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    if save:
        joblib.dump(reg_opt,"mlp_" + tar + ".pkl")
        result = pd.ExcelWriter("mlp_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        feature_importance_df.to_excel(result, sheet_name="importance")
        print(feature_importance_df)
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

        return train_sizes, train_scores, validation_scores
    
def mlp_pre(Xtrain, Ytrain, Xtest, Ytest, tar, model, save = True):
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    Ytrain = Ytrain.values.ravel()
    Ytest = Ytest.values.ravel()

    reg_opt = joblib.load(model)
    train_sizes, train_scores, validation_scores = learning_curve(estimator=reg_opt,X=Xtrain,y=Ytrain,train_sizes=np.linspace(0.1, 1.0, 5),
                                                                    cv=5,scoring='neg_mean_squared_error')
    r2_train = r2_score(Ytrain, reg_opt.predict(Xtrain))
    r2_test = r2_score(Ytest,reg_opt.predict(Xtest))
    mae_train = mean_absolute_error(Ytrain,reg_opt.predict(Xtrain))
    mae_test = mean_absolute_error(Ytest,reg_opt.predict(Xtest))
    rmse_train = mean_squared_error(Ytrain,reg_opt.predict(Xtrain),squared=False)
    rmse_test = mean_squared_error(Ytest,reg_opt.predict(Xtest),squared=False)
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    if save:
        joblib.dump(reg_opt,"mlp_" + tar + ".pkl")
        result = pd.ExcelWriter("mlp_"+tar+".xlsx")
        df_result_train = pd.DataFrame({"Ytrain": Ytrain.reshape(-1),
                                        'Ytrain_pre': reg_opt.predict(Xtrain).ravel()})
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        df_result_test = pd.DataFrame({"Ytest": Ytest.reshape(-1),
                                        'Ytest_pre': reg_opt.predict(Xtest).ravel()})
        df_result_test.to_excel(result, index=False, sheet_name = "data_test")
        importances = permutation_importance(reg_opt, Xtest, Ytest, n_repeats=30, random_state=1129)
        feature_importances = {Xtrain.columns[i]: importances.importances_mean[i] for i in range(len(Xtrain.columns))}
        feature_importance_df = pd.DataFrame(list(feature_importances.values()), 
                                            index=list(feature_importances.keys()), 
                                            columns=['Importance'])
        feature_importance_df.to_excel(result, sheet_name="importance")
        print(feature_importance_df)
        score = pd.DataFrame({"r2_train": [r2_train],
                            "r2_test": [r2_test],
                            'mae_train': [mae_train],
                            'mae_test': [mae_test],
                            'rmse_train': [rmse_train],
                            'rmse_test': [rmse_test]})
        score.to_excel(result, index=False, sheet_name = "score")
        result.close()

    return train_sizes, train_scores, validation_scores