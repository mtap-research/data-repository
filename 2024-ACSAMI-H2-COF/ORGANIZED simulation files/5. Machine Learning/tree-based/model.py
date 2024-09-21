from data import data_split
import warnings
import joblib
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error,r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning

np.int = int
warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=ConvergenceWarning)

def gbr(data_csv,unit, n_job, call, save = True):
    warnings.filterwarnings("ignore", category=FutureWarning, module='xgboost')
    Xtrain, Xtest, Ytrain, Ytest = data_split(data_csv)
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
        # print(result)
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
    rmse_train = root_mean_squared_error(Ytrain,reg_opt.predict(Xtrain))
    rmse_test = root_mean_squared_error(Ytest,reg_opt.predict(Xtest))
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    if save:
        joblib.dump(reg_opt,"gbr_" + unit + ".pkl")
        result = pd.ExcelWriter("gbr_"+unit+".xlsx")

        result_train = {}
        for fea in Xtrain.columns.values:
            result_train[fea] = Xtrain[fea]
        result_train["Ytrain"] = Ytrain.values
        result_train['Ytrain_pre'] = reg_opt.predict(Xtrain)
        df_result_train = pd.DataFrame(result_train)
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        result_test = {}
        for fea in Xtest.columns.values:
            result_test[fea] = Xtest[fea]
        result_test["Ytest"] = Ytest.values
        result_test['Ytest_pre'] = reg_opt.predict(Xtest)
        df_result_test = pd.DataFrame(result_test)
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

def rf(data_csv,unit, n_job, call, save = True):
    Xtrain, Xtest, Ytrain, Ytest = data_split(data_csv)
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
    rmse_train = root_mean_squared_error(Ytrain.values.ravel(),reg_opt.predict(Xtrain))
    rmse_test = root_mean_squared_error(Ytest.values.ravel(),reg_opt.predict(Xtest))
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    if save:
        joblib.dump(reg_opt,"rf_" + unit + ".pkl")
        result = pd.ExcelWriter("rf_"+unit+".xlsx")
        
        result_train = {}
        for fea in Xtrain.columns.values:
            result_train[fea] = Xtrain[fea]
        result_train["Ytrain"] = Ytrain.values
        result_train['Ytrain_pre'] = reg_opt.predict(Xtrain)
        df_result_train = pd.DataFrame(result_train)
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        result_test = {}
        for fea in Xtest.columns.values:
            result_test[fea] = Xtest[fea]
        result_test["Ytest"] = Ytest.values
        result_test['Ytest_pre'] = reg_opt.predict(Xtest)
        df_result_test = pd.DataFrame(result_test)
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

def dt(data_csv, unit, n_job, call, save = False):
    Xtrain, Xtest, Ytrain, Ytest = data_split(data_csv)
    reg = DecisionTreeRegressor()
    space = [Integer(1, 20, name='max_depth'),
             Integer(2, 30, name='min_samples_split'),
             Integer(1, 30, name='min_samples_leaf')]
    @use_named_args(space)
    def objective(**params):
        reg.set_params(**params)
        result = -np.mean(cross_val_score(reg, Xtrain, Ytrain, cv=5, n_jobs=n_job,
                                           scoring="neg_mean_squared_error"))
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
    rmse_train = root_mean_squared_error(Ytrain,reg_opt.predict(Xtrain))
    rmse_test = root_mean_squared_error(Ytest,reg_opt.predict(Xtest))
    print("Training: r2,mae,rmse: ", r2_train,mae_train,rmse_train)
    print("Test: r2,mae,rmse: ", r2_test,mae_test,rmse_test)
    if save:
        joblib.dump(reg_opt,"dt_" + unit + ".pkl")
        result = pd.ExcelWriter("dt_"+unit+".xlsx")
        
        result_train = {}
        for fea in Xtrain.columns.values:
            result_train[fea] = Xtrain[fea]
        result_train["Ytrain"] = Ytrain.values
        result_train['Ytrain_pre'] = reg_opt.predict(Xtrain)
        df_result_train = pd.DataFrame(result_train)
        df_result_train.to_excel(result, index=False, sheet_name = "data_train")
        result_test = {}
        for fea in Xtest.columns.values:
            result_test[fea] = Xtest[fea]
        result_test["Ytest"] = Ytest.values
        result_test['Ytest_pre'] = reg_opt.predict(Xtest)
        df_result_test = pd.DataFrame(result_test)
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
