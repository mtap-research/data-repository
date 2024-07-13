import numpy as np
import pandas as pd

def wt(a):
    b = 100*(0.001*a)/(1+0.001*a)
    return b

def gL(a):
    b = a/11.2
    return b

T_list = [111,231,296]
def all_pre(data_csv,unit):
    data = pd.read_csv(data_csv)
    for name in data["name"]:
        if len(name.split('_')) == 1:
            if unit =="wt":
                ori_list = []
                one_h_list = []
                one_f_list = []
                two_h_list = []
                two_f_list = []
                for t in T_list:
                    tar_ori = wt(data[data["name"]==name][str(t)+"_"+str("1e7")]-data[data["name"]==name][str(t)+"_"+str("5e5")])
                    tar_ori = float(tar_ori.iloc[0])
                    try:
                        one_h = wt(data[data["name"]==name+"_1.0_0.5"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_1.0_0.5"][str(t)+"_"+str("5e5")])
                        if one_h.empty:
                            one_h = [0]
                    except:
                        one_h = [0]
                    try:
                        one_f = wt(data[data["name"]==name+"_1.0_1.0"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_1.0_1.0"][str(t)+"_"+str("5e5")])
                        if one_f.empty:
                            one_f = [0]
                    except:
                        one_f = [0]
                    try:
                        two_h = wt(data[data["name"]==name+"_2.0_0.5"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_2.0_0.5"][str(t)+"_"+str("5e5")])
                        if two_h.empty:
                            two_h = [0]
                    except:
                        two_h = [0]
                    try:
                        two_f = wt(data[data["name"]==name+"_2.0_1.0"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_2.0_1.0"][str(t)+"_"+str("5e5")])
                        if two_f.empty:
                            two_f = [0]
                    except:
                        two_f = [0]
                    ori_list.append(tar_ori)
                    one_h_list.append(one_h)
                    one_f_list.append(one_f)
                    two_h_list.append(two_h)
                    two_f_list.append(two_f)
                ori = np.array(ori_list).flatten()
                one_h = np.array(one_h_list).flatten()
                one_f = np.array(one_f_list).flatten()
                two_h = np.array(two_h_list).flatten()
                two_f = np.array(two_f_list).flatten()
                all_DC = np.hstack((ori,one_h,one_f,two_h,two_f))
                np.save("./data/cif/npy" + "_" + unit + "/" + name + ".npy", all_DC)
            else:
                ori_list = []
                one_h_list = []
                one_f_list = []
                two_h_list = []
                two_f_list = []
                for t in T_list:
                    tar_ori = gL(data[data["name"]==name][str(t)+"_"+str("1e7")]-data[data["name"]==name][str(t)+"_"+str("5e5")])
                    tar_ori = float(tar_ori.iloc[0])
                    try:
                        one_h = wt(data[data["name"]==name+"_1.0_0.5"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_1.0_0.5"][str(t)+"_"+str("5e5")])
                        if one_h.empty:
                            one_h = [0]
                    except:
                        one_h = [0]
                    try:
                        one_f = wt(data[data["name"]==name+"_1.0_1.0"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_1.0_1.0"][str(t)+"_"+str("5e5")])
                        if one_f.empty:
                            one_f = [0]
                    except:
                        one_f = [0]
                    try:
                        two_h = wt(data[data["name"]==name+"_2.0_0.5"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_2.0_0.5"][str(t)+"_"+str("5e5")])
                        if two_h.empty:
                            two_h = [0]
                    except:
                        two_h = [0]
                    try:
                        two_f = wt(data[data["name"]==name+"_2.0_1.0"][str(t)+"_"+str("1e7")]-data[data["name"]==name+"_2.0_1.0"][str(t)+"_"+str("5e5")])
                        if two_f.empty:
                            two_f = [0]
                    except:
                        two_f = [0]
                    ori_list.append(tar_ori)
                    one_h_list.append(one_h)
                    one_f_list.append(one_f)
                    two_h_list.append(two_h)
                    two_f_list.append(two_f)
                ori = np.array(ori_list).flatten()
                one_h = np.array(one_h_list).flatten()
                one_f = np.array(one_f_list).flatten()
                two_h = np.array(two_h_list).flatten()
                two_f = np.array(two_f_list).flatten()
                all_DC = np.hstack((ori,one_h,one_f,two_h,two_f))
                np.save("./data/cif/npy" + "_" + unit + "/" + name + ".npy", all_DC)
        else:
            pass

# all_pre(data_csv="uptake_w.csv",unit="wt")
all_pre(data_csv="uptake_g.csv",unit="gL")