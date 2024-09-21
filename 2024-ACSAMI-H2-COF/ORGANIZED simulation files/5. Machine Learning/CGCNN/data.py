import numpy as np
import pandas as pd
from tqdm import tqdm


def wt(a):
    # b = 100*(0.001*a)/(1+0.001*a)
    b = a 
    return b

def gL(a):
    # b = a/11.2
    b = a 
    return b

proc = {'wt':wt,'gL':gL}

def all_pre(data_csv,unit):
    data = pd.read_csv(data_csv)
    for name in tqdm(data["name"]):
        if len(name.split('_')) == 1:
            break_flag = False
            ori_list = []
            one_h_list = []
            one_f_list = []
            two_h_list = []
            two_f_list = []
            for t in [111,231,296]:
                for p in ['5e5','1e7']:
                    tar_ori = proc[unit](data[data["name"]==name][str(t)+"_"+p].values[0])
                    try:
                        one_h = proc[unit](data[data["name"]==name+"_1.0_0.5"][str(t)+"_"+p].values[0])
                    except:
                        break_flag = True
                        break
                    try:
                        one_f = proc[unit](data[data["name"]==name+"_1.0_1.0"][str(t)+"_"+p].values[0])
                    except:
                        break_flag = True
                        break
                    try:
                        two_h = proc[unit](data[data["name"]==name+"_2.0_0.5"][str(t)+"_"+p].values[0])
                    except:
                        break_flag = True
                        break
                    try:
                        two_f = proc[unit](data[data["name"]==name+"_2.0_1.0"][str(t)+"_"+p].values[0])
                    except:
                        break_flag = True
                        break
                    ori_list.append(tar_ori)
                    one_h_list.append(one_h)
                    one_f_list.append(one_f)
                    two_h_list.append(two_h)
                    two_f_list.append(two_f)
                if break_flag: break
            if break_flag: continue
            ori = np.array(ori_list).flatten()
            one_h = np.array(one_h_list).flatten()
            one_f = np.array(one_f_list).flatten()
            two_h = np.array(two_h_list).flatten()
            two_f = np.array(two_f_list).flatten()
            all_DC = np.hstack((ori,one_h,one_f,two_h,two_f))
            np.save("./data/cif/npy" + "_" + unit + "/" + name + ".npy", all_DC)

        else:
            pass

all_pre(data_csv="uptake_w.csv",unit="wt")
all_pre(data_csv="uptake_g.csv",unit="gL")