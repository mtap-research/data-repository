import os
import multiprocessing

from model import gbr,rf,dt

function_map = {
    'gbr': gbr,
    'rf': rf,
    'dt': dt
}

original_directory = os.getcwd()

def run_train(alg,units):
    opt_dir = alg+'_'+units
    if not os.path.exists(opt_dir): os.mkdir(opt_dir) 
    os.chdir(opt_dir)
    print(alg,units,'is running')
    function_map[alg](data_csv = "../dataset_"+units+".csv",unit=units, n_job=-1, call=100, save = True)
    os.chdir(original_directory)

def make_task(alg, unit):
    def task():
        run_train(alg, unit)
    return task

def run_tasks(units, algs):
    processes = []
    for unit in units:
        for alg in algs:
            task = make_task(alg, unit)  
            p = multiprocessing.Process(target=task)
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    units = ['wt', 'gL']
    algs = ['gbr']
    run_tasks(units, algs)
