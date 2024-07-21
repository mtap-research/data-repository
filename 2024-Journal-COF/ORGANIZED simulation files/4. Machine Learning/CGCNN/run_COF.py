from model.CGCNN_run import FineTune

cgcnn_run = FineTune(root_dir="./data/cif/",save_dir="./result/wt/",unit="wt",tar=True,log_every_n_steps=50,eval_every_n_epochs=1,
                     epoch=500,opti="SGD",lr=0.001,momentum=0.9,weight_decay=1e-6,cif_list="COF_list.csv",batch_size=64,n_conv=3,
                     random_seed = 1129,pin_memory=False)

# cgcnn_run = FineTune(root_dir="./data/cif/",save_dir="./result/gL/",unit="gL",tar=True,log_every_n_steps=50,eval_every_n_epochs=1,
#                      epoch=500,opti="SGD",lr=0.001,momentum=0.9,weight_decay=1e-6,cif_list="COF_list.csv",batch_size=64,n_conv=3,
#                      random_seed = 1129,pin_memory=False)

cgcnn_run.train()
loss, metric = cgcnn_run.test()
cgcnn_run.predict()
