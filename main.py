#TODO:2 add argsparse
#TODO:2 add add wandb
"""
Contains FedAvg and CE-FedAvg functions to run experiments.
"""
import numpy as np 
import os 
import wandb
import time
from tqdm import tqdm

debug = 0
from fed_base_options import args_parser
#wandb login 3ba3d83a8f834e66ec78450600440e4f06066167
wandb.init(project='fed_CEAVG_IOT')
#suspend TF outputs
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
#TODO 3: add to args.parse argument or verbose argument?
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from model_utils import *
from models import *
import pickle
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy as CatCrossEnt 
from tensorflow.keras.optimizers import SGD, Adam
from data_utils import *
from compress_utils import *
    

def get_out_fname(exp_type, model_type, C, E, W, iid, lr, sparsity, seed):
    """ Turn parameters into a formatted string ending with '.pkl'. """
    return '{}-{}-C-{}-E-{}-W-{}-iid-{}-lr-{}-S-{}-seed-{}.pkl'.format(
            exp_type, model_type, C, E, W, iid, lr, sparsity, seed)

#save_data with formatted fname based on configuration
def save_data(fname, data):
    """ Saves data in file with name fname using pickle. """
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

#CE_FED_AVG
def run_ce_fed_avg(dataset, model_fn, C, E, B, W, iid, R, s, seed,args):
    """
    Load dataset and perform R rounds of CE-FedAvg using FedAvg parameters. 
    Saves round errors and accuracies at server in file with exp details.
    
    Parameters:
    dataset (str):          'mnist' or 'cifar'
    model_fn (FedAvgModel): callable *not* instance of model class to use 
    C (float):              fraction of users used per round
    E (int):                number of local epochs
    B (int):                local batch size 
    W (int):                number of users 
    iid (bool):             iid or non-iid data partition
    R (int):                total number of global rounds to run
    s (float):              sparsity 0 <= s < 1
    seed (int):             random seed for trial
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    #train, test are tuples (x_trains, y_trains), (x_test, y_test)
    train, test = load_dataset(dataset, W, iid)

    #  model_fn = lambda: MNIST2NNModel(optim, CatCrossEnt, 784, 10)
    #TODO1 model cnn mlp
    master_model = model_fn()
    #TODO3: replace lambda assignment with if argsparse, 
    # Instead of creating two model we can perform deepcopy everytime the local \
    # model is called:  workmodel=copy.deepcopy(global_model)
    #TODO1: replace this function with deep copy
    worker_model = model_fn()
    ################## Wandbhelper ************
    print(master_model)

    # get which model weights correspond to optim params - see get_corr_optims
    corr_optim_idxs = get_corr_optims(worker_model)
    
    central_errs = []
    central_accs = []
    best_acc=0
    max_round=0
    worker_ids = np.arange(W)
    #worker per round = fraction of user * number of users
    workers_per_round = max(int(C * W), 1)
    
    for r in tqdm(range(R)): #line 6 of algorithm 1: CE-FedAVG
        #get the weights of the server'model and the param of server optim
        #toclarify: if change to deepcopy, we may not need this funciton
        #return a list of of layer params[weight params, bias params]
        round_master_weights = master_model.get_weights()
        #return a list of optim weights ()
        round_master_optims = master_model.get_optim_weights()
        
        # to store aggregate updates
        #the model is construct by layers
        agg_model = zeros_like_model(round_master_weights)
        #the optim is construct by vector v
        agg_optim = zeros_like_optim(round_master_optims)

        if debug:
            agg_model_alt =np.array(round_master_weights)-np.array(round_master_weights)
            agg_optim_alt =np.array(round_master_optims)-np.array(round_master_optims)

        
        round_total_samples = 0
        #TODO2 change batchsize B to max test size
        err, acc = master_model.test(test[0], test[1], B)
        
        if debug:
            print("Original err, acc {},{}".format(err,acc))
            print("Propose err, acc {},{}".format(err_alt,acc_alt))
        print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(r, R, err, acc))
        central_errs.append(err)
        central_accs.append(acc)
        
        # indexes of workers participating in this round
        choices = np.random.choice(worker_ids, workers_per_round, replace=False)
        
        for w in choices:
            model_deltas_list=[]
            optim_deltas_list=[]
            w_samples_list=[]

            # "download" global model
            worker_model.set_weights(round_master_weights)
            worker_model.set_optim_weights(round_master_optims)
            
            #train[0] list of train data for W workers, train[1] labls
            w_samples = train[0][w].shape[0]
            round_total_samples += w_samples
            if debug:
                print("training {} sample".format(w_samples))
            # train worker model for given num local epochs
            for e in range(E):
                w_x, w_y = shuffle_client_data((train[0][w], train[1][w]))
                worker_model.train(w_x, w_y, B)
            
            #return a list of detals = a -b
            #TODO3 is there a better way to calculate delta?
            #replace this with element wise matrix delta
            model_deltas = minus_model_ws( worker_model.get_weights(),
                                            round_master_weights)
            #replace this with element wise vector delta
            optim_deltas = minus_optim_ws(  worker_model.get_optim_weights(),
                                            round_master_optims)
            if debug:
                model_deltas_alt = np.array(worker_model.get_weights())-np.array(master_model.get_weights())
                optim_deltas_alt = np.array(worker_model.get_optim_weights())-np.array(master_model.get_optim_weights())
                model_deltas_list.append(model_deltas_alt)
                optim_deltas_list.append(optim_deltas_alt)

            
            # compress and decompress deltas as per Algorithm 1
            #this function simulated 3 steps
            #sparcification
            #compression
            #decompresstion.

            if s > 0:
                model_deltas, optim_deltas = compress_ce_fed_avg_deltas(
                                                        corr_optim_idxs,
                                                        model_deltas,
                                                        optim_deltas,
                                                        s)
            
            # add to agg model, weighted by num local samples
            
            #Elementwise multiplication
            #TODO1: can we store all params and do it in one shot?
            p_deltas = multiply_model_ws(model_deltas, w_samples)
            p_optims = multiply_optim_ws(optim_deltas, w_samples)
            #Elementwise addition
            agg_model = add_model_ws(agg_model, p_deltas)
            agg_optim = add_optim_ws(agg_optim, p_optims)
        # model_deltas_list=[]
        # optim_deltas_list=[]
        # w_samples_list=[]

        if debug:
            p_deltas_alt =[a*b for a,b in zip(model_deltas_list,w_samples_list)]
            p_optims_alt = [a*b for a,b in zip(optim_deltas_list,w_samples_list)]
        
        # global model is weighted average of client models
        agg_model = divide_model_ws(agg_model, round_total_samples)
        agg_optim = divide_optim_ws(agg_optim, round_total_samples)
        
        master_model.set_weights(add_model_ws(round_master_weights, agg_model))
        master_model.set_optim_weights(add_optim_ws(round_master_optims,
                                                    agg_optim))
                #Check the accurary
        err, test_acc = master_model.test(test[0], test[1], test[0].shape[0])
        test_acc = test_acc*100

        if test_acc > best_acc:
            best_acc = test_acc
            max_round = r+1
        
        wandb.log({
            "Test Acc": test_acc,
            "lr": args.lr,
            'Best_Acc': best_acc,
            'Max round': max_round
        }) 
        if best_acc >= args.target_acc:
            print('Accuracy reached')
            break
        
    err, acc = master_model.test(test[0], test[1], B)
    print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(R, R, err, acc))
    central_errs.append(err)
    central_accs.append(acc)
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))    
    # save stats
    fname = get_out_fname(  'ce_fedavg', master_model.name, C, E, 
                            W, iid, s, None, seed)
    save_data(fname, [central_errs, central_accs])


# def run_fed_avg(dataset, model_fn, C, E, B, W, iid, R, s, lr, seed):
#     """
#     Load dataset and perform R rounds of FedAvg using given FedAvg parameters. 
#     Saves round errors and accuracies at server in file with exp details.
    
#     Parameters:
#     dataset (str):          'mnist' or 'cifar'
#     model_fn (FedAvgModel): callable *not* instance of model class to use 
#     C (float):              fraction of workers used per round
#     E (int):                number of local worker epochs of training
#     B (int):                worker batch size 
#     W (int):                number of workers 
#     iid (bool):             iid or non-iid data partition
#     R (int):                total number of rounds to run
#     s (float):              sparsity 0 <= s < 1
#     lr (float):             SGD learning rate used
#     seed (int):             random seed for trial
#     """
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
    
#     train, test = load_dataset(dataset, W, iid)
    
#     master_model = model_fn()
#     worker_model = model_fn()
    
#     central_errs = []
#     central_accs = []
#     best_acc = 0
#     worker_ids = np.arange(W)
#     workers_per_round = max(int(C * W), 1)
    
#     for r in range(R):
#         round_master_weights = master_model.get_weights()

#         # to store aggregate updates
#         agg_model = zeros_like_model(round_master_weights)
        
#         round_total_samples = 0
        
#         err, acc = master_model.test(test[0], test[1], B)
#         print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(r, R, err, acc))
#         central_errs.append(err)
#         central_accs.append(acc)
        
#         # indexes of workers participating in round
#         choices = np.random.choice(worker_ids, workers_per_round, replace=False)
        
#         for w in choices:
#             # "download" global model
#             worker_model.set_weights(round_master_weights)

#             w_samples = train[0][w].shape[0]
#             round_total_samples += w_samples
            
#             # train worker model for given num epochs
#             for e in range(E):
#                 w_x, w_y = shuffle_client_data((train[0][w], train[1][w]))
#                 worker_model.train(w_x, w_y, B)
            
#             worker_deltas = minus_model_ws( worker_model.get_weights(),
#                                             round_master_weights)
            
#             # compress and decompress deltas as per (part of) Algorithm 1
#             if s > 0:
#                 worker_deltas = compress_fed_avg_deltas(worker_deltas, s)
            
#             # add to aggregate model, weighted by local samples
#             p_deltas = multiply_model_ws(worker_deltas, w_samples)
#             agg_model = add_model_ws(agg_model, p_deltas)
        
#         # global model is weighted average of client models
#         round_deltas = divide_model_ws(agg_model, round_total_samples)
#         master_model.set_weights(add_model_ws(round_master_weights, round_deltas))


        
#     err, acc = master_model.test(test[0], test[1], B)
#     print('Round {}/{}, err = {:.5f}, acc = {:.5f}'.format(R, R, err, acc))
#     central_errs.append(err)
#     central_accs.append(acc)
        
#     # save stats
#     fname = get_out_fname('fedavg', master_model.name, C, E, W, iid, s, lr, seed)
#     save_data(fname, [central_errs, central_accs])


def main():
    start_time = time.time()
    #Initialize WANDB, get args from inputs
    args = args_parser()
    wandb.config.update(args, allow_val_change=True)
    #print experiments details 
    exp_details(args)

    #TODO2: add argsparse.

    # Use for FedAvg
    if args.optimizer =='sgd':
        optim = lambda: SGD(args.lr)
    elif args.optimizer =='adam' :
        optim = lambda: Adam(0.0005, 0.9, 0.999)
    else:
        print('Err: Unknown Optimizer')
    
    if args.model =='mlp':
    # Use with MNIST
        model_fn = lambda: MNIST2NNModel(optim, CatCrossEnt, 784, 10)
    elif args.model =='cnn':
        model_fn = lambda: MNISTCNNModel(optim, CatCrossEnt, 28, 1, 10)
    else:
        print("Err: Unknown model")
    # Use with CIFAR
    # model_fn = lambda: CIFARCNNModel(optim, CatCrossEnt, 32, 3, 10)

    for seed in args.seed:
        # run FedAvg
        # run_fed_avg(DATASET, model_fn, C, E, B, W, IID, R, SPARSITY, LR, seed)
    
        # Run CE-FedAvg
        #TODO:2: add to argsparse
        run_ce_fed_avg(args.dataset, model_fn, args.C, args.E, args.B, args.W, args.iid, args.R, args.S, seed,args)

if __name__ == '__main__':
    main()
