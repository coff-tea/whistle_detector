"""
From https://github.com/coff-tea/whistle_detector
""" 

import sys
import random

import argparse
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import optuna

from Helpers import *



#===============================================================================================
#### Argparser and json####
parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, help="Task to be done.", choices=["cross", "tune", "train"])
parser.add_argument("model_name", type=str, help="Which model to use.", \
                    choices=["simple", "vgg16tf", "vgg16tfd", "vgg16", "vgg16bn", "vgg19", "vgg19bn", "res50", "res101", "res152", "dense161", "dense169", "dense201"])
parser.add_argument("data_format", type=str, help="Data format.", \
                    choices=["single", "all", "stk", "avg", "stkwavg"])
parser.add_argument("-k", "--kfolds", type=int, default=5, dest="k", required=False, help="Number of K folds to use for cross-validation")
parser.add_argument("-c", "--channel", type=int, default=0, dest="c", required=False, help="If using single channel, which channel.")
parser.add_argument("-tf", action="store_true",  help="For non-tf models, use tf data.")
parser.add_argument("-g", action="store_true",  help="Use global average pooling rather than linear layer.")
parser.add_argument("-f", "--freeze", type=int, default=0, dest="f", required=False, help="If freezing part of pre-trained layer, only valid with certain model types.")
parser.add_argument("-o", action="store_true",  help="Do not use pretrained model.")
parser.add_argument("-r", "--retrain", type=str, default="", dest="r", required=False, help="If retraining a saved model, provide the model name.")
parser.add_argument("-p", "--partial", type=float, default=1.0, dest="p", required=False, help="If using only part of a dataset for training.")
parser.add_argument("-l", "--longer", type=int, default=1, dest="l", required=False, help="Use a multiplier for stop_after time if using partial dataset.")
args = parser.parse_args()
paras = json.load(open("Helpers/cross_tune_train.json", "r"))   # Stores relevant information for data, folder system, training process, etc.


#===============================================================================================
#### General setup ####
model = None
opt = None
crit = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights = torch.ones(1)
#--------- Check data formatting conditions
if args.data_format == "single":
    if paras["mdl_channels"] != 1:
        print("Single channel error")
        exit()
elif args.data_format == "all":
    if paras["mdl_channels"] != 1:
        print("All channel error")
        exit()
elif args.data_format == "stk":
    if paras["mdl_channels"] != paras["channels"] and paras["channels"] != 1:
        print("All channel error")
        exit()
elif args.data_format == "avg":
    if paras["mdl_channels"] != 1:
        print("Avg channel error")
        exit()
elif args.data_format == "stkwavg":
    if paras["mdl_channels"] <= paras["channels"]:
        print("All channel error")
        exit()
#--------- Set stopping condition value
stop_after = paras["stop_after"]    
if args.model_name == "simple":     # Simple model requires more training to converge
    stop_after *= 2
elif args.mode != "train":
    stop_after /= 2
stop_after = int(stop_after)
if args.f > 0:
    save_name = "{}-{:d}-{:d}_{}_{}_{}".format(paras["save_spec"], paras["seed"], args.f, args.mode, args.model_name, args.data_format)
else:
    save_name = "{}-{:d}_{}_{}_{}".format(paras["save_spec"], paras["seed"], args.mode, args.model_name, args.data_format)
print("Working on {}".format(save_name))        # Print save_name
if args.g:
    print("\tUse GAP model")
if args.o:
    print("\tNo pretrained weights")
pretrained = not args.o


#===============================================================================================
#### Load data ####
if "tf" in args.model_name or args.tf:      # Using Add-Pre (https://arxiv.org/abs/2211.15406)
    X_pos = []
    if paras["channels"] == 1:
        X_pos.append(spec_data.load_data_tf(np.load("{}/{}_Xpos.npy".format(paras["data_folder"], paras["data_spec"]))))
    else:
        for ch in range(paras["channels"]):
            X_pos.append(spec_data.load_data_tf(np.load("{}/{}_X{:d}pos.npy".format(paras["data_folder"], paras["data_spec"], ch+1))))
    X_neg = []
    if paras["channels"] == 1:
        X_neg.append(spec_data.load_data_tf(np.load("{}/{}_Xneg.npy".format(paras["data_folder"], paras["data_spec"]))))
    else:
        for ch in range(paras["channels"]):
            X_neg.append(spec_data.load_data_tf(np.load("{}/{}_X{:d}neg.npy".format(paras["data_folder"], paras["data_spec"], ch+1))))
else:       # Using Min-Pre
    X_pos = []
    if paras["channels"] == 1:
        X_pos.append(spec_data.load_data(np.load("{}/{}_Xpos.npy".format(paras["data_folder"], paras["data_spec"]))))
    else:
        for ch in range(paras["channels"]):
            X_pos.append(spec_data.load_data(np.load("{}/{}_X{:d}pos.npy".format(paras["data_folder"], paras["data_spec"], ch+1))))
    X_neg = []
    if paras["channels"] == 1:
        X_neg.append(spec_data.load_data(np.load("{}/{}_Xneg.npy".format(paras["data_folder"], paras["data_spec"]))))
    else:
        for ch in range(paras["channels"]):
            X_neg.append(spec_data.load_data(np.load("{}/{}_X{:d}neg.npy".format(paras["data_folder"], paras["data_spec"], ch+1))))
            
idx_dict = dict()
if args.mode == "cross":        # Fold boundaries for k-cross validation
    # Positive indices
    fold_size = round(len(X_pos[0])/args.k)
    fold_bounds = []
    for i in range(args.k):
        fold_bounds.append(i*fold_size)
    fold_bounds.append(len(X_pos[0]))
    idx_dict["pos"] = fold_bounds.copy()
    # Positive indices
    fold_size = round(len(X_neg[0])/args.k)
    fold_bounds = []
    for i in range(args.k):
        fold_bounds.append(i*fold_size)
    fold_bounds.append(len(X_neg[0]))
    idx_dict["neg"] = fold_bounds.copy()
else:       # Which indices to use for which set
    idx_dict["pos"] = tuple(spec_data.split_sets(len(X_pos[0]), [paras["train_perc"], 0.5], seed=paras["seed"]))
    idx_dict["neg"] = tuple(spec_data.split_sets(len(X_neg[0]), [paras["train_perc"], 0.5], seed=paras["seed"]+1))


#===============================================================================================
#### FUNCTION: train_epoch ####
# Train through batches in a dataloader using global parameters of the file. Returns tuple of
# epoch results (loss, accuracy (%), false alarm (%), missed detection (%)).
# PARAMETERS
#   - dataloader (torch.utils.data.DataLoader object)
def train_epoch(dataloader):
    global model
    global opt
    global crit
    global device

    ep_loss = 0.
    y_pred = []
    y_true = []
    model.train()
    for _, data in enumerate(dataloader):
        x, y = data[0].to(device), data[1].to(device)
        y = torch.unsqueeze(y, dim=1).float()

        opt.zero_grad()
        output = model(x.float())
        loss = crit(output, y)
        loss.backward()
        opt.step()

        ep_loss += loss.item()
        pred = torch.zeros_like(output)
        pred[torch.sigmoid(output) >= 0.5] = 1.
        y_pred.extend([p.item() for p in pred])
        y_true.extend([true.item() for true in y])

    cf_matrix = confusion_matrix(y_true, y_pred)
    ep_acc = (cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix) * 100
    ep_fa = cf_matrix[0][1] / np.sum(cf_matrix[0]) * 100
    ep_md = cf_matrix[1][0] / np.sum(cf_matrix[1]) * 100

    return ep_loss / len(dataloader), ep_acc, ep_fa, ep_md


#===============================================================================================
#### FUNCTION: eval_dataloader ####
# Evaluate batches in a dataloader using global parameters of the file. Returns tuple of epoch 
# results (loss, accuracy (%), false alarm (%), missed detection (%)).
# PARAMETERS
#   - dataloader (torch.utils.data.DataLoader object)
def eval_dataloader(dataloader):
    global model
    global crit
    global device

    dl_loss = 0.
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for _, data in enumerate(dataloader):
            x, y = data[0].to(device), data[1].to(device)
            for s in x:
                if torch.mean(s).isnan():
                    print(s)
                    print(torch.max(s))
                    print(torch.min(s))
            y = torch.unsqueeze(y, dim=1).float()

            output = model(x.float())
            loss = crit(output, y)

            dl_loss += loss.item()
            pred = torch.zeros_like(output)
            pred[torch.sigmoid(output) >= 0.5] = 1.
            y_pred.extend([p.item() for p in pred])
            y_true.extend([true.item() for true in y])

    cf_matrix = confusion_matrix(y_true, y_pred)
    dl_acc = (cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix) * 100
    dl_fa = cf_matrix[0][1] / np.sum(cf_matrix[0]) * 100
    dl_md = cf_matrix[1][0] / np.sum(cf_matrix[1]) * 100 

    return dl_loss / len(dataloader), dl_acc, dl_fa, dl_md


#===============================================================================================
#### K-FOLDS CROSSVALIDATION ####
if args.mode == "cross":
    #--------- Dictionaries to retain values, keyed with k# (fold)
    try:        # Try finding in "results_folder"
        cv_perf = np.load("{}/{}.npy".format(paras["results_folder"], save_name), allow_pickle=True).item()
    except:
        cv_perf = dict()
        cv_perf["hyper"] = (paras["def_pos"], paras["def_dropout"], paras["def_lr"], \
                              paras["def_decay"], paras["def_beta1"], paras["def_beta2"])
    folds_done = cv_perf.keys()
    #--------- Prepare to run
    model = detectors.make_detector(args.model_name, paras["mdl_channels"], paras["img_dim"], freeze=args.f, gap=args.g, pre=pretrained)
    model.to(device)
    torch.save(model.state_dict(), "{}/start.pt".format(paras["temp_folder"]))
    pos_idx = [i for i in range(len(X_pos[0]))]
    neg_idx = [i for i in range(len(X_neg[0]))]
    if paras["replicable"]: 
        random.seed(paras["seed"])
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    #--------- Iterate through folds
    for k in range(args.k):
        #--------- Make dictionary key
        d_key = "fold{:d}".format(k+1)
        if d_key in folds_done:
            print("\tSkip {}".format(d_key))
            continue
        print("\tTry {}".format(d_key))
        #------------------ Set up model, criterion, optimizer
        model.load_state_dict(torch.load("{}/start.pt".format(paras["temp_folder"])))
        torch.save(model.state_dict(), "{}/best.pt".format(paras["temp_folder"]))
        class_weights[0] = paras["def_pos"]
        crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        opt = optim.Adam(model.parameters(), lr=paras["def_lr"], \
                         betas=(paras["def_beta1"], paras["def_beta2"]), weight_decay=paras["def_decay"])
        #------------------ Make dataloaders
        pos_test_idx = [i for i in pos_idx[idx_dict["pos"][k]:idx_dict["pos"][k+1]]]
        pos_train_idx = [i for i in pos_idx if i not in pos_test_idx]
        if "tf" in args.model_name or args.tf:
            test_data = spec_data.process_data_tf(X_pos, pos_test_idx, paras["mdl_channels"], paras["img_dim"], tag=1)
            train_data = spec_data.process_data_tf(X_pos, pos_train_idx, paras["mdl_channels"], paras["img_dim"], tag=1)
        else:
            test_data = spec_data.process_data(X_pos, pos_test_idx, args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
            train_data = spec_data.process_data(X_pos, pos_train_idx, args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        neg_test_idx = [i for i in neg_idx[idx_dict["neg"][k]:idx_dict["neg"][k+1]]]
        neg_train_idx = [i for i in neg_idx if i not in neg_test_idx]
        if "tf" in args.model_name or args.tf:
            test_data.extend(spec_data.process_data_tf(X_neg, neg_test_idx, paras["mdl_channels"], paras["img_dim"], tag=0))
            train_data.extend(spec_data.process_data_tf(X_neg, neg_train_idx, paras["mdl_channels"], paras["img_dim"], tag=0))
        else:
            test_data.extend(spec_data.process_data(X_neg, neg_test_idx, args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
            train_data.extend(spec_data.process_data(X_neg, neg_train_idx, args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=paras["batch_size"])
        del train_data
        testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=paras["batch_size"])
        del test_data
        #------------------ Start training
        train_losses = []
        train_accs = []
        no_change = 0
        best_tloss = 1000.
        best_tacc = 0.
        best_at = 0
        stopped_epoch = paras["max_epochs"]
        for epoch in range(paras["max_epochs"]):
            ept_loss, ept_acc, _, _ = train_epoch(trainloader)
            if ept_loss + paras["improve_margin"] < best_tloss:
                best_tloss = ept_loss
                best_tacc = ept_acc
                best_at = epoch
                torch.save(model.state_dict(), "{}/best.pt".format(paras["temp_folder"]))
                no_change = 0
            else:
                no_change += 1
            train_losses.append(ept_loss)
            train_accs.append(ept_acc)
            if no_change >= stop_after:
                stopped_epoch = epoch
                break
            if (epoch+1) % paras["print_status"] == 0:
                print("\t\t\tOn epoch {:d}".format(epoch+1))
        #------------------ Get test performance
        model.load_state_dict(torch.load("{}/best.pt".format(paras["temp_folder"])))
        test_loss, test_acc, test_fa, test_md = eval_dataloader(testloader)
        cv_perf[d_key] = (best_tloss, best_tacc, best_at, test_loss, test_acc, test_fa, test_md)
        #------------------ Save to dictionary
        iter_dict = dict()
        iter_dict["acc"] = train_accs
        iter_dict["loss"] = train_losses
        iter_dict["perf"] = (best_tloss, best_tacc, best_at, test_loss, test_acc, test_fa, test_md)
        cv_perf[d_key] = iter_dict
        np.save("{}/{}".format(paras["results_folder"], save_name), cv_perf, allow_pickle=True)


#===============================================================================================
#### HYPERPARAMETER TUNING ####
if args.mode == "tune":
    #------------------ Dictionaries to retain best hyperparameters, keyed by val_accuracy
    try:        # Try finding in "results_folder" 
        hyp_perf = np.load("{}/{}.npy".format(paras["results_folder"], save_name), allow_pickle=True).item()
    except:
        hyp_perf = dict()
    #------------------ Make dataloaders
    if "tf" in args.model_name or args.tf:
        train_data = spec_data.process_data_tf(X_pos, idx_dict["pos"][0], paras["mdl_channels"], paras["img_dim"], tag=1)
        train_data.extend(spec_data.process_data_tf(X_neg, idx_dict["neg"][0], paras["mdl_channels"], paras["img_dim"], tag=0))
    else:
        train_data = spec_data.process_data(X_pos, idx_dict["pos"][0], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        train_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][0], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=paras["batch_size"])
    del train_data
    if "tf" in args.model_name or args.tf:
        val_data = spec_data.process_data_tf(X_pos, idx_dict["pos"][2], paras["mdl_channels"], paras["img_dim"], tag=1)
        val_data.extend(spec_data.process_data_tf(X_neg, idx_dict["neg"][2], paras["mdl_channels"], paras["img_dim"], tag=0))
    else:
        val_data = spec_data.process_data(X_pos, idx_dict["pos"][2], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        val_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][2], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
    valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=paras["batch_size"])
    del val_data
    #------------------ Objective function
    def objective(trial, model_name, chs_in, spat_dim, max_epochs, prune=True):
        global model
        global crit 
        global opt
        global device 
        global trainloader
        global valloader
        #------------------ Create model, criterion, and optimiser
        if "simple" in args.model_name or "tfd" in args.model_name:
            dropout = trial.suggest_float("dropout", 0.25, 0.8, log=True)
            model = detectors.make_detector(args.model_name, chs_in, paras["img_dim"], freeze=args.f, gap=args.g, dropout=dropout, pre=pretrained)
        else:
            model = detectors.make_detector(args.model_name, chs_in, paras["img_dim"], freeze=args.f, gap=args.g, pre=pretrained)
        model.to(device)
        class_weights[0] = trial.suggest_float("pos", 1, 3, log=True)
        crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
        decay = trial.suggest_float("decay", 1e-10, lr, log=True)
        beta1 = trial.suggest_float("beta1", 0.5, 0.99, log=True)
        beta2 = trial.suggest_float("beta2", 0.5, 0.99, log=True)
        opt = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
        #------------------ Start training
        no_change = 0
        best_vloss = 1000.
        for epoch in range(max_epochs):
            _, _, _, _ = train_epoch(trainloader)
            epv_loss, _, _, _ = eval_dataloader(valloader)
            if epv_loss + paras["improve_margin"] < best_vloss:
                best_vloss = epv_loss
                torch.save(model.state_dict(), "{}/best.pt".format(paras["temp_folder"]))
                no_change = 0
            else:
                trial.report(best_vloss, epoch)
                if prune and trial.should_prune():
                    raise optuna.TrialPruned()
                no_change += 1
            if no_change >= stop_after:
                break
            if (epoch+1) % paras["print_status"] == 0:
                print("\t\t\tOn epoch {:d}".format(epoch+1))
        return best_vloss
    #------------------ Tune model
    study = optuna.create_study(direction="minimize", study_name=save_name)
    if len(hyp_perf.keys()) == 0:       # Starting point
        start_point = dict()
        if "simple" in args.model_name or "tfd" in args.model_name:
            start_point["dropout"] = paras["def_dropout"]
        start_point["lr"] = paras["def_lr"]
        start_point["decay"] = paras["def_decay"]
        start_point["beta1"] = paras["def_beta1"]
        start_point["beta2"] = paras["def_beta2"]
        start_point["pos"] = paras["def_pos"]
        study.enqueue_trial(start_point)
    else:       # Queue up best result from last run
        latest = 0
        key = ""
        for k in hyp_perf.keys():
            parts = k.split("-")
            if int(parts[0]) % 2 == 0 and int(parts[0]) > latest:
                key = k
        print("\t\tUse parameters from: {}".format(key))
        study.enqueue_trial(hyp_perf[key])
    study.optimize(lambda trial:objective(trial, args.model_name, paras["mdl_channels"], paras["img_dim"], int(paras["max_epochs"]/2)), \
                                          n_trials=paras["tune_trials"], gc_after_trial=True)
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
    first_trial = dict()
    for key, value in complete_trials[0].params.items():    # Save starting point
        first_trial[key] = value
    hyp_perf["{:d}-{:.4f}".format(len(hyp_perf)+1, complete_trials[0].value)] = first_trial
    best_trial = dict()
    for key, value in study.best_trial.params.items():      # Save best result
        best_trial[key] = value
    hyp_perf["{:d}-{:.4f}".format(len(hyp_perf)+1, study.best_trial.value)] = best_trial
    np.save("{}/{}.npy".format(paras["results_folder"], save_name), hyp_perf, allow_pickle=True)


#===============================================================================================
#### MODEL TRAINING ####
if args.mode == "train":
    best_paras = json.load(open("Helpers/train_detectors.json", "r"))
    if args.model_name not in best_paras.keys():
        hyper = [{"dropout": 0.5, "pos": 2, "lr": 2e-4, "decay": 1e-6, "beta1": 0.9, "beta2": 0.99}]
        print("\tNo best set of hyp found for model type {}".format(args.model_name))
    else:   
        hyper = best_paras[args.model_name]
        print("\tTraining {} with {:d} best parameter sets".format(args.model_name, len(hyper)))
    #------------------ Cut down training set
    if args.p > 0 and args.p < 1:
        pos_train, _ = tuple(spec_data.split_sets(len(idx_dict["pos"][0]), [args.p], seed=paras["seed"]))
        pos_list = []
        for i in pos_train:
            pos_list.append(idx_dict["pos"][0][i])
        idx_dict["pos"] = (pos_list, idx_dict["pos"][1], idx_dict["pos"][2])
        neg_train, _ = tuple(spec_data.split_sets(len(idx_dict["neg"][0]), [args.p], seed=paras["seed"]+1))
        neg_list = []
        for i in pos_train:
            neg_list.append(idx_dict["neg"][0][i])
        idx_dict["neg"] = (neg_list, idx_dict["neg"][1], idx_dict["neg"][2])
        stop_after *= int(args.l)
    #------------------ Make dataloaders
    if "tf" in args.model_name or args.tf:
        train_data = spec_data.process_data_tf(X_pos, idx_dict["pos"][0], paras["mdl_channels"], paras["img_dim"], tag=1)
        train_data.extend(spec_data.process_data_tf(X_neg, idx_dict["neg"][0], paras["mdl_channels"], paras["img_dim"], tag=0))
    else:
        train_data = spec_data.process_data(X_pos, idx_dict["pos"][0], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        train_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][0], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=paras["batch_size"])
    del train_data
    if "tf" in args.model_name or args.tf:
        test_data = spec_data.process_data_tf(X_pos, idx_dict["pos"][1], paras["mdl_channels"], paras["img_dim"], tag=1)
        test_data.extend(spec_data.process_data_tf(X_neg, idx_dict["neg"][1], paras["mdl_channels"], paras["img_dim"], tag=0))
    else:
        test_data = spec_data.process_data(X_pos, idx_dict["pos"][1], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        test_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][1], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=paras["batch_size"])
    del test_data
    if "tf" in args.model_name or args.tf:
        val_data = spec_data.process_data_tf(X_pos, idx_dict["pos"][2], paras["mdl_channels"], paras["img_dim"], tag=1)
        val_data.extend(spec_data.process_data_tf(X_neg, idx_dict["neg"][2], paras["mdl_channels"], paras["img_dim"], tag=0))
    else:
        val_data = spec_data.process_data(X_pos, idx_dict["pos"][2], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=1, which_ch=args.c)
        val_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][2], args.data_format, paras["mdl_channels"], paras["img_dim"], tag=0, which_ch=args.c))
    valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=paras["batch_size"])
    del val_data
    #------------------ Start training
    for h in range(len(hyper)):
        if "simple" in args.model_name or "tfd" in args.model_name:
            model = detectors.make_detector(args.model_name, paras["mdl_channels"], paras["img_dim"], freeze=args.f, gap=args.g, dropout=hyper[h]["dropout"], pre=pretrained)
        else:
            model = detectors.make_detector(args.model_name, paras["mdl_channels"], paras["img_dim"], freeze=args.f, gap=args.g, pre=pretrained)
        model.to(device)
        if args.r != "":
            model.load_state_dict(torch.load("{}/{}".format(paras["models_folder"], args.r)))
        torch.save(model.state_dict(), "{}/best.pt".format(paras["temp_folder"]))
        class_weights[0] = hyper[h]["pos"]
        crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        opt = optim.Adam(model.parameters(), lr=hyper[h]["lr"], betas=(hyper[h]["beta1"], hyper[h]["beta2"]), \
                         weight_decay=hyper[h]["decay"])
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        no_change = 0
        best_vloss, best_vacc, _, _ = eval_dataloader(valloader)
        best_at = 0
        stopped_epoch = paras["max_epochs"]
        for epoch in range(paras["max_epochs"]):
            ept_loss, ept_acc, _, _ = train_epoch(trainloader)
            epv_loss, epv_acc, _, _ = eval_dataloader(valloader)
            if epv_loss + paras["improve_margin"] < best_vloss:
                best_vloss = epv_loss
                best_vacc = epv_acc
                best_at = epoch+1
                torch.save(model.state_dict(), "{}/best.pt".format(paras["temp_folder"]))
                no_change = 0
            else:
                no_change += 1
            train_losses.append(ept_loss)
            train_accs.append(ept_acc)
            val_losses.append(epv_loss)
            val_accs.append(epv_acc)
            if no_change >= stop_after:
                stopped_epoch = epoch
                break
            if (epoch+1) % paras["print_status"] == 0:
                print("\t\t\tOn epoch {:d}, best is {:.4f} in epoch {:d}".format(epoch+1, best_vloss, best_at))
        #==================== SAVE RESULTS
        train_perf = dict()
        model.load_state_dict(torch.load("{}/best.pt".format(paras["temp_folder"])))
        test_loss, test_acc, test_fa, test_md = eval_dataloader(testloader)
        print("\tTest results: ", test_loss, test_acc)
        if args.p > 0 and args.p < 1:
            train_perf["percent"] = args.p
            train_perf["longer"] = args.l
        train_perf["hyper"] = hyper[h]
        train_perf["train_hist"] = (train_losses, train_accs)
        train_perf["val_hist"] = (val_losses, val_accs)
        train_perf["best"] = (best_vloss, best_vacc, best_at, stopped_epoch)
        train_perf["test"] = (test_loss, test_acc, test_fa, test_md)
        if args.r != "":
            train_perf["starting"] = args.r
        this_save = save_name.replace(args.model_name, "{}-{:d}".format(args.model_name, h))
        np.save("{}/{}".format(paras["results_folder"], this_save), train_perf, allow_pickle=True)
        torch.save(model.state_dict(), "{}/{}.pt".format(paras["models_folder"], this_save))
