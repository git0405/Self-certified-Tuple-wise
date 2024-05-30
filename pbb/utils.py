import math
import numpy as np
import torch

import torch.optim as optim

from tqdm import tqdm, trange
from pbb.models import  testNNet ,trainNNet, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, ResNet18_new,ProbResNet_bn
from pbb.bounds import PBBobj
from pbb import data

from time import time, ctime

model_name = [ResNet18_new,ProbResNet_bn]
 

 


def runexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, 
learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, delta_test=0.01, mc_samples=1000, 
samples_ensemble=10, kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', 
verbose=False, device='cuda', prior_epochs=1, dropout_prob=0.2, perc_train=1.0, verbose_test=False, 
perc_prior=0.2, batch_size=16):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)
    
    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 4,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    rho_prior = math.log(math.exp(sigma_prior)-1.0)


    train, val, test,class_img_labels, class_val,class_num = data.pair_pretrain_on_dataset(name_data,perc_prior=perc_prior)
    print("load train")



    net0 = model_name[0]( ).to(device)
        
        
    train = data.SiameseNetworkDataset(train, should_invert=False)
    print("load val")
    if val:
        val  = data.SiameseNetworkDataset(val, should_invert=False)
    print("load test")
    test = data.SiameseNetworkDataset(test , should_invert=False)

    if prior_type == 'learnt':
        train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(
            train, val, test, loader_kargs, batch_size, prior=True, perc_train=perc_train, perc_prior=perc_prior)
        optimizer = optim.SGD(
            net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior,weight_decay=5e-04,)
        for epoch in trange(prior_epochs):
                    
                    trainNNet(net0, optimizer, epoch, valid_loader, epoch,
                              device=device, verbose=verbose)
                    errornet0 = testNNet(net0, test_loader, device=device)


    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    toolarge = True
    train_size = len(train_loader.dataset)
    classes = 2


    if model == 'cnn':
        toolarge = True

        net =model_name[1](rho_prior, prior_dist=prior_dist,
                          device=device, init_net=net0).to(device)

    else:
        raise RuntimeError(f'Architecture {model} not supported')
    testNNet(net, test_loader, device=device)

    bound = PBBobj(objective, pmin, classes, delta,
                    delta_test, mc_samples, kl_penalty, device, n_posterior = posterior_n_size, n_bound=bound_n_size)

    optimizer_lambda = None
    lambda_var = None
    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    print("bound n size ", bound_n_size)


    for epoch in trange(train_epochs):
        print("begin train epoch##=====",epoch)

        trainPNNet(net, optimizer, bound, epoch, train_loader, lambda_var, optimizer_lambda, verbose)

        if verbose_test and ((epoch+1) % 5 == 0):
            train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge,
            bound, device=device, lambda_var=lambda_var, train_loader=val_bound)

            stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
            post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
            ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

            print(f"***Checkpoint results***")         
            print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
            print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")


    ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)
    t = time()
    print(ens_loss, ens_err)
    print("done test Ensemble",ctime(t))

    stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
    t = time()
    print(stch_loss, stch_err)
    print('test',"done test Stochastic",ctime(t))
    

    post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
    t = time()
    print(post_loss, post_err)
    print("done test Posterior Mean",ctime(t))

    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, device=device,
    lambda_var=lambda_var, train_loader=val_bound)
    t = time()
    print('val',train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train)
    print("done compute Risk Certificates",ctime(t))

    print(f"***Final results***") 
    print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
    print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
