import torch
from pbb.utils import runexp

from time import time, ctime

#============the code is built on the work of https://github.com/mperezortiz/PBB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 16

TRAIN_EPOCHS = 100
prior_epochs = 40

DELTA = 0.025
DELTA_TEST = 0.01
PRIOR = 'learnt'

SIGMAPRIOR= 0.01
PMIN = 1e-5
KL_PENALTY= 1
LEARNING_RATE = 0.01
MOMENTUM= 0.9
LEARNING_RATE_PRIOR = 0.01
MOMENTUM_PRIOR = 0.9
perc_prior = 0.5
dropout_prob = 0.5

# note the number of MC samples used in the paper is 150.000, which usually takes a several hours to compute
MC_SAMPLES = 150
samples_ensemble = 100

t = time()
print("##====== begin======= ",ctime(t))
                
runexp('market','fclassic' ,PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM,
                                       LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, delta_test=DELTA_TEST,
                                       mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, 
                                        prior_epochs=prior_epochs,perc_prior=perc_prior,verbose=True, dropout_prob=dropout_prob,batch_size=BATCH_SIZE, kl_penalty=KL_PENALTY,samples_ensemble=samples_ensemble)

                

