import math
import numpy as np
import torch

from tqdm import tqdm, trange
import torch.nn.functional as F


class PBBobj():
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired 
    training objective and evaluate the risk certificate at the end of training. 

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)
    
    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem
    
    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective
    
    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    mc_samples : int
        number of Monte Carlo samples when estimating the risk

    kl_penalty : float
        penalty for the kl coefficient in the training objective
    
    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, objective='fclassic', pmin=1e-4, classes=10, delta=0.025,
    delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda', n_posterior=30000, n_bound=30000):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.n_bound = n_bound


    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        empirical_risk = F.nll_loss(outputs, targets)
        
        if bounded == True:
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
            
        return empirical_risk

    def compute_losses(self, net, data1,data2, target, clamping=True):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well

        outputs = net(data1,data2, sample=True,
                          clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(
                    outputs, target, clamping)
            
        pred = outputs.max(1, keepdim=True)[1]
             
        correct = pred.eq(
                    target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1-(correct/total)

        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, train_size, lambda_var=None):
        # compute training objectives
        
        if self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(
                kl + np.log((self.n_bound*(self.n_bound-1)/2)/self.delta), 2*np.trunc(train_size/2))
            
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj

    def mcsampling(self, net, input, target, batches=True, clamping=True, data_loader=None):
        # compute empirical risk with Monte Carlo sampling
        error = 0.0
        cross_entropy = 0.0
        batches = True
       
        if batches:
            for batch_id, (data_batch1,data_batch2, target_batch) in enumerate(tqdm(data_loader)):
                try:
                    if len(target_batch[0])>1:
                        mutil_loss =True
                except:
                    mutil_loss = False
                
                if mutil_loss: 
                    label1, label2, label3 = target_batch
                    data_batch1,data_batch2, label1 = data_batch1.to(self.device),data_batch2.to(self.device), label1.to(self.device)
                
                    cross_entropy_mc = 0.0
                    error_mc = 0.0

                    for i in range(self.mc_samples):
                        loss_ce, loss_01, _ = self.compute_losses(net, data_batch1, data_batch2,label1, clamping)
                        cross_entropy_mc += loss_ce
                        error_mc += loss_01
                    # we average cross-entropy and 0-1 error over all MC samples
                    cross_entropy += cross_entropy_mc/self.mc_samples
                    error += error_mc/self.mc_samples
                # we average cross-entropy and 0-1 error over all batches
                
                else:
                 
                    try: 
                        target_batch = target_batch.squeeze(1)
                    except:
                        pass
                    data_batch1, data_batch2,target_batch = data_batch1.to(
                        self.device),data_batch2.to(self.device), target_batch.to(self.device)
                    cross_entropy_mc = 0.0
                    error_mc = 0.0

                    for i in range(self.mc_samples):
                        loss_ce, loss_01, _ = self.compute_losses(net, data_batch1, data_batch2,target_batch, clamping)
                        cross_entropy_mc += loss_ce
                        error_mc += loss_01
                    # we average cross-entropy and 0-1 error over all MC samples
                    cross_entropy += cross_entropy_mc/self.mc_samples
                    error += error_mc/self.mc_samples
                # we average cross-entropy and 0-1 error over all batches
                        
            cross_entropy /= (batch_id+1)
            error /= (batch_id+1)

        return cross_entropy, error

    def train_obj(self, net, input1,input2, target, clamping=True, lambda_var=None):
        # compute train objective and return all metrics
        # outputs = torch.zeros(target.size(0), self.classes).to(self.device)

        try:
            if len(target[0])>1:
                mutil_loss =True
        except:
            mutil_loss = False
        if mutil_loss:
            kl,_ = net.compute_kl()
            loss_ce, loss_01, outputs = self.compute_losses(net,
                                                            input1,input2, target, clamping)
            
            train_obj1 = self.bound(loss_ce[0], kl, self.n_posterior, lambda_var)
            train_obj23 = train_obj1 + 0.5*loss_ce[1]+ 0.5* loss_ce[2]
            train_obj = [train_obj1,train_obj23 ]

        else: 
            kl = net.compute_kl()
            loss_ce, loss_01, outputs = self.compute_losses(net,
                                                            input1,input2, target, clamping)

            train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var)
        return train_obj, kl/self.n_posterior, outputs, loss_ce, loss_01

    def compute_final_stats_risk(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        # compute all final stats and risk certificates

        kl = net.compute_kl()
        try:
            kl,_= kl

        except:
            pass

        if data_loader:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=True,
                                                     clamping=True, data_loader=data_loader)

        empirical_risk_ce = inv_kl(
                error_ce.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01 = inv_kl(
                error_01, np.log(2/self.delta_test)/self.mc_samples)
    
        train_obj = self.bound(empirical_risk_ce, kl, self.n_posterior, lambda_var)
        risk_ce = inv_kl(empirical_risk_ce, (kl + np.log(( self.n_bound*(self.n_bound-1)/2 +1)/self.delta_test))/(np.trunc(self.n_bound/2)))
        risk_01 = inv_kl(empirical_risk_01, (kl + np.log((self.n_bound*(self.n_bound-1)/2 +1)/self.delta_test))/(np.trunc(self.n_bound/2)))


        return train_obj.item(), kl.item()/self.n_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01


def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*math.log(qs/p)+0)
        else:
            ikl = ks-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
