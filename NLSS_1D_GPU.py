import sys
import scipy as sp
import numpy as np
import math
import torch

def PDIST(X,device):
    N=X.size()[0]
    dim=X.size()[1]
    numel=N*N*dim
    # CHANGE MAX SIZE IF RUNNING OUT OF MEMORY
    numel_max = 400000000
    #numel_max = 100000000
    nl=np.ceil(numel/numel_max).astype(int) # divide dim by this
    if nl ==1:
        Tpxy=torch.sum(torch.abs(X - X[:, None]),2)
    else:
        Tpxy=torch.zeros(N,N).to(device)
        Tpxy=Tpxy.type(X.type())
        mdim=np.floor(dim/nl).astype(int) # dim slices
        for i in range(nl+1):
            i1=i*mdim
            i2=np.minimum((i+1)*mdim,dim)
            if i1<i2:
                Xt=X[:,i1:i2]
                Tpxy.add_(torch.sum(torch.abs(Xt - Xt[:, None]),2))
    return Tpxy

def nlss( X, param, r_loc,device):
    #    XSS_ij = exp( -1/param * sum_k |X(i,k) - X(j,k)| / N_features )
    # size(X)= N_frames x N_features
    # size(XSS)= N_frames x N_frames
    #
    # if r_loc is given, it specifies the "search radius" for NLmeans
    # in this case:
    #    XSS_ij is exponentially tapered
    #    XSS_ij = 0 for |i-j|>r_loc
    #    XSS is periodically extended to size N_frames x N_frames + 2*r_loc
    if len(sys.argv)<3:
        r_loc=[]
    
    N = X.size()[0]
    dim = X.size()[1]
    #--- compute distance & similarity
    XM2 = PDIST(X,device)/dim
    XSS0 = torch.exp(-1/param*XM2).double()
    # if not NLM over whole sequence, set similarity beyond search radius to
    # zero and extend boundaries
    if r_loc:
        t1=torch.zeros(N,1).to(device)
        t1[:,0]=torch.arange(0,N)
        T=PDIST(t1,device)
        mparam=-1.3862943611198906/r_loc
        MASK=torch.exp(mparam*T).double()
        XSS=torch.mul(XSS0,MASK)
        XSS = XSS.transpose(1,0)
        
        np1=torch.flip(XSS[0:r_loc,:],(0,))
        np2=torch.flip(XSS[XSS.size()[0]-r_loc:XSS.size()[0],:],(0,))
        XSS = torch.cat((np1, XSS, np2))
        XSS = XSS.transpose(1,0)
    return XSS

def construct_SS_matrix(X, nnh):
    # function [XX] = construct_SS_matrix(X,nnh)
    # 
    # stack nnh left-and right neighbor feature vectors of X using periodic
    # boundary conditions.  
    # size(X)  = N_frames x N_features 
    # size(XX) = N_frames x N_features* 2*nnh 

    # periodic border effects
        XE=np.concatenate((  np.flipud(X[1:nnh+1,:]),X,(np.flipud(X[X.shape[0]-nnh-2:X.shape[0]-1,:]))), axis = 0)
    
        #--- construct neighborhood 
    
        a= np.arange(-nnh,0, dtype = int)
        b = np.arange(1,nnh+1, dtype = int)

        NH_ID = np.concatenate((a,b))# do not include self
        #NH_ID = np.concatenate((a,0, b))% include self (i.e., "center pixel")

        if not NH_ID.any():
        # nothing to do: working with pixel only, without neighbors
            XX=X
        else:
        # stack neighbors
            XX=np.array([]) 
        for inh in NH_ID:
            XTMP= np.roll(XE,inh,axis=0) 
            if not XX.size:  
                XX = XTMP[nnh:XTMP.shape[0]-nnh-1,:]
            else:
                XX=np.hstack((XX,XTMP[nnh:XTMP.shape[0]-nnh-1,:]))
        return XX


        

