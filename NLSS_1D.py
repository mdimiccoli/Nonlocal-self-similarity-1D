import sys
import scipy as sp
import numpy as np
import math
import math
from scipy import spatial
        
def nlss( X, param,r_loc):
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
        
        N = len(X)
        dim = X.shape[1]
        #--- compute distance & similarity
        XM2 = sp.spatial.distance.squareform(sp.spatial.distance.pdist(X,metric = 'cityblock'))/dim
        XSS = np.exp(-1/param*XM2)
        XSS0 = XSS
        # if not NLM over whole sequence, set similarity beyond search radius to
        # zero and extend boundaries
        if r_loc:   
            t=np.arange(1,N)
            for i in range(1,N):
                #print(t)
                #print(np.arange(N-i))
                t = np.concatenate((t, np.arange(1,N-i)), axis=None)
            t = np.transpose(t)
            T=spatial.distance.squareform(t) #%T=T(:,1:min(N1,N2));
            MASK=sp.exp(math.log(0.25)*T/r_loc)
            XSS=np.multiply(XSS0,MASK)
            XSS = np.transpose(XSS)
            XSS= np.concatenate((np.flipud(XSS[0:r_loc,:]),XSS,np.flipud(XSS[XSS.shape[0]-r_loc:XSS.shape[0],:])), axis=0)
            XSS = np.transpose(XSS)
        return XSS

def construct_SS_matrix(X,nnh):
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


        

