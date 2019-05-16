# EXAMPLE USAGE: python main_gpu.py -X "EDUB-Seg_seq1.npy" -N 3 -n 1 -p 0.25
import argparse
import torch
import NLSS_1D_GPU
import sys
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-X", "--data_matrix", type=str)
    parser.add_argument("-N", "--search_radius", type=int)
    parser.add_argument("-n", "--neighborhood_size", type=int)
    parser.add_argument("-p", "--parameter", type=float)
    
    args = parser.parse_args()
    X1 = np.load(args.data_matrix)
    nnh = args.neighborhood_size
    r_loc = args.search_radius
    par00 = args.parameter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not torch.cuda.is_available():
        print('GPU not available. Falling back onto CPU. Use CPU optimized main.py instead.')
    
    #X1=X1-X1.min(); X1=X1/X1.max()*2-1 #normalized to [-1,1]
    #----- do nonlocal mean and compute similarity
    XX = np.empty([])

    XX = NLSS_1D_GPU.construct_SS_matrix(X1,nnh) # stack neighborhood
    XXT=torch.tensor(XX).to(device)
    XSS = NLSS_1D_GPU.nlss(XXT,par00,r_loc,device).cpu().detach().numpy() # - compute (self)-similarity

    X1r=np.concatenate((np.flipud(X1[0:r_loc,:]),X1, np.flipud(X1[X1.shape[0]-r_loc:X1.shape[0],:]))) # pad with search radius (periodic boundary conditions)
    a = np.matmul(np.asarray(XSS),np.asarray(X1r))
    xss = np.transpose(XSS)
    b = np.diag(1./xss.sum(axis=0))
    Xnlm=np.matmul(b,a) # non-local mean
    #Xnlm = Xnlm-Xnlm.min(); Xnlm = Xnlm/Xnlm.max()*2-1 #normalized to [-1,1]

    #----- save to output file
    np.save(args.data_matrix[0:-4]+'_nlm.npy',Xnlm)
