import numpy as np 
import matplotlib.pyplot as plt 
import sys 

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

def fetch_data(T,data,nc,col_id):
    nt = len(T)
    c_all = np.zeros((nt,nc)) - 1

    for it in range(len(T)):
        idx = data[:,0] == it
        c = data[idx,col_id]
        for i in range(len(c)):
            if i >=nc: continue
            c_all[it,i] = c[i]

    return c_all

def main():

    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("usage: python plot_disp.py max_order")
        exit(1)
    
    if len(sys.argv)  == 2:
        ncplot = int(sys.argv[1])
    else:
        ncplot = 100

    # load swd data
    data = np.loadtxt("out/swd.txt",skiprows=1)
    data_cps = np.loadtxt("out/swd.cps.txt",skiprows=1)

    # find unique T
    T = np.loadtxt("out/swd.txt",max_rows=1)
    T1 = np.loadtxt("out/swd.cps.txt",max_rows=1)
    nt = len(T)

    # plot phase
    nc = int(np.max(data[:,-1])) + 1
    ncplot = min(ncplot,nc)
    
    # plotting maps
    cmap = plt.get_cmap("viridis",ncplot)
    norm = mpl.colors.Normalize(vmin=0, vmax=ncplot-1)  # Normalize the color range
    fig,ax = plt.subplots(1,2,figsize=(15,6))

    #plt.scatter(1./data[:,0],data[:,1],s=10,color='k')
    c_all = fetch_data(T,data,nc,1)
    for i in range(ncplot):
        idx = c_all[:,i] > 0
        if np.sum(idx) != 0:
            ax[0].plot(1./T[idx],c_all[idx,i],color=cmap(i))

    c1_all = fetch_data(T1,data_cps,nc,1)
    for i in range(ncplot):
        idx = c1_all[:,i] > 0
        label = None
        if i == 0:
            label = 'T-H'
        if np.sum(idx) != 0:
            ax[0].scatter(1./T1[idx],c1_all[idx,i],s=10,color='k',label=label)

    ax[0].legend()
    ax[0].set_xlabel("Frequency,Hz")
    ax[0].set_ylabel("Phase Velocity, km/s")

    # plot group
    c_all = fetch_data(T,data,nc,2)

    # plot
    for i in range(ncplot):
        idx = c_all[:,i] > 0
        if np.sum(idx) != 0:
            ax[1].plot(1./T[idx],c_all[idx,i],color=cmap(i))

    c1_all = fetch_data(T1,data_cps,nc,2)

    for i in range(ncplot):
        idx = c1_all[:,i] > 0
        label = None 
        if i == 0:
            label = 'T-H'
        if np.sum(idx) != 0:
            ax[1].scatter(1./T1[idx],c1_all[idx,i],color='k',s=10,label=label)
    ax[1].legend()
    ax[1].set_xlabel("Frequency,Hz")
    ax[1].set_ylabel("Group Velocity, km/s")
    sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
    fig.colorbar(sm,ax=ax.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
    fig.savefig("eigenvalues.jpg",dpi=300)
    fig.savefig("eigenvalues.pdf",dpi=300)

main()
    