import numpy as np 
import h5py 
import sys 
import matplotlib.pyplot as plt 
from numba import jit 

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

@jit(nopython=True)
def get_Q_sls_model(Q):

    y_sls_ref = np.array([1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129])
    w_sls_ref = np.array([4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03])
    dy = y_sls_ref * 0 
    y = y_sls_ref * 0
    NSLS = len(y)

    for i in range(NSLS):
        y[i] = y_sls_ref[i] / Q 
    dy[0] = 1. + 0.5 * y[0]

    for i in range(1,NSLS):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]

    #// copy to y_sls/w_sls
    w_sls = w_sls_ref * 1. 
    y_sls = dy * y 

    return w_sls,y_sls 

@jit(nopython=True)
def compute_q_sls_model(y_sls,w_sls,om,exact=False):
    Q_ls = 1. 
    nsls = len(y_sls)
    if exact:
        for p in range(nsls):
            Q_ls += y_sls[p] * om**2 / (om**2 + w_sls[p]**2)

    # denom
    Q_demon = 0.
    for p in range(nsls):
        Q_demon += y_sls[p] * om * w_sls[p] / (om**2 + w_sls[p]**2)
    
    return Q_ls / Q_demon

@jit(nopython=True)
def get_sls_modulus_factor(freq,Q):
    om = 2 * np.pi * freq

    w_sls,y_sls = get_Q_sls_model(Q)
    s = np.sum(1j * om * y_sls / (w_sls + 1j * om))

    return s + 1.

@jit(nopython=True)
def get_sls_Q_deriv(freq,Q):
    y_sls_ref = np.array([1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129])
    w_sls_ref = np.array([4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03])
    dy = y_sls_ref * 0 
    y = y_sls_ref  / Q 

    # corrector
    NSLS = len(y_sls_ref)
    dy[0] = 1. + 0.5 * y[0]
    for i in range(1,NSLS):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]
    dd_dqi = dy * 0
    dd_dqi[0] = 0.5 * y_sls_ref[0]
    for i in range(1,NSLS):
        dd_dqi[i] = dd_dqi[i-1] + (dy[i-1] - 0.5) * y_sls_ref[i-1] + dd_dqi[i-1] * y[i-1] +  0.5 * y_sls_ref[i]

    dd_dqi = dd_dqi * y + dy * y_sls_ref
    om = 2 * np.pi * freq
    dsdqi = np.sum(1j * om * dd_dqi /(w_sls_ref + 1j * om))
    s = np.sum(1j * om * y * dy / (w_sls_ref + 1j * om))

    return s + 1., dsdqi


def get_group(c,r1,r2,bv1,bv2,bh1,bh2,H,om):
    L1 = r1 * bv1**2 
    L2 = r2 * bv2**2 
    gamma2 = np.sqrt((1 - (c / bh2)**2))
    k = om / c
    Omega = k * H * gamma2 * bh2/bv2 * (
            (r1 / r2) * ((c**2 - bh1**2) / (bh2**2 - bh1**2)) +
            (L2 / L1) * ((bh2**2 - c**2) / (bh2**2 - bh1**2))
        )
    u = bh1**2 / c * ((c/bh1)**2 + Omega) / ( 1 + Omega)

    return u


def get_group_att(c,r1,r2,bev1,bev2,beh1,beh2,Qv1,Qv2,Qh1,Qh2,H,om):
    freq = om / (2 * np.pi)

    L1 = r1 * bev1**2 
    L2 = r2 * bev2**2 
    N1 = r1 * beh1**2 
    N2 = r2 * beh2**2 
    L1 *= get_sls_modulus_factor(freq,Qv1)
    L2 *= get_sls_modulus_factor(freq,Qv2)
    N1 *= get_sls_modulus_factor(freq,Qh1)
    N2 *= get_sls_modulus_factor(freq,Qh2)

    # bv1 = np.sqrt(L1 / r1)
    # bv2 =  np.sqrt(L2 / r2)
    # bh1 = np.sqrt(N1 / r1)
    # bh2 = np.sqrt(N2 / r2)
    # gamma2 = np.sqrt((1 - (c / bh2)**2))
    # k = om / c
    # Omega = k * H * gamma2 * bh2/bv2 * (
    #         (r1 / r2) * ((c**2 - bh1**2) / (bh2**2 - bh1**2)) +
    #         (L2 / L1) * ((bh2**2 - c**2) / (bh2**2 - bh1**2))
    #     )
    # u = bh1**2 / c * ((c/bh1)**2 + Omega) / ( 1 + Omega)

    gamma2 = np.sqrt((1 - r2 * c**2/N2) * N2/L2)
    Omega = om/c * H * gamma2 * (
                ((r1 * c**2 - N1) / (N2 - r2 / r1 * N1)) +
                (L2 / L1) * ((N2 - r2*c**2) / (N2 - r2/r1 * N1))
            )
    u = N1 / (c * r1) * ( c**2 * r1 / N1 + Omega) / ( 1 + Omega)

    return u

def get_cQ_kl(dcdm,cc):
    cl = np.real(cc)
    Qi = 2. * np.imag(cc) / cl 
    dcLdm = np.real(dcdm)
    dQiLdm = (np.imag(dcdm) * 2. - Qi * dcLdm) / cl 

    return dcLdm,dQiLdm 

def transform_kernel(dcc_dcL1,dcc_dcL2,dcc_dcN1,dcc_dcN2,dcc_dr1,dcc_dr2,
                     sl1,sl2,sn1,sn2,dsdqil1,dsdqil2,dsdqin1,dsdqin2,
                     L1_r,L2_r,N1_r,N2_r,rho1,rho2,c):
    # convert dm to real parameters
    dcc_dL1,dcc_dQil1 = dcc_dcL1 * sl1, dcc_dcL1 * L1_r * dsdqil1
    dcc_dL2,dcc_dQil2 = dcc_dcL2 * sl2, dcc_dcL2 * L2_r * dsdqil2
    dcc_dN1,dcc_dQin1 = dcc_dcN1 * sn1, dcc_dcN1 * N1_r * dsdqin1
    dcc_dN2,dcc_dQin2 = dcc_dcN2 * sn2, dcc_dcN2 * N2_r * dsdqin2

    # dictionary to save all derivatives
    D = np.zeros((2,5,2)) # (dcordq,5params, 2 layer)
    D[:,0,0] = get_cQ_kl(dcc_dL1,c)
    D[:,0,1] = get_cQ_kl(dcc_dL2,c)
    D[:,1,0] = get_cQ_kl(dcc_dN1,c)
    D[:,1,1] = get_cQ_kl(dcc_dN2,c)
    D[:,2,0] = get_cQ_kl(dcc_dQin1,c)
    D[:,2,1] = get_cQ_kl(dcc_dQin2,c)
    D[:,3,0] = get_cQ_kl(dcc_dQil1,c)
    D[:,3,1] = get_cQ_kl(dcc_dQil2,c)
    D[:,4,0] = get_cQ_kl(dcc_dr1,c)
    D[:,4,1] = get_cQ_kl(dcc_dr2,c)

    # transform to vsv,vsh kernels
    bv1 = np.sqrt(L1_r / rho1); bv2 = np.sqrt(L2_r / rho2)
    bh1 = np.sqrt(N1_r / rho1); bh2 = np.sqrt(N2_r / rho2)

    # kernels
    vsv_kl = np.zeros((2,2))
    vsh_kl = np.zeros((2,2))
    rho_kl = np.zeros((2,2))
    vsv_kl[:,0] = D[:,0,0] * 2. * rho1 * bv1 
    vsv_kl[:,1] = D[:,0,1] * 2. * rho2 * bv2 
    vsh_kl[:,0] = D[:,1,0] * 2. * rho1 * bh1 
    vsh_kl[:,1] = D[:,1,1] * 2. * rho2 * bh2 
    rho_kl[:,0] = D[:,4,0] + bv1**2 * D[:,0,0] + bh1**2 * D[:,1,0]
    rho_kl[:,1] = D[:,4,1] + bv2**2 * D[:,0,1] + bh2**2 * D[:,1,1]

    # copy back
    D[:,0,:] = vsh_kl[:,:]
    D[:,1,:] = vsv_kl[:,:]
    D[:,4,:] = rho_kl[:,:]

    return D


def love_func(c,r1,r2,bev1,bev2,beh1,beh2,Qv1,Qv2,Qh1,Qh2,H,om):
    freq = om / np.pi / 2
    L1_r = r1 * bev1**2 
    L2_r = r2 * bev2**2 
    N1_r = r1 * beh1**2 
    N2_r = r2 * beh2**2 

    # factor and deriv
    sn1,dsdqin1 = get_sls_Q_deriv(freq,Qh1)
    sn2,dsdqin2 = get_sls_Q_deriv(freq,Qh2)
    sl1,dsdqil1 = get_sls_Q_deriv(freq,Qv1)
    sl2,dsdqil2 = get_sls_Q_deriv(freq,Qv2)

    # complex modulus
    L1 = L1_r * sl1
    L2 = L2_r * sl2
    N1 = N1_r * sn1
    N2 = N2_r * sn2

    bv1 = np.sqrt(L1 / r1)
    bv2 = np.sqrt(L2 / r2)
    bh1 = np.sqrt(N1 / r1)
    bh2 = np.sqrt(N2 / r2)

    k = om / c
    t1 = np.sqrt(c**2 - bh1**2)
    t2 = np.sqrt(bh2**2 - c**2) 
    temp = np.tan(k * H / bv1 * t1)
    f = L2 / (L1) * t2 / bv2
    f -= t1 / bv1 * temp 
    #f = L2/L1 * np.sqrt((N2/L2)**2 - (r2 * c/L2)**2) - np.sqrt((c *r1/L1)**2 - (N1/L1)**2) * np.tan((om * H)/c *np.sqrt((c * r1/ L1)**2 - (N1/L1)**2))

    # derivative df/dc
    fd = - L2 * bv1 / (L1 * bv2) * c / t2
    fd += -c / t1 * temp 
    fd += -t1 * (1 + temp**2) * (- om * H / c**2 / bv1 * t1 + om * H / bv1 / t1 )

    # d(\tilde{c})/dL1/L2/N1/N2/rho
    dcc_dcL2=(-L2*(-1/2*N2/L2**2 + (1/2)*c**2*r2/L2**2)/(L1*np.sqrt(N2/L2 - c**2*r2/L2)) - np.sqrt(N2/L2 - c**2*r2/L2)/L1)/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
    dcc_dcL1=(H*om*((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/c + ((1/2)*N1/L1**2 - 1/2*c**2*r1/L1**2)*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/np.sqrt(-N1/L1 + c**2*r1/L1) + L2*np.sqrt(N2/L2 - c**2*r2/L2)/L1**2)/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
    dcc_dcN2=-(1/2)/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
    dcc_dcN1=(-1/2*H*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/(L1*c) - 1/2*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))
    dcc_dr2=(1/2)*c**2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)*(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2))))
    dcc_dr1=((1/2)*H*c*om*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1)/L1 + (1/2)*c**2*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))/(-np.sqrt(-N1/L1 + c**2*r1/L1)*(-H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c**2 + H*om*r1/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)))*(np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)**2 + 1) - c*r1*np.tan(H*om*np.sqrt(-N1/L1 + c**2*r1/L1)/c)/(L1*np.sqrt(-N1/L1 + c**2*r1/L1)) - c*r2/(L1*np.sqrt(N2/L2 - c**2*r2/L2)))

    D = transform_kernel(dcc_dcL1,dcc_dcL2,dcc_dcN1,dcc_dcN2,dcc_dr1,dcc_dr2,
                     sl1,sl2,sn1,sn2,dsdqil1,dsdqil2,dsdqin1,dsdqin2,
                     L1_r,L2_r,N1_r,N2_r,r1,r2,c)
    
    return f,fd,D

def get_deriv_2layer_h5(fio:h5py,gname:str):
    kl_name = ['vsh','vsv','Qvsh','Qvsv','rho']
    D = np.zeros((2,5,2))

    for iker in range(len(kl_name)):
        a = fio[f"{gname}/C_{kl_name[iker]}"][:]
        q = fio[f"{gname}/Q_{kl_name[iker]}"][:]
        a1 = a[0] + a[1]
        a2 = a[2]
        q1 = q[0] + q[1]
        q2 = q[2]

        D[0,iker,:] = [a1,a2]
        D[1,iker,:] = [q1,q2]

    return D

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("usage: python compute_group max_order")
        exit(1)

    if len(sys.argv)  == 2:
        ncplot = int(sys.argv[1])
    else:
        ncplot = 5

    # open h5file
    fio = h5py.File("out/kernels.h5","r")

    # load velocity model
    HAS_ATT = np.loadtxt("model.txt",max_rows=1,dtype=int)[1]
    model = np.loadtxt("model.txt",skiprows=1)
    rho1,bh1,bv1,Qh1,Qv1 = model[0,1:]
    H,rho2,bh2,bv2,Qh2,Qv2 = model[-1,0:]
    print(H,rho2,bv2,bh2,Qv2,Qh2)

    # find phase velocity
    T0 = np.loadtxt("out/swd.txt",max_rows=1)
    nt = len(T0)
    if HAS_ATT ==0:
        #c_bench = get_phase_veloc(rho1,rho2,bv1,bv2,bh1,bh2,H,2*np.pi/T0)
        T1,cin,mode_id = np.loadtxt("no_att.txt",unpack=True)
        c_bench = np.zeros((len(T0),20))
        for it in range(nt):
            t = T0[it]
            idx = T1 == t 
            c_bench[it,:np.sum(idx)] = cin[idx]

        # group velocity
        u_bench = c_bench * 0.
        for it in range(nt):
            idx = c_bench[it,:] > 0
            u_bench[it,idx] = get_group(c_bench[it,idx],rho1,rho2,bv1,bv2,
                                        bh1,bh2,H,2*np.pi/T0[it])
        
        # figures
        fig1 = plt.figure(1,figsize=(14,6))
        fig2 = plt.figure(2,figsize=(8,6))
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)

        # loop every mode
        for imode in range(ncplot):
            gname = f"swd/mode{imode}/"
            if gname not in fio.keys():
                continue

            # load data
            T = fio[f"{gname}/T"][:]
            c = fio[f"{gname}/c"][:]
            u = fio[f"{gname}/u"][:]

            # plot phase
            idx = c_bench[:,imode] > 0
            ax1.plot(1./T,c)
            ax1.scatter(1./T0[idx],c_bench[idx,imode],s=10,color='k')

            # group 
            ax2.plot(1./T,u)
            ax2.scatter(1./T0[idx],u_bench[idx,imode],s=10,color='k')
            if imode == ncplot - 1:
                ax1.scatter(1./T0[idx],c_bench[idx,imode],s=10,color='k',label='benchmark')
                ax2.scatter(1./T0[idx],u_bench[idx,imode],s=10,color='k',label='benchmark')

        # labels
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel("Frequency,Hz")
        ax2.set_xlabel("Frequency,Hz")
        ax1.set_ylabel("Phase Velocity, km/s")
        ax2.set_ylabel("Group Velocity, km/s")
        fig1.savefig("phase.jpg")
        fig2.savefig("group.jpg")
    else:

        fig1,ax1 = plt.subplots(2,2,figsize=(14,12))

        # axese for figure 2
        fig2,ax2 = plt.subplots(2,2,figsize=(15,12))

        # axese for figure 3
        fig3,ax3 = plt.subplots(2,2,figsize=(15,12))

        # colormap
        #cmap = mpl.cm.get_cmap("jet",ncplot)
        cmap = plt.get_cmap("viridis",ncplot)
        norm = mpl.colors.Normalize(vmin=0, vmax=ncplot-1)  # Normalize the color range

        # loop every mode 
        for imode in range(ncplot):
            gname = f"swd/mode{imode}/"
            if gname not in fio.keys():
                continue

            # load data
            T = fio[f"{gname}/T"][:]
            c = fio[f"{gname}/c"][:]
            cQ = fio[f"{gname}/cQ"][:]
            u = fio[f"{gname}/u"][:]
            uQ = fio[f"{gname}/uQ"][:] 

            # compute relative error
            c_cmplx = c * (1 + 1j / (2 * cQ))
            dc = c * 0.0
            ua = c_cmplx * 1.0j
            phase_deriv = np.zeros((2,5,2,len(T)))
            phase_sem = phase_deriv * 0.
            for it in range(len(T)):
                f0,dfdc,D = love_func(c_cmplx[it],rho1,rho2,bv1,bv2,bh1,
                                    bh2,Qv1,Qv2,Qh1,Qh2,H,2*np.pi/T[it])
                ua[it] = get_group_att(c_cmplx[it],rho1,rho2,bv1,bv2,
                                        bh1,bh2,Qv1,Qv2,Qh1,Qh2,
                                        H,2*np.pi/T[it])
                dc[it] = 100 * abs(f0 / dfdc / c_cmplx[it])
                phase_deriv[...,it] = D * 1.

                # compute phase_sem
                idx = np.where(abs(T[it]-T0) < 1.0e-4)[0][0]
                phase_sem[...,it] = get_deriv_2layer_h5(fio,f"kernels/{idx}/mode{imode}")


            # plot
            label3 = "SEM"
            label4 = "Analytical"
            if imode != ncplot -1:
                label3 = None 
                label4 = None

            # phase velocity
            ax1[0,0].plot(1./T,c,color=cmap(imode))
            ax1[0,1].scatter(1./T,dc,s=14,color=cmap(imode))
            ax1[1,0].plot(1./T,u,color=cmap(imode))
            ax1[1,0].scatter(1./T,ua.real,color='k',s=10,label=label4)
            ax1[1,1].plot(1./T,uQ,color=cmap(imode))
            ax1[1,1].scatter(1./T,ua.real/ua.imag *0.5,color='k',s=10,label=label4)

            # derivatives
            for i in range(2):
                for j in range(2):
                    a1 = phase_sem[0,i,j,:]
                    a2 = phase_deriv[0,i,j,:]
                    ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    ax2[i,j].scatter(1./T,a2,color='k',s=10,label=label4)
                    # ax2[i,j].plot(1./T,a1,color=cmap(imode))
                    # ax2[i,j].scatter(1./T,a2,color='k',s=10,label=label4)

                    a1 = phase_sem[0,i+2,j,:]
                    a2 = phase_deriv[0,i+2,j,:]
                    ax3[i,j].plot(1./T,a1,color=cmap(imode))
                    ax3[i,j].scatter(1./T,a2,color='k',s=10,label=label4)
                    # ax3[i,j].plot(1./T,a1,color=cmap(imode))
                    # ax3[i,j].scatter(1./T,a2,color='k',s=10,label=label4)

        
        ax1[0,0].set_title("(a)",loc='left')
        ax1[0,0].set_ylabel("Phase Velocity, km/s")

        ax1[0,1].set_title("(b)",loc='left')
        ax1[0,1].set_yscale("log")
        ax1[0,1].set_ylabel(r"Relative Error % $|\frac{dc}{c}|$")
        
        ax1[1,1].set_xlabel("Frequency,Hz")
        ax1[1,1].set_title("(d)",loc='left')
        ax1[1,1].set_ylabel("Group Q")
        ax1[1,1].legend()
        
        ax1[1,0].set_xlabel("Frequency,Hz")
        ax1[1,0].set_ylabel("Group Velocity, km/s")
        ax1[1,0].set_title("(c)",loc='left')
        ax1[1,0].legend()

        # 
        ax2[0,0].set_ylabel(r"$dc/d\beta_{h1}$")
        ax2[0,1].set_ylabel(r"$dc/d\beta_{h2}$")
        ax2[1,0].set_ylabel(r"$dc/d\beta_{v1}$")
        ax2[1,1].set_ylabel(r"$dc/d\beta_{v2}$")
        for i in range(2): ax2[1,i].set_xlabel("Frquency,Hz")
    
        ax3[0,0].set_ylabel(r"$dc/dQ^{-1}_{h1}$")
        ax3[0,1].set_ylabel(r"$dc/dQ^{-1}_{h2}$")
        ax3[1,0].set_ylabel(r"$dc/dQ^{-1}_{v1}$")
        ax3[1,1].set_ylabel(r"$dc/dQ^{-1}_{v2}$")
        for i in range(2): ax3[1,i].set_xlabel("Frquency,Hz")

        for i in range(2):
            for j in range(2):
                ax2[i,j].legend()
                ax3[i,j].legend()

        #fig2.savefig("group_att.jpg",dpi=300)
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=norm)
        #fig1.colorbar(sm,ax=ax1[0,1],label='order')
        fig1.colorbar(sm,ax=ax1.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        fig1.savefig("love_wave_eigenvalues.jpg",dpi=300)

        # save figure2
        fig2.colorbar(sm,ax=ax2.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        fig2.savefig("love_wave_phase_deriv_veloc.jpg",dpi=300)

        # save figure2
        fig3.colorbar(sm,ax=ax3.ravel().tolist(),label='order',location='bottom',pad=0.075,shrink=0.4,format='%d')
        fig3.savefig("love_wave_phase_deriv_Q.jpg",dpi=300)

if __name__ == "__main__":
    main()