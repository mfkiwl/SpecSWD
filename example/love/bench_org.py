import numpy as np 
import h5py 
import sys 
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=13
mpl.rcParams['ytick.labelsize']=13
mpl.rcParams['axes.labelsize']=13
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20

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

def get_sls_modulus_factor(freq,Q):
    om = 2 * np.pi * freq

    w_sls,y_sls = get_Q_sls_model(Q)
    s = np.sum(1j * om * y_sls / (w_sls + 1j * om))

    return s + 1.

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

    bv1 = np.sqrt(L1 / r1)
    bv2 =  np.sqrt(L2 / r2)
    bh1 = np.sqrt(N1 / r1)
    bh2 = np.sqrt(N2 / r2)
    gamma2 = np.sqrt((1 - (c / bh2)**2))
    k = om / c
    Omega = k * H * gamma2 * bh2/bv2 * (
            (r1 / r2) * ((c**2 - bh1**2) / (bh2**2 - bh1**2)) +
            (L2 / L1) * ((bh2**2 - c**2) / (bh2**2 - bh1**2))
        )
    u = bh1**2 / c * ((c/bh1)**2 + Omega) / ( 1 + Omega)

    return u


def myfunc(c,r1,r2,bv1,bv2,bh1,bh2,H,om):
    L1 = r1 * bv1**2 
    L2 = r2 * bv2**2 

    k = om / c
    f = L2 * bv1 / (L1 * bv2) * np.sqrt(bh2**2 - c**2)
    f -= np.sqrt(c**2 - bh1**2) * np.tan(k * H / bv1 * np.sqrt(c**2 - bh1**2))
    return f

def myfunc_att(carr,r1,r2,bev1,bev2,beh1,beh2,Qv1,Qv2,Qh1,Qh2,H,om):
    freq = om / np.pi / 2
    L1 = r1 * bev1**2 
    L2 = r2 * bev2**2 
    N1 = r1 * beh1**2 
    N2 = r2 * beh2**2 
    L1 *= get_sls_modulus_factor(freq,Qv1)
    L2 *= get_sls_modulus_factor(freq,Qv2)
    N1 *= get_sls_modulus_factor(freq,Qh1)
    N2 *= get_sls_modulus_factor(freq,Qh2)

    bv1 = np.sqrt(L1 / r1)
    bv2 =  np.sqrt(L2 / r2)
    bh1 = np.sqrt(N1 / r1)
    bh2 = np.sqrt(N2 / r2)

    c = carr[0] + 1j * carr[1]
    k = om / c
    f = L2 * bv1 / (L1 * bv2) * np.sqrt(bh2**2 - c**2) 
    f -= np.sqrt(c**2 - bh1**2) * np.tan(k * H / bv1 * np.sqrt(c**2 - bh1**2))

    return np.abs(f)**2

def compute_QL(cc,b1,b2,rho1,rho2,Q1,Q2,om,H):
    # Compute η2 and η1
    beta2,beta1 = b2,b1 
    mu2 = beta2 **2 * rho2 
    mu1 = beta1**2 * rho1 
    nt = len(cc)
    QL = np.zeros(nt)

    w1,y1 = get_Q_sls_model(Q1)
    w2,y2 = get_Q_sls_model(Q2)

    for it in range(nt):
        c = cc[it].real
        k = om[it] / c 
        eta2 = np.sqrt(1 - (c / beta2) ** 2)
        eta1 = np.sqrt((c / beta1) ** 2 - 1)

        # Compute qL
        qL = (k * H / eta2) * (1 - ((mu2 * eta2) / (mu1 * eta1)) ** 2) + (mu2 / mu1) * (c / beta2) ** 2 * (1 / eta2 ** 2 + (beta2 / (beta1 * eta1)) ** 2)

        # Compute p1 and p2
        p1 = (mu2 / mu1) * ((c / beta2) ** 2 * (beta2 / (beta1 * eta1)) ** 2 - 2)
        p2 = (mu2 / mu1) * ((c / beta2) ** 2 / eta2 ** 2 + 2)

        # Compute A and B
        A = p1 / qL
        B = p2 / qL

        # compute Q 
        Q1_e = compute_q_sls_model(y1,w1,om[it],True)
        Q2_e = compute_q_sls_model(y2,w2,om[it],True)
        Qi = A / Q1_e + B / Q2_e 

        QL[it] = 1./ Qi

    return QL

def get_phase_veloc(r1,r2,bv1,bv2,bh1,bh2,H,om):
    nt = len(om)
    nroots = np.zeros((nt),dtype=int)
    out = np.zeros((nt,20))

    # first determine how many roots for each frequency
    for it in range(nt):
        nr = 0
        for i in range(50):
            wn = i * np.pi * bv1 / H / np.sqrt(1 - (bh1/bh2)**2)
            if wn > om[it]:
                break
            nr += 1
        nroots[it] = nr 
    
    # loop every period to find roots
    for it in range(nt):
        args = r1,r2,bv1,bv2,bh1,bh2,H,om[it]

        # bracket roots
        loc = np.zeros((nroots[it],2))
        for ir in range(nroots[it]):
            a = (ir * np.pi * bv1 / om[it] / H)**2
            b =  ((ir + 0.5) * np.pi * bv1 / om[it] / H) ** 2
            if b > 1:
                b = 1 - (bh1/bh2)**2 
            cmax = np.sqrt(bh1**2 / (1-b))
            cmin = np.sqrt(bh1**2 / (1 - a))
            if cmax > bh2:
                cmax = bh2 
            if cmin >= cmax:
                cmin = cmax * 0.999

            dc = (cmax - cmin) / 500
            c = cmin
            c1 = c + dc 
            f0 = myfunc(c,*args)
            f1 = myfunc(c1,*args)
            for i in range(1,500):
                # if it == 46 and ir == 1:
                #     print(i,f0,f1,c,c1,cmin,cmax)
                if np.sign(f0) != np.sign(f1):
                    #print(c,c1,f0,f1)
                    loc[ir,:] = [c,c1]
                    break 
                f0 = f1 * 1. 
                c = c1 * 1. 
                c1 = c1 + dc
                if c1 > cmax:
                    break
                f1 = myfunc(c1,*args)
        
        # interpolation
        for ir in range(nroots[it]):
            c0,c1 = loc[ir,:]
            cmid = 0.5 * (c0 + c1)
            f0 = myfunc(c0,*args)
            f1 = myfunc(c1,*args)
            fmid = myfunc(cmid,*args)

            # bisect
            for i in range(15):
                if np.sign(fmid) == np.sign(f1):
                    f1 = fmid * 1.
                    c1 = cmid 
                else:
                    f0 = fmid 
                    c0 = cmid 
                cmid = 0.5 * (c0 + c1)
                fmid = myfunc(cmid,*args)

            # quadratic interpolation of c
            x = np.array([f0,fmid,f1])
            y = np.array([c0,cmid,c1]) 
            L0 = (0-x[1]) * (0-x[2]) / (x[0] - x[1]) / (x[0] - x[2])
            L1 = (0-x[0]) * (0-x[2]) / (x[1] - x[0]) / (x[1] - x[2])
            L2 = (0-x[0]) * (0-x[1]) / (x[2] - x[0]) / (x[2] - x[1])
            out[it,ir] = y[0] * L0 + y[1] * L1 + y[2] * L2 
        
    return out 

def get_phase_veloc_att(r1,r2,bv1,bv2,bh1,bh2,Qv1,Qv2,Qh1,Qh2,H,om):
    from scipy.optimize import minimize
    nt = len(om)
    nroots = np.zeros((nt),dtype=int)

    # first determine how many roots for each frequency
    for it in range(nt):
        nr = 0
        for i in range(50):
            wn = i * np.pi * bv1 / H / np.sqrt(1 - (bh1/bh2)**2)
            if wn > om[it]:
                break
            nr += 1
        nroots[it] = nr 
    
    # out list
    idx = 0
    out = np.zeros((np.sum(nroots),4))
    for it in range(nt):
        for ir in range(nroots[it]):
            out[idx,0] = (2 * np.pi) / om[it]
            out[idx,3] = ir 
            idx += 1
    
    # loop every period to find roots
    idx = 0
    for it in range(nt):
        for ir in range(nroots[it]):
            # bracket roots
            a = (ir * np.pi * bv1 / om[it] / H)**2
            b =  ((ir + 0.5) * np.pi * bv1 / om[it] / H) ** 2
            if b > 1:
                b = 1 - (bh1/bh2)**2 
            cmax = np.sqrt(bh1**2 / (1-b))
            cmin = np.sqrt(bh1**2 / (1 - a))
            if cmax > bh2:
                cmax = bh2 
            if cmin >= cmax:
                cmin = cmax * 0.999

            # find roots by using L-BFGS-B
            init = np.array([0.5*(cmin + cmax),bh1*0.5/200])
            args = (r1,r2,bv1,bv2,bh1,bh2,Qv1,Qv2,Qh1,Qh2,H,om[it])
            result = minimize(myfunc_att,init,args=args,
                             method='L-BFGS-B',bounds=[(cmin,cmax),(bh1*0.5/1000,bh2*0.5/50)],
                                options={"maxiter":2000})

            # check if result are sucess
            if result.success:
                out[idx,1] = result.x[0]
                out[idx,2] = result.x[1]
            else:
                print("failed",2*np.pi/om[it],ir)
            idx += 1
        
    return out

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
    rho1,bv1,bh1,Qv1,Qh1 = model[0,1:]
    H,rho2,bv2,bh2,Qv2,Qh2 = model[-1,0:]
    H = 35

    # find phase velocity
    T0 = np.loadtxt("out/swd.txt",max_rows=1)
    nt = len(T0)
    if HAS_ATT ==0:
        #c_bench = get_phase_veloc(rho1,rho2,bv1,bv2,bh1,bh2,H,2*np.pi/T0)
        T1,cin = np.loadtxt("no_att.txt",unpack=True)
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
        # initial guess
        # T1,cin0,modeid = np.loadtxt("no_att.txt",unpack=True)
        # c_bench = cin0 * 0j
        # for it in range(len(T1)):
        #     om = np.pi * 2 / T1[it]
        #     init = np.array([cin0[it],cin0[it]*0.5/Qh2])
        #     args = (rho1,rho2,bv1,bv2,bh1,bh2,Qv1,Qv2,Qh1,Qh2,H,om)
        #     result = minimize(myfunc_att,init,args=args,
        #                      method='L-BFGS-B',bounds=[(bh1,bh2),(bh1*0.5/1000,bh2/0.5/50)],
        #                         options={"maxiter":2000})
        #     if result.success and result.fun < 0.01:
        #         c_bench[it] = result.x[0] + 1j * result.x[1]
        # T1,rec,imc,modeid = np.loadtxt("att.txt",unpack=True)
        # c_bench = rec + 1j * imc
        out = get_phase_veloc_att(rho1,rho2,bv1,bv2,bh1,bh2,Qv1,Qv2,Qh1,Qh2,H,2*np.pi/T0)
        T1 = out[:,0]; modeid = out[:,-1]
        rec = out[:,1]; imc = out[:,2]
        c_bench = rec + 1j * imc
        
        u_bench = c_bench * 0j
        for it in range(len(T1)):
            if c_bench[it].real > 0:
                u_bench[it] = get_group_att(c_bench[it],rho1,rho2,bv1,bv2,
                                            bh1,bh2,Qv1,Qv2,Qh1,Qh2,
                                            H,2*np.pi/T1[it]) 

        fig1 = plt.figure(1,figsize=(14,6))
        fig2 = plt.figure(2,figsize=(14,6))
        ax1 = fig1.add_subplot(121)
        ax2 = fig1.add_subplot(122)
        ax3 = fig2.add_subplot(121)
        ax4 = fig2.add_subplot(122)

        # plot phase/group
        cQ_a = c_bench.real * 0.
        uQ_a = c_bench.real * 0.
        idx = c_bench.real > 0
        cQ_a[idx] = c_bench[idx].real / c_bench[idx].imag /2
        uQ_a[idx] = u_bench[idx].real / u_bench[idx].imag / 2

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

            # filter
            idx = (c_bench.real > 0) & (modeid == imode)
            print(modeid[idx])

            # plot
            ax1.plot(1./T,u)
            
            ua = u * 0.j
            for it in range(len(T)):
                ua[it] = get_group_att(c[it]*(1 + 1j / cQ[it] * 0.5),rho1,rho2,bv1,bv2,
                                            bh1,bh2,Qv1,Qv2,Qh1,Qh2,
                                            H,2*np.pi/T[it])
            ax1.scatter(1./T1[idx],u_bench[idx].real,s=14,color='k',label=f'bench-{imode}')
            #ax1.scatter(1./T,ua.real,s=14,color='k',label=f'benchmark')
        
            ax2.plot(1./T,uQ)
            ax2.scatter(1./T1[idx],uQ_a[idx],s=14,color='k',label=f'bench-{imode}')
            #ax2.scatter(1./T,ua.real/ua.imag/2,s=14,color='k')
            # phase
            ax3.plot(1./T,c)
            ax3.scatter(1./T1[idx],c_bench[idx].real,s=14,color='k',label=f'bench-{imode}')
            ax4.plot(1./T,cQ)
            ax4.scatter(1./T1[idx],cQ_a[idx],s=14,color='k',label=f'bench-{imode}')

        
        ax1.legend()
        ax2.legend()
        ax4.legend()
        ax1.set_xlabel("Frequency,Hz")
        ax2.set_xlabel("Frequency,Hz")
        ax3.set_xlabel("Frequency,Hz")
        ax4.set_xlabel("Frequency,Hz")
        ax1.set_ylabel("Group Velocity, km/s")
        ax2.set_ylabel("Group Q")
        ax3.set_ylabel("Phase Velocity, km/s")
        ax4.set_ylabel("Phase Q")
        fig1.savefig("group_att.jpg")
        fig2.savefig("phase_att.jpg")

if __name__ == "__main__":
    main()