import numpy as np 
import h5py 
import sys 
import matplotlib.pyplot as plt 

def get_Q_sls_model(Q):

    y_sls_ref = np.array([0.0096988,  0.00832481, 0.0088744,  0.00735887, 0.00866749])
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

def get_sls_modulus_factor(freq,Q):
    om = 2 * np.pi * freq

    w_sls,y_sls = get_Q_sls_model(Q)
    s = np.sum(1j * om * y_sls / (w_sls + 1j * om))

    return s + 1.

def compute_group_velocity(cc, b1, b2,Q1,Q2, rho1, rho2,om, H):
    # Compute Omega
    nt = len(om)
    U = np.zeros((nt),dtype=complex)

    for it in range(nt):
        mu1 = rho1 * b1**2 
        mu2 = rho2 * b2**2 
        freq = om[it] / (2 * np.pi)
        mu1 *= get_sls_modulus_factor(freq,Q1)
        mu2 *= get_sls_modulus_factor(freq,Q2)
        beta1 = np.sqrt(mu1/rho1)
        beta2 = np.sqrt(mu2/rho2)
        c = cc[it]

        gamma2 = np.sqrt((1 - (c / beta2)**2))
        k = om[it] / c

        Omega = k * H * gamma2 * (
            (rho1 / rho2) * ((c**2 - beta1**2) / (beta2**2 - beta1**2)) +
            (mu2 / mu1) * ((beta2**2 - c**2) / (beta2**2 - beta1**2))
        )
        # Compute group velocity U
        U[it] = (beta1**2 / c) * ((c**2 / beta1**2) + Omega) / (1 + Omega)
    
    return U

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 1:
        print("usage: python compute_group max_order")
        exit(1)

    if len(sys.argv)  == 2:
        ncplot = int(sys.argv[1])
    else:
        ncplot = 5

    # load velocity model
    model = np.loadtxt("model.txt",skiprows=1)
    rho1,beta1,Q1 = model[0,[1,2,4]]
    H,rho2,beta2,Q2 = model[-1,[0,1,2,4]]
    print(rho1,beta1,rho2,beta2,H,Q1,Q2)

    # open h5file
    fio = h5py.File("out/kernels.h5","r")

    fig1 = plt.figure(1,figsize=(14,6))
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
        cQ = fio[f"{gname}/cQ"][:]

        # complex c and estimated group 
        cc = c * (1. + 1j / (2. * cQ))
        uu = compute_group_velocity(cc,beta1,beta2,Q1,Q2,rho1,rho2,2*np.pi/T,H)

        # estimated uL and uQ
        # u = uL ( 1 + i / (2 * uQ))
        u_a = uu.real
        uQ_a =  u_a / uu.imag / 2.

        # group velocity
        u = fio[f"{gname}/u"][:]
        uQ = fio[f"{gname}/uQ"][:]
        if imode == 1:
            print(u)

        # plot
        ax1.plot(1./T,u)
        ax1.plot(1./T,u_a,ls='--',color='k',label=f'bench-{imode}')
    
        ax2.plot(1./T,uQ)
        ax2.plot(1./T,uQ_a,ls='--',color='k',label=f'bench-{imode}')
    
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("Frequency,Hz")
    ax2.set_xlabel("Frequency,Hz")
    ax1.set_ylabel("Group Velocity, km/s")
    ax2.set_ylabel("Group Q")
    fig1.savefig("group_att.jpg")

if __name__ == "__main__":
    main()