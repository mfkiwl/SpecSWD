import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import dual_annealing

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

def compute_q_model(Q0,om,alpha = 0.,om_ref=1.):
    """
    User defined power law Q model: Q = Q0 (om/om_ref)**alpha

    Parameters 
    ============
    Q0: float
        target constant Q
    om: np.ndarray,float
        angular frequency used
    alpha: float
        power factor
    om_ref: float
        reference angular frequency

    Returns
    ============
    q: np.ndarray
        target q model
    """
    q = Q0 * (om / om_ref) ** alpha

    return q

def misfit_func(x,om,q_target,weights = 1):
    NSLS = len(x) // 2
    y_sls = x[:NSLS] * 1. 
    w_sls = x[NSLS:] * 1. 
    q_sls = compute_q_sls_model(y_sls,w_sls,om,False)

    return np.sum(np.log(q_sls/q_target)**2 * weights) * 0.5 

def y_corrector(y_sls,Q_ref,Q):
    # note y_sls is fitted by Q = 20.
    y = y_sls / Q * Q_ref 
    dy = y * 0. 

    # corrector
    dy[0] = 1 + 0.5 * y[0]
    for i in range(1,len(y)):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]
    
    # corrected y_sls
    return dy * y 

def main():
    # set your parameters here 
    ##### target Q power law model Q(w) = Q (w / w_ref)^alpha
    alpha = 0.   # 
    om_ref = 1.
    Q = 200.

    # SLS q model
    NSLS = 5
    min_freq = 0.5 * 1.0e-2
    max_freq = 2 * 100
    nfsamp = 100  # no. of frequency points in om to fit Q model
    weight_by_freq = True

    # optimization method
    OPT_METHOD = 'dual' # 1 for simulated annealing and 2 for dual anneling
    rand_seed = 10

    #### STOP HERE  #####

    # reference Q to fitting
    Q_ref = 1.

    # check OPT_METHOD
    assert(OPT_METHOD in ['dual','simulated'])
    
    # get om vector
    om = np.logspace(np.log10(min_freq),np.log10(max_freq),nfsamp) * np.pi * 2 

    # initial y_sls and w_sls
    w_sls = np.logspace(np.log10(min_freq),np.log10(max_freq),NSLS) * np.pi * 2 
    y_sls = 1.5 * np.ones((NSLS)) / Q_ref
    x = np.append(y_sls,w_sls)

    # weight factor
    weights = om * 0 + 1.
    if weight_by_freq:
        weights = om / np.sum(om)

    # target Q model
    q_target = compute_q_model(Q_ref,om,alpha,om_ref)

    if OPT_METHOD == 'simulated':
        # set random seed
        np.random.seed(rand_seed)

        # now call a simulated annealing optimizer
        Tw = 0.1 
        Ty = 0.1
        chi = misfit_func(x,om,q_target,weights)
        
        x_test = x * 1.
        for it in range(100000):
            for i in range(NSLS):
                x_test[i] = x[i] * (1.0 + (0.5 - np.random.rand()) * Ty)
                x_test[i+NSLS] = x[i+NSLS] * (1.0 + (0.5 - np.random.rand()) * Tw)
            
            # compute Q 
            chi_test = misfit_func(x_test,om,q_target,weights)
            Ty *= 0.995
            Tw *= 0.995
            
            # check if accept this parameter
            if chi_test < chi:
                x[:] = x_test[:] * 1.
                chi = chi_test * 1.

        # return taget model
        print('final misift = ',chi)
        y_opt = x[:NSLS]
        w_opt = x[NSLS:]
    else : # dual annealing
        # set search boundary for w_sls and y_sls
        bounds = []
        for i in range(NSLS):
            y_min = y_sls[i] * 0.8
            y_max = y_sls[i] * 1.5
            bounds.append((y_min,y_max))
        for i in range(NSLS):
            w_min = w_sls[i] * 0.8
            w_max = w_sls[i] * 1.5
            bounds.append((w_min,w_max))
        res = dual_annealing(misfit_func,bounds=bounds,args=(om,q_target,weights),  \
                             x0=x,maxiter=1000,seed=10)
        print(res)
        y_opt = res.x[:NSLS]
        w_opt = res.x[NSLS:]
        pass

    # # save optimized y_sls and w_sls
    # fio = open("include/atteunuation_table.hpp","w")
    # fio.write("#include <array>\n")
    # fio.write("const int NSLS = 5;\n")
    # y_opt_str = ",".join([str(y_opt[i]) for i in range(len(y_opt))])
    # w_opt_str = ",".join([str(w_opt[i]) for i in range(len(y_opt))])
    # fio.write("const std::array<double,NSLS> y_sls_ref = {%s};\n" % (y_opt_str))
    # fio.write("const std::array<double,NSLS> w_sls_ref = {%s};\n" % (w_opt_str))
    # fio.close()

    # plot Q model
    om_plot = np.logspace(np.log10(min_freq*0.01),np.log10(max_freq*100),500) * np.pi * 2 
    y_opt = y_corrector(y_opt,Q_ref,Q)
    Q_opt = compute_q_sls_model(y_opt,w_opt,om_plot,True)
    Q_pow = compute_q_model(Q,om_plot,alpha,om_ref)
    print('optimized y_sls = ',y_opt)
    print('optimized w_sls = ',w_opt)

    plt.semilogx(om_plot/(2*np.pi),1./Q_opt,label='1/Q_opt')
    plt.semilogx(om_plot/(2*np.pi),1./Q_pow,label='1/Q_target')
    plt.xlabel("Frequency,Hz")
    plt.ylabel("$Q^{-1}$")
    plt.axvline(min_freq)
    plt.axvline(max_freq)
    plt.legend()
    plt.savefig("Qmodel.jpg")

if __name__ == "__main__":
    main()
