import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import dual_annealing

def compute_q_sls_model(y_sls,w_sls,om,exact=False):
    """
    compute Q model by a given SLS coefficients:
    Q^{-1}(om) =  D(om) / N(om)
    where N(om) = \sum_p y[p] * om^2 / (om^2 + w[p]^2) and 
    D(om) = y[p] * om^2 / (om^2 + w[p]^2)

    Parameters
    ----------
    y_sls: np.ndarray
        y coefficients, shape(NSLS)
    w_sls: np.ndarray
        w coefficients, shape(NSLS)
    om: np.ndarray
        current angular frequency
    exact: bool
        if True, N(om) and D(om) are all computed. Else set N =1.

    """
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

def _compute_q_model(Q0:float,om,alpha = 0.,om_ref=1.):
    """
    User defined power law Q model: Q = Q0 (om/om_ref)**alpha

    Parameters 
    ----------
    Q0: float
        target constant Q
    om: np.ndarray
        angular frequency used
    alpha: float
        power factor
    om_ref: float
        reference angular frequency

    Returns
    --------
    q: np.ndarray
        target q model
    """
    q = Q0 * (om / om_ref) ** alpha

    return q

def _misfit_func(x,om,q_target,weights = 1):
    """
    misfit function of targeted and synthetic Q model  = 1/2 \int d\om log[(Q_sls / Q_target)]**2 * weights 

    Parameter
    ------------
    x: np.ndarray
        sls coefs [y,w] with shape(NSLS * 2)
    om: np.ndarray
        angular frequency used
    q_target: np.ndarray
        target Q model at om 
    weights: np.darray or float
        weights 

    Returns
    -----------
    f: float
        misfit = 1/2 \int d\om log[(Q_sls / Q_target)]**2 * weights 
    """
    NSLS = len(x) // 2
    y_sls = x[:NSLS] * 1. 
    w_sls = x[NSLS:] * 1. 
    q_sls = compute_q_sls_model(y_sls,w_sls,om,False)

    return np.sum(np.log(q_sls/q_target)**2 * weights) * 0.5 

def y_corrector(y_sls,Q_ref:float,Q:float):
    """
    correction from reference y to target y 

    Parameters
    --------------
    y_sls: np.ndarray
        y factor for refernce sls model
    Q_ref: float
        refernce Q
    Q: float
        target Q
    """
    # note y_sls is fitted
    y = y_sls / Q * Q_ref 
    dy = y * 0. 

    # corrector
    dy[0] = 1 + 0.5 * y[0]
    for i in range(1,len(y)):
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i]
    
    # corrected y_sls
    return dy * y 

def compute_sls_coefs(freqmin:float,freqmax:float, NSLS=5, nfsamp=100,
                      weight_by_freq = True,method='dual',
                      Q_ref = 1.,random_seed=10):
    """
    compute SLS coefs for reference model, by nonlinear inversion

    Parameter
    ------------
    freqmin: float
        minimum frequency
    freqmax: float
        maximum frequency
    NSLS: int
        no. of SLS solids, default = 5
    nfsamp: int
        no. of sampling points in [freqmin,freqmax], default = 100
    weight_by_freq: bool
        if apply frequency weighting in misfit function ,default = True
    method: str
        one of ['dual','simulated'], dual or regular simulated annealling
    Q_ref: float
        reference Q value, default is 1. Donnot change it unless you know what you're doning
    random_seed: int
        random seed, default 10
    
    Returns
    ------------
    y: np.ndarray
        SLS factor
    w: np.ndarray
        SLS angular frequency
    """

    # get min/max freq, with a bigger range
    min_freq = 0.5 * freqmin 
    max_freq = 2 * freqmax 

    # check OPT_METHOD
    assert(method in ['dual','simulated'])
    
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
    q_target = _compute_q_model(Q_ref,om,0.,1.)

    if method == 'simulated':
        # set random seed
        np.random.seed(random_seed)

        # now call a simulated annealing optimizer
        Tw = 0.1 
        Ty = 0.1
        chi = _misfit_func(x,om,q_target,weights)
        
        x_test = x * 1.
        for it in range(100000):
            for i in range(NSLS):
                x_test[i] = x[i] * (1.0 + (0.5 - np.random.rand()) * Ty)
                x_test[i+NSLS] = x[i+NSLS] * (1.0 + (0.5 - np.random.rand()) * Tw)
            
            # compute Q 
            chi_test = _misfit_func(x_test,om,q_target,weights)
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
        res = dual_annealing(_misfit_func,bounds=bounds,args=(om,q_target,weights),  \
                             x0=x,maxiter=1000,seed=random_seed)
        y_opt = res.x[:NSLS]
        w_opt = res.x[NSLS:]

    return y_opt,w_opt

def test():
    # set your parameters here 
    ##### target Q power law model Q(w) = Q (w / w_ref)^alpha
    alpha = 0.   # 
    om_ref = 1.
    Q = 20

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
    q_target = _compute_q_model(Q_ref,om,alpha,om_ref)

    if OPT_METHOD == 'simulated':
        # set random seed
        np.random.seed(rand_seed)

        # now call a simulated annealing optimizer
        Tw = 0.1 
        Ty = 0.1
        chi = _misfit_func(x,om,q_target,weights)
        
        x_test = x * 1.
        for it in range(100000):
            for i in range(NSLS):
                x_test[i] = x[i] * (1.0 + (0.5 - np.random.rand()) * Ty)
                x_test[i+NSLS] = x[i+NSLS] * (1.0 + (0.5 - np.random.rand()) * Tw)
            
            # compute Q 
            chi_test = _misfit_func(x_test,om,q_target,weights)
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
        res = dual_annealing(_misfit_func,bounds=bounds,args=(om,q_target,weights),  \
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
    print('optimized y_sls_ref = ',y_opt)
    print('optimized w_sls_ref = ',w_opt)
    y_opt = y_corrector(y_opt,Q_ref,Q)
    Q_opt = compute_q_sls_model(y_opt,w_opt,om_plot,True)
    Q_pow = _compute_q_model(Q,om_plot,alpha,om_ref)

    plt.semilogx(om_plot/(2*np.pi),1./Q_opt,label='1/Q_opt')
    plt.semilogx(om_plot/(2*np.pi),1./Q_pow,label='1/Q_target')
    plt.xlabel("Frequency,Hz")
    plt.ylabel("$Q^{-1}$")
    plt.axvline(min_freq)
    plt.axvline(max_freq)
    plt.legend()
    plt.savefig("Qmodel.jpg")
