from specd import THSolver,SpecWorkSpace
import numpy as np 
import matplotlib.pyplot as plt 

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize']=15
mpl.rcParams['ytick.labelsize']=15
mpl.rcParams['axes.labelsize']=15
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['savefig.bbox'] = 'tight'

def brocher(vsz):
    vpz = 0.9409 + 2.0947*vsz - 0.8206*vsz**2+  \
            0.2683*vsz**3 - 0.0251*vsz**4
    rhoz = 1.6612 * vpz - 0.4721 * vpz**2 +   \
            0.0671 * vpz**3 - 0.0043 * vpz**4 +   \
            0.000106 * vpz**5

    return vpz,rhoz 

def test_model():
    z = np.linspace(0,120,3)
    nz = len(z)
    vs = 3.0 + 0.02 * z
    vp,rho = brocher(vs)
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    Qa = z * 0 
    Qb = z * 0
    Qb[:nz-1] = 200
    Qb[nz-1] =  400
    Qa = Qb * 9/4.

    return thk,vp,vs,rho,Qa,Qb 

def test_fluid():
    z = np.array([0.,5.,50,100.])
    nz = len(z)
    vs = 3.0 + 0.02 * z
    vs[0] = 0.
    vp,rho = brocher(vs)
    vp[0] = 1.5
    rho[0] = 1.
    thk = np.zeros_like(z)
    thk[0:nz-1] = np.diff(z)
    Qa = z * 0 
    Qb = z * 0
    Qb[:nz-1] = 200
    Qb[nz-1] =  400
    Qa = Qb * 9/4.

    return thk,vp,vs,rho,Qa,Qb 

def cps2spec(thk:np.ndarray,param:np.ndarray):
    nz = len(thk)
    z_spec = np.zeros((nz*2-1))
    param_spec = np.zeros_like(z_spec)
    z = thk * 0.
    z[1:] = np.cumsum(thk)[:nz-1]

    id = 0
    for i in range(nz):
        z_spec[id] = z[i]
        param_spec[id] = param[i]
        id += 1
        if i < nz - 1:
            z_spec[id] = z[i+1]
            param_spec[id] = param[i]
            id += 1
    
    return z_spec,param_spec

def main():
    thk,vp,vs,rho,Qa,Qb = test_fluid()
    z_spec,vp_spec = cps2spec(thk,vp)
    _,vs_spec = cps2spec(thk,vs)
    _,rho_spec = cps2spec(thk,rho)
    print(vs_spec)

    # frequencies
    nt = 100
    freqs = 10**np.linspace(np.log10(0.01),np.log10(0.5),nt)
    T = 1. / freqs

    # compute phase velocity by cps
    sol = THSolver(thk,vp,vs,rho)
    c = T * 0.
    for i in range(nt):
        c[i:i+1] = sol.compute_swd('Rc',0,T[i])

    # init workspace
    ws = SpecWorkSpace()
    ws.initialize(
        'rayl',
        z_spec,
        rho=rho_spec,
        vph = vp_spec * 1.,
        vpv = vp_spec * 1.,
        vsh = vs_spec * 1., 
        vsv = vs_spec * 1.,
        eta = vs_spec * 0 + 1.
    )

    c1 = T * 0.
    for i in range(nt):
        c1[i:i+1] = ws.compute_egn(freqs[i],1,False)

    fig,ax = plt.subplots(1,1,figsize=(8,4))
    ax.plot(freqs,c)
    ax.plot(freqs,c1,ls='--')
    fig.savefig("test.jpg",dpi=300)



if __name__ == "__main__":
    main()