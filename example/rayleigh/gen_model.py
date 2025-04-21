import numpy as np

def brocher(vsz):
    vpz = 0.9409 + 2.0947*vsz - 0.8206*vsz**2+  \
            0.2683*vsz**3 - 0.0251*vsz**4
    rhoz = 1.6612 * vpz - 0.4721 * vpz**2 +   \
            0.0671 * vpz**3 - 0.0043 * vpz**4 +   \
            0.000106 * vpz**5

    return vpz,rhoz 

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

f = open("model.txt.cps","w")
for i in range(nz):
    f.write("%f %f %f %f %f %f 1.\n"%(thk[i],rho[i],vp[i],vp[i],vs[i],vs[i]))
f.close()

# write specswd file
f = open("model.txt","w")
f.write("1 0\n")
for i in range(nz):
    f.write("%f %f %f %f %f 1. %f %f %f\n"%(z[i],rho[i],vp[i],vp[i],vs[i],Qa[i],Qa[i],Qb[i]))
    if i < nz-1:
        f.write("%f %f %f %f %f 1. %f %f %f\n"%(z[i+1],rho[i],vp[i],vp[i],vs[i],Qa[i],Qa[i],Qb[i]))
f.close()