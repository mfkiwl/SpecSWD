import numpy as np

z = np.array([0,35,35])
nz = len(z)
vs = np.array([3.0,3.0,5.0])
rho = np.array([2.8,2.8,3.2])

f = open("model.txt","w")
f.write("0 0\n")

for i in range(len(z)):
    f.write("%f %f %f %f 200. 200.\n"%(z[i],rho[i],vs[i],vs[i]))

f.close()