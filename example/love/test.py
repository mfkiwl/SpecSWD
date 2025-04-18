import numpy as np 
from specd import SpecWorkSpace

model = np.array([
    [0.0000000, 2.800000, 3.300000, 3.000000, 220., 200.],
    [35.000000, 2.800000, 3.300000, 3.000000, 220., 200.],
    [35.000000, 3.200000, 5.500000, 5.000000, 330., 300.]
])
z,rho,vsh,vsv,_,_ = (model[:, i].reshape(3) for i in range(6))

ws = SpecWorkSpace('love',z,rho,vsh=vsh,vsv=vsv,disp=True)
c = ws.compute_egn(1.,-1,False)
print(c)