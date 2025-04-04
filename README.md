# SPECSWD
**SpecSWD** utilizes spectral element method and quadratic eigenvalue solver to compute visco-elastic surface wave dispersion, eigen functions and sensitivity kernels for **general layered models**.

It can handle dispersion values for several type of complex media, such as VTI, full anisotropy, and (visco)acoustic-(visco)elastic coupling and at any frequency, and any mode. And it can support both 1-D model with discontinuities (continous variation in both fluid and elastic domain).

## Requirements and Installation
1. **Compilers:** C++/Fortran compilers which support c++17 (tested on `GCC >=7.5`), `cmake >= 3.12`

2. packages:
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.4.0
* [MKL/LAPACK](https://www.netlib.org/lapack/lapacke.html)
* [doxygen](https://www.doxygen.nl/) for api document generation, optional.

3. Install:
```bash
mkdir -p build; cd build;
cmake .. -DCXX=g++ -DFC=gfortran  -DEIGEN_INC=/path/to/eigen/
make -j4; make install 
```
This program also provides python libraries (`.so`) and you can install them by adding ```-DBUILD_PYLIB=TRUE -DPYTHON_EXECUTABLE=`which python` ```

4. install API docs
```bash
cd doxygen
doxygen config.cfg
```

# Gallery
### Benchmark: SWDTTI with CPS330
![image](example//rayleigh/phase.jpg)
### HTI model: Phase velocity vs. Azimuthal angle
![image](example/tti/group-direc.jpg)

### Fluid-Elastic Coupling phase and group velocity
![image](example/ac/phase.jpg)
![image](example/ac/group.jpg)
### Acoustic 

