#ifndef SPECSWD_LIB_GLOB_H_
#define SPECSWD_LIB_GLOB_H_

#include "mesh/mesh.hpp"
#include "vti/vti.hpp"

#include <memory>
#include <complex>

#define CNAME(a) extern "C" void a

// global vars for solver/mesh

namespace specswd_pylib
{

extern std::unique_ptr<specswd::Mesh> M_;
extern std::unique_ptr<specswd::SolverLove> LoveSol_;
extern std::unique_ptr<specswd::SolverRayl> RaylSol_;

// global vars for eigenvalues/eigenvectors 
extern std::vector<float> egnr_,egnl_,c_,u_;
extern std::vector<specswd::scmplx> cegnr_,cegnl_,cc_,cu_;

} // namespace specswd_pylib

#endif