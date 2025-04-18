
#include "libswd/specswd.hpp"
#include "libswd/global.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iostream>


namespace py = pybind11;
using namespace py::literals;
using py::arg;

const auto FCST = (py::array::c_style | py::array::forcecast) ;
typedef py::array_t<float,FCST> vec;
typedef py::array_t<std::complex<float>,FCST> cvec;

void init_love(const vec &z,const vec &rho,
             const vec &vsh, const vec &vsv,
             const vec &QN, const vec &QL,
             bool HAS_ATT,bool print_info)
{
    // init GQtable
    specswd_init_GQTable();

    // get pointer
    const float *qn = nullptr, *ql = nullptr;
    if(HAS_ATT) {
        qn = QN.data();
        ql = QL.data();
    }

    // init mesh
    int nz = z.size();
    specswd_init_mesh_love(
        nz,z.data(),rho.data(),vsh.data(),vsv.data(),
        qn,ql,HAS_ATT,print_info
    );
}

void init_rayl(const vec &z,const vec &rho,
             const vec &vph, const vec &vpv,
             const vec &vsv, const vec &QA,
             const vec &QC, const vec &QL,
             bool HAS_ATT,bool print_info)
{
    // init GQtable
    specswd_init_GQTable();

    // get pointer
    const float *qa = nullptr, 
                *ql = nullptr,
                *qc = nullptr;
    if(HAS_ATT) {
        qa = QA.data();
        ql = QL.data();
        qc = QC.data();
    }

    int nz = z.size();
    specswd_init_mesh_rayl(
        nz,z.data(),rho.data(),vph.data(),vpv.data(),
        vsv.data(),qa,qc,ql,HAS_ATT,print_info
    );
}

template <typename T> py::array_t<T,FCST> 
compute_swd(float freq,int max_order,bool use_qz) 
{
    using namespace specswd_pylib;
    py::array_t<T,FCST> c_out;
    const int SWD_TYPE = M_->SWD_TYPE;
    const bool HAS_ATT = M_->HAS_ATT;

    // get phase velocities
    switch (SWD_TYPE)
    {
    case 0:
        specswd_egn_love(freq,use_qz);
        break;
    case 1:
        specswd_egn_rayl(freq,use_qz);
        break;
    default:
        break;
    }

    // allocate space
    int nc = (HAS_ATT) ? cc_.size()  : c_.size();
    const T *c_glob = nullptr;
    if constexpr (std::is_same_v<T,float>) {
        c_glob = c_.data();
    }
    else {
        c_glob = cc_.data();
    }
    if (max_order < 0 || max_order > nc ) {
        max_order = nc;
    }
    c_out.resize({nc});

    // copy phase velocity to c_out
    auto c = c_out.template mutable_unchecked<1>();
    for(int ic = 0; ic < max_order; ic ++) {
        c(ic) = c_glob[ic];
    }

    return c_out;
}

template <typename T> py::array_t<T,FCST> 
compute_group_vel(int max_order)
{
    using namespace specswd_pylib;
    py::array_t<T,FCST> u_out;
    const int SWD_TYPE = M_->SWD_TYPE;
    const bool HAS_ATT = M_->HAS_ATT;

    // get phase velocities
    switch (SWD_TYPE)
    {
    case 0:
        specswd_group_love();
        break;
    case 1:
        specswd_group_rayl();
        break;
    default:
        break;
    }

    // allocate space
    int nc = (HAS_ATT) ? cu_.size()  : u_.size();
    const T *u_glob = nullptr;
    if constexpr (std::is_same_v<T,float>) {
        u_glob = u_.data();
    }
    else {
        u_glob = cu_.data();
    }
    if (max_order < 0 || max_order > nc ) {
        max_order = nc;
    }
    u_out.resize({nc});

    // copy phase velocity to c_out
    auto u = u_out.template mutable_unchecked<1>();
    for(int ic = 0; ic < max_order; ic ++) {
        u(ic) = u_glob[ic];
    }

    return u_out;
}

PYBIND11_MODULE(libswd,m){
    m.doc() = "Surface wave dispersion and sensivity kernel\n";
    m.def("init_love",&init_love,arg("z"),arg("rho"),arg("vsh"),
          arg("vsv"),arg("QN"),arg("QL"),
          arg("HAS_ATT") = false,
          arg("print_info") = false,
          "initialize global vars for love wave");
        
    m.def("init_rayl",&init_rayl,arg("z"),arg("rho"),arg("vph"),
          arg("vpv"),arg("vsv"),arg("QA"),arg("QC"),
          arg("QL"),arg("HAS_ATT") = false,
          arg("print_info") = false,
          "initialize global vars for rayleigh wave");
    
    m.def("compute_egn",&compute_swd<float>,
          arg("freq"),arg("max_order"),arg("use_qz")=true,
          "compute dispersions for elastic wave");

    m.def("compute_egn_att",&compute_swd<std::complex<float>>,
          arg("freq"),arg("max_order"),arg("use_qz")=true,
          "compute dispersions for visco-elastic wave");

    m.def("group_vel",&compute_group_vel<float>,
            arg("max_order"),
         "compute group velocity for elastic wave");

    m.def("group_vel_att",&compute_group_vel<std::complex<float>>,
            arg("max_order"),
         "compute group velocity for visco-elastic wave");
}