#include "mesh/mesh.hpp"
#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

#include <iostream>

#include <memory>
#include <complex>

// global vars

// global vars for solver/mesh
namespace specswd_pylib
{
std::unique_ptr<specswd::Mesh> M_;
std::unique_ptr<specswd::SolverLove> LoveSol_;
std::unique_ptr<specswd::SolverRayl> RaylSol_;

// global vars for eigenvalues/eigenvectors 
std::vector<float> egnr_,egnl_,c_,u_;
std::vector<specswd::scmplx> cegnr_,cegnl_,cc_,cu_;
}

extern "C"  void 
specswd_init_GQTable() {
    GQTable::initialize();
}


extern "C" void 
specswd_init_mesh_love(
    int nz, const float *z,const float *rho,const float *vsh,
    const float *vsv,const float *QN, const float *QL,
    bool HAS_ATT,bool print_tomo_info = false
)
{
    using namespace specswd_pylib;

    // allocate memory
    auto &M = M_;
    M.reset(new specswd::Mesh);

    // allocate space for tomo, and set model
    M -> allocate_1D_model(nz,0,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        M -> rho_tomo[i] = rho[i];
        M -> vsv_tomo[i] = vsv[i];
        M -> vsh_tomo[i] = vsh[i];
        M -> depth_tomo[i] = z[i];
        if (HAS_ATT) {
            M -> QN_tomo[i] = QN[i];
            M -> QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    M -> create_model_attributes();
    if (print_tomo_info) {
        M -> print_model();
    }

    // initialize solver
    LoveSol_.reset(new specswd::SolverLove);
}

extern "C" void 
specswd_init_mesh_rayl(
    int nz, const float *z,const float *rho,
    const float *vph,const float* vpv,const float *vsv,
    const float *QA, const float *QC,const float *QL,
    bool HAS_ATT,bool print_tomo_info = false
)
{
    using namespace specswd_pylib;

    // allocate memory
    auto &M = M_;
    M.reset(new specswd::Mesh);

    // allocate space for tomo, and set model
    M -> allocate_1D_model(nz,1,HAS_ATT);
    for(int i = 0; i < nz; i ++) {
        M -> rho_tomo[i] = rho[i];
        M -> vsv_tomo[i] = vsv[i];
        M -> vph_tomo[i] = vph[i];
        M -> vpv_tomo[i] = vpv[i];
        M -> depth_tomo[i] = z[i];
        if (HAS_ATT) {
            M -> QA_tomo[i] = QA[i];
            M -> QC_tomo[i] = QC[i];
            M -> QL_tomo[i] = QL[i];
        }
    }

    // create attributes
    M -> create_model_attributes();
    if (print_tomo_info) {
        M -> print_model();
    }

    // initialize solver
    RaylSol_.reset(new specswd::SolverRayl);
}

extern "C" void 
specswd_egn_love(float freq,bool use_qz)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = M_->HAS_ATT;

    // create database
    M_ -> create_database(freq,0.);
    
    // prepare all matrices
    LoveSol_ -> prepare_matrices(*M_);

    // compute
    if(!HAS_ATT) {
        LoveSol_ -> compute_egn(*M_,c_,egnr_,use_qz);
    }
    else {
        LoveSol_ -> compute_egn_att(*M_,cc_,cegnr_,use_qz);
    }
}

extern "C" void 
specswd_egn_rayl(float freq,bool use_qz)
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = M_->HAS_ATT;

    // create database
    M_ -> create_database(freq,0.);

    // prepare all matrices
    RaylSol_ -> prepare_matrices(*M_);

    if(!HAS_ATT) {
        RaylSol_ -> compute_egn(*M_,c_,egnr_,egnl_,use_qz);
    }
    else {
        RaylSol_ -> compute_egn_att(*M_,cc_,cegnr_,cegnl_,use_qz);
    }
}

extern "C" void 
specswd_group_love()
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = M_->HAS_ATT;
    int ng = M_ -> nglob_el;

    int nc = 0;
    if(HAS_ATT) {
        nc = c_.size();
        cu_.resize(nc);
    
        for(int ic = 0; ic < nc; ic ++ ) {
            cu_[ic] = LoveSol_ -> group_vel_att(*M_,cc_[ic],&cegnr_[ic*ng]);
        }
    }
    else {
        nc = cc_.size();
        u_.resize(nc);
        for(int ic = 0; ic < nc; ic ++ ) {
            u_[ic] = LoveSol_ -> group_vel(*M_,c_[ic],&egnr_[ic*ng]);
        }
    }
}

extern "C" void 
specswd_group_rayl()
{
    using namespace specswd_pylib;

    // get contants
    bool HAS_ATT = M_->HAS_ATT;
    int ng = M_->nglob_el * 2 + M_-> nglob_ac;


    int nc = 0;
    if(HAS_ATT) { 
        nc = c_.size();
        cu_.resize(nc);
    
        for(int ic = 0; ic < nc; ic ++ ) {
            cu_[ic] = RaylSol_ -> group_vel_att(*M_,cc_[ic],&cegnr_[ic*ng],&cegnl_[ic*ng]);
        }
    }
    else {
        nc = cc_.size();
        u_.resize(nc);

        for(int ic = 0; ic < nc; ic ++ ) {
            u_[ic] = RaylSol_ -> group_vel(*M_,c_[ic],&egnr_[ic*ng],&egnl_[ic*ng]);
        }
    }
}
 