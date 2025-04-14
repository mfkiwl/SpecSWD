#include "aniso/aniso.hpp"
#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd
{

/**
 * @brief compute Love wave dispersion and eigenfunctions, elastic case
 * 
 * @param freq current frequency
 * @param c dispersion, shape(nc)
 * @param egn eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param save_qz if true, save QZ matrix
 */
void SolverAni::
compute_egn(const Mesh &mesh,float freq,
            std::vector<float> &c,
            std::vector<scmplx> &egn,
            bool save_qz=false)
{
    typedef Eigen::MatrixX<realw> crmat2;
    using Eigen::all; using Eigen::seq;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_el*3 + mesh.nglob_ac;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXf> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> H(Hmat.data(),ng,ng);
    
    // construct A  = (om^2 M - E), B = K
    const crealw imag_i{0.,1.};
    realw om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    crmat2 A(ng*2,ng*2), B(ng*2,ng*2);
    auto idx1 = seq(0,ng-1), idx2 = seq(ng,ng*2-1);
    A.setZero(); B.setZero();
    A(idx1,idx2).setIdentity();
    A(idx2,idx1) = omega2 * crmat2(M.cast<crealw>().asDiagonal()) - E.cast<crealw>();
    A(idx2,idx2) = H.cast<crealw>() *imag_i;
    B(idx1,idx1).setIdentity();
    B(idx2,idx2) = K.cast<crealw>();

    // get eigen values/eigen vectors
    crmat2 A_cp = A, B_cp = B;
    Eigen::Array<realw,-1,1> k(ng*2);
    LAPACKE_CMPLX(hegvd)(
        LAPACK_COL_MAJOR,1,'V','U',ng*2,
        (LCREALW*)A_cp.data(),ng*2,
        (LCREALW*)B_cp.data(),ng*2,k.data()
    );

    Eigen::ArrayX<realw> c_all = om / k;
    auto mask = ((c_all >= mesh.PHASE_VELOC_MIN)&& 
                (c_all <= mesh.PHASE_VELOC_MAX) && 
                k.real().abs() >= 10 *k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }

    // sort according to ascending order 
    int nc = idx0.size();
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]] < c_all[idx0[i2]];}); 

    // copy to c/displ
    c.resize(nc); egn.resize(nc * ng);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];

        // normalize
        scmplx scale = A_cp(seq(0,ng-1),id).norm();
        for(int i = 0; i < ng; i ++) {
            egn[ic*ng+i] = A_cp(i,id) / scale;
        }
    }

    if(save_qz) {
        schur_qz(ng*2,A,B,cQmat_,cZmat_,cSmat_,cSpmat_);
    }
}

} // namespace specswd
