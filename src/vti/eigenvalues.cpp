#include "vti/vti.hpp"

#include "shared/schur.hpp"

#include <algorithm>
#include <iostream>

namespace specswd {

/**
 * @brief compute Love wave dispersion and eigenfunctions, elastic case
 * 
 * @param c dispersion, shape(nc)
 * @param egn eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if false, only compute phase velocities
 */
void SolverLove::
compute_egn(const Mesh &mesh,
            std::vector<float> &c,
            std::vector<float> &egn,
            bool use_qz)
{
    typedef Eigen::MatrixX<realw> rmat2;
    using Eigen::all;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXf> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);
    
    // construct A  = (om^2 M - E), B = K
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    rmat2 A = rmat2(((realw)1. / K.array()).matrix().cast<realw>().asDiagonal()) *
             (-E.cast<realw>() + rmat2(M.cast<realw>().asDiagonal() * omega2));
    rmat2 displ_all = A * 0.;

    // get eigen values/eigen vectors
    Eigen::Array<realw,-1,1> kr(ng),ki(ng);
    LAPACKE_REAL(geev)(
        LAPACK_COL_MAJOR,'N','V',ng,A.data(),
        ng,kr.data(),ki.data(),nullptr,
        ng,displ_all.data(),ng
    );

    // filter swd 
    Eigen::Array<crealw,-1,1> k =  (kr + crealw{0,1.} * ki).sqrt();
    Eigen::Array<realw,-1,1> c_all = (om / k).real();
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k.real().abs() >= 10 *k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    for(int i = 0; i < c_all.size(); i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }

    // sort to ascending order 
    int nc = idx0.size();
    std::vector<int> idx(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]] < c_all[idx0[i2]];}); 

    // copy to c/displ
    int size = displ_all.rows();
    c.resize(nc); egn.resize(nc * size);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];

        // scale
        float scale = displ_all(all,id).norm();
        for(int i = 0; i < size; i ++) {
            egn[ic * size + i] = displ_all(i,id) / scale;
        }
    } 

    // compute QZ if required
    if(use_qz) {
        // get hessenburg form by Schur decomposition
        A = -E.cast<realw>() + rmat2(M.cast<realw>().asDiagonal()* omega2);
        rmat2 B = rmat2(K.cast<realw>().asDiagonal());

        schur_qz(ng,A,B,Qmat_,Zmat_,Smat_,Spmat_);
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, visco-elastic case
 * 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param egn eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix
 */
void SolverLove::
compute_egn_att(const Mesh &mesh,
                std::vector<scmplx> &c,
                std::vector<scmplx> &displ,
                bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;

    // construct matrix
    int ng = mesh.nglob_el;
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXcf> K(CKmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> E(CEmat.data(),ng,ng);

    // construct A  = K^{-1}(om^2 M - E)
    float freq = mesh.freq;
    float om = 2. * M_PI * freq; 
    realw omega2 = std::pow(om,2);
    crmat2 A = (1. / K.array()).cast<crealw>().matrix().asDiagonal() * 
                (-E + M.asDiagonal().toDenseMatrix() * omega2).cast<crealw>();
    // crmat2 A = (crmat2((1. / K.array()).cast<crealw>().matrix().asDiagonal()) * 
    //             (-E.cast<crealw>() + crmat2(M.cast<crealw>().asDiagonal()) * omega2));
    
    // solve it 
    crmat2 displ_all = A;
    Eigen::ArrayX<crealw> k(ng);
    LAPACKE_CMPLX(geev)(
        LAPACK_COL_MAJOR,'N','V',ng,
        (LCREALW *)A.data(),ng,(LCREALW*)k.data(),
        nullptr,ng,(LCREALW *)displ_all.data(),ng
    );
    k = k.sqrt();

    // filter SWD 
    using Eigen::all;
    Eigen::ArrayX<crealw> c_all = om / k;
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k.real().abs() >= k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }
    int nc = idx0.size();

    // sort according to ascending order 
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

    // copy to c/displ
    c.resize(nc); displ.resize(nc * ng);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];

        // scale factor
        crealw scale = displ_all(all,id).norm();
        for(int i = 0; i < ng; i ++) {
            displ[ic * ng + i] = displ_all(i,id) / scale;
        }
    } 

    // save QZ matrix
    if(use_qz) {
        // get hessenburg form by Schur decomposition
        //typedef lapack_complex_double ldcmplx;
        A = -E.cast<crealw>() + crmat2(M.cast<crealw>().asDiagonal()) * omega2;
        crmat2 B = K.cast<crealw>().asDiagonal();
        
        schur_qz(ng,A,B,cQmat_,cZmat_,cSmat_,cSpmat_);
    } 
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, elastic case
 * 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverRayl::
compute_egn(const Mesh &mesh,
            std::vector<float> &c,
            std::vector<float> &ur,
            std::vector<float> &ul,
            bool use_qz)
{
    typedef Eigen::MatrixX<realw> rmat2;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_ac + mesh.nglob_el * 2; 
    Eigen::Map<const Eigen::VectorXf> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<float,-1,-1,1>> E(Emat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq;
    realw omega2 = om * om;

    // solve this system
    rmat2 A = omega2 * rmat2(M.cast<realw>().asDiagonal()) - E.cast<realw>();
    rmat2 B = K.cast<realw>();
    Eigen::ArrayX<realw> alphar(ng),alphai(ng),beta(ng);
    rmat2 vsl(ng,ng),vsr(ng,ng);
    
    // generalized eigenvalue problem
    // A x = k B x
    LAPACKE_REAL(ggev)(
        LAPACK_COL_MAJOR,'V','V',ng,A.data(),ng,B.data(),ng,
        alphar.data(),alphai.data(),beta.data(),vsl.data(),ng,
        vsr.data(),ng);

    //eigenvalue
    const crealw imag_i = {0,1.};
    Eigen::ArrayX<crealw> k = ((alphar + imag_i * alphai) / beta).sqrt();

    // filter SWD 
    using Eigen::all;
    Eigen::ArrayX<realw> c_all = (om / k).real();
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
    c.resize(nc); ur.resize(nc * ng); ul.resize(nc*ng);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];

        // normalize factor
        float sr = vsr(all,id).norm();
        float sl = vsl(all,id).norm();

        // copy to ur/ul
        for(int i = 0; i < ng; i ++) {
            ur[ic*ng+i] = vsr(i,id) / sr;
            ul[ic*ng+i] = vsl(i,id) / sl;
        }
    } 

    // save qz matrix
    if(use_qz) {
        schur_qz(ng,A,B,Qmat_,Zmat_,Smat_,Spmat_);
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, visco-elastic case
 * 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverRayl::
compute_egn_att(const Mesh &mesh,
                std::vector<scmplx> &c,
                std::vector<scmplx> &ur,
                std::vector<scmplx> &ul,
                bool use_qz)
{
    typedef Eigen::MatrixX<crealw> crmat2;

    // mapping M,K,E to matrix
    int ng = mesh.nglob_ac + mesh.nglob_el * 2; 

    Eigen::Map<const Eigen::VectorXcf> M(CMmat.data(),ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> K(CKmat.data(),ng,ng);
    Eigen::Map<const Eigen::Matrix<scmplx,-1,-1,1>> E(CEmat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    float freq = mesh.freq;
    realw om = 2. * M_PI * freq;
    realw omega2 = om * om;

    // solve this system
    crmat2 A = crmat2(M.cast<crealw>().asDiagonal()) * omega2 - E.cast<crealw>();
    crmat2 B = K.cast<crealw>();
    Eigen::ArrayX<crealw> alpha(ng),beta(ng);
    crmat2 vsl(ng,ng),vsr(ng,ng);

    // generalized eigenvalue problem
    // A x = k B x
    // eigenvalues
    LAPACKE_CMPLX(ggev)(
        LAPACK_COL_MAJOR,'V','V',ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),ng,
        (LCREALW*)alpha.data(),(LCREALW*)beta.data(),(LCREALW*)vsl.data(),ng,
        (LCREALW*)vsr.data(),ng
    );

    Eigen::ArrayX<crealw> k = (alpha / beta).sqrt();
    
    // filter SWD 
    using Eigen::all;
    Eigen::ArrayX<crealw> c_all = om / k;
    auto mask = ((c_all.real() >= mesh.PHASE_VELOC_MIN)&& 
                (c_all.real() <= mesh.PHASE_VELOC_MAX) && 
                k.real().abs() >= k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    int nc_all = c_all.size();
    for(int i = 0; i < nc_all; i ++) {
        if(mask[i]) {
            idx0.push_back(i);
        }
    }
    int nc = idx0.size();

    // sort according to ascending order 
    std::vector<int> idx;
    idx.resize(nc);
    for(int i = 0; i < nc; i ++ ) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
        [&c_all,&idx0](size_t i1, size_t i2) {return c_all[idx0[i1]].real() < c_all[idx0[i2]].real();}); 

    // copy to c/displ
    c.resize(nc); ur.resize(nc * ng); ul.resize(nc*ng);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
    
        // normalize factor
        realw sr = vsr(all,id).norm();
        realw sl = vsl(all,id).norm();

        for(int i = 0; i < ng; i ++) {
            ul[ic * ng + i] = vsl(i,id) / sl;
            ur[ic * ng + i] = vsr(i,id) / sr;
        }
    }

   // save qz matrix
    if(use_qz) {
        schur_qz(ng,A,B,cQmat_,cZmat_,cSmat_,cSpmat_);
    }
}

} // end namespace specswd