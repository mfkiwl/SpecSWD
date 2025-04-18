#include "solver/solver.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <iostream>

typedef std::complex<double> dcmplx;


/**
 * @brief compute Love wave dispersion and eigenfunctions
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc)
 * @param displ eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix, where K^{-1} (om^2 M -E) = Q^T Sigma Q
 */
void SolverSEM:: 
compute_slegn(double freq,std::vector<double> &c,
            std::vector<double> &displ,bool use_qz)
{
    this -> prepare_matrices_love_(freq);
    typedef Eigen::Matrix<double,-1,-1,1> dmat2;
    using Eigen::MatrixXd;
    using Eigen::all;

    // mapping M,K,E to matrix
    int ng = nglob_el;
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXd> K(Kmat.data(),ng);
    Eigen::Map<const dmat2> E(Emat.data(),ng,ng);
    
    // construct A  = (om^2 M - E), B = K
    double om = 2. * M_PI * freq; 
    double omega2 = std::pow(om,2);
    MatrixXd A = MatrixXd((1. / K.array()).matrix().asDiagonal()) *
                        (-E + dmat2(M.asDiagonal()) * omega2);
    MatrixXd displ_all = A * 0.;

    // get eigen values/eigen vectors
    Eigen::Array<double,-1,1> kr(ng),ki(ng);
    LAPACKE_dgeev(LAPACK_COL_MAJOR,'N','V',ng,A.data(),ng,kr.data(),ki.data(),
                 nullptr,ng,displ_all.data(),ng);

    // filter swd 
    Eigen::Array<dcmplx,-1,1> k =  (kr + dcmplx{0,1.} * kr).sqrt();
    Eigen::Array<double,-1,1> c_all = (om / k).real();
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
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
    c.resize(nc); displ.resize(nc * size);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
        for(int i = 0; i < size; i ++) {
            displ[ic * size + i] = displ_all(i,id);
        }
    } 

    // compute QZ if required
    if(use_qz) {
        // resize all matrices
        Qmat_.resize(ng*ng); Zmat_.resize(ng*ng);
        Smat_.resize(ng*ng); Spmat_.resize(ng*ng);

        // get hessenburg form by Schur decomposition
        A = -E + dmat2(M.asDiagonal()) * omega2;
        MatrixXd B = MatrixXd(K.asDiagonal());
        Eigen::VectorXd alphar(ng),alphai(ng),beta(ng);

        // decompose A = Q @ S @ Z.T, B = Q@
        int sdim;
        LAPACKE_dgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,
                      ng,A.data(),ng,B.data(),ng,&sdim,alphar.data(),
                      alphai.data(),beta.data(),Qmat_.data(),ng,
                      Zmat_.data(),ng);
        memcpy(Smat_.data(),A.data(),A.size()*sizeof(double));
        memcpy(Spmat_.data(),B.data(),B.size()*sizeof(double));
        
    }
}

/**
 * @brief compute Love wave dispersion and eigenfunctions, visco-elastic case
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param displ eigen functions(displ at y direction), shape(nc,nglob_el)
 * @param use_qz if true, save QZ matrix, where K^{-1} (om^2 M -E) = Q^T Sigma Q
 */
void SolverSEM:: 
compute_slegn_att(double freq,std::vector<std::complex<double>> &c,
                 std::vector<std::complex<double>> &displ,
                bool use_qz)
{
    this -> prepare_matrices_love_att_(freq);
    using Eigen::MatrixXcd;
    typedef Eigen::Matrix<dcmplx,-1,-1,1> cdmat2;

    // construct matrix
    int ng = nglob_el;
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),ng);
    Eigen::Map<const Eigen::VectorXcd> K(CKmat.data(),ng);
    Eigen::Map<const cdmat2> E(CEmat.data(),ng,ng);

    // construct A  = K^{-1}(om^2 M - E)
    double om = 2. * M_PI * freq; 
    double omega2 = std::pow(om,2);
    MatrixXcd A = (cdmat2((1. / K.array()).matrix().asDiagonal()) * 
                        (-E + cdmat2(M.cast<dcmplx>().asDiagonal()) * omega2));
    
    // solve it 
    typedef lapack_complex_double lscmplx;
    MatrixXcd displ_all = A,A_cp = A;
    Eigen::ArrayXcd k(ng);
    LAPACKE_zgeev(LAPACK_COL_MAJOR,'N','V',ng,
                (lscmplx*)A_cp.data(),ng,(lscmplx*)k.data(),
                 nullptr,ng,(lscmplx*)displ_all.data(),ng);
    k = k.sqrt();

    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXcd c_all = om / k;
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
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
        for(int i = 0; i < ng; i ++) {
            displ[ic * ng + i] = displ_all(i,id);
        }
    } 

    // save QZ matrix
    if(use_qz) {
        // resize all matrices
        cQmat_.resize(ng*ng); cZmat_.resize(ng*ng);
        cSmat_.resize(ng*ng); cSpmat_.resize(ng*ng);

        // get hessenburg form by Schur decomposition
        typedef lapack_complex_double ldcmplx;
        A = -E + cdmat2(M.asDiagonal()) * omega2;
        MatrixXcd B = MatrixXcd(K.asDiagonal());
        Eigen::VectorXcd alpha(ng),beta(ng);

        // decompose A = Q @ S @ Z.H, B = Q@ S' @ Z.H
        int sdim;
        LAPACKE_zgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,
                      ng,(ldcmplx*)A.data(),ng,(ldcmplx*)B.data(),
                      ng,&sdim,(ldcmplx*)alpha.data(),(ldcmplx*)beta.data(),
                      (ldcmplx*)cQmat_.data(),ng,
                      (ldcmplx*)cZmat_.data(),ng);
        memcpy(cSmat_.data(),A.data(),A.size()*sizeof(dcmplx));
        memcpy(cSpmat_.data(),B.data(),B.size()*sizeof(dcmplx));
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, elastic case
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverSEM::
compute_sregn(double freq,std::vector<double> &c,
            std::vector<double> &ur,
            std::vector<double> &ul,
            bool use_qz)
{
    typedef Eigen::Matrix<double,-1,-1,1> dmat2;
    using Eigen::MatrixXd; using Eigen::ArrayXd;

    // prepare matrices
    this -> prepare_matrices_rayl_(freq);

    // mapping M,K,E to matrix
    int ng = nglob_ac + nglob_el * 2; 
    
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),ng);
    Eigen::Map<const dmat2> K(Kmat.data(),ng,ng);
    Eigen::Map<const dmat2> E(Emat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    double om = 2. * M_PI * freq;
    double omega2 = om * om;

    // solve this system
    MatrixXd A = omega2 * dmat2(M.asDiagonal()) - E;
    MatrixXd B = K;
    ArrayXd alphar(ng),alphai(ng),beta(ng);
    MatrixXd vsl(ng,ng),vsr(ng,ng);
    
    // generalized eigenvalue problem
    // A x = k B x
    LAPACKE_dggev(LAPACK_COL_MAJOR,'V','V',ng,A.data(),ng,B.data(),ng,
                 alphar.data(),alphai.data(),beta.data(),vsl.data(),ng,
                 vsr.data(),ng);

    // eigenvalue
    const std::complex<double> imag_i = {0,1.};
    Eigen::ArrayXcd k = ((alphar + imag_i * alphai) / beta).sqrt();
    
    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXd c_all = (om / k).real();
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
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
    c.resize(nc); ur.resize(nc * nglob); ul.resize(nc*nglob);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
        for(int i = 0; i < nglob; i ++) {
            ur[ic*nglob+i] = vsr(i,id);
            ul[ic*nglob+i] = vsl(i,id);
        }
    } 

    // save qz matrix
    if(use_qz) {
        Qmat_.resize(ng*ng); Zmat_.resize(ng*ng);
        Smat_.resize(ng*ng); Spmat_.resize(ng*ng);

        // get hessenburg form by Schur decomposition
        // A = Q @ S @ Z.T;  B = Q @ S' @ Z.T
        int sdim = 0;
        LAPACKE_dgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,ng,
                      A.data(),ng,B.data(),ng,&sdim,alphar.data(),
                      alphai.data(),beta.data(),Qmat_.data(),ng,
                      Zmat_.data(),ng);
        memcpy(Smat_.data(),A.data(),A.size()*sizeof(double));
        memcpy(Spmat_.data(),B.data(),B.size()*sizeof(double));
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, visco-elastic case
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param ur/ul left/right eigenvectors, shape(nc,nglob_el*2+nglob_ac)
 * @param use_qz if true, save QZ matrix
 */
void SolverSEM::
compute_sregn_att(double freq,std::vector<dcmplx> &c,
                std::vector<dcmplx> &ur,
                std::vector<dcmplx> &ul,
                bool use_qz)
{
    typedef Eigen::Matrix<dcmplx,-1,-1,1> cdmat2;
    typedef lapack_complex_double lscmplx;

    // prepare matrices
    this -> prepare_matrices_rayl_att_(freq);

    // mapping M,K,E to matrix
    int ng = nglob_ac + nglob_el * 2; 

    Eigen::Map<const Eigen::VectorXcd> M(CMmat.data(),ng);
    Eigen::Map<const cdmat2> K(CKmat.data(),ng,ng);
    Eigen::Map<const cdmat2> E(CEmat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    double om = 2. * M_PI * freq;
    double omega2 = om * om;

    // solve this system
    Eigen::MatrixXcd A = omega2 * cdmat2(M.asDiagonal()) - E;
    Eigen::MatrixXcd B = K;
    Eigen::ArrayXcd alpha(ng),beta(ng);
    Eigen::MatrixXcd vsl(ng,ng),vsr(ng,ng);

    // generalized eigenvalue problem
    // A x = k B x
    // eigenvalues
    LAPACKE_zggev(LAPACK_COL_MAJOR,'V','V',ng,(lscmplx*)A.data(),ng,(lscmplx*)B.data(),ng,
                 (lscmplx*)alpha.data(),(lscmplx*)beta.data(),(lscmplx*)vsl.data(),ng,
                 (lscmplx*)vsr.data(),ng);
    Eigen::ArrayXcd k = (alpha / beta).sqrt();
    
    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXcd c_all = om / k;
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
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
        for(int i = 0; i < ng; i ++) {
            ul[ic * ng + i] = vsl(i,id);
            ur[ic * ng + i] = vsr(i,id);
        }
    }

   // save qz matrix
    if(use_qz) {
        cQmat_.resize(ng*ng); cZmat_.resize(ng*ng);
        cSmat_.resize(ng*ng); cSpmat_.resize(ng*ng);
        
        // Schur decomposition
        int sdim = 0;
        LAPACKE_zgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,ng,
            (lscmplx*)A.data(),ng,(lscmplx*)B.data(),ng,&sdim,(lscmplx*)alpha.data(),
            (lscmplx*)beta.data(),(lscmplx*)cQmat_.data(),ng,
            (lscmplx*)cZmat_.data(),ng);


        memcpy(cSmat_.data(),A.data(),A.size()*sizeof(A(0,0)));
        memcpy(cSpmat_.data(),B.data(),B.size()*sizeof(B(0,0)));
    }
}