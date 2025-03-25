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
 * @param displ eigen functions(displ at y direction), shape(nc,nglob)
 * @param save_qz if true, save QZ matrix, where K^{-1} (om^2 M -E) = Q^T Sigma Q
 */
void SolverSEM:: 
compute_slegn(double freq,std::vector<double> &c,
            std::vector<double> &displ,bool save_qz)
{
    this -> prepare_matrices_love_();
    typedef Eigen::Matrix<double,-1,-1,1> dmat2;

    // mapping M,K,E to matrix
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),nglob);
    Eigen::Map<const Eigen::VectorXd> K(Kmat.data(),nglob);
    Eigen::Map<const dmat2> E(Emat.data(),nglob,nglob);
    
    // construct A  = K^{-1}(om^2 M - E)
    double om = 2. * M_PI * freq; 
    double omega2 = std::pow(om,2);
    Eigen::MatrixXd displ_all = (-E + dmat2(M.asDiagonal()) * omega2);
    Eigen::MatrixXd B = Eigen::MatrixXd(K.asDiagonal());
    
    // get eigen values/eigen vector
    Eigen::Array<double,-1,1> k1(nglob);
    LAPACKE_dsygv(LAPACK_COL_MAJOR,1,'V','U',nglob,displ_all.data(),nglob,B.data(),nglob,k1.data());  

    // normalize displ_all
    using Eigen::all;
    for(int i = 0; i < nglob; i ++) {
        displ_all(all,i) /= displ_all(all,i).norm();
    }

    // filter swd 
    Eigen::Array<dcmplx,-1,1> k =  k1.cast<dcmplx>().sqrt();
    Eigen::Array<double,-1,1> c_all = (om / k).real();
    auto mask = ((c_all >= PHASE_VELOC_MIN)&& (c_all <= PHASE_VELOC_MAX));
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

    // save Q matrix if required
    if(save_qz) {
        Qmat_.resize(nglob*nglob);
        for(int j = 0; j < nglob; j ++) {
        for(int i = 0; i < nglob; i ++) {
            Qmat_[j*nglob+i] = displ_all(i,j);
        }}

    }
}


/**
 * @brief compute Love wave dispersion and eigenfunctions, visco-elastic case
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param displ eigen functions(displ at y direction), shape(nc,nglob)
 * @param save_qz if true, save QZ matrix, where K^{-1} (om^2 M -E) = Q^T Sigma Q
 */
void SolverSEM:: 
compute_slegn_att(double freq,std::vector<std::complex<double>> &c,
                 std::vector<std::complex<double>> &displ,
                bool save_qz)
{
    this -> prepare_matrices_love_att_();
    typedef Eigen::Matrix<dcmplx,-1,-1,1> cdmat2;

    // construct matrix
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),nglob);
    Eigen::Map<const Eigen::VectorXcd> K(CKmat.data(),nglob);
    Eigen::Map<const cdmat2> E(CEmat.data(),nglob,nglob);

    // construct A  = K^{-1}(om^2 M - E)
    double om = 2. * M_PI * freq; 
    double omega2 = std::pow(om,2);
    Eigen::MatrixXcf A = (cdmat2((1. / K.array()).matrix().asDiagonal()) * 
                        (-E + cdmat2(M.cast<dcmplx>().asDiagonal()) * omega2))
                        .cast<Eigen::scomplex>();
    
    // solve it 
    typedef lapack_complex_float lscmplx;
    Eigen::MatrixXcf displ_all = A,A_cp = A;
    Eigen::ArrayXcf k(nglob);
    LAPACKE_cgeev(LAPACK_COL_MAJOR,'N','V',nglob,
                (lscmplx*)A_cp.data(),nglob,(lscmplx*)k.data(),
                 nullptr,nglob,(lscmplx*)displ_all.data(),nglob);
    k = k.sqrt();

    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXcf c_all = om / k;
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
                k.real().abs() >= k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    for(size_t i = 0; i < c_all.size(); i ++) {
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
    c.resize(nc); displ.resize(nc * nglob);
    for(int ic = 0; ic < nc; ic ++) {
        int id = idx0[idx[ic]];
        c[ic] = c_all[id];
        for(int i = 0; i < nglob; i ++) {
            displ[ic * nglob + i] = displ_all(i,id);
        }
    } 

    // save QZ matrix
    if(save_qz) {
        cQmat_.resize(nglob*nglob);
        for(int j = 0; j < nglob; j ++) {
        for(int i = 0; i < nglob; i ++) {
            cQmat_[j*nglob+i] = displ_all(i,j);
        }}
    }
}

/**
 * @brief compute rayleigh wave dispersion and eigenfunctions, elastic case
 * 
 * @param freq current frequency
 * @param vmin,vmax min/max velocity for your model 
 * @param c dispersion, shape(nc) c = c0(1 + iQL^{-1})
 * @param displ eigen functions(displ at y direction), shape(nc,nglob)
 * @param save_qz if true, save QZ matrix, where K^{-1} (om^2 M -E) = Q^T Sigma Q
 */
void SolverSEM::
compute_sregn(double freq,std::vector<double> &c,
            std::vector<double> &ur,
            std::vector<double> &ul,
            bool save_qz)
{
    // prepare matrices
    this -> prepare_matrices_rayl_(freq);

    // mapping M,K,E to matrix
    int ng = nglob_ac + nglob_el * 2; 
    typedef Eigen::Matrix<double,-1,-1,1> dmat2;
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),ng);
    Eigen::Map<const dmat2> K(Kmat.data(),ng,ng);
    Eigen::Map<const dmat2> E(Emat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    double om = 2. * M_PI * freq;
    double omega2 = om * om;

    // solve this system
    Eigen::MatrixXf A = (omega2 * dmat2(M.asDiagonal()) - E).cast<float>();
    Eigen::MatrixXf B = K.cast<float>();
    Eigen::ArrayXf alphar(ng),alphai(ng),beta(ng);
    Eigen::MatrixXf vsl(ng,ng),vsr(ng,ng);
    
    // generalized eigenvalue problem
    // A x = k B x
    // where A = vsl @ S vsr.T, B = vsl @ S' @ vsr.T 
    LAPACKE_sgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,ng,
                  A.data(),ng,B.data(),ng,0,alphar.data(),
                  alphai.data(),beta.data(),vsl.data(),ng,
                  vsr.data(),ng);

    // eigenvalue
    const std::complex<float> imag_i = {0,1.};
    Eigen::ArrayXf k = ((alphar + imag_i * alphai) / beta).sqrt().real();
    
    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXf c_all = om / k;
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX));
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    for(size_t i = 0; i < c_all.size(); i ++) {
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
    if(save_qz) {
        Qmat_.resize(ng*ng); Zmat_.resize(ng*ng);
        Smat_.resize(ng*ng); Spmat_.resize(ng*ng);

        for(int j = 0; j < ng; j ++) {
        for(int i = 0; i < ng; i ++) {
            Qmat_[j * ng + i] = vsl(i,j);
            Zmat_[j * ng + i] = vsr(i,j);
            Smat_[j * ng + i] = A(i,j);
            Spmat_[j * ng + i] = B(i,j);
        }}
    }
}

void SolverSEM::
compute_sregn_att(double freq,std::vector<dcmplx> &c,
                std::vector<dcmplx> &ur,
                std::vector<dcmplx> &ul,
                bool save_qz)
{
    // prepare matrices
    this -> prepare_matrices_rayl_att_(freq);

    // mapping M,K,E to matrix
    int ng = nglob_ac + nglob_el * 2; 
    typedef Eigen::Matrix<dcmplx,-1,-1,1> cdmat2;
    typedef lapack_complex_float lscmplx;
    Eigen::Map<const Eigen::VectorXcd> M(CMmat.data(),ng);
    Eigen::Map<const cdmat2> K(CKmat.data(),ng,ng);
    Eigen::Map<const cdmat2> E(CEmat.data(),ng,ng);

    // prepare matrix A = om^2 M -E
    double om = 2. * M_PI * freq;
    double omega2 = om * om;

    // solve this system
    Eigen::MatrixXcf A = (omega2 * cdmat2(M.asDiagonal()) - E).cast<Eigen::scomplex>();
    Eigen::MatrixXcf B = K.cast<Eigen::scomplex>();
    Eigen::ArrayXcf alpha(ng),beta(ng);
    Eigen::MatrixXcf vsl(ng,ng),vsr(ng,ng);

    // generalized eigenvalue problem
    // A x = k B x
    // where A = vsl @ S vsr.T, B = vsl @ S' @ vsr.T 
    LAPACKE_cgges(LAPACK_COL_MAJOR,'V','V','N',nullptr,ng,
                  (lscmplx*)A.data(),ng,(lscmplx*)B.data(),ng,0,(lscmplx*)alpha.data(),
                  (lscmplx*)beta.data(),(lscmplx*)vsl.data(),ng,
                  (lscmplx*)vsr.data(),ng);

    // eigenvalues
    Eigen::ArrayXcf k = (alpha / beta).sqrt();
    
    // filter SWD 
    using Eigen::all;
    Eigen::ArrayXcf c_all = om / k;
    auto mask = ((c_all.real() >= PHASE_VELOC_MIN)&& 
                (c_all.real() <= PHASE_VELOC_MAX) && 
                k.real().abs() >= k.imag().abs());
    std::vector<int> idx0; idx0.reserve(mask.cast<int>().sum());
    for(size_t i = 0; i < c_all.size(); i ++) {
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
    if(save_qz) {
        cQmat_.resize(ng*ng); cZmat_.resize(ng*ng);
        cSmat_.resize(ng*ng); cSpmat_.resize(ng*ng);

        for(int j = 0; j < ng; j ++) {
        for(int i = 0; i < ng; i ++) {
            cQmat_[j * ng + i] = vsl(i,j);
            cZmat_[j * ng + i] = vsr(i,j);
            cSmat_[j * ng + i] = A(i,j);
            cSpmat_[j * ng + i] = B(i,j);
        }}
    }
}