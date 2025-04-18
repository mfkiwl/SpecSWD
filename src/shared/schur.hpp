#ifndef SPECSWD_SCHUR_H_
#define SPECSWD_SCHUR_H_

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#ifdef SPECSWD_EGN_DOUBLE
typedef double realw;
#define LAPACKE_REAL(name) LAPACKE_d ## name
#define LAPACKE_CMPLX(name) LAPACKE_z ## name
#define LCREALW lapack_complex_double
 
#else
typedef float realw;
#define LAPACKE_REAL(name) LAPACKE_s ## name
#define LAPACKE_CMPLX(name) LAPACKE_c ## name
#define LCREALW lapack_complex_float
#endif

typedef std::complex<realw> crealw; 


namespace specswd {

/**
 * @brief QZ decomposition of matrix A and B
 * @param ng rows/cols of A, B
 * @param A,B two matrices, type = COMMTP
 * @param Q,Z,S,SP QZ matrix, where A = Q @ S @ Z.H, B = Q @ S' @ Z.H
 */
template<typename COMMTP=double,typename SAVETP=float>  void
schur_qz(int ng,
    Eigen::MatrixX<COMMTP> &A, 
    Eigen::MatrixX<COMMTP> &B,
    std::vector<SAVETP> &Qmat_,std::vector<SAVETP> &Zmat_,
    std::vector<SAVETP> &Smat_,std::vector<SAVETP> &Spmat_
)
{
    static_assert(std::is_same_v<SAVETP,float> || 
                    std::is_same_v<SAVETP,scmplx>);

    // resize all matrices
    Qmat_.resize(ng*ng); Zmat_.resize(ng*ng);
    Smat_.resize(ng*ng); Spmat_.resize(ng*ng);

    // run QZ
    int sdim = 0;
    if constexpr (std::is_same_v<SAVETP,float>){
        Eigen::VectorX<COMMTP> alphar(ng),alphai(ng),beta(ng);
        if constexpr (std::is_same_v<realw,float>) {
            LAPACKE_REAL(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,A.data(),ng,B.data(),ng,&sdim,alphar.data(),
                alphai.data(),beta.data(),Qmat_.data(),ng,
                Zmat_.data(),ng
            );
            memcpy(Smat_.data(),A.data(),A.size()*sizeof(A(0,0)));
            memcpy(Spmat_.data(),B.data(),B.size()*sizeof(B(0,0)));
        }
        else {
            Eigen::MatrixX<COMMTP> Q1(ng,ng),Z1(ng,ng);
            LAPACKE_REAL(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,A.data(),ng,B.data(),ng,&sdim,alphar.data(),
                alphai.data(),beta.data(),Q1.data(),ng,
                Z1.data(),ng
            );

            // copy to Smat/Spmat
            for(int j = 0; j < ng; j ++) {
            for(int i = 0; i < ng; i ++) {
                int idx = j * ng + i;
                Smat_[idx] = A(i,j);
                Spmat_[idx] = B(i,j);
                Qmat_[idx] = Q1(i,j);
                Zmat_[idx] = Z1(i,j);
            }}
        }
    }
    else { // save type is scmplex
        Eigen::VectorX<COMMTP> alpha(ng),beta(ng);

        if constexpr (std::is_same_v<realw,float>) {
            LAPACKE_CMPLX(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),
                ng,&sdim,(LCREALW*)alpha.data(),(LCREALW*)beta.data(),
                (LCREALW*)Qmat_.data(),ng,
                (LCREALW*)Zmat_.data(),ng
            );
            memcpy(Smat_.data(),A.data(),A.size()*sizeof(A(0,0)));
            memcpy(Spmat_.data(),B.data(),B.size()*sizeof(B(0,0)));
        }
        else {
            Eigen::MatrixX<COMMTP> Q1(ng,ng),Z1(ng,ng);
            LAPACKE_CMPLX(gges)(
                LAPACK_COL_MAJOR,'V','V','N',nullptr,
                ng,(LCREALW*)A.data(),ng,(LCREALW*)B.data(),
                ng,&sdim,(LCREALW*)alpha.data(),(LCREALW*)beta.data(),
                (LCREALW*)Q1.data(),ng,
                (LCREALW*)Z1.data(),ng
            );

            // copy to Smat/Spmat
            for(int j = 0; j < ng; j ++) {
            for(int i = 0; i < ng; i ++) {
                int idx = j * ng + i;
                Smat_[idx] = A(i,j);
                Spmat_[idx] = B(i,j);
                Qmat_[idx] = Q1(i,j);
                Zmat_[idx] = Z1(i,j);
            }}
        } 
    }
}
    
}

#endif