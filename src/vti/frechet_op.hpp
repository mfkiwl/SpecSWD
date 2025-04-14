#ifndef SPECSWD_FRECHET_OP_H_
#define SPECSWD_FRECHET_OP_H_

/**
 * @brief derivative operators:
 * @note y.H @ (dA / dm - alpha dB / dm) @ x 
 */

#include "shared/attenuation.hpp"
#include "shared/GQTable.hpp"

#define GET_REAL(dc_dm,loc) frekl_r[loc*size+id1] = dc_dm.real(); \
                            frekl_i[loc*size+id1] = dc_dm.imag();

namespace specswd
{

/**
 * @brief convert df_complx/dm to df_real/dm and dfQi_dm, where f_complx = f_real (1 + 0.5 i * fQi) = f_real + i f_imag
 * @param dfdm, Frechet kernel for some quantity, rst m
 * @param f  user defiend quantity
 * @param dcLdm,dQiLdm dc / dm and dQi / dm
 */
void inline
get_fQ_kl(int npts,std::complex<float> f_cmplx,
          const float *frekl_r,
          float *__restrict frekl_i)
{
    float f_real = f_cmplx.real();
    float f_imag = f_cmplx.imag();
    float fQi = 2. * f_imag / f_real;

    for(int ipt = 0; ipt < npts; ipt ++) {
        float dQidm = (frekl_i[ipt] * 2. - fQi * frekl_r[ipt]) / f_real;
        frekl_i[ipt] = dQidm;
    }
}

// /**
//  * @brief compute coef * y^H @ d( A - alpha B)/dm_i @ x dm_i, where A = om^2 M-E, B=K
//  * @param freq current frequency
//  * @param c current phase velocity
//  * @param coef scaling factor 
//  * @param egn eigen vector,shape(nglob_el)
//  * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
//  * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+NGRL)
//  * @param jaco jacobian matrix, shape (nspec_el + 1)
//  * @param xN/xL/xQN/xQL/rho model parameters, shape(nspec_el*NGLL+NGRL)
//  * @param frekl_r real part of this operation, in which m = (N/L/rho) (elastic) or m = (N/L/QN/QL/rho) (anelstic)
//  * @param frekl_i imag part of this operation, in which nullptr or imag part of (N/L/QN/QL/rho) (anelstic)
//  */
// template<typename T = float > 
// void love_deriv_op(float freq, T c, T coef,const T *y,const T *x, 
//                     int nspec_el,int nglob_el, const int *ibool_el,
//                     const float *jaco, const float *xN,
//                     const float *xL,const float *xQN, 
//                     const float *xQL, float * __restrict frekl_r,
//                     float * __restrict frekl_i)
// {
//     // check template type
//     static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

//     using namespace GQTable;
//     std::array<T,NGRL> rW,lW;
//     size_t size = nspec_el*NGLL + NGRL;
//     T om = 2 * M_PI * freq;
//     T k2 = (om * om) / (c * c);
//     for(int ispec = 0; ispec < nspec_el + 1; ispec ++) {
//         const float *hp = &hprime[0];
//         const float *w = &wgll[0];    
//         float J = jaco[ispec]; // jacobians in this layers
//         int NGL = NGLL;
//         int id = ispec * NGLL;

//         // GRL layer
//         if(ispec == nspec_el) {
//             hp = &hprime_grl[0];
//             w = &wgrl[0];
//             NGL = NGRL;
//         }

//         // cache displ in a element
//         for(int i = 0; i < NGL; i ++) {
//             int iglob = ibool_el[id+i];
//             rW[i] = x[iglob] * coef;
//             lW[i] = y[iglob];
//             if constexpr (std::is_same_v<T,std::complex<float>>) {
//                 lW[i] = std::conj(lW[i]);
//             }
//         }

//         // compute kernels
//         T dc_drho{}, dc_dN{}, dc_dL{};
//         T dc_dqni{}, dc_dqli{};
//         T sn = 1., sl = 1.;
//         T dsdqni{}, dsdqli{};
//         for(int m = 0; m < NGL; m ++) {
//             dc_drho = w[m] * J * om * om * rW[m] * lW[m];

//             // get sls derivative if required
//             if constexpr (std::is_same_v<T,std::complex<float>>) {
//                 get_sls_Q_derivative(freq,xQN[id+m],sn,dsdqni);
//                 get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqli);
//                 dsdqni *= xN[id+m];
//                 dsdqli *= xL[id+m];
//             }

//             // N kernel
//             T temp = -k2 * rW[m] * lW[m]  * J * w[m];
//             dc_dN = temp * sn;
//             dc_dqni = temp * dsdqni;

//             // L kernel
//             T sx{},sy{};
//             for(int i = 0; i < NGL; i ++) {
//                 sx += hp[m*NGL+i] * rW[i];
//                 sy += hp[m*NGL+i] * lW[i];
//             }
//             temp = -sx * sy * w[m] / J;
//             dc_dL = temp * sl;
//             dc_dqli = temp * dsdqli;

//             // copy to frekl
//             int id1 = id + m;
//             if constexpr (std::is_same_v<T,std::complex<float>>) {
//                 GET_REAL(dc_dN,0); GET_REAL(dc_dL,1);
//                 GET_REAL(dc_dqni,2); GET_REAL(dc_dqli,3);
//                 GET_REAL(dc_drho,4);
//             }
//             else {
//                 frekl_r[0*size+id1] = dc_dN;
//                 frekl_r[1*size+id1] = dc_dL;
//                 frekl_r[2*size+id1] = dc_drho;
//             }
//         }
//     }
// }

/**
 * @brief compute y^H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * @tparam Mname which matrix you want to find derivative M/K/E = 1,2,3
 * @param freq current frequency
 * @param c_M,c_K,C_E scaling coefs for each matrix
 * @param y/x left/right eigen vector,shape(nglob_el)
 * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
 * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+NGRL)
 * @param jaco jacobian matrix, shape (nspec_el + 1)
 * @param xN/xL/xQN/xQL/rho model parameters, shape(nspec_el*NGLL+NGRL)
 * @param frekl_r dc/d(N/L/rho) (elastic) or dc/d(N/L/QN/QL/rho) (anelstic)
 * @param frekl_i nullptr or dqc/d(N/L/QN/QL/rho) (anelstic)
 */
template<typename T = float > void 
love_op_matrix(float freq,T c_M, T c_K,T c_E, const T *y,const T *x, 
               int nspec_el,int nglob_el, const int *ibool_el,
               const float *jaco, const float *xN,
               const float *xL,const float *xQN, 
               const float *xQL,float * __restrict frekl_r,
               float * __restrict frekl_i)
{
    // check template type
    static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

    using namespace GQTable;
    std::array<T,NGRL> rW,lW;
    size_t size = nspec_el*NGLL + NGRL;
    for(int ispec = 0; ispec < nspec_el + 1; ispec ++) {   
        int id = ispec * NGLL;
        float J = jaco[ispec]; // jacobians in this layers

        const bool is_gll = (ispec != nspec_el);
        const float *w = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;;

        // cache displ in a element
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id+i];
            rW[i] = x[iglob];
            lW[i] = y[iglob];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                lW[i] = std::conj(lW[i]);
            }
        }

        // compute kernels
        T dc_drho{}, dc_dN{}, dc_dL{};
        T dc_dqni{}, dc_dqli{};
        T sn = 1., sl = 1.;
        T dsdqni{}, dsdqli{};
        for(int m = 0; m < NGL; m ++) {
            dc_drho = w[m] * J * rW[m] * lW[m] * c_M;           

            // get sls derivative if required
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,xQN[id+m],sn,dsdqni);
                get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqli);
                dsdqni *= xN[id+m];
                dsdqli *= xL[id+m];
            }

            // N kernel
            T temp = rW[m] * lW[m]  * J * w[m] * c_K;
            dc_dN = temp * sn;
            dc_dqni = temp * dsdqni;

            // L kernel
            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * rW[i];
                sy += hp[m*NGL+i] * lW[i];
            }
            temp = sx * sy * w[m] / J * c_E;
            dc_dL = temp * sl;
            dc_dqli = temp * dsdqli;

            // copy to frekl
            int id1 = id + m;
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                GET_REAL(dc_dN,0); GET_REAL(dc_dL,1);
                GET_REAL(dc_dqni,2); GET_REAL(dc_dqli,3);
                GET_REAL(dc_drho,4);
            }
            else {
                frekl_r[0*size+id1] = dc_dN;
                frekl_r[1*size+id1] = dc_dL;
                frekl_r[2*size+id1] = dc_drho;
            }
        }
    }
}

/**
 * @brief compute y^H @ d(c_M * M + c_K * K + c_E * E )/dm_i @ x dm_i where
 * @param freq current frequency
 * @param c_M,c_K,C_E scaling coefs for each matrix
 * @param y/x  dot vector,shape(nglob_el*2+nglob_ac)
 * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
 * @param nspec_ac/nglob_ac mesh nelemnts/unique points for acoustic
 * @param nspec_el/ac_grl no. of GRL elements
 * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+nspec_el_grl*NGRL)
 * @param ibool_ac elastic connectivity matrix, shape(nspec_ac*NGLL+nspec_ac_grl*NGRL)
 * @param jaco jacobian matrix, shape (nspec_el + 1)
 * @param xA/xC/xL/xeta/xQA/xQC/xQL/xrho elastic model parameters,ibool_el.shape
 * @param xkappa_ac/xQk_ac/xrho_ac acoustic model parameters, ibool_ac.shape 
 * @param frekl_r dc/d(A/C/L/kappa/rho) (elastic) or dc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
 * @param frekl_i nullptr or dqc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
 */
template<typename T = float >
void
rayl_op_matrix(float freq,T c_M,T c_K, T c_E,const T *y, const T *x,
                int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,int nglob_el,
                int nglob_ac, const int *el_elmnts,const int *ac_elmnts,
                const int* ibool_el, const int* ibool_ac,
                const float *jaco,const float *xrho_el,const float *xrho_ac,
                const float *xA, const float *xC,const float *xL,const float *xeta, 
                const float *xQA, const float *xQC,const float *xQL, 
                const float *xkappa_ac, const float *xQk_ac,
                float *__restrict frekl_r,
                float *__restrict frekl_i)
{
    // check template type
    static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

    // constants
    using namespace GQTable;
    size_t size = nspec_el * NGLL + nspec_el_grl * NGRL + 
                  nspec_ac * NGLL + nspec_ac_grl * NGRL;

    // loop elastic elements
    std::array<T,NGRL> U,V,lU,lV; //left/right eigenvectors in on element
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;

        // jacobian
        float J = jaco[iel];

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *w = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache U,V and lU,lV
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            U[i] = x[iglob];
            V[i] = x[iglob + nglob_el];
            lU[i] = y[iglob]; 
            lV[i] = y[iglob + nglob_el];
            if  constexpr (std::is_same_v<T,std::complex<float>>) {
                lU[i] = std::conj(lU[i]);
                lV[i] = std::conj(lV[i]);
            }
        }

       // compute kernel
        T dc_drho{}, dc_dA{}, dc_dC{}, dc_dL{};
        T dc_deta{}, dc_dQci{},dc_dQai{},dc_dQli{};
        const T two = 2.;
        for(int m = 0; m < NGL; m ++) {
            T temp = weight[m] * J * c_M;
            dc_drho = temp * (U[m] * lU[m] + V[m] * lV[m]);

            // get sls factor if required
            T sa = 1.,sl = 1.,sc = 1.;
            T dsdqai{},dsdqci{},dsdqli{};
            T C = xC[id+m], A = xA[id+m], L = xL[id+m], eta = xeta[m];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,xQA[id+m],sa,dsdqai);
                get_sls_Q_derivative(freq,xQC[id+m],sc,dsdqci);
                get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqli);
                dsdqai *= A;
                dsdqci *= C;
                dsdqli *= L;
            }

            // K matrix
            // dc_dA
            temp = weight[m] * J * U[m] * lU[m] * c_K;
            dc_dA = temp * sa; dc_dQai = temp * dsdqai;

            // dc_dL
            temp = weight[m] * J * V[m] * lV[m] * c_K;
            dc_dL = temp * sl; dc_dQli = temp * dsdqli;

            // Ematrix
            T sx{},sy{},lsx{},lsy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * U[i];
                sy += hp[m*NGL+i] * V[i];
                lsx += hp[m*NGL+i] * lU[i];
                lsy += hp[m*NGL+i] * lV[i];
            }

            // E1
            temp = weight[m] / J * sx * lsx * c_E;
            dc_dL += temp * sl; dc_dQli += temp * dsdqli;
            
            // E3
            temp = weight[m] / J * sy * lsy * c_E;
            dc_dC = temp * sc; dc_dQci = temp * dsdqci;

            // K2,  d / dm_k sum_{ij} w_j F_j hpT(i,j) U_j lV_i 
            // = \sum_{i} w_k dF/dm_k hpT(i,k) U_k lV_i = lsy * w_k * U_k * dF/dm_k 
            temp = weight[m] * U[m] * lsy * c_K;
            dc_deta = temp * (A*sa - two*L*sl); 
            temp *= eta;
            dc_dA += temp * sa; dc_dQai += temp * dsdqai;
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            // K2, -d / dm_k sum_{ij} w_i L_i hp(i,j) U_j lV_i
            // = - \sum_{j} w_k dL/dm_k hp(j,k) U_j lV_k = -sx * w_k * lV_k dL/dm_k
            temp = -weight[m] * lV[m] * sx * c_K; 
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            //E2 \sum_{j} w_k dF/dm_k hp(j,k) V_j lU_k = -sx * w_k * lV_k dF/dm_k
            temp = weight[m] * lU[m] * sy * c_E; 
            dc_deta = temp * (A*sa - two*L*sl); 
            temp *= eta;
            dc_dA += temp * sa; dc_dQai += temp * dsdqai;
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            // E2 -lsx * w_k * V_k * dL/dm_k 
            temp = -weight[m] * V[m] * lsx * c_E;
            dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

            // copy them to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,float>) {
                frekl_r[0*size+id1] = dc_dA;
                frekl_r[1*size+id1] = dc_dC;
                frekl_r[2*size+id1] = dc_dL;
                frekl_r[3*size+id1] = dc_deta;
                frekl_r[5*size+id1] = dc_drho;
            }
            else {
                GET_REAL(dc_dA,0);
                GET_REAL(dc_dC,1);
                GET_REAL(dc_dL,2);
                GET_REAL(dc_deta,3);
                GET_REAL(dc_dQai,4);
                GET_REAL(dc_dQci,5);
                GET_REAL(dc_dQli,6);
                GET_REAL(dc_drho,9);
            }
        }
    }

    // acoustic eleemnts
    std::array<T,NGRL> chi,lchi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = ac_elmnts[ispec];
        int id = ispec * NGLL;

        // const arrays
        const bool is_gll = (ispec != nspec_ac);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // jacobians
        float J = jaco[iel];

        // cache chi and lchi in one element
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_ac[id + i];
            chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*2];
            lchi[i] = (iglob == -1) ? 0.: y[iglob+nglob_el*2];
            if  constexpr (std::is_same_v<T,std::complex<float>>) {
                lchi[i] = std::conj(lchi[i]);
            }
        }

        // derivatives
        T dc_dkappa{},dc_drho{}, dc_dqki{};
        T sk = 1., dskdqi = 0.;
        for(int m = 0; m < NGL; m ++ ){
            // copy material 
            float rho = xrho_ac[id+m];
            float kappa = xkappa_ac[id+m];
            if constexpr (std::is_same_v<T,std::complex<float>>) {
                get_sls_Q_derivative(freq,xQk_ac[id+m],sk,dskdqi);
                dskdqi *= kappa;
            }
            
            // kappa kernel
            T temp = -c_M * weight[m]* J*chi[m] * lchi[m] / (sk * kappa) / (sk * kappa);
            dc_dkappa = temp * sk;
            dc_dqki =  temp * dskdqi;

            dc_drho = - c_K * weight[m]* J * chi[m] * lchi[m] / rho / rho; 

            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * chi[i];
                sy += hp[m*NGL+i] * lchi[i];
            }
            dc_drho += -c_E * weight[m] / J / (rho*rho) * sx * sy;

            // copy to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,float>) {
                frekl_r[4*size+id1] = dc_dkappa;
                frekl_r[5*size+id1] = dc_drho;
            }
            else {
                GET_REAL(dc_dkappa,7);
                GET_REAL(dc_dqki,8);
                GET_REAL(dc_drho,9);
            }
        }
    }
}

// /**
//  * @brief compute coef * y^dag @ d( (w^2 M -E) - k^2 K)/dm_i @ x dm_i
//  * @param freq current frequency
//  * @param c current phase velocity
//  * @param coef derivative scaling coefs 
//  * @param y/x  dot vector,shape(nglob_el*2+nglob_ac)
//  * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
//  * @param nspec_ac/nglob_ac mesh nelemnts/unique points for acoustic
//  * @param nspec_el/ac_grl no. of GRL elements
//  * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+nspec_el_grl*NGRL)
//  * @param ibool_ac elastic connectivity matrix, shape(nspec_ac*NGLL+nspec_ac_grl*NGRL)
//  * @param jaco jacobian matrix, shape (nspec_el + 1)
//  * @param xA/xC/xL/xeta/xQA/xQC/xQL/xrho elastic model parameters,ibool_el.shape
//  * @param xkappa_ac/xQk_ac/xrho_ac acoustic model parameters, ibool_ac.shape 
//  * @param frekl_r dc/d(A/C/L/kappa/rho) (elastic) or dc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
//  * @param frekl_i nullptr or dqc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
//  * @note frekl_r and frekl_i should be set to 0 before calling this routine
//  */
// template<typename T = float >
// void
// rayl_deriv_op(float freq,T c,T coef,const T *y, const T *x,
//                 int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,int nglob_el,
//                 int nglob_ac, const int *el_elmnts,const int *ac_elmnts,
//                 const int* ibool_el, const int* ibool_ac,
//                 const float *jaco,const float *xrho_el,const float *xrho_ac,
//                 const float *xA, const float *xC,const float *xL,const float *xeta, 
//                 const float *xQA, const float *xQC,const float *xQL, 
//                 const float *xkappa_ac, const float *xQk_ac,
//                 float *__restrict frekl_r,
//                 float *__restrict frekl_i)
// {
//     // check template type
//     static_assert(std::is_same_v<float,T> || std::is_same_v<std::complex<float>,T>);

//     // constants
//     using namespace GQTable;
//     size_t size = nspec_el *  NGLL + nspec_el_grl * NGRL + 
//                   nspec_ac * NGLL + nspec_ac_grl * NGRL;
//     T om = 2 * M_PI * freq;
//     T k2 = std::pow(om / c,2);

//     // loop elastic elements
//     std::array<T,NGRL> U,V,lU,lV;
//     for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
//         int iel = el_elmnts[ispec];
//         int id = ispec * NGLL;

//         const float *weight = wgll.data();
//         const float *hp = hprime.data();
//         int NGL = NGLL;

//         // jacobian
//         float J = jaco[iel];

//         // grl case
//         if(ispec == nspec_el) {
//             weight = wgrl.data();
//             hp = hprime_grl.data();
//             NGL = NGRL;
//         }  

//         // cache U,V and lU,lV
//         for(int i = 0; i < NGL; i ++) {
//             int iglob = ibool_el[id + i];
//             U[i] = x[iglob] * coef; 
//             V[i] = x[iglob + nglob_el] * coef;
//             lU[i] = y[iglob]; 
//             lV[i] = y[iglob + nglob_el];
//             if  constexpr (std::is_same_v<T,std::complex<float>>) {
//                 lU[i] = std::conj(lU[i]);
//                 lV[i] = std::conj(lV[i]);
//             }
//         }

//        // compute kernel
//         T dc_drho{}, dc_dA{}, dc_dC{}, dc_dL{};
//         T dc_deta{}, dc_dQci{},dc_dQai{},dc_dQli{};
//         const T two = 2.;
//         for(int m = 0; m < NGL; m ++) {
//             T temp = weight[m] * J;
//             dc_drho = temp * om * om * 
//                 (U[m] * lU[m] + V[m] * lV[m]);

//             // get sls factor if required
//             T sa = 1.,sl = 1.,sc = 1.;
//             T dsdqai{},dsdqci{},dsdqli{};
//             float C = xC[id+m], A = xA[id+m],
//                    L = xL[id+m], eta = xeta[m];
//             if constexpr (std::is_same_v<T,std::complex<float>>) {
//                 get_sls_Q_derivative(freq,xQA[id+m],sa,dsdqai);
//                 get_sls_Q_derivative(freq,xQC[id+m],sc,dsdqai);
//                 get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqai);
//                 dsdqai *= A;
//                 dsdqci *= C;
//                 dsdqli *= L;
//             }

//             // K matrix
//             // dc_dA
//             temp = -weight[m] * J * k2 * U[m] * lU[m];
//             dc_dA = temp * sa; dc_dQai = temp * dsdqai;

//             // dc_dL
//             temp = -weight[m] * J * k2 * V[m] * lV[m];
//             dc_dL = temp * sl; dc_dQli = temp * dsdqli;

//             // Ematrix
//             T sx{},sy{},lsx{},lsy{};
//             for(int i = 0; i < NGL; i ++) {
//                 sx += hp[m*NGL+i] * U[i];
//                 sy += hp[m*NGL+i] * V[i];
//                 lsx += hp[m*NGL+i] * lU[i];
//                 lsy += hp[m*NGL+i] * lV[i];
//             }
//             temp = -weight[m] / J * sx * lsx;
//             dc_dL += temp * sl; dc_dQli += temp * dsdqli;

//             temp = - weight[m] / J * sy * lsy;
//             dc_dC = temp * sc; dc_dQci = temp * dsdqci;

//             // eta
//             temp = - (k2 * weight[m] * U[m] * lsy + 
//                         weight[m] * lU[m] * sy);
//             dc_deta = temp * (A*sa - two*L*sl);

//             temp *= eta;
//             dc_dA += temp * sa; dc_dQai += temp * dsdqai;
//             dc_dL += - temp * two * sl; dc_dQli += -temp * two * dsdqli;

//             temp = k2 * weight[m] * lV[m] * sx + weight[m] * V[m] * lsx;
//             temp *= coef;
//             dc_dL +=  temp * sl;
//             dc_dQli += temp * dsdqli;

//             // copy them to frekl
//             int id1 = iel * NGLL + m;
//             if constexpr (std::is_same_v<T,float>) {
//                 frekl_r[0*size+id1] = dc_dA;
//                 frekl_r[1*size+id1] = dc_dC;
//                 frekl_r[2*size+id1] = dc_dL;
//                 frekl_r[3*size+id1] = dc_deta;
//                 frekl_r[5*size+id1] = dc_drho;
//             }
//             else {
//                 GET_REAL(dc_dA,0);
//                 GET_REAL(dc_dC,1);
//                 GET_REAL(dc_dL,2);
//                 GET_REAL(dc_deta,3);
//                 GET_REAL(dc_dQai,4);
//                 GET_REAL(dc_dQci,5);
//                 GET_REAL(dc_dQli,6);
//                 GET_REAL(dc_drho,9);
//             }
//         }
//     }

//     // acoustic eleemnts
//     std::array<T,NGRL> chi,lchi;
//     for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
//         int iel = ac_elmnts[ispec];
//         int id = ispec * NGLL;
//         const float *weight = wgll.data();
//         const float *hp = hprime.data();
//         int NGL = NGLL;

//         // jacobians
//         float J = jaco[iel];

//         // grl case
//         if(ispec == nspec_ac) {
//             weight = wgrl.data();
//             hp = hprime_grl.data();
//             NGL = NGRL;
//         }   

//         // cache chi and lchi in one element
//         for(int i = 0; i < NGL; i ++) {
//             int iglob = ibool_ac[id + i];
//             chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*2] * coef;
//             lchi[i] = (iglob == -1) ? 0.: y[iglob+nglob_el*2];
//             if  constexpr (std::is_same_v<T,std::complex<float>>) {
//                 lchi[i] = std::conj(lchi[i]);
//             }
//         }

//         // derivatives
//         T dc_dkappa{},dc_drho{}, dc_dqki{};
//         T sk = 1., dskdqi = 0.;
//         for(int m = 0; m < NGL; m ++ ){
//             // copy material 
//             float rho = xrho_ac[id+m];
//             float kappa = xkappa_ac[id+m];
//             if constexpr (std::is_same_v<T,std::complex<float>>) {
//                 get_sls_Q_derivative(freq,xQk_ac[id+m],sk,dskdqi);
//                 dskdqi *= kappa;
//             }
            
//             // kappa kernel
//             T temp = std::pow(om/(sk * kappa),2) *weight[m]* J*
//                         chi[m] * lchi[m];
//             dc_dkappa = temp * sk;
//             dc_dqki =  temp * dskdqi;

//             dc_drho = -k2 * std::pow(om/rho,2) *weight[m]* J*
//                         chi[m] * lchi[m]; 

//             T sx{},sy{};
//             for(int i = 0; i < NGL; i ++) {
//                 sx += hp[m*NGL+i] * chi[i];
//                 sy += hp[m*NGL+i] * lchi[i];
//             }
//             dc_drho += weight[m] / J / (rho*rho) * sx * sy;

//             // copy to frekl
//             int id1 = iel * NGLL + m;
//             if constexpr (std::is_same_v<T,float>) {
//                 frekl_r[4*size+id1] = dc_dkappa;
//                 frekl_r[5*size+id1] = dc_drho;
//             }
//             else {
//                 GET_REAL(dc_dkappa,7);
//                 GET_REAL(dc_dqki,8);
//                 GET_REAL(dc_drho,9);
//             }
//         }
//     }
// }

    
} // namespace specswd

#undef GET_REAL

#endif