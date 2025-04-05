#ifndef SPECSWD_FRECHET_H_
#define SPECSWD_FRECHET_H_

#include "shared/attenuation.hpp"
#include "GQTable.hpp"


/**
 * @brief convert d\tilde{c}/dm to dcL/dm, dQiL/dm, where \tilde{c} = c(1 + 1/2 i Qi)
 * @param dcdm, Frechet kernel for complex phase velocity, rst m
 * @param c  complex phase velocity
 * @param dcLdm,dQiLdm dc / dm and dQi / dm
 */
template <typename T>  void 
get_cQ_kl(T &dcdm,T c,
          double &dcLdm,double &dQiLdm)
{   
    static_assert(std::is_same_v<std::complex<double>,T>);
    double cl = c.real();
    double Qi = 2. * c.imag() / cl;
    dcLdm += dcdm.real();
    dQiLdm += (dcdm.imag() * 2. - Qi * dcLdm) / cl;
}

/**
 * @brief compute coef * y^T @ d( (w^2 M -E) - k^ K)/dm_i @ x dm_i
 * @param freq current frequency
 * @param c current phase velocity
 * @param coef scaling coefs
 * @param egn eigen vector,shape(nglob_el)
 * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
 * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+NGRL)
 * @param jaco jacobian matrix, shape (nspec_el + 1)
 * @param xN/xL/xQN/xQL/rho model parameters, shape(nspec_el*NGLL+NGRL)
 * @param frekl_c dc/d(N/L/rho) (elastic) or dc/d(N/L/QN/QL/rho) (anelstic)
 * @param frekl_q nullptr or dqc/d(N/L/QN/QL/rho) (anelstic)
 * @note frekl_c and frekl_q should be set to 0 before calling this routine
 */
template<typename T = double > 
void get_deriv_love_(double freq, T c, T coef,const T *y,const T *x, 
                    int nspec_el,int nglob_el, const int *ibool_el,
                    const double *jaco, const double *xN,
                    const double *xL,const double *xQN, 
                    const double *xQL, double * __restrict frekl_c,
                    double * __restrict frekl_q)
{
    // check template type
    static_assert(std::is_same_v<double,T> || std::is_same_v<std::complex<double>,T>);

    using namespace GQTable;
    std::array<T,NGRL> rW,lW;
    size_t size = nspec_el*NGLL + NGRL;
    double om = 2 * M_PI * freq;
    T k2 = (om * om) / (c * c);
    for(int ispec = 0; ispec < nspec_el + 1; ispec ++) {
        const double *hp = &hprime[0];
        const double *w = &wgll[0];    
        double J = jaco[ispec]; // jacobians in this layers
        int NGL = NGLL;
        int id = ispec * NGLL;

        // GRL layer
        if(ispec == nspec_el) {
            hp = &hprime_grl[0];
            w = &wgrl[0];
            NGL = NGRL;
        }

        // cache displ in a element
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id+i];
            rW[i] = x[iglob];
            lW[i] = y[iglob];
        }

        // compute kernels
        T dc_drho{}, dc_dN{}, dc_dL{};
        T dc_dqni{}, dc_dqli{};
        T sn = 1., sl = 1.;
        T dsdqni{}, dsdqli{};
        for(int m = 0; m < NGL; m ++) {
            dc_drho = w[m] * J * om * om * rW[m] * lW[m] * coef;

            // get sls derivative if required
            if constexpr (std::is_same_v<T,std::complex<double>>) {
                get_sls_Q_derivative(freq,xQN[id+m],sn,dsdqni);
                get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqli);
                dsdqni *= xN[id+m];
                dsdqli *= xL[id+m];
            }

            // N kernel
            T temp = -k2 * rW[m] * lW[m]  * J * w[m] * coef;
            dc_dN = temp * sn;
            dc_dqni = temp * dsdqni;

            // L kernel
            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * rW[i];
                sy += hp[m*NGL+i] * lW[i];
            }
            temp = -sx * sy * w[m] / J * coef;
            dc_dL = temp * sl;
            dc_dqli = temp * dsdqli;

            // copy to frekl
            int id1 = id + m;
            if constexpr (std::is_same_v<T,std::complex<double>>) {
                get_cQ_kl(dc_dN,c,frekl_c[0*size+id1],frekl_q[0*size+id1]);
                get_cQ_kl(dc_dL,c,frekl_c[1*size+id1],frekl_q[1*size+id1]);
                get_cQ_kl(dc_dqni,c,frekl_c[2*size+id1],frekl_q[2*size+id1]);
                get_cQ_kl(dc_dqli,c,frekl_c[3*size+id1],frekl_q[3*size+id1]);
                get_cQ_kl(dc_drho,c,frekl_c[4*size+id1],frekl_q[4*size+id1]);
            }
            else {
                frekl_c[0*size+id1] += dc_dN;
                frekl_c[1*size+id1] += dc_dL;
                frekl_c[2*size+id1] += dc_drho;
            }
        }
    }
}


/**
 * @brief compute coef * y^dag @ d( (w^2 M -E) - k^2 K)/dm_i @ x dm_i
 * @param freq current frequency
 * @param c current phase velocity
 * @param coef derivative scaling coefs 
 * @param y/x  dot vector,shape(nglob_el*2+nglob_ac)
 * @param nspec_el/nglob_el mesh nelemnts/unique points for elastic
 * @param nspec_ac/nglob_ac mesh nelemnts/unique points for acoustic
 * @param nspec_el/ac_grl no. of GRL elements
 * @param ibool_el elastic connectivity matrix, shape(nspec_el*NGLL+nspec_el_grl*NGRL)
 * @param ibool_ac elastic connectivity matrix, shape(nspec_ac*NGLL+nspec_ac_grl*NGRL)
 * @param jaco jacobian matrix, shape (nspec_el + 1)
 * @param xA/xC/xL/xeta/xQA/xQC/xQL/xrho elastic model parameters,ibool_el.shape
 * @param xkappa_ac/xQk_ac/xrho_ac acoustic model parameters, ibool_ac.shape 
 * @param frekl_c dc/d(A/C/L/kappa/rho) (elastic) or dc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
 * @param frekl_q nullptr or dqc/d(A/C/L/QA/QC/QL/kappa/Qk/rho) (anelstic)
 * @note frekl_c and frekl_q should be set to 0 before calling this routine
 */
template<typename T = double >
void
get_deriv_rayl_(double freq,T c,T coef,const T *y, const T *x,
                int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,int nglob_el,
                int nglob_ac, const int *el_elmnts,const int *ac_elmnts,
                const int* ibool_el, const int* ibool_ac,
                const double *jaco,const double *xrho_el,const double *xrho_ac,
                const double *xA, const double *xC,const double *xL,const double *xeta, 
                const double *xQA, const double *xQC,const double *xQL, 
                const double *xkappa_ac, const double *xQk_ac,
                double *__restrict frekl_c,
                double *__restrict frekl_q)
{
    // check template type
    static_assert(std::is_same_v<double,T> || std::is_same_v<std::complex<double>,T>);

    // constants
    using namespace GQTable;
    size_t size = nspec_el *  NGLL + nspec_el_grl * NGRL + 
                  nspec_ac * NGLL + nspec_ac_grl * NGRL;
    double om = 2 * M_PI * freq;
    T k2 = std::pow(om / c,2);

    // loop elastic elements
    std::array<T,NGRL> U,V,lU,lV;
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;

        const double *weight = wgll.data();
        const double *hp = hprime.data();
        int NGL = NGLL;

        // jacobian
        double J = jaco[iel];

        // grl case
        if(ispec == nspec_el) {
            weight = wgrl.data();
            hp = hprime_grl.data();
            NGL = NGRL;
        }  

        // cache U,V and lU,lV
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            U[i] = x[iglob]; 
            V[i] = x[iglob + nglob_el];
            lU[i] = y[iglob]; 
            lV[i] = y[iglob + nglob_el];
            if  constexpr (std::is_same_v<T,std::complex<double>>) {
                lU[i] = std::conj(lU[i]);
                lV[i] = std::conj(lV[i]);
            }
        }

       // compute kernel
        T dc_drho{}, dc_dA{}, dc_dC{}, dc_dL{};
        T dc_deta{}, dc_dQci{},dc_dQai{},dc_dQli{};
        for(int m = 0; m < NGL; m ++) {
            T temp = weight[m] * J * coef;
            dc_drho = temp * om * om * 
                (U[m] * lU[m] + V[m] * lV[m]);

            // get sls factor if required
            T sa = 1.,sl = 1.,sc = 1.;
            T dsdqai{},dsdqci{},dsdqli{};
            double C = xC[id+m], A = xA[id+m],
                   L = xL[id+m], eta = xeta[m];
            if constexpr (std::is_same_v<T,std::complex<double>>) {
                get_sls_Q_derivative(freq,xQA[id+m],sa,dsdqai);
                get_sls_Q_derivative(freq,xQC[id+m],sc,dsdqai);
                get_sls_Q_derivative(freq,xQL[id+m],sl,dsdqai);
                dsdqai *= A;
                dsdqci *= C;
                dsdqli *= L;
            }

            // K matrix
            // dc_dA
            temp = -weight[m] * J * k2 * U[m] * lU[m] * coef;
            dc_dA = temp * sa; dc_dQai = temp * dsdqai;

            // dc_dL
            temp = -weight[m] * J * k2 * V[m] * lV[m] * coef;
            dc_dL = temp * sl; dc_dQli = temp * dsdqli;

            // Ematrix
            T sx{},sy{},lsx{},lsy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * U[i];
                sy += hp[m*NGL+i] * V[i];
                lsx += hp[m*NGL+i] * lU[i];
                lsy += hp[m*NGL+i] * lV[i];
            }
            temp = -weight[m] / J * sx * lsx * coef ;
            dc_dL += temp * sl; dc_dQli += temp * dsdqli;

            temp = - weight[m] / J * sy * lsy * coef;
            dc_dC = temp * sc; dc_dQci = temp * dsdqci;

            // eta
            temp = - (k2 * weight[m] * U[m] * lsy + 
                        weight[m] * lU[m] * sy) * coef;
            dc_deta = temp * (A*sa - 2.*L*sl);

            temp *= eta;
            dc_dA += temp * sa; dc_dQai += temp * dsdqai;
            dc_dL += - temp * 2. * sl; dc_dQli += -temp * 2. * dsdqli;

            temp = k2 * weight[m] * lV[m] * sx + weight[m] * V[m] * lsx;
            temp *= coef;
            dc_dL +=  temp * sl;
            dc_dQli += temp * dsdqli;

            // copy them to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,double>) {
                frekl_c[0*size+id1] += dc_dA;
                frekl_c[1*size+id1] += dc_dC;
                frekl_c[2*size+id1] += dc_dL;
                frekl_c[3*size+id1] += dc_deta;
                frekl_c[5*size+id1] += dc_drho;
            }
            else {
                get_cQ_kl(dc_dA,c,frekl_c[0*size+id1],frekl_q[0*size+id1]);
                get_cQ_kl(dc_dC,c,frekl_c[1*size+id1],frekl_q[1*size+id1]);
                get_cQ_kl(dc_dL,c,frekl_c[2*size+id1],frekl_q[2*size+id1]);
                get_cQ_kl(dc_deta,c,frekl_c[3*size+id1],frekl_q[3*size+id1]);
                get_cQ_kl(dc_dQai,c,frekl_c[4*size+id1],frekl_q[4*size+id1]);
                get_cQ_kl(dc_dQci,c,frekl_c[5*size+id1],frekl_q[5*size+id1]);
                get_cQ_kl(dc_dQli,c,frekl_c[6*size+id1],frekl_q[6*size+id1]);
                get_cQ_kl(dc_drho,c,frekl_c[9*size+id1],frekl_q[9*size+id1]);
            }
        }
    }

    // acoustic eleemnts
    std::array<T,NGRL> chi,lchi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = ac_elmnts[ispec];
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hp = hprime.data();
        int NGL = NGLL;

        // jacobians
        double J = jaco[iel];

        // grl case
        if(ispec == nspec_ac) {
            weight = wgrl.data();
            hp = hprime_grl.data();
            NGL = NGRL;
        }   

        // cache chi and lchi in one element
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_ac[id + i];
            chi[i] = (iglob == -1) ? 0: x[iglob+nglob_el*2];
            lchi[i] = (iglob == -1) ? 0.: y[iglob+nglob_el*2];
            if  constexpr (std::is_same_v<T,std::complex<double>>) {
                lchi[i] = std::conj(lchi[i]);
            }
        }

        // derivatives
        T dc_dkappa{},dc_drho{}, dc_dqki{};
        T sk = 1., dskdqi = 0.;
        for(int m = 0; m < NGL; m ++ ){
            // copy material 
            double rho = xrho_ac[id+m];
            double kappa = xkappa_ac[id+m];
            if constexpr (std::is_same_v<T,std::complex<double>>) {
                get_sls_Q_derivative(freq,xQk_ac[id+m],sk,dskdqi);
                dskdqi *= kappa;
            }
            
            // kappa kernel
            T temp = std::pow(om/(sk * kappa),2) *weight[m]* J*
                        chi[m] * lchi[m] * coef;
            dc_dkappa = temp * sk;
            dc_dqki =  temp * dskdqi;

            dc_drho = -k2 * std::pow(om/rho,2) *weight[m]* J*
                        chi[m] * lchi[m] * coef; 

            T sx{},sy{};
            for(int i = 0; i < NGL; i ++) {
                sx += hp[m*NGL+i] * chi[i];
                sy += hp[m*NGL+i] * lchi[i];
            }
            dc_drho += weight[m] / J / (rho*rho) * sx * sy * coef;

            // copy to frekl
            int id1 = iel * NGLL + m;
            if constexpr (std::is_same_v<T,double>) {
                frekl_c[4*size+id1] = dc_dkappa;
                frekl_c[5*size+id1] = dc_drho;
            }
            else {
                get_cQ_kl(dc_dkappa,c,frekl_c[7*size+id1],frekl_q[7*size+id1]);
                get_cQ_kl(dc_dqki,c,frekl_c[8*size+id1],frekl_q[8*size+id1]);
                get_cQ_kl(dc_drho,c,frekl_c[9*size+id1],frekl_q[9*size+id1]);
            }

        }
    }
}

#endif