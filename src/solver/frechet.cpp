#include "solver/solver.hpp"
#include "GQTable.hpp"
#include "shared/attenuation_table.hpp"
#include <Eigen/Core>

typedef std::complex<double> dcmplx;

/**
 * @brief compute group velocity and kernels for love wave phase velocity, elastic case
 * 
 * @param freq current frequency
 * @param c  current phase velocity
 * @param displ eigen function, shape(nglob)
 * @param frekl Frechet kernels (N/L/rho) for elastic parameters, shape(3,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
double SolverSEM::
compute_love_kl(double freq,double c,const double *displ, std::vector<double> &frekl) const
{
    Eigen::Map<const Eigen::ArrayXd> x(displ,nglob);
    Eigen::Map<const Eigen::ArrayXd> K(Kmat.data(),nglob);
    Eigen::Map<const Eigen::ArrayXd> M(Mmat.data(),nglob);

    // group velocity
    // u = x^T K x / (c x^T M x)
    double u = (x * K * x).sum() / (c * (x * M * x).sum());

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    double k = om / c;
    double coef = - 0.5 * std::pow(c,3) / std::pow(om,2) / (x * K * x).sum();

    // allocate phase velocity kernels
    using namespace GQTable;
    int size = ibool_el.size();
    frekl.resize(3*size); // N/L/rho
    auto *N_kl = &frekl[0];
    auto *L_kl = &frekl[xrho_el.size()];
    auto *rho_kl = &frekl[xrho_el.size() * 2];

    // loop every element to compute kenrels
    std::array<double,NGRL> W;
    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        const double *hp = &hprime[0];
        const double *w = &wgll[0];    
        double J = jaco[ispec]; // jacobians in this layers
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == nspec) {
            hp = &hprime_grl[0];
            w = &wgrl[0];
            NGL = NGRL;
        }

        // cache displ in a element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = ibool[id];
            W[i] = displ[iglob];
        }

        // compute kernels
        for(int m = 0; m < NGL; m ++) {
            int id = id0 + m;
            rho_kl[id] = w[m] * J * std::pow(om*W[m],2);
            N_kl[id] = -std::pow(W[m],2) * w[m] * J;
            double temp = W[k] / J;

            double s{};
            for(int i = 0; i < NGL; i ++) {
                s += hp[m*NGL+i] * W[i];
            }
            L_kl[id] = - k * k * s * s * temp; 
        }
    }

    // scale all kernels
    for(int iker = 0; iker < 3; iker ++) {
        for(int i = 0; i < size; i ++) {
            frekl[iker*size+i] *= coef;
        }
    }

    // return group velocity
    return u;
}

/**
 * @brief convert d\tilde{c}/dm to dcL/dm, dQiL/dm, where \tilde{c} = c(1 + 1/2 i Qi)
 * @param dcdm, Frechet kernel for complex phase velocity, rst m
 * @param c,Qi phase velocity and phase attenutation
 * @param dcLdm,dQiLdm dc / dm and dQi / dm
 */
static void  
convert_cQ_kernels(dcmplx &dcdm,double c, double Qi,
            double &dcLdm,double &dQiLdm)
{
    dcLdm = dcdm.real();
    dQiLdm = (dcdm.imag() * 2. - Qi * dcLdm) / c;
}

/**
 * @brief compute group velocity and kernels for love wave phase velocity, visco-elastic case
 * 
 * @param freq current frequency
 * @param c  current complex phase velocity
 * @param displ eigen function, shape(nglob)
 * @param frekl_c dRe(c)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @param frekl_q d(qi)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
dcmplx SolverSEM:: 
compute_love_kl_att(double freq,dcmplx c,const dcmplx *displ, 
                    std::vector<double> &frekl_c,
                    std::vector<double> &frekl_q) const
{
    Eigen::Map<const Eigen::ArrayXcd> x(displ,nglob);
    Eigen::Map<const Eigen::ArrayXcd> K(CKmat.data(),nglob);
    Eigen::Map<const Eigen::ArrayXd> M(Mmat.data(),nglob);

    // group velocity
    dcmplx u = (x * K * x).sum() / (c * (x * M * x).sum());

    // get c_L and Qi_L  c= c_L(1 + i Qi/2)
    double c_L = c.real();
    double Qi_L = c.imag() / c_L * 2.;

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    dcmplx k = om / c;
    dcmplx coef = - 0.5 * std::pow(c,3) / std::pow(om,2) / (x * K * x).sum();

    // allocate phase velocity kernels
    using namespace GQTable;
    int size = ibool_el.size();
    frekl_c.resize(5*size); // dc_L/d(N/L/QN/QL/rho)
    frekl_q.resize(5*size); // d(Qi_L)/d(N/L/QN/QL/rho)

    // loop every element to compute kenrels
    std::array<dcmplx,NGRL> W;
    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        const double *hp = &hprime[0];
        const double *w = &wgll[0];    
        double J = jaco[ispec]; // jacobians in this layers
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == nspec) {
            hp = &hprime_grl[0];
            w = &wgrl[0];
            NGL = NGRL;
        }

        // cache displ in a element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = ibool[id];
            W[i] = displ[iglob];
        }

        // compute kernels
        for(int m = 0; m < NGL; m ++) {
            int id = id0 + m;
            dcmplx dc_drho = w[m] * J * om * W[m] * W[m];

            // get SLS factor and derivative
            dcmplx sn,dsdqni; 
            dcmplx sl,dsdqli;
            get_sls_Q_derivative(freq,xQN[id],sn,dsdqni);
            get_sls_Q_derivative(freq,xQL[id],sl,dsdqli);
            double N = xN[id], L = xL[id];

            // derivative for N/Qn^{-1}
            dcmplx dc_dN = -std::pow(W[m],2) * w[m] * J * sn;
            dcmplx dc_dQni = -std::pow(W[m],2) * w[m] * J * N * dsdqni;

            dcmplx s{};
            for(int i = 0; i < NGL; i ++) {
                s += hp[m*NGL+i] * W[i];
            }
            dcmplx dc_dL = - k * k * s * s * W[m] / J * sl; 
            dcmplx dc_dQli = - k * k * s * s * W[m] / J * L * dsdqli; 

            // scaling
            dc_drho *= coef; dc_dL *= coef;
            dc_dN *= coef; dc_dQni *= coef; dc_dQli *= coef;

            // get dcL / dparam and dQi_L / dparam
            convert_cQ_kernels(dc_dN,c_L,Qi_L,frekl_c[0*size+id],frekl_q[0*size+id]);
            convert_cQ_kernels(dc_dL,c_L,Qi_L,frekl_c[1*size+id],frekl_q[1*size+id]);
            convert_cQ_kernels(dc_dQni,c_L,Qi_L,frekl_c[2*size+id],frekl_q[2*size+id]);
            convert_cQ_kernels(dc_dQli,c_L,Qi_L,frekl_c[3*size+id],frekl_q[3*size+id]);
            convert_cQ_kernels(dc_drho,c_L,Qi_L,frekl_c[4*size+id],frekl_q[4*size+id]);
        }
    }

    return u;
}