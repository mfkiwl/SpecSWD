#include "solver/solver.hpp"
#include "solver/frechet.hpp"

#include <Eigen/Core>

typedef std::complex<double> dcmplx;


/**
 * @brief compute group velocity and kernels for love wave phase velocity, elastic case
 * 
 * @param freq current frequency
 * @param c  current phase velocity
 * @param displ eigen function, shape(nglob_el)
 * @param frekl Frechet kernels (N/L/rho) for elastic parameters, shape(3,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
double SolverSEM::
compute_love_kl(double freq,double c,const double *displ, std::vector<double> &frekl) const
{
    int ng = nglob_el;
    Eigen::Map<const Eigen::ArrayXd> x(displ,ng);
    Eigen::Map<const Eigen::ArrayXd> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::ArrayXd> M(Mmat.data(),ng);

    // group velocity
    // u = x^T K x / (c x^T M x)
    double u = (x * K * x).sum() / (c * (x * M * x).sum());

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    double coef = -0.5 * std::pow(c,3) / std::pow(om,2) / (x * K * x).sum();

    // allocate phase velocity kernels
    int size = ibool_el.size();
    frekl.resize(3*size,0); // N/L/rho
    get_deriv_love_(freq,c,coef,displ,displ,nspec_el,nglob_el,
                    ibool_el.data(),jaco.data(),
                    nullptr,nullptr,nullptr,nullptr,
                    frekl.data(),nullptr);

    // return group velocity
    return u;
}



/**
 * @brief compute group velocity and kernels for love wave phase velocity, visco-elastic case
 * 
 * @param freq current frequency
 * @param c  current complex phase velocity
 * @param displ eigen function, shape(nglob_el)
 * @param frekl_c dRe(c)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @param frekl_q d(qi)/d(N/L/QN/QL/rho) shape(5,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
dcmplx SolverSEM:: 
compute_love_kl_att(double freq,dcmplx c,const dcmplx *displ, 
                    std::vector<double> &frekl_c,
                    std::vector<double> &frekl_q) const
{
    int ng = nglob_el;
    Eigen::Map<const Eigen::ArrayXcd> x(displ,ng);
    Eigen::Map<const Eigen::ArrayXcd> K(CKmat.data(),ng);
    Eigen::Map<const Eigen::ArrayXd> M(Mmat.data(),ng);

    // group velocity
    dcmplx u = (x * K * x).sum() / (c * (x * M * x).sum());

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    dcmplx coef = -0.5 * std::pow(c,3) / std::pow(om,2) / (x * K * x).sum();

    // allocate phase velocity kernels
    int size = ibool_el.size();
    frekl_c.resize(5*size,0); // dc_L/d(N/L/QN/QL/rho)
    frekl_q.resize(5*size,0); // d(Qi_L)/d(N/L/QN/QL/rho)

    // get kernels
    get_deriv_love_(freq,c,coef,displ,displ,nspec_el,nglob_el,ibool_el.data(),
                    jaco.data(),xN.data(),xL.data(),
                    xQN.data(),xQL.data(),frekl_c.data(),
                    frekl_q.data());

    return u;
}


/**
 * @brief compute group velocity and kernels for love wave
 * 
 * @param freq current frequency
 * @param c  current phase velocity
 * @param displ eigen function, shape(nglob * 2)
 * @param frekl Frechet kernels A/C/L/eta/rho_kl kernels for elastic parameters, shape(5,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
double SolverSEM:: 
compute_rayl_kl(double freq,double c,const double *displ, 
                const double *ldispl, std::vector<double> &frekl) const
{
    int ng = nglob_el*2 + nglob_ac;
    typedef Eigen::Matrix<double,-1,-1,Eigen::RowMajor> dmat2;
    Eigen::Map<const Eigen::VectorXd> x(displ,ng),y(ldispl,ng);
    Eigen::Map<const dmat2> K(Kmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorXd> M(Mmat.data(),ng);

    // group velocity
    // u = x^T K x / (c x^T M x)
    double u_nume = (y.transpose() * K * x);
    double u_deno = c* y.transpose() * M.asDiagonal() * x;
    double u = u_nume / u_deno;

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    double coef = - 0.5 * std::pow(c,3) / std::pow(om,2) / (y.transpose() * K * x).sum();

    // allocate phase velocity kernels
    using namespace GQTable;
    int size = ibool.size();
    frekl.resize(6*size,0); // A/C/L/eta/kappa_ac/rho_kl

    // compute phase kernels
    get_deriv_rayl_(freq,c,coef,ldispl,displ,nspec_el,nspec_ac,
                    nspec_el_grl,nspec_ac_grl,nglob_el,nglob_ac,
                    el_elmnts.data(),ac_elmnts.data(),
                    ibool_el.data(),ibool_ac.data(),jaco.data(),
                    xrho_el.data(),xrho_ac.data(),xA.data(),xC.data(),
                    xL.data(),xeta.data(),xQA.data(),xQC.data(),
                    xQL.data(),xkappa_ac.data(),xQk_ac.data(),
                    frekl.data(),nullptr);

    return u;
}



/**
 * @brief compute group velocity and kernels for love wave
 * 
 * @param freq current frequency
 * @param c  current phase velocity
 * @param displ eigen function, shape(nglob * 2)
 * @param frekl Frechet kernels A/C/L/eta/rho_kl kernels for elastic parameters, shape(5,nspec*NGLL + NGRL) 
 * @return double u group velocity
 */
dcmplx SolverSEM:: 
compute_rayl_kl_att(double freq,dcmplx c,const dcmplx *displ, 
                const dcmplx *ldispl, std::vector<double> &frekl_c,
                std::vector<double> &frekl_q) const
{
    int ng = nglob_el*2 + nglob_ac;
    typedef Eigen::Matrix<dcmplx,-1,-1,Eigen::RowMajor> dcmat2;
    Eigen::Map<const Eigen::VectorXcd> x(displ,ng),y(ldispl,ng);
    Eigen::Map<const dcmat2> K(CKmat.data(),ng,ng);
    Eigen::Map<const Eigen::VectorXcd> M(CMmat.data(),ng);

    // group velocity
    // u = x^T K x / (c x^T M x)
    dcmplx u_nume = (y.adjoint() * K * x);
    dcmplx u_deno = c* y.adjoint() * M.asDiagonal() * x;
    dcmplx u = u_nume / u_deno;

    // coefs for phase kernel
    double om = 2 * M_PI * freq;
    dcmplx coef = - 0.5 * std::pow(c,3) / std::pow(om,2) / (y.adjoint() * K * x).sum();

    // allocate phase velocity kernels
    using namespace GQTable;
    int size = ibool.size();
    frekl_c.resize(10*size,0); // A/C/L/eta/kappa_ac/rho_kl
    frekl_q.resize(10*size,0); // A/C/L/eta/kappa_ac/rho_kl

    // compute phase kernels
    get_deriv_rayl_(freq,c,coef,ldispl,displ,nspec_el,nspec_ac,
                    nspec_el_grl,nspec_ac_grl,nglob_el,nglob_ac,
                    el_elmnts.data(),ac_elmnts.data(),
                    ibool_el.data(),ibool_ac.data(),jaco.data(),
                    xrho_el.data(),xrho_ac.data(),xA.data(),xC.data(),
                    xL.data(),xeta.data(),xQA.data(),xQC.data(),
                    xQL.data(),xkappa_ac.data(),xQk_ac.data(),
                    frekl_c.data(),frekl_q.data());
    
    return u;
}