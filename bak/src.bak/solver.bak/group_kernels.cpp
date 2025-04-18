#include "solver/solver.hpp"
#include "solver/frechet.hpp"

#include <Eigen/Dense>

void SolverSEM:: 
compute_love_group_kl(double freq,double c,const double *displ,
                    std::vector<double> &frekl) const
{
    int ng = nglob_el;
    Eigen::Map<const Eigen::ArrayXd> x(displ,ng);
    Eigen::Map<const Eigen::ArrayXd> K(Kmat.data(),ng);
    Eigen::Map<const Eigen::ArrayXd> M(Mmat.data(),ng);

    // set frekl to 0
    frekl.resize(3*ibool_el.size(),0);

    // group velocity
    // du / alpha = du/dc * dc/dalpha
    double om = 2 * M_PI * freq;
    double k2 = std::pow(om/c,2);
    double xTKx = (x * K * x).sum(), xTMx = (x * M * x).sum();
    double du_dalpha = -xTKx / (c*c * xTMx);
    du_dalpha *= -0.5 * std::pow(c,3) / std::pow(om,2) / xTKx;

    // du /dx
    Eigen::VectorXd du_dx = 2. * K * x / (c * xTMx) - 
                           2. * xTKx * M * x / std::pow(c*xTMx,2);
    
    // mapping Q,Z,S,Sp
    Eigen::Map<const Eigen::MatrixXd> Q(Qmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXd> Z(Zmat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXd> S(Smat_.data(),ng,ng);
    Eigen::Map<const Eigen::MatrixXd> Sp(Spmat_.data(),ng,ng);

    // solve (A.T - k^2 B.T) lambda = du_dx
    du_dx = Z.transpose() * du_dx;
    Eigen::VectorXd lamb = (S - k2 * Sp).eval().triangularView<Eigen::Upper>().
                            solve(du_dx);
    lamb = Q * lamb;

    // - lambda^T (dA / dm_i - k^2 dB / dm) x
    get_deriv_love_(freq,c,-1.,lamb.data(),displ,nspec_el,
                    nglob_el,ibool_el.data(),jaco.data(),
                    nullptr,nullptr,nullptr,nullptr,
                    frekl.data(),nullptr);
    
    // c1+ c2
    double c12 = - (du_dalpha + (lamb.array() * K * x).sum()) / xTKx;

    // -(c1+c2) x^T (dA / dm_i - k^2 dB / dm) x
    get_deriv_love_(freq,c,-c12,displ,displ,nspec_el,
                    nglob_el,ibool_el.data(),jaco.data(),
                    nullptr,nullptr,nullptr,nullptr,
                    frekl.data(),nullptr);
}