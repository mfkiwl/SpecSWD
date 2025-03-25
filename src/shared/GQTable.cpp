#include "GQTable.hpp"
#include "shared/quadrature.hpp"

namespace GQTable
{

std::array<double,NGLL> xgll,wgll;
std::array<double,NGRL> xgrl,wgrl;
std::array<double,NGLL*NGLL> hprimeT,hprime; // hprimeT(i,j) = l'_i(xi_j)
std::array<double,NGRL*NGRL> hprimeT_grl,hprime_grl;

/**
 * @brief initialize GLL/GRL nodes/weights
 * 
 */
void initialize()
{
    // GLL nodes/weights
    gauss_legendre_lobatto(xgll.data(),wgll.data(),NGLL);

    // compute hprime and hprimeT
    double poly[NGLL];
    for(int i = 0; i < NGLL; i ++) {
        double xi = xgll[i];
        lagrange_poly(xi,NGLL,xgll.data(),poly,&hprime[i*NGLL]);
    }
    for(int i = 0; i < NGLL; i ++) {
    for(int j = 0; j < NGLL; j ++) {
        hprimeT[i * NGLL + j] = hprime[j * NGLL + i];
    }}


    // GRL nodes/weights
    gauss_radau_laguerre(xgrl.data(),wgrl.data(),NGRL);

    // compute hprime_grl and hprimeT_grl
    for(int i = 0; i < NGRL; i ++) {
    for(int j = 0; j < NGRL; j ++) {
        int id = i * NGRL + j;
        if(i != j) {
            hprimeT_grl[id] = laguerre_func(NGRL,xgrl[j]) / laguerre_func(NGRL,xgrl[i]) / (xgrl[j] - xgrl[i]);
        }
        else if(i == j && i == 0) {
            hprimeT_grl[id] = -NGRL / 2.;
        }
        else {
            hprimeT_grl[id] = 0.;
        }
    }} 
    for(int i = 0; i < NGRL; i ++) {
    for(int j = 0; j < NGRL; j ++) {
        int id = i * NGRL + j;
        hprime_grl[j * NGRL + i] = hprimeT_grl[id];
    }}
}
    
} // namespace GQTable
