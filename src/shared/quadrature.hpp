#ifndef SPECSWD_QUADRATURE_H_
#define SPECSWD_QUADRATURE_H_
#include <cmath>

//GLL
void gauss_legendre_lobatto(double* knots, double* weights, size_t length);
void lagrange_poly(double xi,size_t nctrl,const double *xctrl,
                double *h,double*  hprime);

// GRL
void gauss_radau_laguerre(double *xgrl,double *wgrl,size_t length);
double laguerre_func(size_t n, double x);

#endif