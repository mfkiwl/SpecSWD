
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include <complex>
const int NSLS = 5;

void 
get_Q_sls_model(double Q,double *y_sls,double *w_sls);

std::complex<double> get_sls_modulus_factor(double freq,const double Q);
void 
get_sls_Q_derivative(double freq,const double Q,std::complex<double> &s,
                    std::complex<double>dsdqi);

#endif