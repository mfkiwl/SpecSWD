
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include <complex>
const int NSLS = 5;

void 
get_Q_sls_model(double Q,double *y_sls,double *w_sls);

std::complex<double> get_sls_modulus_factor(double freq,const double Q);
void 
get_sls_Q_derivative(double freq,const double Q,std::complex<double> &s,
                    std::complex<double> &dsdqi);

void set_C21_att_model(double freq,const double *Qm,int nQmodel,
                       std::complex<double>* __restrict c21,
                       int funcid=0,bool do_deriv=false);

#endif