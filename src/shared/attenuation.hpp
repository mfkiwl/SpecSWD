
#ifndef SPECSWD_ATT_TABLE_H_
#define SPECSWD_ATT_TABLE_H_

#include <complex>

namespace specswd
{
    
const int NSLS = 5;

std::complex<float> get_sls_modulus_factor(float freq,float Q);
void 
get_sls_Q_derivative(float freq,float Qm,std::complex<float> &s,
                    std::complex<float> &dsdqi);

void get_C21_att(float freq,const float *Qm,int nQmodel,
                       std::complex<float> * c21,
                       int funcid=1);

void 
reset_ref_Q_model(const double *w_sls, const double *y_sls);

}

#endif