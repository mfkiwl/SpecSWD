#include <array>
#include <complex>

// only valid for frequency range [0.01,100]
const int NSLS = 5;
const std::array<double,NSLS> y_sls_ref = {0.0096988,  0.00832481, 0.0088744,  0.00735887, 0.00866749};
const std::array<double,NSLS> w_sls_ref = {4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03};

/**
 * @brief correct y from reference model to target model
 * 
 * @param Q target Q 
 * @param y_sls reference y_sls parameters
 */
void 
get_Q_sls_model(double Q,double *y_sls,double *w_sls)
{
    double dy[NSLS];
    double y[NSLS];
    for(int i = 0; i < NSLS; i ++) {
        y[i] = y_sls_ref[i] / Q; 
    }
    dy[0] = 1. + 0.5 * y[0];
    for(int i = 1; i < NSLS; i ++) {
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i];
    }

    // copy to y_sls/w_sls
    for(int i = 0; i < NSLS; i ++) {
        w_sls[i] = w_sls_ref[i];
        y_sls[i] = dy[i] * y[i];
    }
}

/**
 * @brief get SLS Q terms on the elastic modulus
 * 
 * @param freq current frequency
 * @param Qa Qvalue 
 * @param xA 
 */
std::complex<double> get_sls_modulus_factor(double freq,const double Q)
{
    double y_sls[NSLS], w_sls[NSLS];
    double om = 2 * M_PI * freq;
    const std::complex<double> I = {0.,1.};

    get_Q_sls_model(Q,y_sls,w_sls);
    std::complex<double> s {};
    for(int j = 0; j < NSLS; j ++) {
        s += I * om * y_sls[j] / (w_sls[j] + I * om);
    }

    return s + 1.;
}

/**
 * @brief Get the Q factor and derivative for SLS model
 * 
 * @param freq frequency 
 * @param Q current Q
 * @param s modulus factor  mu = mu * s 
 * @param dsdqi Q^{-1} derivative ds / dQi
 */
void 
get_sls_Q_derivative(double freq,const double Q,std::complex<double> &s,std::complex<double>dsdqi)
{
    double dy[NSLS],dd_dqi[NSLS];
    double y[NSLS];
    const std::complex<double> I = {0.,1.};
    double om = 2 * M_PI * freq;

    // compute corrector
    for(int i = 0; i < NSLS; i ++) {
        y[i] = y_sls_ref[i] / Q; 
    }
    dy[0] = 1. + 0.5 * y[0];
    for(int i = 1; i < NSLS; i ++) {
        dy[i] = dy[i-1] + (dy[i-1] - 0.5) * y[i-1] + 0.5 * y[i];
    }

    dd_dqi[0] = 0.5 * y_sls_ref[0];
    for(int i = 1; i < NSLS; i ++) {
        dd_dqi[i] = dd_dqi[i-1] + (dy[i-1] - 0.5) * y_sls_ref[i-1] + dd_dqi[i-1] * y[i-1] +  0.5 * y_sls_ref[i];

        // y' = delta * y 
        // dy'/dqi = d delta /dqi * y + delta * dy/dqi
        dd_dqi[i] = dd_dqi[i] * y[i] + dy[i] * y_sls_ref[i];
    }

    // sum together
    s = 0.; dsdqi = 0.;
    for(int j = 0; j < NSLS; j ++) {
        dsdqi += I * om * dd_dqi[j] / (w_sls_ref[j] + I * om);
        s += I * om * y[j] * dy[j] / (w_sls_ref[j] + I * om);
    }
    s = s + 1.;
}