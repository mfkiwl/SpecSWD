#include <array>
#include <complex>

// only valid for frequency range [0.01,100]
const int NSLS = 5;
const std::array<double,NSLS> y_sls_ref = {1.93044501, 1.64217132, 1.73606189, 1.42826439, 1.66934129};
const std::array<double,NSLS> w_sls_ref = {4.71238898e-02, 6.63370885e-01, 9.42477796e+00, 1.14672436e+02,1.05597079e+03};


// complex
typedef std::complex<double> dcmplx;

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
dcmplx get_sls_modulus_factor(double freq,const double Q)
{
    double y_sls[NSLS], w_sls[NSLS];
    double om = 2 * M_PI * freq;
    const dcmplx I = {0.,1.};

    get_Q_sls_model(Q,y_sls,w_sls);
    dcmplx s {};
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
get_sls_Q_derivative(double freq,const double Q,dcmplx &s,dcmplx &dsdqi)
{
    double dy[NSLS],dd_dqi[NSLS];
    double y[NSLS];
    const dcmplx I = {0.,1.};
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
    }

    // sum together
    s = 0.; dsdqi = 0.;
    for(int i = 0; i < NSLS; i ++) {
        s += I * om * y[i] * dy[i]/ (w_sls_ref[i] + I * om);

        // y' = delta * y 
        // dy'/dqi = d delta /dqi * y + delta * dy/dqi
        double dyp_dqi = dd_dqi[i] * y[i] + dy[i] * y_sls_ref[i];
        dsdqi += I * om * dyp_dqi / (w_sls_ref[i] + I * om);
    }
    s = s + 1.;
}

static int Index(int m,int n)
{
    if(m > n) {
        std::swap(m,n);
    }
    int idx = m * 6 + n - (m * (m + 1)) / 2;

    return idx;
}

/**
 * @brief only set Qkappa and Qmu to C21 
 * @see Carcione and Cavallini (1995d), delta = 2, M3 = M4 = M2 -> Qmu
 */
static void 
set_C21_iso_bkgd(dcmplx Qk_fac,dcmplx Qmu_fac,dcmplx __restrict *c21)
{
    // get kappa and mu by using average
    #define C(p,q) c21[Index(p,q)]
    dcmplx eps = (1./3.) * (C(0,0) + C(1,1) + C(2,2));
    dcmplx mu = (1./3.) * (C(3,3) + C(4,4) + C(5,5));
    dcmplx kappa = eps - (4. / 3.) * mu;

    // add back to c21
    for(int i = 0; i < 3; i ++) {
        C(i,i) = C(i,i) - eps + kappa * Qk_fac + (4./3.) * mu * Qmu_fac;
    }
    for(int i = 0; i < 3; i ++) {
        for(int j = i + 1; j < 3; j ++) {
            C(i,j) = C(i,j) - eps + kappa * Qk_fac + 2.* mu * (1. - 1./3 * Qmu_fac);
        }
    }

    for(int i = 3; i < 6; i ++) C(i,i) *= Qmu_fac;

    #undef C

}

void set_C21_att_model(double freq,const double *Qm,int nQmodel,
                       dcmplx __restrict *c21,int funcid,bool do_deriv)
{
    // get all sls factor
    std::array<dcmplx,21> Qfac;
    for(int im = 0; im < nQmodel; im ++) {
        Qfac[im] = get_sls_modulus_factor(freq,Qm[im]);
    }

    // choose sls factor
    switch (funcid)
    {
    case 1:
        set_C21_iso_bkgd(Qfac[0],Qfac[1],c21);
        break;
    
    default:
        printf("not implemented!\n");
        exit(1);
        break;
    }
}