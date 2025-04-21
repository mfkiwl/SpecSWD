#include "shared/GQTable.hpp"
#include "shared/attenuation.hpp"
#include "libswd/global.hpp"

extern "C"  void 
specswd_init_GQTable() {
    GQTable::initialize();
}

extern "C"  void 
specswd_reset_Qmodel(const double *w,const double *y)
{
    specswd::reset_ref_Q_model(w,y);
}

/**
 * @brief compute kernel size (nker,nz)
 * @param nkers no. of kernels
 * @param nz size of input tomo model
 */
extern "C"  void 
specswd_kernel_size(int *nkers,int *nz)
{
    using namespace specswd_pylib;
    bool HAS_ATT = M_ -> HAS_ATT;
    int SWD_TYPE = M_ -> SWD_TYPE; 
    *nz = M_ -> nz_tomo;
    int nker{};

    if(SWD_TYPE == 0) {
        nker = 3;
        if (HAS_ATT) {
            nker = 5;
        }
    }
    else if (SWD_TYPE == 1) {
        nker = 5;
        if (HAS_ATT) {
            nker = 10;
        }
    }
    else {
        nker = 22;
        if(HAS_ATT) {
            nker += M_ -> nQmodel_ani;
        }
    }

    // copy to output
    *nkers = nker;
}