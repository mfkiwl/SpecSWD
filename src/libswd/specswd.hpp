#ifndef SPECSWD_LIB_UTILS_H_
#define SPECSWD_LIB_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void 
specswd_init_GQTable();

void specswd_reset_Qmodel(const double *w,const double *y);

void specswd_kernel_size(int *nkers,int *nz);

void
specswd_init_mesh_love(
    int nz, const float *z,const float *rho,const float *vsh,
    const float *vsv,const float *QN, const float *QL,
    bool HAS_ATT,bool print_tomo_info
);

void 
specswd_init_mesh_rayl(
    int nz, const float *z,const float *rho,
    const float *vph,const float* vpv,const float *vsv,
    const float *eta,const float *QA, const float *QC,
    const float *QL, bool HAS_ATT,bool print_tomo_info = false
);

void 
specswd_egn_love(float freq,bool use_qz);

void 
specswd_egn_rayl(float freq,bool use_qz);

void 
specswd_group_love();

void 
specswd_group_rayl();

void specswd_phase_kl(int imode,float *frekl_c,float *frekl_q);
void specswd_group_kl(int imode,float *frekl_c,float *frekl_q);



#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif