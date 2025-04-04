#include "solver/solver.hpp"
#include "GQTable.hpp"

template<typename T = double>
void egn2displ_love_(int nspec,const int *ibool_el,const T *egn, 
                    T * __restrict displ)
{
    using namespace GQTable;
    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        int NGL = NGLL;
        if(ispec == nspec) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[ispec*NGLL+i];
            displ[ispec*NGLL+i] = (iglob!=-1) ?egn[iglob] : 0;
        }
    }   
}


template<typename T = double>
void egn2displ_rayl_(int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,
                    int nglob_el,int nglob_ac,const double* jaco,const int *ibool_el,
                    const int *ibool_ac, const int *el_elmnts,
                    const int *ac_elmnts,const double *xrho_ac,
                    const T *egn, double freq,T c, T * __restrict displ)
{

    // get wave number
    T k = 2 * M_PI * freq / c;

    // size
    using namespace GQTable;
    int npts = (nspec_el + nspec_ac) * NGLL + (nspec_ac_grl + nspec_el_grl) * NGRL;

    // loop elastic elements
    for(int ispec = 0; ispec < nspec_el+nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id0+i];
            displ[0*npts + id1+i] = egn[iglob];
            displ[1*npts + id1+i] = egn[iglob + nglob_el] / k; // this is V\bar = kV
        }
    }   

    // loop each acoustic element
    std::array<T,NGRL> chi;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec += 1) {
        int iel = ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        const double *hp = &hprime[0];
        const double J = jaco[iel];

        // GRL layer
        if(ispec == nspec_ac) {
            NGL = NGRL;
            hp = &hprime_grl[0];
        }

        // cache chi in an element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = ibool_ac[id];
            chi[i] = (iglob == -1) ? (T)0.: egn[nglob_el * 2 + iglob] / k;
        }


        // compute derivative  dchi / dz
        for(int i = 0; i < NGL; i ++) {
            T dchi{};
            for(int j = 0; j < NGL; j ++) {
                dchi += chi[j] * hp[i * NGL + j];
            }
            dchi /= J;

            // set value to displ
            displ[0*npts + id1+i] = k / xrho_ac[id0 + i] * chi[i];
            displ[1*npts + id1+i] = dchi / xrho_ac[id0 + i];
        }
    }

}

void SolverSEM:: 
egn2displ_vti(double freq,double c,const double *egn, double * __restrict displ) const 
{
    switch (SWD_TYPE)
    {
    case 0:
        egn2displ_love_(nspec,ibool_el.data(),egn,displ);
        break;
    case 1:
        egn2displ_rayl_(nspec_el,nspec_ac,nspec_el_grl,nspec_ac_grl,
                        nglob_el,nglob_ac,jaco.data(),ibool_el.data(),ibool_ac.data(),
                        el_elmnts.data(),ac_elmnts.data(),xrho_ac.data(),
                        egn,freq,c,displ);
        break;
    default:
        printf("this function is only for VTI case!\n");
        exit(1);
        break;
    }
}

void SolverSEM:: 
egn2displ_vti_att(double freq,dcmplx c,const dcmplx *egn, dcmplx * __restrict displ) const 
{
    switch (SWD_TYPE)
    {
    case 0:
        egn2displ_love_(nspec,ibool_el.data(),egn,displ);
        break;
    case 1:
        egn2displ_rayl_(nspec_el,nspec_ac,nspec_el_grl,nspec_ac_grl,
                        nglob_el,nglob_ac,jaco.data(),ibool_el.data(),ibool_ac.data(),
                        el_elmnts.data(),ac_elmnts.data(),xrho_ac.data(),
                        egn,freq,c,displ);
        break;
    default:
        printf("this function is only for VTI case!\n");
        exit(1);
        break;
    }
}


void SolverSEM::
egn2displ_aniso(double freq,dcmplx c,double phi,const dcmplx *egn,
                dcmplx * __restrict displ) const
{

    // get wave number
    dcmplx wavnum = 2 * M_PI * freq / c;
    dcmplx k[2] = {wavnum * std::cos(phi),wavnum * std::sin(phi)};

    // size
    using namespace GQTable;
    int npts = (nspec_el + nspec_ac) * NGLL + (nspec_ac_grl + nspec_el_grl) * NGRL;

    // loop elastic elements
    for(int ispec = 0; ispec < nspec_el+nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id0+i];
            for(int j = 0; j < 3; j ++) {
                displ[j*npts + id1+i] = egn[iglob + nglob_el*j];
            }
        }
    }   

    // loop each acoustic element
    std::array<dcmplx,NGRL> chi;
    const dcmplx I = {0,1.};
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec += 1) {
        int iel = ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;
        int id1 = iel * NGLL;
        const double *hp = &hprime[0];
        const double J = jaco[iel];

        // GRL layer
        if(ispec == nspec_ac) {
            NGL = NGRL;
            hp = &hprime_grl[0];
        }

        // cache chi in an element
        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int iglob = ibool_ac[id];
            chi[i] = (iglob == -1) ? 0.: egn[nglob_el * 3 + iglob];
        }


        // compute derivative  dchi / dz
        for(int i = 0; i < NGL; i ++) {
            dcmplx dchi{};
            for(int j = 0; j < NGL; j ++) {
                dchi += chi[j] * hp[i * NGL + j];
            }
            dchi /= J;

            // set value to displ
            displ[0*npts + id1+i] = -I * k[0] / xrho_ac[id0 + i] * chi[i];
            displ[1*npts + id1+i] = -I * k[1] / xrho_ac[id0 + i] * chi[i];
            displ[2*npts + id1+i] = dchi / xrho_ac[id0 + i];
        }
    }
}

static void
check_kernel_size(const std::vector<double> &frekl,int npts,
                int SWD_TYPE, bool HAS_ATT,int nQmodel)
{
    int nker = frekl.size() / npts;
    int nker0 = 0;
    if(SWD_TYPE == 0) {
        nker0 = 3;
        if(HAS_ATT) nker0 = 5;
    }
    else if(SWD_TYPE == 1) {
        nker0 = 6;
        if(HAS_ATT) nker0 = 10;
    }
    else {
        nker0 = 23; // 21 (elastic) + 1(acoustic) + rho
        if(HAS_ATT) {
            nker0 = 23 + nQmodel + 1; 
        }
    }

    if(nker0 != nker) {
        printf("target/current number of kernels = %d %d\n",nker0,nker);
        printf("please check the size of frekl!\n");
        exit(1);
    }
}

/**
 * @brief transform modulus kernel to velocity kernel
 * @param frekl frechet kernels, the shape depends on:
 *   - `1`: elastic love wave: N/L/rho -> vsh/vsv/rho  
 *   - `2`: anelastic love wave: N/L/QNi/QLi/rho -> vsh/vsv/QNi/QLi/rho
 *   - `3`: elastic rayleigh wave: A/C/L/eta/kappa/rho -> vph/vpv/vsv/eta/vp/rho  
 *   - `4`: anelastic rayleigh wave: A/C/L/eta/QAi/QCi/QLi/kappa/Qki/rho -> vph/vpv/vsv/eta/QAi/QCi/QLi/vp/Qki/rho 
 */
void SolverSEM:: 
transform_kernels(std::vector<double> &frekl) const
{
    using namespace GQTable;
    int npts = nspec * NGLL + NGRL;

    // check kenrel size
    check_kernel_size(frekl,npts,SWD_TYPE,HAS_ATT,nQmodel_ani);
   
    if(SWD_TYPE == 0) {
        
        for(int ipt = 0; ipt < npts; ipt ++) {
            double N_kl,L_kl,rho_kl;
            N_kl = frekl[0 * npts + ipt];
            L_kl = frekl[1 * npts + ipt];

            int i = 2;
            if(HAS_ATT) i =4;
            rho_kl = frekl[i * npts + ipt];

            // get variables
            double L = xL[ipt], N = xN[ipt], rho = xrho_el[ipt];
            double vsh = std::sqrt(N / rho), vsv = std::sqrt(L / rho);

            // transform kernels
            double vsh_kl = 2. * rho * vsh * N_kl, vsv_kl = 2. * rho * vsv * L_kl;
            double r_kl = vsh * vsh * N_kl + 
                         vsv * vsv * L_kl + rho_kl;
            
            // copy back to frekl array
            frekl[0 * npts + ipt] = vsh_kl;
            frekl[1 * npts + ipt] = vsv_kl;
            frekl[i * npts + ipt] = r_kl;
        }
    }
    else if (SWD_TYPE == 1) {
        for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec += 1) {
            int iel = el_elmnts[ispec];
            int NGL = NGLL;
            int id0 = ispec * NGLL;

            // GRL layer
            if(ispec == nspec_el) {
                NGL = NGRL;
            }

            for(int i = 0; i < NGL; i ++) {
                int id = id0 + i;
                int ipt = iel * NGLL + i;
                double A_kl{},C_kl{}, L_kl{}, rho_kl{};
                
                // loc of rho
                int loc = 5;
                if(HAS_ATT) loc = 9;

                // kernels
                A_kl = frekl[0 * npts + ipt];
                C_kl = frekl[1 * npts + ipt];
                L_kl = frekl[2 * npts + ipt];
                rho_kl = frekl[loc * npts + ipt];

                // compute vph/vpv/vsh/vsv/
                double rho = xrho_el[ipt];
                double vph = std::sqrt(xA[id] / rho);
                double vpv = std::sqrt(xC[id] / rho);
                double vsv = std::sqrt(xL[id] / rho);

                double vph_kl = 2. * rho * vph * A_kl;
                double vpv_kl = 2. * rho * vpv * C_kl;
                double vsv_kl = 2. * rho * vsv * L_kl;
                double r_kl = vph * vph * A_kl + vpv * vpv * C_kl + 
                              + vsv * vsv * L_kl + rho_kl;
                frekl[0 * npts + ipt] = vph_kl;
                frekl[1 * npts + ipt] = vpv_kl;
                frekl[2 * npts + ipt] = vsv_kl;
                frekl[loc * npts + ipt] = r_kl;
            }
        }

        // acoustic domain
        for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec += 1) {
            int iel = ac_elmnts[ispec];
            int NGL = NGLL;
            int id0 = ispec * NGLL;

            // GRL layer
            if(ispec == nspec_ac) {
                NGL = NGRL;
            }

            for(int i = 0; i < NGL; i ++) {
                int id = id0 + i;
                int ipt = iel * NGLL + i;

                // kernels
                double kappa_kl{}, rho_kl{};

                // loc
                int loc_k = 4, loc_r = 5;
                if(HAS_ATT) {
                    loc_k = 7; 
                    loc_r = 9;
                }

                // modulus kernels
                kappa_kl = frekl[loc_k * npts + ipt];
                rho_kl = frekl[loc_r * npts + ipt];

                // velocity
                double rho = xrho_el[id];
                double vp = std::sqrt(xkappa_ac[id] / rho);

                //velocity kernels
                double vp_kl = 2. * rho* vp * kappa_kl;
                double r_kl = vp * vp * kappa_kl + rho_kl;
                frekl[loc_k * npts + ipt] = vp_kl;
                frekl[loc_r * npts + ipt] = r_kl;
            }
        }
    }
}


/**
 * @brief find location of z0 in ascending list z
 * 
 * @param z depth list, shape(nlayer)
 * @param z0 current loc, must be inside z 
 * @param nlayer 
 * @return int location of z0 in z, satisfies  z0 >= z[i] && z0 < z[i + 1]
 */
static int 
find_loc(const float *z,float z0,int nz) 
{

    int i = 0;
    while(i < nz - 1) {
        if(z0 >= z[i] && z0 < z[i + 1]) {
            break;
        }
        i += 1;
    }

    return i;
}

/**
 * @brief project kernels to original 1-D model
 * @param frekl derivatives, shape(nspec*NGLL+NGRL)
 * @param kl_out derivatives on original 1-Dmodel, shape(nz_)
 */
void SolverSEM:: 
project_kl(const double *frekl, double *kl_out) const
{
    using GQTable :: NGLL; using GQTable :: NGRL;

    // zero out kl_out
    for(int i = 0; i < nz_; i ++) kl_out[i] = 0.;

    // loop every region
    for(int ireg = 0; ireg < nregion_; ireg ++) {
        int istart = region_bdry[ireg*2];
        int iend = region_bdry[ireg*2+1];
        int npts = iend - istart + 1;

        // choose domain
        const int *mat_elmnts = nullptr;
        int nel = 0;
        for(int im = 0; im < 2; im ++) {
            if(im == 0) {
                nel = nspec_el + nspec_el_grl;
                mat_elmnts = el_elmnts.data(); 
            }
            else {
                nel = nspec_ac + nspec_ac_grl;
                mat_elmnts = ac_elmnts.data();
            }

            // elastic region
            for(int ispec_md = 0; ispec_md < nel; ispec_md ++) {
                int ispec = mat_elmnts[ispec_md];
                if(ireg != iregion_flag[ispec]) continue;

                // check if GRL 
                int NGL = NGLL;
                if(ispec == nspec) {
                    NGL = NGRL;
                }

                // find interpolate points
                for(int i = 0; i < NGL; i ++) {
                    double z0 = znodes[ispec * NGLL + i];
                    int id = ispec * NGLL + i;

                    // find loc in this region
                    int j = find_loc(&depth_[istart],z0,npts) + istart;
                    if(j < istart || j >= iend) {
                        j = j < istart ? istart : iend;
                        kl_out[j] += frekl[id];
                    }
                    else {
                        float dz = depth_[j + 1] - depth_[j];
                        float coef = (z0 - depth_[j]) / dz;
                        kl_out[j] += (1 - coef) * frekl[id];
                        kl_out[j+1] += coef * frekl[id];
                    }
                }
            }
        }
    }
}