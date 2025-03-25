#include "solver/solver.hpp"
#include "GQTable.hpp"

template<typename T = double>
void egn2displ_love_(int nspec,const int *ibool,const T *egn, 
                    T * __restrict displ)
{
    using namespace GQTable;
    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        int NGL = NGLL;
        if(ispec == nspec) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool[ispec*NGLL+i];
            displ[ispec*NGLL+i] = egn[iglob];
        }
    }   
}

void SolverSEM:: 
egn2displ_love(const double *egn, double * __restrict displ) const
{
    egn2displ_love_(nspec,ibool.data(),egn,displ);
}

void SolverSEM:: 
egn2displ_love_att(const dcmplx *egn, dcmplx * __restrict displ) const
{
    egn2displ_love_(nspec,ibool.data(),egn,displ);
}

void SolverSEM:: 
transform_kernels(std::vector<double> &frekl) const
{
    using namespace GQTable;
    if(SWD_TYPE == 0) {
        int npts = nspec * NGLL + NGRL;
        for(int ipt = 0; ipt < npts; ipt ++) {
            double N_kl,L_kl,rho_kl,Qn_kl,Ql_kl;
            N_kl = frekl[0 * npts + ipt];
            L_kl = frekl[1 * npts + ipt];
            rho_kl = frekl[2 * npts + ipt];
            if(HAS_ATT) {
                Qn_kl = frekl[2*npts + ipt];
                Ql_kl = frekl[3*npts + ipt];
                rho_kl = frekl[4*npts + ipt];
            }

            // get variables
            double L = xL[ipt], N = xN[ipt], rho = xrho_el[ipt];
            double vsh = std::sqrt(N / rho), vsv = std::sqrt(L / rho);

            // transform kernels
            double vsh_kl = 2. * rho * vsh * N_kl, vsv_kl = 2. * rho * vsv * L_kl;
            double r_kl = vsh * vsh * N_kl + 
                         vsv * vsv * L_kl + rho_kl;
            
            // copy back to frekl array
            frekl[0 * npts + ipt] = r_kl;
            frekl[1 * npts + ipt] = vsv_kl;
            frekl[2 * npts + ipt] = vsh_kl;
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
project_kl(const double *frekl, float *kl_out) const
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