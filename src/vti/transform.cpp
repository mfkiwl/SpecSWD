#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

namespace specswd {

template<typename T = float>
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

void SolverLove::
egn2displ(const Mesh &M,float freq,float c, const float *egn,float * __restrict displ ) const 
{
    egn2displ_love_(M.nspec,M.ibool_el.data(),egn,displ);
}

void SolverLove::
egn2displ_att(const Mesh &M,float freq,scmplx c, const scmplx *egn,scmplx * __restrict displ ) const 
{
    egn2displ_love_(M.nspec,M.ibool_el.data(),egn,displ);
}



template<typename T = float>
void egn2displ_rayl_(int nspec_el,int nspec_ac,int nspec_el_grl,int nspec_ac_grl,
                    int nglob_el,int nglob_ac,const float* jaco,const int *ibool_el,
                    const int *ibool_ac, const int *el_elmnts,
                    const int *ac_elmnts,const float *xrho_ac,
                    const T *egn, float freq,T c, T * __restrict displ)
{

    // get wave number
    T k = (T)(M_PI * 2.) * freq / c;

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
        const float *hp = &hprime[0];
        const float J = jaco[iel];

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

void SolverRayl::
egn2displ(const Mesh &M,float freq,float c, const float *egn,float * __restrict displ ) const 
{
    egn2displ_rayl_(
        M.nspec_el,M.nspec_ac,M.nspec_el_grl,M.nspec_ac_grl,
        M.nglob_el,M.nglob_ac,M.jaco.data(),M.ibool_el.data(),
        M.ibool_ac.data(),M.el_elmnts.data(),M.ac_elmnts.data(),
        M.xrho_ac.data(),egn,freq,c,displ
    );
}

void SolverRayl::
egn2displ_att(const Mesh &M,float freq,scmplx c, const scmplx *egn,scmplx * __restrict displ ) const 
{
    egn2displ_rayl_(
        M.nspec_el,M.nspec_ac,M.nspec_el_grl,M.nspec_ac_grl,
        M.nglob_el,M.nglob_ac,M.jaco.data(),M.ibool_el.data(),
        M.ibool_ac.data(),M.el_elmnts.data(),M.ac_elmnts.data(),
        M.xrho_ac.data(),egn,freq,c,displ
    );
}

/**
 * @brief transform modulus kernel to velocity kernel, Love wave case
 * @param frekl frechet kernels, the shape depends on:
 *   - `1`: elastic love wave: N/L/rho -> vsh/vsv/rho  
 *   - `2`: anelastic love wave: N/L/QNi/QLi/rho -> vsh/vsv/QNi/QLi/rho 
 */
void SolverLove:: 
transform_kernels(const Mesh &M,std::vector<float> &frekl) const
{
    // check no. of kernels
    using namespace GQTable;
    int npts = M.nspec * NGLL + NGRL;
    int nker = frekl.size() / npts;
    int nker0 = 3;
    if(M.HAS_ATT) nker0 = 5;
    if(nker0 != nker) {
        printf("target/current number of kernels = %d %d\n",nker0,nker);
        printf("please check the size of frekl!\n");
        exit(1);
    }

    for(int ipt = 0; ipt < npts; ipt ++) {
        double N_kl,L_kl,rho_kl;
        N_kl = frekl[0 * npts + ipt];
        L_kl = frekl[1 * npts + ipt];

        int i = 2;
        if(M.HAS_ATT) i =4;
        rho_kl = frekl[i * npts + ipt];

        // get variables
        double L = M.xL[ipt], N = M.xN[ipt], rho = M.xrho_el[ipt];
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

/**
 * @brief transform modulus kernel to velocity kernel, Rayleigh wave case
 * @param frekl frechet kernels, the shape depends on:
 *   - `1`: elastic rayleigh wave: A/C/L/eta/kappa/rho -> vph/vpv/vsv/eta/vp/rho  
 *   - `2` anelastic rayleigh wave: A/C/L/eta/QAi/QCi/QLi/kappa/Qki/rho -> vph/vpv/vsv/eta/QAi/QCi/QLi/vp/Qki/rho 
 */
void SolverRayl:: 
transform_kernels(const Mesh &M,std::vector<float> &frekl) const
{
    // check no. of kernels
    using namespace GQTable;
    int npts = M.nspec * NGLL + NGRL;
    int nker = frekl.size() / npts;
    int nker0 = 6;
    if(M.HAS_ATT) nker0 = 10;
    if(nker0 != nker) {
        printf("target/current number of kernels = %d %d\n",nker0,nker);
        printf("please check the size of frekl!\n");
        exit(1);
    }

    for(int ispec = 0; ispec < M.nspec_el + M.nspec_el_grl; ispec += 1) {
        int iel = M.el_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == M.nspec_el) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int ipt = iel * NGLL + i;
            double A_kl{},C_kl{}, L_kl{}, rho_kl{};
            
            // loc of rho
            int loc = 5;
            if(M.HAS_ATT) loc = 9;

            // kernels
            A_kl = frekl[0 * npts + ipt];
            C_kl = frekl[1 * npts + ipt];
            L_kl = frekl[2 * npts + ipt];
            rho_kl = frekl[loc * npts + ipt];

            // compute vph/vpv/vsh/vsv/
            double rho = M.xrho_el[ipt];
            double vph = std::sqrt(M.xA[id] / rho);
            double vpv = std::sqrt(M.xC[id] / rho);
            double vsv = std::sqrt(M.xL[id] / rho);

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
    for(int ispec = 0; ispec < M.nspec_ac + M.nspec_ac_grl; ispec += 1) {
        int iel = M.ac_elmnts[ispec];
        int NGL = NGLL;
        int id0 = ispec * NGLL;

        // GRL layer
        if(ispec == M.nspec_ac) {
            NGL = NGRL;
        }

        for(int i = 0; i < NGL; i ++) {
            int id = id0 + i;
            int ipt = iel * NGLL + i;

            // kernels
            double kappa_kl{}, rho_kl{};

            // loc
            int loc_k = 4, loc_r = 5;
            if(M.HAS_ATT) {
                loc_k = 7; 
                loc_r = 9;
            }

            // modulus kernels
            kappa_kl = frekl[loc_k * npts + ipt];
            rho_kl = frekl[loc_r * npts + ipt];

            // velocity
            double rho = M.xrho_ac[id];
            double vp = std::sqrt(M.xkappa_ac[id] / rho);

            //velocity kernels
            double vp_kl = 2. * rho* vp * kappa_kl;
            double r_kl = vp * vp * kappa_kl + rho_kl;
            frekl[loc_k * npts + ipt] = vp_kl;
            frekl[loc_r * npts + ipt] = r_kl;
        }
    }
}

}