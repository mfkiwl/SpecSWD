#include "shared/GQTable.hpp"
#include "aniso/aniso.hpp"

namespace specswd
{


using std::vector;

template<typename T = float >
static void 
prepare_aniso_(float freq,int nspec_el,int nspec_ac,int nspec_el_grl,
                int nspec_ac_grl,int nglob_el,int nglob_ac,int nQmodel,
                const int *el_elmnts,const int *ac_elmnts,
                const float *xrho_el,const float *xrho_ac,
                const int* ibool_el, const int* ibool_ac,
                const float *jaco,const float *xc21,
                const float *xQani,const float *xkappa_ac,
                const float *xQk_ac,int nfaces_bdry,
                const int* ispec_bdry,const char *bdry_norm_direc,
                float phi,vector<T> &Mmat,
                vector<T> &Kmat,vector<T> &Hmat,vector<T> &Emat)
{
    // allocate space and set zero
    int ng = nglob_ac + nglob_el * 3;
    Mmat.resize(ng);
    Emat.resize(ng * ng);
    Kmat.resize(ng); Hmat.resize(ng*ng);
    std::fill(Mmat.begin(),Mmat.end(),(T)0.);
    std::fill(Emat.begin(),Emat.end(),(T)0.);
    std::fill(Kmat.begin(),Kmat.end(),(T)0.);
    std::fill(Hmat.begin(),Hmat.end(),(T)0.);

    // direction
    float k[2] = {std::cos(phi),std::sin(phi)};

    // temp arrays to save elastic tensor
    using namespace GQTable;
    const int size_el = nspec_el*NGLL + nspec_el_grl * NGRL;
    std::array<T,NGRL*21> sumC21;
    #define C21(i,j,p,q,a) sumC21[a*NGRL + Index(i,j,p,q)]

    // compute M/K/H/E for gll/grl layer, elastic
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;
        float J = jaco[iel];

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const float *hp = is_gll? hprime.data(): hprime_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            for(int idx = 0; idx < 21; idx ++) {
                sumC21[i*NGRL+idx] = xc21[idx*size_el+i];
            }

            // apply Q model to C21 if required
            if constexpr (std::is_same_v<T,dcmplx>) {
                std::array<float,21> Qm;
                for(int q = 0; q < nQmodel; q ++) {
                    Qm[q] = xQani[q*size_el+i];
                }
                set_C21_att_model(freq,Qm.data(),nQmodel,
                                &sumC21[i*NGRL]);
            }

            
            // add other terms
            for(int idx = 0; idx < 21; idx ++) {
                sumC21[i*NGRL+idx] *= J * weight[i];
            }
        }

        // assemble H/E
        for(int a = 0; a < NGL; a ++) {
            int iglob = ibool_el[id + a];
            for(int b = 0; b < NGL; b ++) {
                int iglob1 = ibool_el[id+b];

                // loop each component
                for(int i = 0; i < 3; i ++) {
                for(int p = 0; p < 3; p ++) {
                    int idx = (i*nglob_el+iglob)*ng+(p*nglob_el+iglob1);

                    // E
                    T sx{};
                    for(int s = 0; s < NGL; s ++) {
                        sx += C21(i,2,p,2,s) * hpT[a*NGL+s] * hpT[b*NGL+s];
                    }
                    Emat[idx] += sx / (J * J);

                    // H
                    T temp1 = C21(i,0,p,2,a) * k[0] + 
                                C21(i,1,p,2,a) * k[1];
                    T temp2 = C21(i,2,p,0,b) * k[0] + 
                                C21(i,2,p,1,b) * k[1];
                    Hmat[idx] += temp1 * hp[a*NGL+b] - 
                                 temp2 * hpT[a*NGL+b];
                }}
            }
        }

        // compute M/K
        for(int a = 0; a < NGL; a ++) {
            int iglob = ibool_el[id + a];

            // compute mass matrix
            T M0 = weight[a] * J * xrho_el[id + a];
            for(int i = 0; i < 3; i ++) {
                Mmat[iglob + nglob_el * i] += M0;
                for(int p = 0; p < 3; p ++) {
                    T temp = C21(i,0,p,0,a) * k[0] * k[0] + 
                                C21(i,0,p,1,a) * k[0] * k[1] + 
                                C21(i,1,p,0,a) * k[0] * k[1] +
                                C21(i,1,p,1,a) * k[1] * k[1];
                    int idx = (i*nglob_el+iglob) * ng + (p*nglob_el+iglob);
                    Kmat[idx] += temp;
                }
            }
        }
    }

    // acoustic case
    std::array<T,NGRL> sumL;
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = ac_elmnts[ispec];
        int id = ispec * NGLL;

        // get const arrays
        const bool is_gll = (ispec != nspec_el);
        const float *weight = is_gll? wgll.data(): wgrl.data();
        const float *hpT = is_gll? hprimeT.data(): hprimeT_grl.data();
        const int NGL = is_gll? NGLL : NGRL;

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            sumL[i] =  weight[i] / jaco[iel] / xrho_ac[id+i];
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int ig0 = ibool_ac[id + i];
            if(ig0 == -1) continue;
            int iglob = ig0 + nglob_el * 3;
            T temp = weight[i] * jaco[iel];

            // assemble M and K
            T sk = 1.;
            if constexpr (std::is_same_v<T,dcmplx>) {
                sk = get_sls_modulus_factor(freq,xQk_ac[id+i]);
            }
            Mmat[iglob] += temp / (sk * xkappa_ac[id + i]);
            Kmat[iglob * ng + iglob] += temp / xrho_ac[id + i];

            // assemble E
            for(int j = 0; j < NGL; j ++) {
                int ig1 = ibool_ac[id + j];
                if(ig1 == -1) continue;
                int iglob1 = ig1 + nglob_el * 3;
                T s{};
                for(int m = 0; m < NGL; m ++) {
                    s += sumL[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += s;
            }
        }
    }

    // acoustic-elastic boundary
    float om = M_PI * 2 * freq;
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        int ispec_ac = ispec_bdry[iface * 2 + 0];
        int ispec_el = ispec_bdry[iface * 2 + 1];
        T norm = -1.;
        int igll_el = 0;
        int igll_ac = NGLL - 1;
        if(!bdry_norm_direc[iface]) {
            norm = 1.;
            igll_ac = 0;
            igll_el = NGLL - 1;
        }

        // get ac/el global loc
        int iglob_el = ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = ibool_ac[ispec_ac * NGLL + igll_ac];

        // add contribution to E mat, elastic case
        // E(nglob_el + iglob_el, nglob_el*2 + iglob_ac) += 
        int id = (nglob_el*2 + iglob_el) * ng + (nglob_el * 3 + iglob_ac);
        Emat[id] += (T)(om * om * norm);
        
        // acoustic case
        // E(nglob_el*2 + iglob_ac, nglob_el + iglob_el) += norm
        id = (nglob_el*3 + iglob_ac) * ng + (nglob_el*2 + iglob_el);
        Emat[id] += (T)norm;
    }

    #undef C21
}

void SolverAni::
prepare_matrices(float freq,float phi,const Mesh &M)
{
    if(!M.HAS_ATT) {
        prepare_aniso_(
            freq,M.nspec_el,M.nspec_ac,M.nspec_el_grl,M.nspec_ac_grl,
            M.nglob_el,M.nglob_ac,M.nQmodel_ani,M.el_elmnts.data(),
            M.ac_elmnts.data(),M.xrho_el.data(),M.xrho_ac.data(),
            M.ibool_el.data(),M.ibool_ac.data(),M.jaco.data(),
            M.xC21.data(),M.xQani.data(),M.xkappa_ac.data(),
            M.xQk_ac.data(),M.nfaces_bdry,M.ispec_bdry.data(),
            M.bdry_norm_direc.data(),phi,Mmat,Kmat,Hmat,Emat
        );
    }
    else {
        prepare_aniso_(
            freq,M.nspec_el,M.nspec_ac,M.nspec_el_grl,M.nspec_ac_grl,
            M.nglob_el,M.nglob_ac,M.nQmodel_ani,M.el_elmnts.data(),
            M.ac_elmnts.data(),M.xrho_el.data(),M.xrho_ac.data(),
            M.ibool_el.data(),M.ibool_ac.data(),M.jaco.data(),
            M.xC21.data(),M.xQani.data(),M.xkappa_ac.data(),
            M.xQk_ac.data(),M.nfaces_bdry,M.ispec_bdry.data(),
            M.bdry_norm_direc.data(),phi,CMmat,CKmat,CHmat,CEmat
        );
    }
}
    
    
} // namespace specswd