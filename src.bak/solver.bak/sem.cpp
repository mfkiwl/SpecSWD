#include "solver/solver.hpp"
#include "GQTable.hpp"
#include "shared/attenuation.hpp"

typedef std::complex<double> dcmplx;
using std::vector;

template<typename T = dcmplx>
static void 
prepare_love_mat(double freq,int nspec,int nglob_el,const double *xrho, 
                const int* ibool_el,const double *jaco,const double *xL, 
                const double *xN,const double *xQL,const double *xQN,
                vector<double> &Mmat,vector<T> &Kmat,vector<T> &Emat)
{
    using namespace GQTable;
    std::array<T,NGRL> sum_terms;

    // allocate space and set zero
    int nglob = nglob_el;
    Mmat.resize(nglob); Kmat.resize(nglob);
    Emat.resize(nglob*nglob);
    std::fill(Mmat.begin(),Mmat.end(),0.);
    std::fill(Kmat.begin(),Kmat.end(),(T)0);
    std::fill(Emat.begin(),Emat.end(),(T)0);

    for(int ispec = 0; ispec < nspec + 1; ispec ++) {
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        int NGL = NGLL;

        // grl case
        if(ispec == nspec) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            NGL = NGRL;
        }

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            T sl = 1.;
            if constexpr (std::is_same_v<T,dcmplx>) {
                sl = get_sls_modulus_factor(freq,xQL[id+i]);
            }
            sum_terms[i] = xL[id + i] * sl * weight[i] / jaco[ispec];
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            if(iglob == -1) continue;

            double temp = weight[i] * jaco[ispec];
            T sn = 1.; 
            if constexpr(std::is_same_v<T,dcmplx>) {
                sn = get_sls_modulus_factor(freq,xQN[id+i]);
            }
            Mmat[iglob] += temp * xrho[id + i];
            Kmat[iglob] += temp * xN[id + i] * sn;

            for(int j = 0; j < NGL; j ++) {
                int iglob1 = ibool_el[id + j];
                if(iglob1 == -1) continue;
                T s{};
                for(int m = 0; m < NGL; m ++) {
                    s += sum_terms[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * nglob + iglob1] += s;
            }
        }
    }
}


/**
 * @brief prepare M/K/E matrices for Love wave, elastic case
 * 
 */
void SolverSEM:: 
prepare_matrices_love_(double freq)
{  
    prepare_love_mat<double>(freq,nspec_el,nglob_el,xrho_el.data(),ibool_el.data(),
                     jaco.data(),xL.data(),xN.data(),nullptr,nullptr,
                     Mmat,Kmat,Emat);
}

/**
 * @brief prepare M/K/E matrices for Love wave, visco-elastic case, K,E are complex
 * 
 */
void SolverSEM:: 
prepare_matrices_love_att_(double freq)
{
    prepare_love_mat<dcmplx>(freq,nspec_el,nglob_el,xrho_el.data(),ibool_el.data(),
                     jaco.data(),xL.data(),xN.data(),xQL.data(),
                     xQN.data(),Mmat,CKmat,CEmat);
}

template<typename T = dcmplx>
static void 
prepare_rayl_mat(double freq,int nspec_el,int nspec_ac,
                int nspec_el_grl,int nspec_ac_grl,int nglob_el,int nglob_ac,
                const int *el_elmnts,const int *ac_elmnts,
                const double *xrho_el,const double *xrho_ac,
                const int* ibool_el, const int* ibool_ac,const double *jaco,
                const double *xA,const double *xC,const double *xL,const double *xeta,
                const double *xQA, const double *xQC, const double *xQL,
                const double *xkappa_ac, const double *xQk_ac,int nfaces_bdry,
                const int* ispec_bdry,const char *bdry_norm_direc,vector<T> &Mmat,
                vector<T> &Kmat,vector<T> &Emat)
{
    // allocate space and set zero
    int ng = nglob_ac + nglob_el * 2;
    Mmat.resize(ng);
    Emat.resize(ng * ng);
    Kmat.resize(ng * ng);
    std::fill(Mmat.begin(),Mmat.end(),(T)0.);
    std::fill(Emat.begin(),Emat.end(),(T)0.);
    std::fill(Kmat.begin(),Kmat.end(),(T)0.);

    // compute M/K/E for gll/grl layer, elastic
    using namespace GQTable;
    std::array<T,NGRL> A,L,C,F;
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        const double *hp = hprime.data();
        int NGL = NGLL;

        // jacobian
        double J = jaco[iel];

        // grl case
        if(ispec == nspec_el) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            hp = hprime_grl.data();
            NGL = NGRL;
        }   
        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            T sl = 1.,sa = 1.,sc = 1.;
            if constexpr (std::is_same_v<T,dcmplx>) {
                sl = get_sls_modulus_factor(freq,xQL[id+i]);
                sa = get_sls_modulus_factor(freq,xQA[id+i]);
                sc = get_sls_modulus_factor(freq,xQC[id+i]);
            }
            // C[i] = xC[id + i] * weight[i] / jaco[iel];
            // L[i] = xL[id + i] * weight[i] / jaco[iel];
            C[i] = xC[id+i] * sc;
            L[i] = xL[id+i] * sl;
            A[i] = xA[id+i] * sa;
            F[i] = xeta[id+i] * (A[i] - 2. * L[i]);
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            T temp = weight[i] * J;

            // element wise M/K1/K3
            T M0 = temp * xrho_el[id + i];
            T K1 = temp * A[i];
            T K3 = temp * L[i];

            // assemble
            Mmat[iglob] += M0;
            Mmat[iglob + nglob_el] += M0;
            Kmat[iglob * ng + iglob] += K1;
            Kmat[(nglob_el + iglob) * ng + (nglob_el + iglob)] += K3;

            // other matrices
            for(int j = 0; j < NGL; j ++) {
                int iglob1 = ibool_el[id + j];
                T E1{},E3{};
                for(int m = 0; m < NGL; m ++) {
                    E1 += L[m] * weight[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                    E3 += C[m] * weight[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += E1 / J;
                Emat[(iglob + nglob_el) * ng + (iglob1 + nglob_el)] += E3 / J;

                // K2/E2
                T K2 = weight[j] * F[j] * hpT[i * NGL + j] - 
                       weight[i] * L[i] * hp[i * NGL + j];
                T E2 = weight[i] * F[i] * hp[i * NGL + j] - 
                       weight[j] * L[j] * hpT[i * NGL + j];
                Kmat[(nglob_el + iglob) * ng + iglob1] += K2;
                Emat[iglob * ng + nglob_el +  iglob1] += E2;
            }
        }
    }

    // acoustic case
    for(int ispec = 0; ispec < nspec_ac + nspec_ac_grl; ispec ++) {
        int iel = ac_elmnts[ispec];
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        int NGL = NGLL;

        // jacobian
        double J = jaco[iel];

        // grl case
        if(ispec == nspec_ac) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            NGL = NGRL;
        }   
        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            L[i] =  weight[i] / (J * xrho_ac[id+i]);
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int ig0 = ibool_ac[id + i];
            if(ig0 == -1) continue;
            int iglob = ig0 + nglob_el * 2;
            T temp = weight[i] * J;


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
                int iglob1 = ig1 + nglob_el * 2;
                T s{};
                for(int m = 0; m < NGL; m ++) {
                    s += L[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += s;
            }
        }
    }

    // acoustic-elastic boundary
    double om = M_PI * 2 * freq;
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
}

void SolverSEM::
prepare_matrices_rayl_(double freq)
{
    prepare_rayl_mat(freq,nspec_el,nspec_ac,
                    nspec_el_grl,nspec_ac_grl,nglob_el,nglob_ac,
                    el_elmnts.data(),ac_elmnts.data(),
                     xrho_el.data(),xrho_ac.data(),ibool_el.data(),
                    ibool_ac.data(),jaco.data(),xA.data(),xC.data(),xL.data(),
                     xeta.data(),nullptr,nullptr,nullptr,xkappa_ac.data(),
                     nullptr,nfaces_bdry,ispec_bdry.data(),bdry_norm_direc.data(),
                     Mmat,Kmat,Emat);
}


void SolverSEM::
prepare_matrices_rayl_att_(double freq)
{
    prepare_rayl_mat(freq,nspec_el,nspec_ac,
                    nspec_el_grl,nspec_ac_grl,nglob_el,nglob_ac,
                    el_elmnts.data(),ac_elmnts.data(),
                     xrho_el.data(),xrho_ac.data(),ibool_el.data(),
                    ibool_ac.data(),jaco.data(),xA.data(),xC.data(),xL.data(),
                    xeta.data(),xQA.data(),xQC.data(),xQL.data(),xkappa_ac.data(),
                    xQk_ac.data(),nfaces_bdry,ispec_bdry.data(),bdry_norm_direc.data(),
                    CMmat,CKmat,CEmat);
}

const int voigt_index[3][3] = {
    {0, 5, 5},  // (1,1) -> 1, (1,2) -> 6, (1,3) -> 5
    {5, 1, 3},  // (2,1) -> 6, (2,2) -> 2, (2,3) -> 4
    {4, 3, 2}   // (3,1) -> 5, (3,2) -> 4, (3,3) -> 3
};


static int Index(int i,int j,int k, int l)
{
    int m = voigt_index[i][j];
    int n = voigt_index[k][l];
    if(m > n) {
        std::swap(m,n);
    }
    int idx = m * 6 + n - (m * (m + 1)) / 2;

    return idx;
}

template<typename T = dcmplx>
static void 
prepare_aniso_mat(double freq,int nspec_el,int nspec_ac,int nspec_el_grl,
                int nspec_ac_grl,int nglob_el,int nglob_ac,int nQmodel,
                const int *el_elmnts,const int *ac_elmnts,
                const double *xrho_el,const double *xrho_ac,
                const int* ibool_el, const int* ibool_ac,
                const double *jaco,const double *xc21,
                const double *xQani,const double *xkappa_ac,
                const double *xQk_ac,int nfaces_bdry,
                const int* ispec_bdry,const char *bdry_norm_direc,
                double phi,vector<T> &Mmat,
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
    double k[2] = {std::cos(phi),std::sin(phi)};

    // temp arrays to save elastic tensor
    using namespace GQTable;
    const int size_el = nspec_el*NGLL + nspec_el_grl * NGRL;
    std::array<T,NGRL*21> sumC21;
    #define C21(i,j,p,q,a) sumC21[a*NGRL + Index(i,j,p,q)]

    // compute M/K/H/E for gll/grl layer, elastic
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        const double *hp = hprime.data();
        int NGL = NGLL;
        double J = jaco[iel];

        // grl case
        if(ispec == nspec_el) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            hp = hprime_grl.data();
            NGL = NGRL;
        } 

        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            for(int idx = 0; idx < 21; idx ++) {
                sumC21[i*NGRL+idx] = xc21[idx*size_el+i];
            }

            // apply Q model to C21 if required
            if constexpr (std::is_same_v<T,dcmplx>) {
                std::array<double,21> Qm;
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
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_ac) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            NGL = NGRL;
        }   
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
    double om = M_PI * 2 * freq;
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

void SolverSEM:: 
prepare_mat_aniso_(double freq,double phi)
{
    prepare_aniso_mat(freq,nspec_el,nspec_ac,nspec_el_grl,nspec_ac_grl,
                     nglob_el,nglob_ac,nQmodel_ani,el_elmnts.data(),ac_elmnts.data(),
                     xrho_el.data(),xrho_ac.data(),ibool_el.data(),
                     ibool_ac.data(),jaco.data(),xC21.data(),xQani.data(),
                     xkappa_ac.data(),xQk_ac.data(),
                     nfaces_bdry,ispec_bdry.data(),bdry_norm_direc.data(),
                     phi,Mmat,Kmat,Hmat,Emat);
}