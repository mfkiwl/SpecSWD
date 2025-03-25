#include "solver/solver.hpp"
#include "GQTable.hpp"

typedef std::complex<double> dcmplx;
using std::vector;

template<typename T = dcmplx>
static void 
prepare_love_mat(int nspec,int nglob,const double *xrho, 
                const int* ibool,const double *jaco,const T *xL, 
                const T *xN,vector<double> &Mmat,vector<T> &Kmat,
                vector<T> &Emat)
{
    using namespace GQTable;
    std::array<T,NGRL> sum_terms;

    // allocate space and set zero
    Mmat.resize(nglob); Kmat.resize(nglob);
    Emat.resize(nglob*nglob);
    std::fill(Mmat.begin(),Mmat.end(),0);
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
            sum_terms[i] = xL[id + i] * weight[i] / jaco[ispec];
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool[id + i];
            double temp = weight[i] * jaco[ispec];
            Mmat[iglob] += temp * xrho[id + i];
            Kmat[iglob] += temp * xN[id + i];

            for(int j = 0; j < NGL; j ++) {
                int iglob1 = ibool[id + j];
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
prepare_matrices_love_()
{  
    prepare_love_mat(nspec,nglob,xrho_el.data(),ibool.data(),
                            jaco.data(),xL.data(),xN.data(),Mmat,Kmat,Emat);
}

/**
 * @brief prepare M/K/E matrices for Love wave, visco-elastic case, K,E are complex
 * 
 */
void SolverSEM:: 
prepare_matrices_love_att_()
{
    prepare_love_mat(nspec,nglob,xrho_el.data(),ibool.data(),
                     jaco.data(),cxL.data(),cxN.data(),Mmat,CKmat,CEmat);
}

template<typename T = dcmplx>
static void 
prepare_rayl_mat(double freq,int nspec_el,int nspec_ac,
                int nspec_el_grl,int nspec_ac_grl,int nglob_el,int nglob_ac,
                const int *el_elmnts,const int *ac_elmnts,
                const double *xrho_el,const double *xrho_ac,
                const int* ibool_el, const int* ibool_ac,const double *jaco,
                const T *xL, const T *xA,const T *xF,const T *xC, 
                const T *xkappa_ac,int nfaces_bdry,
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
    std::array<T,NGRL> sumC,sumL;
    for(int ispec = 0; ispec < nspec_el + nspec_el_grl; ispec ++) {
        int iel = el_elmnts[ispec];
        int id = ispec * NGLL;
        const double *weight = wgll.data();
        const double *hpT = hprimeT.data();
        const double *hp = hprime.data();
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_el) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            hp = hprime_grl.data();
            NGL = NGRL;
        }   
        // cache temporary arrays
        for(int i = 0; i < NGL; i ++) {
            sumC[i] = xC[id + i] * weight[i] / jaco[iel];
            sumL[i] = xL[id + i] * weight[i] / jaco[iel];
        }

        // compute M/K/E
        for(int i = 0; i < NGL; i ++) {
            int iglob = ibool_el[id + i];
            T temp = weight[i] * jaco[iel];

            // element wise M/K1/K3
            T M0 = temp * xrho_el[id + i];
            T K1 = temp * xA[id + i];
            T K3 = temp * xL[id + i];

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
                    E1 += sumL[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                    E3 += sumC[m] * hpT[i * NGL + m] * hpT[j * NGL + m];
                }
                Emat[iglob * ng + iglob1] += E1;
                Emat[(iglob + nglob_el) * ng + (iglob1 + nglob_el)] += E3;

                // K2/E2
                T K2 = weight[j] * xF[id + j] * hpT[i * NGL + j] - 
                       weight[i] * xL[id + i] * hp[i * NGL + j];
                T E2 = weight[i] * xF[id + i] * hp[i * NGL + j] - 
                       weight[j] * xL[id + j] * hpT[i * NGL + j];
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
        const double *hp = hprime.data();
        int NGL = NGLL;

        // grl case
        if(ispec == nspec_ac) {
            weight = wgrl.data();
            hpT = hprimeT_grl.data();
            hp = hprime_grl.data();
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
            int iglob = ig0 + nglob_el * 2;
            T temp = weight[i] * jaco[iel];

            // assemble M and K
            Mmat[iglob] += temp / xkappa_ac[id + i];
            Kmat[iglob * ng + iglob] += temp / xrho_ac[id + i];

            // assemble E
            for(int j = 0; j < NGL; j ++) {
                int ig1 = ibool_ac[id + j];
                if(ig1 == -1) continue;
                int iglob1 = ig1 + nglob_el * 2;
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
        int id = (nglob_el + iglob_el) * ng + (nglob_el * 2 + iglob_ac);
        Emat[id] += (T)(om * om * norm);
        
        // acoustic case
        // E(nglob_el*2 + iglob_ac, nglob_el + iglob_el) += norm
        id = (nglob_el*2 + iglob_ac) * ng + (nglob_el + iglob_el);
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
                    ibool_ac.data(),jaco.data(),xL.data(),xA.data(),xF.data(),
                     xC.data(),xkappa_ac.data(),nfaces_bdry,
                     ispec_bdry.data(),bdry_norm_direc.data(),
                    Mmat,Kmat,Emat);
}


void SolverSEM::
prepare_matrices_rayl_att_(double freq)
{
    prepare_rayl_mat(freq,nspec_el,nspec_ac,
                    nspec_el_grl,nspec_ac_grl,nglob_el,nglob_ac,
                    el_elmnts.data(),ac_elmnts.data(),
                     xrho_el.data(),xrho_ac.data(),ibool_el.data(),
                    ibool_ac.data(),jaco.data(),cxL.data(),cxA.data(),cxF.data(),
                     cxC.data(),cxkappa_ac.data(),nfaces_bdry,
                     ispec_bdry.data(),bdry_norm_direc.data(),
                    CMmat,CKmat,CEmat);
}