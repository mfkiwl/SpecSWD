#include "solver/solver.hpp"
#include "GQTable.hpp"
#include <algorithm>


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
 * @brief interpolate elastic/acoustic model by using coordinates
 * 
 * @param param input model parameter, shape(nz_)
 * @param elmnts all elements used, ispec = elmnts[i]
 * @param md model required to interpolate, shape(nspec_el*NGLL + nspec_el_grl * NGRL)
 */
void SolverSEM::
interp_model(const float *param,const std::vector<int> &elmnts,std::vector<double> &md) const
{
    using GQTable :: NGLL; using GQTable :: NGRL;
    int nel = elmnts.size();

    for(int ireg = 0; ireg < nregion_; ireg ++) {
        int istart = region_bdry[ireg*2];
        int iend = region_bdry[ireg*2+1];
        int npts = iend - istart + 1;

        for(int ispec_md = 0; ispec_md < nel; ispec_md ++) {
            int ispec = elmnts[ispec_md];
            int NGL = NGLL;
            if(ispec == nspec) {
                NGL = NGRL;
            }

            if(ireg != iregion_flag[ispec]) continue;

            // interpolate
            for(int i = 0; i < NGL; i ++) {
                int id = ispec_md * NGLL + i;
                
                double z0 = znodes[ispec * NGLL + i];

                // find loc in this region
                int j = find_loc(&depth_[istart],z0,npts) + istart;
                if(j < istart || j >= iend) {
                    j = j < istart ? istart : iend;
                    md[id] = param[j];
                }
                else {
                    float dzinv = 1./ (depth_[j + 1] - depth_[j]);
                    md[id] = param[j] + (param[j+1]-param[j]) * dzinv * (z0-depth_[j]); 
                }
            
            }
        }
    }

}

void SolverSEM:: 
create_material_info_()
{
    using namespace GQTable;
    ac_elmnts.resize(0); el_elmnts.resize(0);
    ac_elmnts.reserve(nspec_ac + nspec_ac_grl);
    el_elmnts.reserve(nspec_el + nspec_el_grl);
    is_elastic.resize(nspec + nspec_grl);
    is_acoustic.resize(nspec + nspec_grl);
    for(int ispec = 0; ispec < nspec + nspec_grl; ispec ++) {
        if(is_elastic[ispec]) {
            el_elmnts.push_back(ispec);
        }
        if(is_acoustic[ispec]) {
            ac_elmnts.push_back(ispec);
        }
    }

    // get nglob_el for elastic 
    ibool_el.resize(nspec_el * NGLL + nspec_el_grl * NGRL);
    nglob_el = 0;
    int idx = -1;
    for(int i = 0; i < nspec_el + nspec_el_grl; i += 1) {
        int ispec = el_elmnts[i];
        if(idx == ibool[ispec * NGLL]) nglob_el -= 1;

        int NGL = NGLL;
        if(i == nspec_el) NGL = NGRL;
        for(int igll = 0; igll < NGL; igll ++) {
            ibool_el[i * NGLL + igll] = nglob_el;
            nglob_el += 1;
        }
        idx = ibool[ispec * NGLL + NGLL-1];
    }

    // regular boundary condition at infinity
    // if(nspec_el_grl == 1) {
    //     nglob_el -= 1;
    //     ibool_el[nspec_el*NGLL + NGRL-1] = -1;
    // }

    // get nglob_ac for acoustic
    ibool_ac.resize(nspec_ac * NGLL + nspec_ac_grl * NGRL);
    idx = -10;
    nglob_ac = 0;
    if(is_acoustic[0]) nglob_ac = -1; // the top point of acoustic wave is 0
    for(int i = 0; i < nspec_ac + nspec_ac_grl; i += 1) {
        int ispec = ac_elmnts[i];
        if(idx == ibool[ispec * NGLL]) nglob_ac -= 1;

        int NGL = NGLL;
        if(i == nspec_ac) NGL = NGRL;
        for(int igll = 0; igll < NGL; igll ++) {
            ibool_ac[i * NGLL + igll] = nglob_ac;
            nglob_ac += 1;
        }
        idx = ibool[ispec * NGLL + NGLL-1];
    }

    // elastic-acoustic boundary
    nfaces_bdry = 0;
    for(int i = 0; i < nspec; i ++) {
        if(is_elastic[i] != is_elastic[i+1]) {
            nfaces_bdry += 1;
        }
    }
    ispec_bdry.resize(nfaces_bdry*2);
    bdry_norm_direc.resize(nfaces_bdry);
    idx = 0;
    for(int i = 0; i < nspec; i ++) {
        if(is_elastic[i] != is_elastic[i+1]) {
            int iloc_el = i + 1, iloc_ac = i;
            if(is_elastic[i]) {
                bdry_norm_direc[i] = 0;
                iloc_el = i;
                iloc_ac = i + 1;
            }
            else {
                bdry_norm_direc[idx] = 1;
            }
            auto it = std::find(el_elmnts.begin(),el_elmnts.end(),iloc_el);
            int ispec_el = it - el_elmnts.begin();
            it = std::find(ac_elmnts.begin(),ac_elmnts.end(),iloc_ac);
            int ispec_ac = it - ac_elmnts.begin();
            ispec_bdry[idx * 2 + 0] = ispec_ac;
            ispec_bdry[idx * 2 + 1] = ispec_el;
            idx += 1;
        }
    }

#ifdef SPECSWD_DEBUG
    // debug
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        printf("\nface %d, ispec_ac,ispec_el = %d %d\n",iface,ispec_bdry[iface*2],ispec_bdry[iface*2+1]);
        printf("acoustic -> elastic  = %d\n",bdry_norm_direc[iface]);
    }
#endif
}

/**
 * @brief Create SEM database by using input model info
 * 
 * @param freq current frequency
 * @param phi directional angle
 */
void SolverSEM::
create_database(double freq,double phi)
{
    using namespace GQTable;

    std::vector<float> vmin,vmax;
    this -> compute_minmax_veloc_(phi,vmin,vmax);

    // find min/max vs
    PHASE_VELOC_MAX = -1.;
    PHASE_VELOC_MIN = 1.0e20;
    for(int i = 0; i < nregion_; i ++) {
        PHASE_VELOC_MAX = std::max((double)vmax[i],PHASE_VELOC_MAX);
        PHASE_VELOC_MIN = std::min((double)vmin[i],PHASE_VELOC_MIN);
    }
    PHASE_VELOC_MIN *= 0.85;
    
    // loop every region to find best element size
    nspec = 0;
    std::vector<int> nel(nregion_ - 1);
    for(int ig = 0; ig < nregion_ - 1; ig ++) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        
        float maxdepth = depth_[iend] - depth_[istart];
        nel[ig] = 1.5 * (maxdepth *  freq) / vmin[ig] + 1;
        if(nel[ig] <=0) nel[ig] = 1;
        nspec += nel[ig];
    }
    nspec_grl = 1;

    // allocate space
    size_t size = nspec * NGLL + NGRL;
    ibool.resize(size); znodes.resize(size);
    jaco.resize(nspec+1); iregion_flag.resize(nspec+1);
    skel.resize(nspec*2+2);

    // connectivity matrix
    int idx = 0;
    for(int ispec = 0; ispec < nspec; ispec ++) {
        for(int igll = 0; igll < NGLL; igll ++) {
            ibool[ispec * NGLL + igll] = idx;
            idx += 1;
        }
        idx -= 1;
    }
    for(int i = 0; i < NGRL; i ++) {
        ibool[nspec * NGLL + i] = idx;
        idx += 1;
    }
    nglob = ibool[nspec * NGLL + NGRL - 1] + 1;

    // skeleton and nspec/ac/el
    int id = 0;
    nspec_ac = 0; nspec_el = 0;
    is_elastic.resize(nspec+1); 
    is_acoustic.resize(nspec+1);
    for(int ig = 0; ig < nregion_ - 1; ig ++) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        double h = (depth_[iend] - depth_[istart]) / nel[ig];
        for(int j = 0; j < nel[ig]; j ++) {
            skel[id * 2 + 0] = depth_[istart] + h * j;
            skel[id * 2 + 1] = depth_[istart] + h * (j + 1.);
            is_elastic[id] = is_el_reg[ig];
            is_acoustic[id] = is_ac_reg[ig];
            if(is_elastic[id]) nspec_el += 1;
            if(is_acoustic[id]) nspec_ac += 1;

            // set iregion flag
            iregion_flag[id] = ig;

            id += 1;
        }
    }

    // half space skeleton
    nspec_el_grl = 0; nspec_ac_grl = 0;
    double scale = PHASE_VELOC_MAX / freq / xgrl[NGRL-1] * 30;  // up to 50 wavelength
    skel[nspec * 2 + 0] = depth_[nz_-1];
    skel[nspec * 2 + 1] = depth_[nz_-1] + xgrl[NGRL-1] * scale;
    iregion_flag[nspec] = nregion_ - 1;

    // half space material type
    is_elastic[nspec] = is_el_reg[nregion_-1];
    is_acoustic[nspec] = is_ac_reg[nregion_-1];
    if(is_acoustic[nspec]) nspec_ac_grl = 1;
    if(is_elastic[nspec]) nspec_el_grl = 1;

    // jacobians and coordinates
    for(int ispec = 0; ispec < nspec; ispec ++) {
        double h = skel[ispec*2+1] - skel[ispec*2+0];
        jaco[ispec] = h / 2.;
        for(int i = 0; i < NGLL; i ++) {
            double xi = xgll[i];
            znodes[ispec * NGLL + i] = skel[ispec*2] + h * 0.5 * (xi + 1);
        }
    }
    // compute coordinates and jaco in GRL layer
    for(int ispec = nspec; ispec < nspec + 1; ispec ++) {
        jaco[ispec] = scale;
        for(int i = 0; i < NGRL; i ++) {
            double xi = xgrl[i];
            znodes[ispec*NGLL+i] = skel[ispec*2] + xi * scale;
        }
    }

    // UNIQUE coordinates
    zstore.resize(nglob);
    for(int i = 0; i < nspec * NGLL + NGRL; i ++) {
        int iglob = ibool[i];
        zstore[iglob] = znodes[i];
    }

    // create connectivity matrix for each material
    this -> create_material_info_();

    // interpolate model
    switch (SWD_TYPE)
    {
    case 0:
        this -> create_db_love_(freq);
        break;
    case 1:
        this -> create_db_rayl_(freq);
        break;
    default:
        this -> create_db_aniso_(freq);
        break;
    }
}
