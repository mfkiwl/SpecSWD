#include "mesh/mesh.hpp"
#include "shared/GQTable.hpp"

namespace specswd
{

void solve_christoffel(float phi, const float *c21,float &cmin,float &cmax);


void Mesh:: 
compute_minmax_veloc_(float phi,std::vector<float> &vmin,std::vector<float> &vmax)
{
    vmin.resize(nregions);
    vmax.resize(nregions);

    for(int ig = 0; ig < nregions; ig ++ ) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        float v0 = 1.0e20, v1 = -1.0e20;

        for(int i = istart; i <= iend; i ++) {
            if(SWD_TYPE == 0) { // love wave
                float vsh = vsh_tomo[i];
                v0 = std::min(v0,vsh);
                // if(HAS_ATT) {
                //     vsh *= 1. + 0.125 / std::pow(QN_[i],2); // correction to second order
                // }
                v1 = std::max(v1,vsh);
            }
            else if (SWD_TYPE == 1) { // rayleigh
                if(is_el_reg[ig]) {
                    v0 = std::min(v0,vsv_tomo[i]);
                    v1 = std::max(v1,vsv_tomo[i]);
                }
                else {
                    v0 = std::min(v0,vpv_tomo[i]);
                    v1 = std::max(v1,vpv_tomo[i]);
                }
            }
            else { // aniso
                float temp[21],cmin,cmax;
                for(int j = 0; j < 21; j ++) {
                    temp[j] = c21_tomo[j*21+i];
                }
                solve_christoffel(phi,temp,cmin,cmax);
                v0 = std::min(cmin,v0);
                v1 = std::max(cmax,v1);
            }
        }

        // set value 
        vmin[ig] = v0;
        vmax[ig] = v1;
    }
}


/**
 * @brief Create SEM database by using input model info
 * 
 * @param freq0 current frequency, in Hz
 * @param phi0 directional angle,in deg
 */
void Mesh::
create_database(float freq0,float phi0)
{
    // copy constants
    this -> freq = freq0;
    this -> phi = phi0 * M_PI / 180.;

    using namespace GQTable;

    std::vector<float> vmin,vmax;
    this -> compute_minmax_veloc_(phi,vmin,vmax);

    // find min/max vs
    PHASE_VELOC_MAX = -1.;
    PHASE_VELOC_MIN = 1.0e20;
    for(int i = 0; i < nregions; i ++) {
        PHASE_VELOC_MAX = std::max(vmax[i],PHASE_VELOC_MAX);
        PHASE_VELOC_MIN = std::min(vmin[i],PHASE_VELOC_MIN);
    }
    PHASE_VELOC_MIN *= 0.85;
    
    // loop every region to find best element size
    nspec = 0;
    std::vector<int> nel(nregions - 1);
    for(int ig = 0; ig < nregions - 1; ig ++) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        
        float maxdepth = depth_tomo[iend] - depth_tomo[istart];
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
    for(int ig = 0; ig < nregions - 1; ig ++) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        float h = (depth_tomo[iend] - depth_tomo[istart]) / nel[ig];
        for(int j = 0; j < nel[ig]; j ++) {
            skel[id * 2 + 0] = depth_tomo[istart] + h * j;
            skel[id * 2 + 1] = depth_tomo[istart] + h * (j + 1.);
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
    float scale = PHASE_VELOC_MAX / freq / xgrl[NGRL-1] * 20;  // up to 50 wavelength
    skel[nspec * 2 + 0] = depth_tomo[nz_tomo-1];
    skel[nspec * 2 + 1] = depth_tomo[nz_tomo-1] + xgrl[NGRL-1] * scale;
    iregion_flag[nspec] = nregions - 1;

    // half space material type
    is_elastic[nspec] = is_el_reg[nregions-1];
    is_acoustic[nspec] = is_ac_reg[nregions-1];
    if(is_acoustic[nspec]) nspec_ac_grl = 1;
    if(is_elastic[nspec]) nspec_el_grl = 1;

    // jacobians and coordinates
    for(int ispec = 0; ispec < nspec; ispec ++) {
        float h = skel[ispec*2+1] - skel[ispec*2+0];
        jaco[ispec] = h / 2.;
        for(int i = 0; i < NGLL; i ++) {
            float xi = xgll[i];
            znodes[ispec * NGLL + i] = skel[ispec*2] + h * 0.5 * (xi + 1);
        }
    }
    // compute coordinates and jaco in GRL layer
    for(int ispec = nspec; ispec < nspec + 1; ispec ++) {
        jaco[ispec] = scale;
        for(int i = 0; i < NGRL; i ++) {
            float xi = xgrl[i];
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
        this -> create_db_love_();
        break;
    case 1:
        this -> create_db_rayl_();
        break;
    default:
        this -> create_db_aniso_();
        break;
    }
}


/**
 * @brief create database for Love wave
 */
void Mesh:: 
create_db_love_()
{
    size_t size = ibool_el.size();

    // interpolate base model
    xrho_el.resize(size); 
    xL.resize(size);
    xN.resize(size);
    this -> interp_model(rho_tomo.data(),el_elmnts,xrho_el);
    this -> interp_model(vsh_tomo.data(),el_elmnts,xN);
    this -> interp_model(vsv_tomo.data(),el_elmnts,xL);
    for(size_t i = 0; i < size; i ++) {
        float r = xrho_el[i];
        xN[i] = std::pow(xN[i],2) * r;
        xL[i] = std::pow(xL[i],2) * r;
    }

    // Q model
    nQmodel_ani = 0;
    if(HAS_ATT) {
        // interpolate Q model
        xQL.resize(size); xQN.resize(size);
        this -> interp_model(QL_tomo.data(),el_elmnts,xQL);
        this -> interp_model(QN_tomo.data(),el_elmnts,xQN);
        nQmodel_ani = 2;
    }
}


/**
 * @brief create database for Love wave
 */
void Mesh:: 
create_db_rayl_()
{
    size_t size_el = ibool_el.size();
    size_t size_ac = ibool_ac.size();

    // allocate space for density
    xrho_el.resize(size_el);
    xrho_ac.resize(size_ac);

    // interpolate xrho
    this -> interp_model(rho_tomo.data(),el_elmnts,xrho_el);
    this -> interp_model(rho_tomo.data(),ac_elmnts,xrho_ac);

    // temp arrays
    xA.resize(size_el); xL.resize(size_el);
    xC.resize(size_el); xeta.resize(size_el);
    xkappa_ac.resize(size_ac);

    // interpolate parameters in elastic domain
    this -> interp_model(vph_tomo.data(),el_elmnts,xA);
    this -> interp_model(vpv_tomo.data(),el_elmnts,xC);
    this -> interp_model(vsv_tomo.data(),el_elmnts,xL);
    this -> interp_model(eta_tomo.data(),el_elmnts,xeta);
    for(size_t i = 0; i < size_el; i ++) {
        double r = xrho_el[i];
        xA[i] = xA[i] * xA[i] * r;
        xC[i] = xC[i] * xC[i] * r;
        xL[i] = xL[i] * xL[i] * r;
    }

    // acoustic domain
    this -> interp_model(&vph_tomo[0],ac_elmnts,xkappa_ac);
    for(size_t i = 0; i < size_ac; i ++) {
        xkappa_ac[i] = xkappa_ac[i] * xkappa_ac[i] * xrho_ac[i];
    }

    nQmodel_ani = 0;
    if(HAS_ATT) {
        // allocate space for  Q
        xQL.resize(size_el); xQA.resize(size_el);
        xQC.resize(size_el); xQk_ac.resize(size_ac);
        this -> interp_model(QL_tomo.data(),el_elmnts,xQL);
        this -> interp_model(QC_tomo.data(),el_elmnts,xQC);
        this -> interp_model(QA_tomo.data(),el_elmnts,xQA);
        this -> interp_model(QC_tomo.data(),ac_elmnts,xQk_ac);

        nQmodel_ani = 3;
    }
}

/**
 * @brief create database for Love wave
 */
void Mesh:: 
create_db_aniso_()
{
    size_t size_el = ibool_el.size();
    size_t size_ac = ibool_ac.size();

    // allocate space for density
    xrho_el.resize(size_el);
    xrho_ac.resize(size_ac);

    // interpolate xrho
    this -> interp_model(rho_tomo.data(),el_elmnts,xrho_el);
    this -> interp_model(rho_tomo.data(),ac_elmnts,xrho_ac);

    // allocate elastic model
    std::vector<float> xtemp_el(size_el);
    xC21.resize(21*size_el);
    xkappa_ac.resize(size_ac);
    for(int i = 0; i < 21; i ++) {
        this -> interp_model(&c21_tomo[21*nz_tomo],el_elmnts,xtemp_el);
        for(size_t j = 0; j < size_el;j ++) {
            xC21[i*size_el+j] = xtemp_el[j];
        }
    }

    // acoustic domain
    this -> interp_model(&c21_tomo[0],ac_elmnts,xkappa_ac);
    if(HAS_ATT) {
        for(int iq = 0; iq < nQmodel_ani; iq ++) {
            this -> interp_model(&Qani_tomo[iq*nz_tomo],el_elmnts,xtemp_el);
            for(size_t j = 0; j < size_el;j ++) {
                xQani[iq*size_el+j] = xtemp_el[j];
            }
        }
        this -> interp_model(&Qani_tomo[0],ac_elmnts,xQk_ac);
    }
}

} // namespace specswd
