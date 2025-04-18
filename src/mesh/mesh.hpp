#ifndef SPECSWD_MESH_H_
#define SPECSWD_MESH_H_

#include <complex>
#include <vector>
#include <array>

namespace specswd
{

struct Mesh {

    // SEM Mesh
    int nspec,nspec_grl; // no. of elements for gll/grl layer
    int nglob; // no. of unique points
    std::vector<int> ibool; // connectivity matrix, shape(nspec * NGLL + NGRL)
    std::vector<float> skel;  // skeleton, shape(nspec * 2 + 2)
    std::vector<float> znodes; // shape(nspec * NGLL + NGRL)
    std::vector<float> jaco; // jacobian for GLL, shape(nspec + 1) dz / dxi
    std::vector<float> zstore; // shape(nglob)  

    // element type for each medium
    int nspec_ac,nspec_el;
    int nspec_ac_grl,nspec_el_grl;
    std::vector<char> is_elastic, is_acoustic;
    std::vector<int> el_elmnts,ac_elmnts; // elements for each media, shape(nspec_? + nspec_?_grl)

    // unique array for acoustic/elastic
    int nglob_ac, nglob_el;
    std::vector<int> ibool_el, ibool_ac; // connectivity matrix, shape shape(nspec_? + nspec_?_grl)

    // density and elastic parameters
    std::vector<float> xrho_ac; // shape(nspec_ac * NGLL + nspec_ac_grl * NGRL)
    std::vector<float> xrho_el; // shape (nsepc_el * NGLL + nspec_el_grl * NGRL)

    // attenuation/type flag
    bool HAS_ATT;
    int SWD_TYPE; // =0 Love wave, = 1 for Rayleigh = 2 full aniso
    
    // vti media
    std::vector<float> xA,xC,xL,xeta,xN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL)
    std::vector<float> xQA,xQC,xQL,xQN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL), Q model

    // full anisotropy
    int nQmodel_ani; // no. of Q used for anisotropy
    std::vector<float> xC21; // shape(21,nspec_el * NGLL+ nspec_el_grl * NGRL)
    std::vector<float> xQani; // shape(nQmodel_ani,nspec_el * NGLL+ nspec_el_grl * NGRL)

    // fluid vti
    std::vector<float> xkappa_ac,xQk_ac;

    // fluid-elastic boundary
    int nfaces_bdry;
    std::vector<int> ispec_bdry; // shape(nfaces_bdry,2) (i,:) = [ispec_ac,ispec_el]
    std::vector<char> bdry_norm_direc; //  shape(nfaces_bdry), = 1 point from acoustic -> z direc elastic

    int nz_tomo, nregions;
    std::vector<float> rho_tomo;
    std::vector<float> vpv_tomo,vph_tomo,vsv_tomo,vsh_tomo,eta_tomo;
    std::vector<float> QC_tomo,QA_tomo,QL_tomo,QN_tomo;
    std::vector<float> c21_tomo,Qani_tomo;
    std::vector<float> depth_tomo;
    std::vector<int> region_bdry; // shape(nregions,2)
    std::vector<int> iregion_flag; // shape(nspec + 1), return region flag

    // interface with layered model
    std::vector<char> is_el_reg, is_ac_reg; // shape(nregions)

    float PHASE_VELOC_MIN,PHASE_VELOC_MAX;

    // current frequency
    float freq;

    // public functions
    void read_model(const char *filename);
    void create_database(float freq,float phi);
    void print_model() const;
    void print_database() const;
    void allocate_1D_model(int nz0,int swd_type,int has_att);
    void create_model_attributes();

    // interpolate model
    void interp_model(const float *param,const std::vector<int> &elmnts,std::vector<float> &md) const;
    void project_kl(const float *frekl, float *kl_out) const;

    // private functions below
    // ==============================
    //
    void create_material_info_();

    // 1-D model
    void read_model_header_(const char *filename);
    void read_model_love_(const char *filename);
    void read_model_rayl_(const char *filename);
    void read_model_full_aniso_(const char *filename);

    // create SEM database
    void compute_minmax_veloc_(float phi,std::vector<float> &vmin,std::vector<float> &vmax);
    void create_db_love_();
    void create_db_rayl_();
    void create_db_aniso_();
};
    
} // namespace specswd




#endif