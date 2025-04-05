#ifndef SPECSWD_SOLVER_H_
#define SPECSWD_SOLVER_H_

#include <complex>
#include <vector>
#include <array>

class SolverSEM {

    typedef std::complex<double> dcmplx;

public:
    // GLL/GRL nodes and weight
    // SEM Mesh
    int nspec,nspec_grl; // no. of elements for gll/grl layer
    int nglob; // no. of unique points
    std::vector<int> ibool; // connectivity matrix, shape(nspec * NGLL + NGRL)
    std::vector<float> skel;  // skeleton, shape(nspec * 2 + 2)
    std::vector<double> znodes; // shape(nspec * NGLL + NGRL)
    std::vector<double> jaco; // jacobian for GLL, shape(nspec + 1) dz / dxi
    std::vector<double> zstore; // shape(nglob)  

    // element type for two medium
    int nspec_ac,nspec_el;
    int nspec_ac_grl,nspec_el_grl;
    std::vector<char> is_elastic, is_acoustic;
    std::vector<int> el_elmnts,ac_elmnts; // elements for each media, shape(nspec_? + nspec_?_grl)

    // unique array for acoustic/elastic
    int nglob_ac, nglob_el;
    std::vector<int> ibool_el, ibool_ac; // connectivity matrix, shape shape(nspec_? + nspec_?_grl)

    // density and elastic parameters
    std::vector<double> xrho_ac; // shape(nspec_ac * NGLL + nspec_ac_grl * NGRL)
    std::vector<double> xrho_el; // shape (nsepc_el * NGLL + nspec_el_grl * NGRL)

    // attenuation/type flag
    bool HAS_ATT;
    int SWD_TYPE; // =0 Love wave, = 1 for Rayleigh = 2 full aniso
    
    // vti media
    std::vector<double> xA,xC,xL,xeta,xN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL)
    std::vector<double> xQA,xQC,xQL,xQN; // shape(nspec_el * NGLL+ nspec_el_grl * NGRL), Q model

    // full anisotropy
    int nQmodel_ani; // no. of Q used for anisotropy
    std::vector<double> xC21; // shape(21,nspec_el * NGLL+ nspec_el_grl * NGRL)
    std::vector<double> xQani; // shape(nQmodel_ani,nspec_el * NGLL+ nspec_el_grl * NGRL)

    // fluid vti
    std::vector<double> xkappa_ac,xQk_ac;

    // fluid-elastic boundary
    int nfaces_bdry;
    std::vector<int> ispec_bdry; // shape(nfaces_bdry,2) (i,:) = [ispec_ac,ispec_el]
    std::vector<char> bdry_norm_direc; //  shape(nfaces_bdry), = 1 point from acoustic -> z direc elastic

private:
    double PHASE_VELOC_MIN,PHASE_VELOC_MAX;

    int nz_, nregion_;
    std::vector<float> rho_;
    std::vector<float> vpv_,vph_,vsv_,vsh_,eta_;
    std::vector<float> QC_,QA_,QL_,QN_;
    std::vector<float> c21_,Qc21_;
    std::vector<float> depth_;
    std::vector<int> region_bdry; // shape(nregion_,2)
    std::vector<int> iregion_flag; // shape(nspec + 1), return region flag

    // interface with layered model
    std::vector<char> is_el_reg, is_ac_reg; // shape(nregion_)

    // solver matrices
    std::vector<double> Mmat,Emat,Kmat,Hmat;
    std::vector<dcmplx> CMmat,CEmat,CKmat,CHmat;

    // QZ matrix all are column major
    std::vector<double> Qmat_,Zmat_,Smat_,Spmat_; // column major!
    std::vector<dcmplx> cQmat_,cZmat_,cSmat_,cSpmat_;


//functions
public:
    SolverSEM(){};
    void read_model(const char *filename);
    void create_database(double freq,double phi);
    void print_model() const;
    void print_database() const;
    void allocate_1D_model(int nz0,int swd_type,int has_att);
    void create_model_attributes();

    // 1-D model info
    int tomo_size()const  {return nz_;} 

    // eigenfunction
    void compute_slegn(double freq,std::vector<double> &c,
                        std::vector<double> &displ,
                        bool save_qz=false);
    void compute_slegn_att(double freq,std::vector<dcmplx> &c,
                        std::vector<dcmplx> &displ,
                        bool save_qz=false);
    void compute_sregn(double freq,std::vector<double> &c,
                        std::vector<double> &ur,
                        std::vector<double> &ul,
                        bool save_qz = false);
    void compute_sregn_att(double freq,std::vector<dcmplx> &c,
                        std::vector<dcmplx> &ur,
                        std::vector<dcmplx> &ul,
                        bool save_qz = false);
    
    void compute_egn_aniso(double freq,std::vector<double> &c,
                            std::vector<double> &ur,
                            std::vector<double> &ul,
                            bool save_qz = false);
    void compute_egn_aniso_att(double freq,std::vector<dcmplx> &c,
                        std::vector<dcmplx> &ur,
                        std::vector<dcmplx> &ul,
                        bool save_qz = false);

    // group velocity
    double get_group_vti(double freq,double c,const double *legn,const double *regn) const; 

    // group velocity and phase velocity kernels
    double compute_love_kl(double freq,double c,const double *displ, std::vector<double> &frekl) const;
    dcmplx compute_love_kl_att(double freq,dcmplx c,const dcmplx *displ, 
                                std::vector<double> &frekl_c,
                                std::vector<double> &frekl_q) const;
    double compute_rayl_kl(double freq,double c,const double *displ, 
                          const double *ldispl, std::vector<double> &frekl) const;
    dcmplx compute_rayl_kl_att(double freq,dcmplx c,const dcmplx *displ, 
                            const dcmplx *ldispl, 
                            std::vector<double> &frekl_c,
                            std::vector<double> &frekl_q) const;

    // group velocity kernels
    void compute_love_group_kl(double freq,double c,const double *displ, std::vector<double> &frekl) const;

    void interp_model(const float *param,const std::vector<int> &elmnts,std::vector<double> &md) const;
    void project_kl(const double *frekl, double *kl_out) const;

    // transformation
    void egn2displ_vti(double freq,double c,const double *egn, 
                    double * __restrict displ) const;
    void egn2displ_vti_att(double freq,dcmplx c,const dcmplx *egn,
                        dcmplx * __restrict disp) const;
    void egn2displ_aniso(double freq,dcmplx c,double phi,const dcmplx *egn,
                        dcmplx * __restrict disp) const;
    void transform_kernels(std::vector<double> &frekl) const;

private:
    void create_material_info_();

    // 1-D model
    void read_model_header_(const char *filename);
    void read_model_love_(const char *filename);
    void read_model_rayl_(const char *filename);
    void read_model_full_aniso_(const char *filename);

    // create SEM database
    void compute_minmax_veloc_(double phi,std::vector<float> &vmin,std::vector<float> &vmax);
    void create_db_love_(double freq);
    void create_db_rayl_(double freq);
    void create_db_aniso_(double freq);

    // eigenvalue part
    void prepare_matrices_love_(double freq);
    void prepare_matrices_love_att_(double freq);
    void prepare_matrices_rayl_(double freq);
    void prepare_matrices_rayl_att_(double freq);
    void prepare_mat_aniso_(double freq,double phi);

};

#endif