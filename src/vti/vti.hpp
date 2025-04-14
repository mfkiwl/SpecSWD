#ifndef SPECSWD_SOLVER_H_
#define SPECSWD_SOLVER_H_

#include "mesh/mesh.hpp"

#include <complex>
#include <vector>


namespace specswd
{

typedef std::complex<float>  scmplx;

class SolverLove {

private:
    // solver matrices
    std::vector<float> Mmat,Emat,Kmat;
    std::vector<scmplx> CMmat,CEmat,CKmat;

    // QZ matrix all are column major
    std::vector<float> Qmat_,Zmat_,Smat_,Spmat_; // column major!
    std::vector<scmplx> cQmat_,cZmat_,cSmat_,cSpmat_;

public:

    // eigenfunctions/values
    void prepare_matrices(float freq,const Mesh &M);
    void compute_egn(const Mesh &M,float freq,
                    std::vector<float> &c,
                    std::vector<float> &egn,
                    bool save_qz=false);
    void compute_egn_att(const Mesh &M,float freq,
                        std::vector<scmplx> &c,
                        std::vector<scmplx> &egn,
                        bool save_qz=false);
    
    // group velocity
    float group_vel(const Mesh &M,float freq,
                    float c,const float *egn) const;
    scmplx group_vel_att(const Mesh &M,float freq,
                         scmplx c, const scmplx *egn) const ;

    // phase velocity kernels
    void compute_phase_kl(const Mesh &M,float freq,
                        float c,const float *egn,
                        std::vector<float> &frekl) const;
    void compute_phase_kl_att(const Mesh &M,float freq,
                        scmplx c, const scmplx *egn,
                        std::vector<float> &frekl_c,
                        std::vector<float> &frekl_q) const;
    // group kernel
    void compute_group_kl(const Mesh &M,float freq,
                        float c,const float *egn,
                        std::vector<float> &frekl) const;
    
    void compute_group_kl_att(const Mesh &M,float freq,
                        scmplx c, scmplx u, const scmplx *egn,
                        std::vector<float> &frekl_u,
                        std::vector<float> &frekl_q) const;
    // tranforms
    void egn2displ(const Mesh &M,float freq,float c,
                    const float*egn, float * __restrict displ) const;
    void egn2displ_att(const Mesh &M,float freq,scmplx c,const scmplx *egn,
                      scmplx * __restrict displ) const;
    void transform_kernels(const Mesh &M,std::vector<float> &frekl) const;
};

class SolverRayl {

private:
    // solver matrices
    std::vector<float> Mmat,Emat,Kmat;
    std::vector<scmplx> CMmat,CEmat,CKmat;

    // QZ matrix all are column major
    std::vector<float> Qmat_,Zmat_,Smat_,Spmat_; // column major!
    std::vector<scmplx> cQmat_,cZmat_,cSmat_,cSpmat_;

public:
    void prepare_matrices(float freq,const Mesh &M);
    void compute_egn(const Mesh &M,float freq,
                    std::vector<float> &c,
                    std::vector<float> &ur,
                    std::vector<float> &ul,
                    bool save_qz=false);
    void compute_egn_att(const Mesh &M,float freq,
                        std::vector<scmplx> &c,
                        std::vector<scmplx> &ur,
                        std::vector<scmplx> &ul,
                        bool save_qz=false);
    
    // group velocity
    float group_vel(const Mesh &M,float freq,
                    float c,const float *ur,
                    const float *ul) const;
    scmplx group_vel_att(const Mesh &M,float freq,
                        scmplx c, const scmplx *ur,
                        const scmplx *ul) const;

    // phase velocity kernels
    void compute_phase_kl(const Mesh &M,float freq,
                        float c,const float *ur,
                        const float *ul,
                        std::vector<float> &frekl) const;
    void compute_phase_kl_att(const Mesh &M,float freq,
                        scmplx c, const scmplx *ur,
                        const scmplx *ul,
                        std::vector<float> &frekl_c,
                        std::vector<float> &frekl_q) const;

    // group velocity kernels
    void compute_group_kl(const Mesh &M,float freq,
                        float c,const float *ur,
                        const float *ul,
                        std::vector<float> &frekl) const;
    void compute_group_kl_att(const Mesh &M,float freq,
                        scmplx c, scmplx u,
                        const scmplx *ur,const scmplx *ul,
                        std::vector<float> &frekl_u,
                        std::vector<float> &frekl_q) const;


    // transforms
    void egn2displ(const Mesh &M,float freq,float c,
                    const float*egn, float * __restrict displ) const;
    void egn2displ_att(const Mesh &M,float freq,scmplx c,const scmplx *egn,
                      scmplx * __restrict displ) const;
    void transform_kernels(const Mesh &M,std::vector<float> &frekl) const;
};
    

} // namespace specswd




#endif 