#include "vti/vti.hpp"

#include <Eigen/Core>

namespace specswd
{

template <typename T>  T
get_love_group_vel(int ng,float freq,
                    T c, const T *egn, 
                    const float *Mmat,
                    const T *Kmat)
{
    Eigen::Map<const Eigen::ArrayX<T>> x(egn,ng);
    Eigen::Map<const Eigen::ArrayX<T>> K(Kmat,ng);
    Eigen::Map<const Eigen::ArrayX<float>> M(Mmat,ng);

    T u = (x * K * x).sum() / (c * (x * M * x).sum());

    return u;
}

/**
 * @brief compute velocity of love wave, elastic case
 */
float SolverLove:: 
group_vel(const Mesh &M,
          float c,const float *egn) const
{
    return  get_love_group_vel(M.nglob_el,M.freq,c,egn,Mmat.data(),Kmat.data());
}

/**
 * @brief compute velocity of love wave, anelastic case
 */
scmplx SolverLove:: 
group_vel_att(const Mesh &M,
          scmplx c,const scmplx *egn) const
{
    return get_love_group_vel(M.nglob_el,M.freq,c,egn,Mmat.data(),CKmat.data());
}


template <typename T = float >  T
get_rayl_group_vel(int ng,float freq,
                    T c, const T *ur,const T *ul,
                    const T *Mmat,
                    const T *Kmat)
{
    typedef Eigen::Matrix<T,-1,-1,Eigen::RowMajor> mat2;
    Eigen::Map<const Eigen::VectorX<T>> x(ur,ng),y(ul,ng);
    Eigen::Map<const mat2> K(Kmat,ng,ng);
    Eigen::Map<const Eigen::VectorX<T>> M(Mmat,ng);

    T u_nume = (y.adjoint() * K * x);
    T u_deno = c* y.adjoint() * M.asDiagonal() * x;
    T u = u_nume / u_deno;

    return u;
}

/**
 * @brief compute velocity of love wave, elastic case
 */
float SolverRayl:: 
group_vel(const Mesh &M,
          float c,const float *ur,
          const float *ul) const
{
    int ng = M.nglob_el*2 + M.nglob_ac;
    return get_rayl_group_vel(ng,M.freq,c,ur,ul,Mmat.data(),Kmat.data());
}

/**
 * @brief compute velocity of love wave, visco-elastic case
 */
scmplx SolverRayl:: 
group_vel_att(const Mesh &M,
             scmplx c,const scmplx *ur,
             const scmplx *ul) const
{
    int ng = M.nglob_el*2 + M.nglob_ac;
    return get_rayl_group_vel(ng,M.freq,c,ur,ul,CMmat.data(),CKmat.data());
}

} // namespace specswd
