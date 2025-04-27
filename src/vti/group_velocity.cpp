#include "vti/vti.hpp"
#include "shared/GQTable.hpp"

#include <Eigen/Core>

namespace specswd
{

template <typename T>  T
get_love_group_vel(int ng,T c, const T *egn, 
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
    return  get_love_group_vel(M.nglob_el,c,egn,Mmat.data(),Kmat.data());
}

/**
 * @brief compute velocity of love wave, anelastic case
 */
scmplx SolverLove:: 
group_vel_att(const Mesh &M,
          scmplx c,const scmplx *egn) const
{
    return get_love_group_vel(M.nglob_el,c,egn,Mmat.data(),CKmat.data());
}


template <typename T = float >  T
get_rayl_group_vel(
    const Mesh &mesh,
    T c,
    const T *ur,
    const T *ul,
    const T *Mmat,
    const T *Kmat
)
{
    int nglob_el = mesh.nglob_el;
    int ng = nglob_el*2 + mesh.nglob_ac;
    typedef Eigen::Matrix<T,-1,-1,Eigen::RowMajor> mat2;
    Eigen::Map<const Eigen::VectorX<T>> x(ur,ng),y(ul,ng);
    Eigen::Map<const mat2> K(Kmat,ng,ng);
    Eigen::Map<const Eigen::VectorX<T>> M(Mmat,ng);

    T u_nume = (y.adjoint() * K * x);
    T u_deno = c * (y.array().conjugate() * M.array() * x.array()).sum();

    // add fluid contribution
    using GQTable::NGLL;
    float om = M_PI * 2 * mesh.freq;
    T twokinv = 0.5f * c / om;
    T dwde{};
    for(int iface = 0; iface < mesh.nfaces_bdry; iface ++) {
        int ispec_ac = mesh.ispec_bdry[iface * 2 + 0];
        int ispec_el = mesh.ispec_bdry[iface * 2 + 1];
        const bool is_pos = mesh.bdry_norm_direc[iface];
        T norm = is_pos ? -1 : 1.;
        int igll_el = is_pos ? 0 : NGLL - 1;
        int igll_ac = is_pos ? NGLL - 1 : 0;

        // get ac/el global loc
        int iglob_el = mesh.ibool_el[ispec_el * NGLL + igll_el];
        int iglob_ac = mesh.ibool_ac[ispec_ac * NGLL + igll_ac];

        T x0 = x[nglob_el * 2 + iglob_ac];
        T y0 = y[nglob_el + iglob_el];
        if constexpr (std::is_same_v<T,scmplx>) {
            y0 = std::conj(y0);
        }
        dwde += (T)(2.0) * om * norm * x0 * y0;
    }
    u_deno -= dwde  * twokinv;

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
    return get_rayl_group_vel(
        M,c,ur,ul,Mmat.data(),Kmat.data()
    );
}

/**
 * @brief compute velocity of love wave, visco-elastic case
 */
scmplx SolverRayl:: 
group_vel_att(const Mesh &M,
             scmplx c,const scmplx *ur,
             const scmplx *ul) const
{
    return get_rayl_group_vel(
        M,c,ur,ul,CMmat.data(),CKmat.data()
    );
}

} // namespace specswd
