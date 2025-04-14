#include <Eigen/Eigenvalues>

// Fast Voigt index lookup table
const int voigt_lookup[3][3] = {
    {0, 5, 4},
    {5, 1, 3},
    {4, 3, 2}
};

static int inline 
Voigt_index(int i, int j) 
{
    return voigt_lookup[i][j];
}

namespace specswd
{


/**
 * @brief find the min/max phase velocity by solving Christoffel equations G_{ik} g_k = v^2 g_i
 * @param phi direction angle, in rad
 * @param c21 c21 tensor, shape(21)
 * @param cmin/cmax min/max phase velocity
 */
void solve_christoffel(float phi, const float *c21,float &cmin,float &cmax)
{
    // direction
    double direc[3] = {std::cos(phi),std::sin(phi),0.};

    // allocate Chirstoffel matrix
    Eigen::Matrix<double,3,3,1> G; G.setZero();
    using Eigen::dcomplex;

    // set value
    for(int i = 0; i < 3; i ++) {
    for(int j = 0; j < 3; j ++) {
        int m = Voigt_index(i,j);
        for(int p = 0; p < 3; p ++) {
        for(int q = 0; q < 3; q ++) {
            int n = Voigt_index(p,q);

            // get c21 index
            int m1 = m;
            int n1 = n;
            if(m1 > n1) std::swap(m1,n1);
            int idx = m * 6 + n - (m * (m + 1)) / 2;

            // sum G_{ik} = c_{ijkl} n_j n_l
            G(i,p) += c21[idx] * direc[j] * direc[q];
        }}
    }}

    // find eigenvalues
    Eigen::Array3d vr,vi;
    LAPACKE_dgeev(
        LAPACK_ROW_MAJOR,'N','N',3,G.data(),3,
        vr.data(),vi.data(),nullptr,3,nullptr,3
    );
    Eigen::Array3d v = (vr + Eigen::dcomplex{0,1.} * vi).array().sqrt().real().cast<double>();

    // sort 
    std::sort(v.data(),v.data() + 3);

    // find the min/max 
    cmin = v[0];
    cmax = v[1]; // we need S wave 
}
    
} // namespace specswd