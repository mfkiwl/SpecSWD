#include <Eigen/Eigenvalues>
#include <iostream>

template <typename T>  T
get_cQ_kl(T a, double &b, const double &c)
{   
    if(std::empty(c)) {
        return a + b;
    }
    else {
        return a + b + c;
    }
}


int main() {
    Eigen::Matrix2d A,B,A1,B1;
    A.setRandom();
    A = (A + A.transpose()) * 0.5;
    B << 1.,0.5,0.5,1.2;
    std::cout << B << "\n";

    A1 = A;
    B1 = B;

    int ng = 2;
    int sdim;
    Eigen::Vector2d alphar,alphai,beta;
    Eigen::Matrix2d Qmat,Zmat;
    LAPACKE_dgges(LAPACK_COL_MAJOR,'V','V','N',NULL,
        ng,A.data(),ng,B.data(),ng,&sdim,alphar.data(),
        alphai.data(),beta.data(),Qmat.data(),ng,
        Zmat.data(),ng);

    Eigen::Matrix2d BB = B.triangularView<Eigen::Upper>();

    std::cout << Qmat * A * Zmat.transpose() - A1 << "\n";
    std::cout << Qmat * B * Zmat.transpose() - B1 << "\n";
    std::cout << BB << "\n";

    double b = 2., c = 3.;
    double a = get_cQ_kl(1.,b,c);
    std::cout << get_cQ_kl(1.,b,c) << " " << get_cQ_kl(1.,b) << "\n";
}