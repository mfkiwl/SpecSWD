#include <iostream>
#include <Eigen/Core>

int main(){
    Eigen::VectorXd K(5);
    K.setRandom();

    Eigen::MatrixXd B = K.asDiagonal();

    std::cout << B << "\n";
}