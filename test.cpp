#include <Eigen/Core>
#include <iostream>

int main() {
    Eigen::MatrixXf a(10,10);
    a.setOnes();

    using Eigen::seq;
    for(int i = 0; i < 10; i ++) {   // forward substitution
        float s = a(i,seq(0,i-1)).sum();
        std::cout << s << "\n";
    }
}