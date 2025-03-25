#ifndef SWD_CONST_H_
#define SWD_CONST_H_

#include <array>

namespace GQTable
{
    
const int NGLL = 7, NGRL = 20;
extern std::array<double,NGLL> xgll,wgll;
extern std::array<double,NGRL> xgrl,wgrl;
extern std::array<double,NGLL*NGLL> hprimeT,hprime; // hprimeT(i,j) = l'_i(xi_j)
extern std::array<double,NGRL*NGRL> hprimeT_grl,hprime_grl;

void initialize();

} // GQTable


#endif