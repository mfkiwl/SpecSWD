#ifndef SPECSWD_GQTABLE_H_
#define SPECSWD_GQTABLE_H_

#include <array>

namespace GQTable
{
    
const int NGLL = 7, NGRL = 20;
extern std::array<float,NGLL> xgll,wgll;
extern std::array<float,NGRL> xgrl,wgrl;
extern std::array<float,NGLL*NGLL> hprimeT,hprime; // hprimeT(i,j) = l'_i(xi_j)
extern std::array<float,NGRL*NGRL> hprimeT_grl,hprime_grl;

void initialize();

} // GQTable


#endif