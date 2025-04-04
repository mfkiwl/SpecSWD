#include "solver/solver.hpp"
#include <iostream>

void solve_christoffel(double phi, const float *c21,float &cmin,float &cmax);

void SolverSEM:: 
compute_minmax_veloc_(double phi,std::vector<float> &vmin,std::vector<float> &vmax)
{
    vmin.resize(nregion_);
    vmax.resize(nregion_);

    for(int ig = 0; ig < nregion_; ig ++ ) {
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];
        float v0 = 1.0e20, v1 = -1.0e20;

        for(int i = istart; i <= iend; i ++) {
            if(SWD_TYPE == 0) { // love wave
                float vsh = vsh_[i];
                v0 = std::min(v0,vsh);
                // if(HAS_ATT) {
                //     vsh *= 1. + 0.125 / std::pow(QN_[i],2); // correction to second order
                // }
                v1 = std::max(v1,vsh);
            }
            else if (SWD_TYPE == 1) { // rayleigh
                if(is_el_reg[ig]) {
                    v0 = std::min(v0,std::min(vsv_[i],vsh_[i]));
                    v1 = std::max(v1,std::max(vsv_[i],vsh_[i]));
                }
                else {
                    v0 = std::min(v0,vpv_[i]);
                    v1 = std::max(v1,vpv_[i]);
                }
            }
            else { // aniso
                float temp[21],cmin,cmax;
                for(int j = 0; j < 21; j ++) {
                    temp[j] = c21_[j*21+i];
                }
                solve_christoffel(phi,temp,cmin,cmax);
                v0 = std::min(cmin,v0);
                v1 = std::max(cmax,v1);
            }
        }

        // set value 
        vmin[ig] = v0;
        vmax[ig] = v1;
    }
}

/**
 * @brief create database for Love wave
 * @param freq current frequency,in Hz
 */
void SolverSEM:: 
create_db_love_(double freq)
{
    size_t size = ibool_el.size();

    // interpolate macro
    #define INTP(A,v1d) this -> interp_model(&v1d[0],el_elmnts,xtemp); \
                       for(size_t i = 0; i < size; i ++) {\
                            A[i] = xtemp[i] * xtemp[i] * xrho_el[i];}

    // interpolate base model
    xrho_el.resize(size); 
    xL.resize(size);
    xN.resize(size);
    this -> interp_model(&rho_[0],el_elmnts,xrho_el);
    this -> interp_model(&vsh_[0],el_elmnts,xN);
    this -> interp_model(&vsv_[0],el_elmnts,xL);
    for(size_t i = 0; i < size; i ++) {
        xN[i] = std::pow(xN[i],2) * xrho_el[i];
        xL[i] = std::pow(xL[i],2) * xrho_el[i];
    }

    // Q model
    nQmodel_ani = 0;
    if(HAS_ATT) {
        // interpolate Q model
        xQL.resize(size); xQN.resize(size);
        this -> interp_model(&QL_[0],el_elmnts,xQL);
        this -> interp_model(&QN_[0],el_elmnts,xQN);
        nQmodel_ani = 2;
    }
}

/**
 * @brief create database for Love wave
 * @param freq current frequency,in Hz
 */
void SolverSEM:: 
create_db_rayl_(double freq)
{
    size_t size_el = ibool_el.size();
    size_t size_ac = ibool_ac.size();

    // allocate space for density
    xrho_el.resize(size_el);
    xrho_ac.resize(size_ac);

    // interpolate xrho
    this -> interp_model(rho_.data(),el_elmnts,xrho_el);
    this -> interp_model(rho_.data(),ac_elmnts,xrho_ac);

    // temp arrays
    std::vector<double> xtemp_el(size_el);
    xA.resize(size_el); xL.resize(size_el);
    xC.resize(size_el); xeta.resize(size_el);
    xkappa_ac.resize(size_ac);

    // interpolate parameters in elastic domain
    this -> interp_model(&vph_[0],el_elmnts,xA);
    this -> interp_model(&vpv_[0],el_elmnts,xC);
    this -> interp_model(&vsv_[0],el_elmnts,xL);
    this -> interp_model(&eta_[0],el_elmnts,xeta);
    for(size_t i = 0; i < size_el; i ++) {
        double r = xrho_el[i];
        xA[i] = xA[i] * xA[i] * r;
        xC[i] = xC[i] * xC[i] * r;
        xL[i] = xL[i] * xL[i] * r;
    }

    // acoustic domain
    this -> interp_model(&vph_[0],ac_elmnts,xkappa_ac);
    for(size_t i = 0; i < size_ac; i ++) {
        xkappa_ac[i] = xkappa_ac[i] * xkappa_ac[i] * xrho_ac[i];
    }

    nQmodel_ani = 0;
    if(HAS_ATT) {
        // allocate space for  Q
        xQL.resize(size_el); xQA.resize(size_el);
        xQC.resize(size_el); xQk_ac.resize(size_ac);
        this -> interp_model(QL_.data(),el_elmnts,xQL);
        this -> interp_model(QC_.data(),el_elmnts,xQC);
        this -> interp_model(QA_.data(),el_elmnts,xQA);
        this -> interp_model(QC_.data(),ac_elmnts,xQk_ac);

        nQmodel_ani = 3;
    }
}

/**
 * @brief create database for Love wave
 * @param freq current frequency,in Hz
 */
void SolverSEM:: 
create_db_aniso_(double freq)
{
    
}