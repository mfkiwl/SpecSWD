#include "solver/solver.hpp"
#include <iostream>
#include <fstream>
#include <sstream>


template<typename T,typename ... Args>
void 
allocate(int n,T &vec1,Args& ...args)
{
    vec1.resize(n);
    std::fill(vec1.begin(),vec1.end(),0);
    if constexpr(sizeof...(args) > 0){
        allocate(n,args...);
    }
}

void SolverSEM:: 
allocate_1D_model(int nz0,int swd_type,int has_att)
{
    // copy value to mesh type 
    SWD_TYPE = swd_type;
    nz_ = nz0;
    HAS_ATT = has_att;

    // allocate space
    switch (SWD_TYPE)
    {
    case 0:
        allocate(nz_,vsv_,vsh_,rho_);
        if(HAS_ATT) {
            allocate(nz_,QN_,QL_);
        }
        break;
    case 1:
        allocate(nz_,vpv_,vph_,vsv_,rho_,eta_);
        if(HAS_ATT) allocate(nz_,QC_,QA_,QL_);
        break;
    
    case 2:
        allocate(nz_*21,c21_);
        allocate(nz_,rho_);
        if(HAS_ATT) allocate(nz_*21,Qc21_);

        break;
    default:
        printf("SWD_TYPE should in [0,1,2]!\n");
        printf("current value is %d\n",SWD_TYPE);
        exit(1);
        break;
    }

    // allocate depth
    allocate(nz_,depth_);
}


/**
 * @brief read header of 1D model, including wave type, attenutation flag, 
 * attenuation model flag
 * @param filename model filename
 */
void SolverSEM:: 
read_model_header_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    if(infile.fail()) {
        printf("cannot open %s\n",filename);
        exit(1);
    }

    // read first line
    std::string line;
    std::getline(infile,line);
    
    // read SWD_TYPE and HAS_ATT
    int dummy[2];
    {
        std::istringstream info(line);
        info >> dummy[0] >> dummy[1];
    }

    // find how many depth points in this file
    int nz = 0;
    while (std::getline(infile,line))
    {
        nz += 1;
    }
    infile.close();

    // allocate model
    this -> allocate_1D_model(nz,dummy[0],dummy[1]);

    // allocate depth
    float z = 0.;

    // read depth in file
    infile.open(filename);
    std::getline(infile,line);
    for(int i = 0; i < nz; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        info >> depth_[i];
        info.clear();

        if(i >= 1) {
            // make sure depth is no descreasing
            if(depth_[i] - z < 0) {
                printf("depth should not decrease!\n");
                printf("current/previous depth = %f %f\n",depth_[i],z);
                exit(1);
            }
            z = depth_[i];
        }
    }
}

/**
 * @brief read 1D VTI model for Love wave
 * @param filename 1D model file
 */
void SolverSEM:: 
read_model_love_(const char *filename)
{
    std::string line;
    std::ifstream infile; infile.open(filename);

    // skip header
    std::getline(infile,line);

    float temp;
    for(int i = 0; i < nz_; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        info >> temp >> rho_[i] >> vsh_[i] >> vsv_[i];
        if(HAS_ATT) {
            info >> QN_[i] >> QL_[i];
        }
        info.clear();
    }
    infile.close();
}

/**
 * @brief read 1D VTI model for Rayleigh wave
 * @param filename 1D model file
 */
void SolverSEM::
read_model_rayl_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    std::string line;

    // skip header
    std::getline(infile,line);

    for(int i = 0; i < nz_; i ++) {
        std::getline(infile,line);
        std::istringstream info(line);
        float temp;
        info >> temp >> rho_[i] >> vph_[i] >> vpv_[i] 
               >> vsv_[i] >> eta_[i];
        if(HAS_ATT) {
            info >> QA_[i] >> QC_[i] >> QL_[i];
        }
        info.clear();
    }
    infile.close();
}

/**
 * @brief read 1D full anisotropy model model for Rayleigh wave
 * @param filename 1D model file
 */
void SolverSEM::
read_model_full_aniso_(const char *filename)
{
    std::ifstream infile; infile.open(filename);
    std::string line;

    // skip header
    std::getline(infile,line);

    for(int i = 0; i < nz_; i ++) {
        float temp;
        infile >> temp >> rho_[i];
        for(int j = 0; j < 21; j ++ ) {
            infile >> c21_[j*21+i];
        }
        if(HAS_ATT) {
            for(int j = 0; j < 21; j ++) {
                infile >> Qc21_[j*21+i];
            }
        }
    }

    // close 
    infile.close();
}

/**
 * @brief read 1D model
 * @param filename 1D model file
 */
void SolverSEM::
read_model(const char *filename)
{
    this -> read_model_header_(filename);
    switch (SWD_TYPE)
    {
    case 0:
        this -> read_model_love_(filename);
        break;
    case 1:
        this -> read_model_rayl_(filename);
        break;
    case 2:
        this -> read_model_full_aniso_(filename);
        break;
    default:
        printf("SWD_TYPE should in [0,1,2]");
        printf("current SWD_TYPE = %d\n",SWD_TYPE);
        exit(1);
    }
}

static bool 
check_fluid_c21(const float *c21)
{
    bool flag = true;
    float c0 = c21[0];
    flag = flag & (c0 > 0);
    for(int i = 2; i < 21; i ++) {
        if(i == 1 || i == 6 || i == 11 ) {
            flag = flag && (c21[i] == c0);
        }
        else if(i == 2 || i == 7) {
            flag = flag && (c21[i] == 2 * c0);
        }
        else {
            flag = flag && (c21[i] == 0.);
        }
    }

    return flag;

}

void SolverSEM:: 
create_model_attributes()
{
    // first check discontinuities
    region_bdry.resize(0);
    region_bdry.reserve(10);
    int ndis = 0;
    int ipt0 = 0,ipt1 = 0;
    for(int i = 1; i < nz_; i ++) {
        if(depth_[i] == depth_[i-1]) {
            ndis += 1;
            ipt1 = i-1;

            // add to region_bdry
            region_bdry.push_back(ipt0);
            region_bdry.push_back(ipt1);
            ipt0 = ipt1 + 1;
        }
    }

    // check a discontinuity is add to half space
    if(region_bdry[region_bdry.size() - 1] != nz_ - 2) {
        printf("Please add a discontinuity at half space !\n");
        exit(1);
    }

    // half space is another region
    region_bdry.push_back(nz_-1);
    region_bdry.push_back(nz_-1);
    nregion_ = region_bdry.size() / 2;

    // now check where the fluid is
    std::vector<char> is_ac_pts; 
    is_ac_pts.resize(nz_);
    for(int i = 0; i < nz_; i ++) {
        is_ac_pts[i] = 0;
        if(SWD_TYPE == 0) { // Love
            if(vsh_[i] < 1.0e-6 || vsv_[i] < 1.0e-6) {
                printf("Love wave cannot exist in fluid layers!\n");
                printf("current velocity  vsv = %f vsh = %f\n",vsv_[i],vsh_[i]);
                exit(1);
            }
        }
        else if(SWD_TYPE == 1) { // Rayleigh 
            if(vsv_[i] < 1.0e-6) {
                is_ac_pts[i] = 1;

                // check if vpv == vph || Qvpv == Qvph
                bool flag = vpv_[i] == vph_[i];
                if(HAS_ATT) flag = flag &(QC_[i] == QA_[i]);
                if(!flag) {
                    printf("vpv and vph should be same in fluid layers\n");
                    printf("current velocity vpv = %f vph = %f\n",vpv_[i],vph_[i]);
                    printf("current velocity Qvpv = %f Qvph = %f\n",QC_[i],QA_[i]);
                    exit(1);
                }
            }
        }
        else { // full aniso
            float temp_c21[21], temp_Qc21[21];
            for(int j = 0; j < 21; j ++) {
                temp_c21[j] = c21_[j*21+i];
            }
            bool flag = check_fluid_c21(temp_c21);
            
            if(HAS_ATT) {
                for(int j = 0; j < 21; j ++) {
                    temp_Qc21[j] = Qc21_[j*21+i];
                }
                flag = flag && check_fluid_c21(temp_Qc21);
            }

            if(flag) {
                is_ac_pts[i] = 1;
            }
        }
    }

    // allocate material flag
    allocate(nregion_,is_ac_reg,is_el_reg);

    // check if all points in a region is fluid/elastic only
    for(int ig = 0; ig < nregion_; ig ++) {
        int startid = region_bdry[ig*2+0];
        int endid = region_bdry[ig*2+1];
        bool flag = is_ac_pts[startid];
        for(int i = startid+1; i <= endid; i ++) {
            if(flag != is_ac_pts[i]) {
                printf("in one region, you can only have one material !\n");
                printf("Problem region %d, index= %d - %d",ig,startid,endid);
                exit(1);
            }
        }

        // set flag
        is_ac_reg[ig] = is_ac_pts[startid];
        is_el_reg[ig] = !is_ac_pts[startid];
    }
}

void SolverSEM::
print_model() const
{
    printf("\n====================================\n");
    printf("========= Model Description ========\n");
    printf("====================================\n\n");

    std::string outinfo = "elastic";
    if(HAS_ATT) {
        outinfo = "visco-elastic";
    }

    if(SWD_TYPE == 0) { // love wave 
        printf("compute dispersions for %s Love wave\n",outinfo.c_str());
    }
    else if(SWD_TYPE == 1) { // rayleigh wave 
        printf("compute dispersions for %s Rayleigh wave\n",outinfo.c_str());
    }
    else {
        printf("compute dispersions for %s fully anisotropic wave\n",outinfo.c_str());
    }

    for(int ig = 0; ig < nregion_; ig ++) {
        if(ig == nregion_ - 1) {
            printf("\nhalf space begin at depth = %f\n",depth_[nz_ - 1]);
        }
        printf("\nregion %d:\n",ig + 1);
        printf("=======================\n");
        int istart = region_bdry[ig*2+0];
        int iend = region_bdry[ig*2+1];

        if(SWD_TYPE == 0) {
            printf("depth\t rho\t vsh\t vsv (QN QL)\t \n");
            for(int i = istart; i <= iend; i ++) {
                printf("%f %f %f %f",
                        depth_[i],rho_[i], vsh_[i],vsv_[i]);
                if(HAS_ATT) {
                    printf(" %f %f\n",QN_[i],QL_[i]);
                }
                else {
                    printf("\n");
                }
            }
        }
        else if (SWD_TYPE == 1) {
            printf("depth\t rho\t vph\t vpv\t vsv\t eta (Qvpv Qvph Qvsv)\n");
            for(int i = istart; i <= iend; i ++) {
                printf("%f %f %f %f %f",
                        depth_[i],rho_[i], vph_[i],vpv_[i],vsv_[i]);
                if(HAS_ATT) {
                    printf(" %f %f %f\n",QA_[i],QC_[i],QL_[i]);
                }
                else {
                    printf("\n");
                }
            }
        }
        else {

        }
    }
}

void SolverSEM::
print_database() const
{

    // print SEM mesh information for debug
    printf("\n====================================\n");
    printf("========= DATABASE Description ========\n");
    printf("====================================\n\n");

    printf("elements:\n");
    printf("=========================\n");
    printf("no. of nelemnts = %d\n",nspec + nspec_grl);
    printf("no. of elastic GLL/GRL nelemnts = %d %d\n",nspec_el,nspec_el_grl);
    printf("no. of acoustic GLL/GRL nelemnts = %d %d\n",nspec_ac,nspec_ac_grl);
    printf("no. of elastic wavefield points = %d\n",nglob_el);
    printf("no. of acoustic wavefield points = %d\n",nglob_ac);

    printf("\nSimulation parameters:\n");
    printf("=========================\n");
    printf("phase velocity min/max = %f %f\n",PHASE_VELOC_MIN,PHASE_VELOC_MAX);

    printf("\nElastic-Acoustic Boundary:\n");
    printf("=========================\n");
    printf("no. of E-A boundaries = %d\n",nfaces_bdry);
    for(int iface = 0; iface < nfaces_bdry; iface ++) {
        int ispec_ac = ispec_bdry[iface * 2 + 0];
        int ispec_el = ispec_bdry[iface * 2 + 1];
        printf("boundary %d:\n",iface);
        printf("\tispec_ac = %d ispec_el = %d\n",ispec_ac,ispec_el);
        int top_is_fluid = bdry_norm_direc[iface];
        printf("top material is fluid = %d\n",top_is_fluid);
    }

}