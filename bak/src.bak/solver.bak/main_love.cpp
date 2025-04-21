#include "solver/solver.hpp"
#include "GQTable.hpp"
#include "swdio.hpp"

#include <iostream>
#include <filesystem>

int main (int argc, char **argv){
    // read model name
    if(argc != 5) {
        printf("Usage: ./surflove modelfile f1 f2 nt\n");
        printf("freqs = logspace(log10(f1),log10(f2),nt)\n");
        exit(1);
    }

    // initialize GLL
    GQTable:: initialize();

    // read model
    const char *filename = argv[1];
    SolverSEM sol;
    sol.read_model(filename);
    sol.create_model_attributes();
    int nz = sol.tomo_size();

    // check if it's love wave
    if(sol.SWD_TYPE != 0) {
        printf("THis module can only handle love wave!\n");
        exit(1);
    }

    // print info to debug
    sol.print_model();

    // Period
    int nt;
    float f1,f2;
    sscanf(argv[2],"%g",&f1); sscanf(argv[3],"%g",&f2);
    sscanf(argv[4],"%d",&nt);
    f1 = std::log10(f1); f2 = std::log10(f2);
    if(f1 > f2) std::swap(f1,f2);
    std::vector<double> freq(nt);
    for(int it = 0; it < nt; it ++) {
        double coef = (nt - 1);
        if(coef == 0.) coef = 1.;
        coef = 1. / coef;
        double f = f1 + (f2 - f1) * coef * it;
        freq[it] = std::pow(10,f);
    }

    // create output dir
    if(!std::filesystem::exists("out/"))
        std::filesystem::create_directory("out/");
    
    // open file to write out data
    FILE *fp = fopen("out/swd.txt","w");
    FILE *fio = fopen("out/database.bin","wb");
#ifdef SPEC_DEBUG
    FILE *fdb = fopen("mesh.bin","wb");
#endif
    for(int it = 0; it < nt; it ++) {
        fprintf(fp,"%g ",1. / freq[it]);
    }
    fprintf(fp,"\n");

    // write meta data int database
    int nkers = 3,ncomp = 1;
    write_binary_f(fio,&sol.SWD_TYPE,1);
    write_binary_f(fio,&sol.HAS_ATT,1);
    if(sol.HAS_ATT) {
        nkers = 5;
    }
    write_binary_f(fio,&nz,1);
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // compute phase velocity for each frequency
    typedef std::complex<double> dcmplx;
    for(int it = 0; it < nt; it ++) {
        // create database
        sol.create_database(freq[it],0.);

        // write coordinates
        write_binary_f(fio,sol.znodes.data(),sol.znodes.size());

#ifdef SPEC_DEBUG
        // write mesh
        using namespace GQTable;
        write_binary_f(fdb,sol.ibool_el.data(),sol.ibool_el.size());
        write_binary_f(fdb,sol.jaco.data(),sol.jaco.size());
        write_binary_f(fdb,sol.xN.data(),sol.xN.size());
        write_binary_f(fdb,sol.xQN.data(),sol.xQN.size());
        write_binary_f(fdb,wgll.data(),wgll.size());
        write_binary_f(fdb,wgrl.data(),wgrl.size());
#endif

        // get database dimension
        int ng = sol.nglob_el;

        if(!sol.HAS_ATT) {
            std::vector<double> c,egn,u,frekl;
            std::vector<double> frekl_tomo;
            std::vector<double> displ;
            sol.compute_slegn(freq[it],c,egn,true);

            // allocate group velocity
            int nc = c.size();
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                u[ic] = sol.compute_love_kl(freq[it],c[ic],&egn[ic*ng],frekl);

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %d\n",it,c[ic],u[ic],ic);

                // write displ
                displ.resize(sol.ibool_el.size());
                sol.egn2displ_vti(freq[it],c[ic],&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // transform kernels
                sol.transform_kernels(frekl);
                frekl_tomo.resize(nkers*nz);
                int npts = sol.ibool_el.size();
                for(int iker = 0; iker < nkers; iker ++) {
                    sol.project_kl(&frekl[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
        else {
            std::vector<dcmplx> c,egn,legn,u;
            std::vector<double> frekl_c,frekl_q;
            std::vector<double> frekl_tomo;
            std::vector<dcmplx> displ;
            sol.compute_slegn_att(freq[it],c,egn,true);
            
            // allocate group velocity
            int nc = c.size();
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                u[ic] = sol.compute_love_kl_att(freq[it],c[ic],&egn[ic*ng],frekl_c,frekl_q);

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic].real(),c[ic].imag(),
                                                u[ic].real(),u[ic].imag(),ic);

                // write displ
                displ.resize(sol.ibool_el.size());
                sol.egn2displ_vti_att(freq[it],c[ic],&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // write kernels
                sol.transform_kernels(frekl_c);
                sol.transform_kernels(frekl_q);
                frekl_tomo.resize(nkers*nz);
                int npts = sol.ibool_el.size();
                for(int iker = 0; iker < nkers; iker ++) {
                    sol.project_kl(&frekl_c[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                for(int iker = 0; iker < nkers; iker ++) {
                    sol.project_kl(&frekl_q[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);
    
#ifdef SPEC_DEBUG
    fclose(fdb);
#endif

    return 0;
}