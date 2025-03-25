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
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // compute phase velocity for each frequency
    typedef std::complex<double> dcmplx;
    for(int it = 0; it < nt; it ++) {
        // create database
        sol.create_database(freq[it],0.);

        // write coordinates
        write_binary_f(fio,sol.znodes.data(),sol.znodes.size());

        // get database dimension
        int ng = sol.nglob_el;

        if(!sol.HAS_ATT) {
            std::vector<double> c,egn,u,frekl;
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
                sol.egn2displ_love(&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // write kernels
                sol.transform_kernels(frekl);
                write_binary_f(fio,frekl.data(),frekl.size());
            }
        }
        else {
            std::vector<dcmplx> c,egn,legn,u;
            std::vector<double> frekl_c,frekl_q;
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
                sol.egn2displ_love_att(&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // write kernels
                sol.transform_kernels(frekl_c);
                sol.transform_kernels(frekl_q);
                write_binary_f(fio,frekl_c.data(),frekl_c.size());
                write_binary_f(fio,frekl_q.data(),frekl_q.size());
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);

    return 0;
}