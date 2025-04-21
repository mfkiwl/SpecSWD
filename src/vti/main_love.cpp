#include "vti/vti.hpp"
#include "shared/iofunc.hpp"
#include "shared/GQTable.hpp"

#include <iostream>
#include <filesystem>
#include <memory>


int main (int argc, char **argv){
    // read model name
    if(argc != 5 &&  argc != 6) {
        printf("Usage: ./surflove modelfile f1 f2 nt [KERNEL_TYPE = 0]\n");
        printf("freqs = logspace(log10(f1),log10(f2),nt)\n");
        exit(1);
    }

    // initialize GLL
    GQTable:: initialize();

    // read mesh 
    const char *filename = argv[1];
    specswd::Mesh mesh;
    mesh.read_model(filename);
    mesh.create_model_attributes();
    int nz = mesh.nz_tomo;

    // check if it's love wave
    if(mesh.SWD_TYPE != 0) {
        printf("THis module can only handle love wave!\n");
        exit(1);
    }

    // print info to debug
    mesh.print_model();

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

    int KERNEL_TYPE = 1;
    if(argc == 6) {
        sscanf(argv[5],"%d",&KERNEL_TYPE);
    }

    // create output dir
    if(!std::filesystem::exists("out/"))
        std::filesystem::create_directory("out/");
    
    // open file to write out meta data
    FILE *fp = fopen("out/swd.txt","w");
    FILE *fio = fopen("out/database.bin","wb");
    for(int it = 0; it < nt; it ++) {
        fprintf(fp,"%g ",1. / freq[it]);
    }
    fprintf(fp,"\n");

    // write meta data int database
    using specswd::write_binary_f;
    int nkers = 3,ncomp = 1;
    write_binary_f(fio,&mesh.SWD_TYPE,1);
    write_binary_f(fio,&mesh.HAS_ATT,1);
    if(mesh.HAS_ATT) {
        nkers = 5;
    }
    write_binary_f(fio,&nz,1);
    write_binary_f(fio,&nkers,1);
    write_binary_f(fio,&ncomp,1);

    // initialize solver
    std::unique_ptr<specswd::SolverLove> sol(new specswd::SolverLove);

    // compute phase velocity for each frequency
    for(int it = 0; it < nt; it ++) {
        // create database
        mesh.create_database(freq[it],0.);

        // write coordinates
        write_binary_f(fio,mesh.znodes.data(),mesh.znodes.size());

        // get database dimension
        int ng = mesh.nglob_el;

        // prepare all matrices
        sol -> prepare_matrices(mesh);

        if(!mesh.HAS_ATT) {
            std::vector<float> c,egn,u,frekl;
            std::vector<float> frekl_tomo;
            std::vector<float> displ;

            // compute eigenvalue
            sol -> compute_egn(mesh,c,egn,true);

            // allocate group velocity
            int nc = c.size();
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                u[ic] = sol -> group_vel(mesh,c[ic],&egn[ic*ng]);
                switch (KERNEL_TYPE) {
                    case 0:
                        sol -> compute_phase_kl(
                            mesh,c[ic],
                            &egn[ic*ng],frekl
                        );
                        break;
                    case 1:
                        sol -> compute_group_kl(
                            mesh,c[ic],
                            &egn[ic*ng],frekl
                        );
                        break;
                    default: {
                        printf("KERNEL_TYPE = %d is not implemented!\n",KERNEL_TYPE);
                        exit(1);
                        break;
                    }
                }

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %d\n",it,c[ic],u[ic],ic);

                // write displ
                displ.resize(mesh.ibool_el.size());
                sol->egn2displ(mesh,c[ic],&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // transform kernels
                sol -> transform_kernels(mesh,frekl);
                frekl_tomo.resize(nkers*nz);
                int npts = mesh.ibool_el.size();
                for(int iker = 0; iker < nkers; iker ++) {
                    mesh.project_kl(&frekl[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
        else {
            using specswd::scmplx;
            std::vector<scmplx> c,egn,legn,u;
            std::vector<float> frekl_c,frekl_q;
            std::vector<float> frekl_tomo;
            std::vector<scmplx> displ;

            // compute eigenvalues
            sol -> compute_egn_att(mesh,c,egn,true);
            
            // allocate group velocity
            int nc = c.size();
            u.resize(nc);

            for(int ic = 0; ic < nc; ic ++) {
                u[ic] = sol -> group_vel_att(mesh,c[ic],&egn[ic*ng]);
                switch (KERNEL_TYPE) {
                    case 0:
                        sol -> compute_phase_kl_att (
                            mesh,c[ic],
                            &egn[ic*ng],frekl_c,frekl_q
                        );
                        break;
                    case 1:
                        sol -> compute_group_kl_att(
                            mesh,c[ic],u[ic],
                            &egn[ic*ng],frekl_c,frekl_q
                        );
                        break;
                    default: {
                        printf("KERNEL_TYPE = %d is not implemented!\n",KERNEL_TYPE);
                        exit(1);
                        break;
                    }
                }

                // write T,c,u,mode
                fprintf(fp,"%d %g %g %g %g %d\n",it,c[ic].real(),u[ic].real(),
                                                c[ic].imag(),u[ic].imag(),ic);

                // write displ
                displ.resize(mesh.ibool_el.size());
                sol -> egn2displ_att(mesh,c[ic],&egn[ic*ng],displ.data());
                write_binary_f(fio,displ.data(),displ.size());

                // write kernels
                sol->transform_kernels(mesh,frekl_c);
                sol->transform_kernels(mesh,frekl_q);
                frekl_tomo.resize(nkers*nz);
                int npts = mesh.ibool_el.size();
                for(int iker = 0; iker < nkers; iker ++) {
                    mesh.project_kl(&frekl_c[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());

                for(int iker = 0; iker < nkers; iker ++) {
                    mesh.project_kl(&frekl_q[iker*npts],&frekl_tomo[iker*nz]);
                }
                write_binary_f(fio,frekl_tomo.data(),frekl_tomo.size());
            }
        }
    }

    // close file
    fclose(fio);
    fclose(fp);
    
    return 0;
}