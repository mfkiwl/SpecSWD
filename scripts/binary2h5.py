import numpy as np 
from scipy.io import FortranFile
import h5py
import sys 

def main():
    if len(sys.argv) != 4:
        print("Usage: python binary2h5.py binfile swdfile outfile")
        exit(1)
    
    # get input 
    binfile = sys.argv[1]
    swdfile = sys.argv[2]
    outfile = sys.argv[3]

    # read attributes
    fin:FortranFile = FortranFile(binfile,"r")
    SWD_TYPE = fin.read_ints('i4')[0]
    HAS_ATT = fin.read_ints('?')[0]
    nz = int(fin.read_ints('i4')[0])
    nkers = fin.read_ints('i4')[0]
    ncomps = fin.read_ints('i4')[0]

    # open swd file to read T and swd
    T = np.loadtxt(swdfile,max_rows=1,ndmin=1)
    data = np.loadtxt(swdfile,skiprows=1)
    Tid = np.int32(data[:,0])
    modeid = np.int32(data[:,-1])
    max_modes = int(np.max(data[:,-1])) + 1

    # open outfile
    fout:h5py.File = h5py.File(outfile,"w")
    fout.create_group("swd")
    for imode in range(max_modes):
        gname = f"swd/mode{imode}/"
        fout.create_group(f"{gname}")
        idx = modeid == imode 
        data1 = data[idx,:]
        nt1 = data1.shape[0]

        fout.create_dataset(f"{gname}/T",shape = (nt1),dtype='f4')
        fout.create_dataset(f"{gname}/c",shape = (nt1),dtype='f4')
        fout.create_dataset(f"{gname}/u",shape = (nt1),dtype='f4')
        fout[f'{gname}/T'][:] = T[Tid[idx]] 
        if HAS_ATT:
            fout.create_dataset(f"{gname}/cQ",shape = (nt1),dtype='f4')
            fout.create_dataset(f"{gname}/uQ",shape = (nt1),dtype='f4')
            fout[f'{gname}/c'][:] = data1[:,1]
            fout[f'{gname}/u'][:] = data1[:,3]
            fout[f'{gname}/cQ'][:] = 0.5 * data1[:,1] / data1[:,2]
            fout[f'{gname}/uQ'][:] = 0.5 * data1[:,3] / data1[:,4]
        else:
            fout[f'{gname}/c'][:] = data1[:,1]
            fout[f'{gname}/u'][:] = data1[:,2]

    # write kernels
    # fin:FortranFile = FortranFile(binfile,"r")
    # SWD_TYPE = fin.read_ints('i4')[0]
    # HAS_ATT = fin.read_ints('?')[0]
    # nz = int(fin.read_ints('i4')[0])
    # nkers = fin.read_ints('i4')[0]
    # ncomps = fin.read_ints('i4')[0]
    fout.attrs['HAS_ATT'] = HAS_ATT
    if SWD_TYPE == 0:
        comp_name = ['W']
        fout.attrs['WaveType'] = 'Love'
        fout.attrs['ModelType'] = 'VTI'
        PTYPE = 'f8'
        if HAS_ATT:
            dname = ['C','Q']
            #kl_name = ['rho','vsv','vsh','Qvsv','Qvsh']
            kl_name = ['vsh','vsv','Qvsh','Qvsv','rho']
            PTYPE = 'c16'
        else:
            dname = ['C']
            kl_name = ['rho','vsh','vsv']
            
    elif SWD_TYPE == 1:
        comp_name = ['U','V']
        fout.attrs['WaveType'] = 'Rayleigh'
        fout.attrs['ModelType'] = 'VTI'
        PTYPE = 'f8'

        if HAS_ATT:
            dname = ['C','Q']
            kl_name = ['rho','vph','vpv','vsv','eta','Qvph','Qvpv','Qvsv']
            PTYPE = 'c16'
        else:
            dname = ['C']
            kl_name = ['rho','vph','vpv','vsv','eta']
    else:
        comp_name = ['U','W','V']
        kl_name = ['rho_kl','vpv_kl','vph_kl','vsv_kl','vsh_kl','eta_kl','theta_kl','phi_kl']
        fout.attrs['WaveType'] = 'Full'
        fout.attrs['ModelType'] = 'TTI'
        PTYPE = 'c16'
    fout.create_group("kernels")

    for it in range(len(T)):
        idx = Tid == it 
        max_mode = np.max(modeid[idx]) + 1
        #fout.attrs[f"kernels/{it}/T"] = T[it]

        # read coordinates
        zcords = fin.read_reals('f8')
        npts = zcords.size
        fout.create_dataset(f"kernels/{it}/zcords",dtype='f4',shape =(npts))
        fout[f'kernels/{it}/zcords'][:] = zcords[:]
        for imode in range(max_mode):
            gname = f"kernels/{it}/mode{imode}"
            fout.create_group(gname)

            # read eigenfuncs
            displ = fin.read_record(PTYPE)
            displ = displ.reshape((ncomps,npts))
            for icomp in range(ncomps):
                fout.create_dataset(f"{gname}/{comp_name[icomp]}",dtype=PTYPE,shape=(npts))
                fout[f"{gname}/{comp_name[icomp]}"][:] = displ[icomp,:]
            
            # read kernels
            for iname in range(len(dname)):
                prefix = dname[iname]
                kernel = fin.read_reals('f8').reshape((nkers,nz))
                for iker in range(nkers):
                    #print(f"{gname}/{prefix}_{kl_name[iker]}")
                    fout.create_dataset(f"{gname}/{prefix}_{kl_name[iker]}",dtype='f8',shape=(nz))
                    fout[f"{gname}/{prefix}_{kl_name[iker]}"][:] = kernel[iker,:]
                    #print(f"{gname}/{prefix}_{kl_name[iker]}")

    # close 
    fin.close()
    fout.close()

    

if __name__ == "__main__":
    main()