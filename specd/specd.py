from .lib import libswd
import numpy as np 

def _model_sanity_check(wavetype:str,vph=None,vpv=None,vsh=None,vsv=None,
                Qa=None,Qc=None,Qn=None,Ql=None,
                c21=None,nQani=None,Qani=None):
    if wavetype == "love":
        if (vsh is None) or (vsv is None):
            print("love wave model mustn't have None vsh/vsv!")
            exit(1)
    elif wavetype == "rayl":
        if (vsv is None) or (vpv is None) or (vph is None):
            print("rayleigh wave model mustn't have None vph/vpv/vsv!")

    pass

class SpecWorkSpace:

    def __init__(self):
        self._max_mode = 0
        self._has_att = None 
        self._use_qz = None 
        self._wavetype = None  

        pass

    def initialize(self,wavetype:str,z:np.ndarray,rho:np.ndarray,
                vph=None,vpv=None,vsh=None,vsv=None,
                eta=None,Qa=None,Qc=None,Qn=None,Ql=None,
                c21=None,nQani=None,Qani=None,
                disp=False):
        """
        initialize working space for SEM

        Parameters
        ----------
        wavetype: str
            wavetype, one of ['love','rayl','aniso']
        z: np.ndarray
            depth vector, should include discontinuities at half sapce
        rho: np.ndarray
            density, shape_like(z)
        vph/vpv/vsh/vsv: np.ndarray
            VTI parameters, shape_like(z)
        Qa/Qc/Qn/Ql: np.ndarray
            VTI quality factors, shape_like(z)
        c21: np.ndarray
            21 elastic tensor, shape(21,nz)
        nQani: int
            no. of Q models for fully anisotropy
        Qnai: np.ndarray
            quality factors, shape(nQnai,nz)
        disp: bool
            if True print model information
        
        Note
        ------------
        The input parameters should be carefully chose by user. For 
        Love wave, only vsh/vsv/Qsh/Qsv can be enabled, and for 
        Rayleigh wave, only vph/vpv/vsv can ve enabled, and 
        other params are for full anisotropy
        """

        self._wavetype = wavetype.lower()
        self._use_qz = False
        self._nz = len(z)
        assert(self._wavetype in ['love','rayl','aniso'])

        _model_sanity_check(wavetype,vph,vpv,vsh,vsv,Qa,
                            Qc,Qn,Ql,c21,nQani,Qani)
        
        # init work space by calling libswd
        self._has_att = False
        if self._wavetype == "love":
            if (Qn is not None) and (Ql is not None):
                self._has_att = True
            libswd.init_love(z,rho,vsh,vsv,Qn,Ql,self._has_att,disp)

        elif self._wavetype == 'rayl':
            if (Qn is not None) and (Qa is not None) and (Qc is not None):
                self._has_att = True
            libswd.init_rayl(z,rho,vph,vpv,vsv,eta,Qa,Qc,Ql,
                            self._has_att,disp)
        else:
            print("not implemented!")
            assert(0 == 1)

    def compute_egn(self,freq:float,max_order:int = -1,use_qz=True) -> np.ndarray:
        """
        compute eigenvalues/eigenvectors

        Parameters
        ---------
        freq: float
            current frequency
        max_order: int
            no. of orders returns
        use_qz: bool
            if False, only compute eigenvalues (phase velocities)
            if True, phase velocities and eigenfunctions will be computed

        Returns
        --------
        c: np.ndarray
            phase velocities
        """

        # save use_qz
        self._use_qz = use_qz

        if self._has_att:
            c = libswd.compute_egn_att(freq,max_order,use_qz)
        else:
            c = libswd.compute_egn(freq,max_order,use_qz)

        # save current mode number
        self._max_mode = len(c)

        return c

    def group_velocity(self,max_order:int = -1) -> np.ndarray:
        """
        compute group velocites

        Parameters
        ----------
        max_order: int
            no. of orders returns

        Returns
        --------
        u: np.ndarray
            group velocities at current frequency

        Note
        ----------
        before calling this routine, use_qz should be True in self.compute_egn
        """
        assert self._use_qz ,"please enable use_qz in self.compute_egn"

        if self._has_att:
            u = libswd.group_vel(max_order)
        else:
            u = libswd.group_vel_att(max_order)

        return u
        
    def get_phase_kl(self,imode:int):
        """
        compute phase velocity sensitivity kernels for mode {imode}

        Parameters
        -------------
        imode : int 
            compute kernels at imode, imode \belong [0,max_mode)

        Returns
        -----------
        frekl_c: np.ndarray
            phase velocity kernels, shape(nkers,self._nz)
        frekl_q: np.ndarray
            phase velocity Q kernels,shape(nkers,self._nz), only returns when
            self._has_att = True
        """
        if(imode >= self._max_mode): 
            print(f"imode should inside [0,{self._max_mode})")
        frekl_c,frekl_q = libswd.phase_kl(imode,self._has_att)

        if self._has_att:
            return frekl_c,frekl_q
        else:
            return frekl_c
        
    def get_group_kl(self,imode:int):
        """
        compute group velocity sensitivity kernels for mode {imode}

        Parameters
        -------------
        imode : int 
            compute kernels at imode, imode \belong [0,max_mode)

        Returns
        -----------
        frekl_c: np.ndarray
            group velocity kernels, shape(nkers,self._nz)
        frekl_q: np.ndarray
            group velocity Q kernels,shape(nkers,self._nz), only returns when
            self._has_att = True
        """
        if(imode >= self._max_mode): 
            print(f"imode should inside [0,{self._max_mode})")
        frekl_c,frekl_q = libswd.group_kl(imode,self._has_att)

        if self._has_att:
            return frekl_c,frekl_q
        else:
            return frekl_c