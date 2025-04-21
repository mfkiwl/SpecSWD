from .lib import cps330
import numpy as np 

class THSolver:
    def __init__(self,thick:np.ndarray,vp:np.ndarray,
                 vs:np.ndarray,rho:np.ndarray,spherical:bool=False
                 ):
        """
        Initialize Thompson-Haskell Matrix Dispersion Solver

        Parameters
        ------------
        thick: np.ndarray
            thickness of layered model,in km
        vp: np.ndarray
            P wave velocity, km/s
        vs: np.ndarray
            S wave velocity, km/s
        rho: np.ndarray
            density
        """
        # backup input params
        self._thick = thick 
        self._vp = vp 
        self._vs = vs 
        self._rho = rho
        self._spherical = spherical
    
    def compute_swd(self,wavetype:str,mode:int,T:np.ndarray) -> np.ndarray:
        """
        compute dispersion (phase/group) for a give wavetype

        Parameters
        ------------
        wavetype: str
            wave type, one of ['Rc','Rg','Lc','Lg']
        mode: int
            which mode it will return, >=0
        T: np.ndarray
            period, in s

        Returns
        -----------
        cg: np.ndarray
            phase velocity (for Rc,Lc) or group velocity (Rg,Lg)
        """
        # check input args
        assert wavetype in ['Rc','Rg','Lc','Lg'], "check wave type!"
        c,_ = cps330.forward(
            self._thick,self._vp,self._vs,
            self._rho,T,wavetype,mode,
            self._spherical)

        return c