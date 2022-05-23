#!/usr/bin/env python
from importlib import import_module
from importlib.metadata import version
import typing
import warnings
import sys, os
__author__ = "Kayahan Saritas"
__version__ = "0.0.1"
__email__ = "saritaskayahan@gmail.com"
__status__ = "Development"
__citation__ = "https://arxiv.org/abs/2205.00300, J. Phys.: Condens. Matter 27 (2015) 166002"

# Currently supported version dependencies
currently_supported = {
    'numpy'      : (1,   13,  1),
    'pandas'     : (1,    3,  5), 
    'matplotlib' : (3,    5,  0),
    # 'xml'        : (4,    6,  4),
    # 're'         : (2,    2,  1),
    }

required_dependencies = set(['numpy', 'xml', 'matplotlib', 'pandas'])
missing_dependencies = []
used_dependencies = ['xml', 're', 'glob']

def check_modules():
    """
    Track version of package dependencies
    """
    module_list = currently_supported.keys()
    for name in module_list:
        try:
            import_module(name)
            ver = tuple([int(x) for x in version(name).split(".")])
            if ver < currently_supported[name]:
                warnings.warn("Check backwards compatibility: ", name, "@", ver)
            used_dependencies.append(name)
        except:
            if name in required_dependencies:
                print("Error: Module ", name, ">=", currently_supported[name], " is a core dependency, please install\n")
                sys.exit(1)
            else:
                warnings.warn("Module ", name, " is an optional dependency, some functionality may be missing\n")
                missing_dependencies.append(name)

check_modules()

for name in used_dependencies:
    if name == 'matplotlib':
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib as mpl
        from matplotlib import rc
        mpl.use('tkagg')
        rc('text',usetex=True)
        rc('text.latex', preamble=r'\usepackage{color}') 
        plt.style.use('science')
    if name == 'numpy':
        import numpy as np
    if name == 'pandas':
        import numpy as pd
    if name == 're':
        import re
    if name == 'xml':
        import xml.etree.ElementTree as ET
    if name == 'glob':
        import glob

constants = {
             'kb' :  8.617 * 1e-2, # meV/K
             'Ry' : 27.211/2.0, # Rydberg to EVeV
             'Ha' : 27.211 # Hartree to EVeV
            }

def normal(x, mu, sig=0.05):
    # Gaussian-1D
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def normal2d(x, y, mux, muy, sig=0.1):
    # Gaussian-2D
    return np.exp(-(np.power(x - mux, 2.) + np.power(y - muy, 2.)) / (2 * np.power(sig, 2.)))

class AngularMoment:
    def __init__(self) -> None:
        """
        Angular momentum density matrices on real spherical harmonics (SH)
        We first write the density matrices on complex SH
        Then, a unitary transformation is applied to convert into real SH
        """
        # p-orbitals <mi|Lx|mj> (lx_ij)
        self.lxp_ij = np.zeros((3,3))
        ii = 0
        l = 1
        for mi in range(-1, 2, 1):
            ij = 0
            for mj in range(-1, 2, 1):
                # Lx = 1./2(L+ + L-)
                # < l, mi| L+ |l,mj> = \sqrt((l-mj)*(l+mj+1)) * \delta_{i,j+1)
                # < l, mi| L- |l,mj> = \sqrt((l+mj)*(l-mj+1)) * \delta_{i,j-1)
                l_plus  = np.sqrt(l*(l+1) - mj*(mj+1)) * ((mi ==  mj + 1)* 1.0)
                l_minus = np.sqrt(l*(l+1) - mj*(mj-1)) * ((mi ==  mj - 1)* 1.0)
                self.lxp_ij[ii, ij] = 1./2 * (l_plus + l_minus)
                ij += 1
            ii += 1

        # d-orbitals <mi|Lx|mj> (lx_ij)
        self.lx_ij = np.zeros((5,5))
        ii = 0
        l = 2
        for mi in range(-2, 3, 1):
            ij = 0
            for mj in range(-2, 3, 1):
                # Lx = 1./2(L+ + L-)
                # < l, mi| L+ |l,mj> = \sqrt((l-mj)*(l+mj+1)) * \delta_{i,j+1)
                # < l, mi| L- |l,mj> = \sqrt((l+mj)*(l-mj+1)) * \delta_{i,j-1)
                l_plus  = np.sqrt(l*(l+1) - mj*(mj+1)) * ((mi ==  mj + 1)* 1.0)
                l_minus = np.sqrt(l*(l+1) - mj*(mj-1)) * ((mi ==  mj - 1)* 1.0)
                self.lx_ij[ii, ij] = 1./2 * (l_plus + l_minus)
                ij += 1
            ii += 1

        # p-orbital <mi|Lz|mj> (lz_ij)
        self.lzp_ij = np.zeros((3,3))
        l = 1
        ii = 0
        for mi in range(-1, 2, 1):
            ij = 0
            for mj in range(-1, 2, 1):
                if (mi == mj):
                    self.lzp_ij[ii, ij] = mi
                ij += 1
            ii += 1

        # d-orbital <mi|Lz|mj> (lz_ij)
        self.lz_ij = np.zeros((5,5))
        l = 2
        ii = 0
        for mi in range(-2, 3, 1):
            ij = 0
            for mj in range(-2, 3, 1):
                if (mi == mj):
                    self.lz_ij[ii, ij] = mi
                ij += 1
            ii += 1

        # Unitary transformation matrix
        # It is to be applied on a 3x3 matrix with m = -1, 0, 1 in this order
        # 0-pz   = Y10
        # 1-px   = 1./sqrt(2)(Y1-1 - Y11)
        # 2-py   = i./sqrt(2)(Y1-1 + Y11)

        sq2 = np.sqrt(2)
        # Use formulas above, applies on real spherical
        u_pm = np.array([[0, 1, 0], 
                            [1./sq2, 0, -1./sq2], 
                            [1j/sq2, 0, 1j/sq2]])
        self.lxp12 = u_pm @ self.lxp_ij @ u_pm.T.conj()
        self.lzp12 = u_pm @ self.lzp_ij @ u_pm.T.conj()

        # Unitary transformation matrix
        # It is to be applied on a 5x5 matrix with m = -2, -1, 0, 1, 2 in this order
        # 0-dz2   = Y20
        # 1-dzx   = 1./sqrt(2)(Y2-1 - Y21)
        # 2-dyz   = i./sqrt(2)(Y2-1 + Y21)
        # 3-dx2y2 = 1./sqrt(2)(Y2-2 + Y22)
        # 4-dxy   = i./sqrt(2)(Y2-2 - Y22)

        u_dm = np.array([[0, 0, 1, 0, 0], 
                            [0, 1./sq2, 0, -1./sq2, 0], 
                            [0, 1j/sq2, 0, 1j/sq2, 0], 
                            [1./sq2, 0, 0, 0, 1./sq2], 
                            [1j/sq2, 0, 0, 0, -1j/sq2]])
        self.lxd12 = u_dm @ self.lx_ij @ u_dm.T.conj()
        self.lzd12 = u_dm @ self.lz_ij @ u_dm.T.conj()

class QE_XML(object):
    def __init__(self, 
                 filename_data : str, 
                 filename_proj : str, 
                 wfc_dict : typing.Dict,
                 orbital : str = None,
                 data_only = False):
        """Reads any collinear atomic_proj.xml/data-file-schema.xml files 
        from Quantum Espresso (QE). Currently can read QE > 7.0.0. 
        atomic_proj.xml is produced from projwfc.x executable. 
        Both of these files should be produced from the same DFT calculation.
        Currently stores only the information that is relevant for MAE calculations.

        Args:
            filename (str): Name of the XML file
            atmwfc (typing.Dict): atomic wavefunctions to be read. 
            atmwfc has to include all the states printed out in the projwfc.x file
            Example:
            FeCl2 has three atoms in the simulation cell
            Fe has s,s,p,d states, Cl has s and p states. Therefore:
            atmwfc = {'Fe':['s', 's', 'p', 'd'],'Cl1':['s', 'p'], 'Cl2':['s', 'p']}
            data_only(bool): If True, reads only datafile
        """
        self._read_datafile(filename_data, wfc_dict)
        if not data_only:
            self._read_proj_xml(filename_proj, orbital=orbital)
        # TODO: must be checked for metallic states
    
    def _read_datafile(self, filename : str, wfc_dict : typing.Dict):
        """Reads data-file-schema.xml into QE_XML object
        """
        try:
            tree = ET.parse(filename)
        except:
            print('Provide filename/correct xml filename')
            exit()
            
        root = tree.getroot()
        output = root.find('output')
        atomic_pos = output.find('atomic_structure').find('atomic_positions')
        self.atoms = []
        self.index = []
        self.atomwfcs = []
        import pdb
        orb_dict = {'s':['s'],
                    'p':['pz', 'px', 'py'],
                    'd':['dz2', 'dzx', 'dzy', 'dx2y2', 'dxy']}
        for i in atomic_pos:
            specie_name = i.attrib['name']
            specie_index = specie_name + i.attrib['index']
            self.index.append(specie_index)
            self.atoms.append(specie_name)
            # pdb.set_trace()
            for orb in wfc_dict[specie_name]:
                for orb_i in orb_dict[orb]:
                    self.atomwfcs.append(specie_index+'_'+orb_i)
        
        bs = output.find('band_structure')
        self.nbnd_up   = int(bs.find('nbnd_up').text)
        self.nbnd_down = int(bs.find('nbnd_dw').text)
        self.num_wfc   = int(bs.find('num_of_atomic_wfc').text)
        self.e_fermi   = float(bs.find('fermi_energy').text) * constants['Ha']
        self.num_kpts  = int(bs.find('nks').text)

        try:
            assert(len(self.atomwfcs) == self.num_wfc)
        except:
            print("Atomic wavefunction dictionary is incorrect!")
            print("Input read here (compare to projwfc.x output):")
            for i, item in enumerate(self.atomwfcs):
                print("State # ", i, " wfc ", item)
            exit(1)
        
        self.kw        = np.zeros(self.num_kpts)
        self.eig       = np.zeros((self.num_kpts, 2, self.nbnd_up))
        self.occ       = np.zeros((self.num_kpts, 2, self.nbnd_up))
        
        ik = 0
        for bs_el in bs:
            if bs_el.tag.startswith('ks_energies'):
                for ks in bs_el:
                    if ks.tag.startswith('k_point'):
                        kw = float(ks.attrib['weight'])
                        self.kw[ik] = kw
                    elif ks.tag.startswith('eigenvalues'):
                        eig = np.array([float(e) for e in ks.text.strip().split()])
                        self.eig[ik,0] = eig[0:self.nbnd_up] * constants['Ha']
                        self.eig[ik,1] = eig[self.nbnd_up:]  * constants['Ha']
                    elif ks.tag.startswith('occupations'):
                        occ = np.array([float(e) for e in ks.text.strip().split()])
                        self.occ[ik,0] = occ[0:self.nbnd_up]
                        self.occ[ik,1] = occ[self.nbnd_up:]
                    #end if
                #end for
                ik += 1
            #end if
        #end for
        self.unit = 'eV'
        self.eig -= self.e_fermi
        self.nup = np.round(np.sum(np.sum(self.occ[:,0,:], axis=1) * self.kw),3)
        self.ndw = np.round(np.sum(np.sum(self.occ[:,1,:], axis=1) * self.kw),3)
        print("{} is read".format(filename))
    #end def _read_datafile
    
    def _read_proj_xml(self, 
                       filename : str,
                       orbital : str = None):
        """Reads atomic-proj.xml file to QE_XML object
        
        This leaves out most of the basic information in the atomic-proj.xml file
        as these are initially read from the data-file-schema.xml file. 
        Below, there are some assertions that check if both files have matching information.
        """
        try:
            tree = ET.parse(filename)
        except:
            print('Provide filename/correct xml filename')
            exit()

        root = tree.getroot()
        header = root.find('HEADER')
        nbands = int(header.attrib['NUMBER_OF_BANDS'])
        nkpts = int(header.attrib['NUMBER_OF_K-POINTS'])
        nspin = int(header.attrib['NUMBER_OF_SPIN_COMPONENTS'])
        natomwfc = int(header.attrib['NUMBER_OF_ATOMIC_WFC'])
        
        # Check if atomic-proj.xml and data-file-schema.xml are 
        # related to the same calculation. 
        
        assert(self.nbnd_up==nbands)
        assert(self.num_kpts==nkpts)
        assert(self.num_wfc==natomwfc)
        
        eig = root.find('EIGENSTATES')
        ik = -1
        if orbital == None:
            # Read all the orbitals
            proj = np.zeros((nkpts, nspin, natomwfc, nbands)) + 0j
            min_index = 0
        else:
            orb_id = orbital.split('_')[0]
            orb_index = orbital.split('_')[-1]
            orb_specie = orbital.split(orb_index)[0]
            orb_kind = orbital[-1]
            if orb_kind == 's':
                natomwfc = 1
            elif orb_kind == 'p':
                natomwfc = 3
            elif orb_kind == 'd':
                natomwfc = 5
            proj = np.zeros((nkpts, nspin, natomwfc, nbands)) + 0j
            atomic_wfc_indices = []
            for idx, i in enumerate(self.atomwfcs):
                if i.startswith(orbital):
                    atomic_wfc_indices.append(idx)
            print("Will read wfc indices only: ", atomic_wfc_indices)
            atomic_wfc_indices = np.array(atomic_wfc_indices, dtype=int)
            min_index = np.min(atomic_wfc_indices)

        for eig_state in eig:
            if eig_state.tag == ('K-POINT'):
                ik += 1
                ik %=nkpts
            if eig_state.tag == ('PROJS'):
                for wfc in eig_state:
                    index = int(wfc.attrib['index'])-1
                    ispin = int(wfc.attrib['spin'])-1
                    # print(ik, ispin, index, proj.shape)
                    read = False
                    if orbital == None:
                        read = True
                    else:
                        if index in atomic_wfc_indices:
                            read = True
                    if read:
                        proj[ik, ispin, index-min_index] = np.array([ (lambda x,y: complex(float(x),float(y)))(*c.split()) for c in wfc.text.strip().splitlines()])
                
        self.proj = proj
        self.atomproj = np.zeros((nkpts, nspin, nbands, natomwfc)) + 0j
        for ik in range(nkpts):
            for ispin in range(nspin):
                for ib in range(nbands):
                    for iatom in range(natomwfc):
                        self.atomproj[ik, ispin, ib, iatom] = self.proj[ik, ispin, iatom, ib].conj()
        print("{} is read".format(filename))
    #end def _read_proj_xml

    def get_partial_proj(self, orbitals : typing.List) -> np.ndarray:
        """Returns only select atomic projections

        Args:
            orbitals (typing.List): List of orbitals to be selected. Returns their projection.
            Example: ['Fe1_d', 'Cl2_p', 'Fe_dxy', 'O_px']

        Returns:
            np.ndarray: returns a slice of the atomproj using "orbitals" indices
        """
        orb_pick = []
        for orb in orbitals:
            num_orb = len(orb_pick)
            for ind, wfc in enumerate(self.atomwfcs):
                if wfc.startswith(orb):
                    orb_pick.append(ind)
            
            # if num_orb != len(orb_pick) + 1:
            #     print("Orbital ", orb, "is either not unique or not found in the atomwfcs set")
            #     print(num_orb, len(orb_pick))
            #     print("Please rerun!")
            #     return None
                
        orb_pick = list(set(orb_pick))
        print("Picked states indices: ", orb_pick)
        proj = self.atomproj[:,:,:,orb_pick]
        return proj

    def print_atomwfc(self):
        for i, item in enumerate(self.atomwfcs):
            print("State # ", i, " wfc ", item)

class MAE(AngularMoment):
    def __init__(self, wfc_dict : typing.Dict,
                 filename_data : str = './data-file-schema.xml', 
                 filename_proj : str = './atomic-proj.xml', 
                 orbital : str = None) -> None:
        """MAE (Magnetic Anisotropy Energy) class
        The code here uses second order perturbation theory formalism 
        to calculate the magnetic anisotropy energy from collinear 
        DFT calculations. This code has been heavily utilized in 
        "https://arxiv.org/abs/2205.00300".
        Therefore, please cite that work if you get to use this code. 
        The code post-processes Quantum Espresso output by using 
        "data-file-schema.xml and atomic-proj.xml"
        files that are printed out at the end of projwfc.x runs. 
        
        Unfortunately, these files do not include the explicit list of 
        atomic wavefunctions, only the indices of each atomic wavefunction 
        are given.  This information is found in projwfc.x output. However, 
        for each pseudopotential, there is a certain set of atomic wavefunctions.
        Therefore, this list can be provided in "wfc_dict" without any loss of 
        generality.

        Args:
            wfc_dict (typing.Dict): Atomic wavefunction dictionary.
            Example: 
            wfc_dict = {'Fe':['s','s','p','d'], 'Si':['s','p'], 'O':['s','p']}
            The ordering in the pseudopotential must be preserved in the lists. 
            
            filename_data (str, optional): Name/location of data-file-schema.xml. Defaults to './data-file-schema.xml'.
            filename_proj (str, optional): Name/location of atomic-proj.xml. Defaults to './atomic-proj.xml'.
        """
        super().__init__()
        self.axml = QE_XML(filename_data, filename_proj, wfc_dict, orbital=orbital)
        self.orbital = orbital
        # Print some basic information
        print("Up/down bands: ", self.axml.nbnd_up, '/', self.axml.nbnd_down)
        print("Up/down electrons: ", self.axml.nup, '/', self.axml.ndw)
        print("Fermi energy {0:.3f}".format(self.axml.e_fermi), ' eV')
        
    def get_proj(self, orbitals : typing.List = None) -> typing.Dict:
        """Get only a subset of band projections onto atomic wavefunctions
        This is useful if you need to run MAE analysis on a subset of atoms or orbitals
        Examples:
        mae = MAE()
        proj = mae.read_proj(['Fe1_d'])
        proj = mae.read_proj(['Fe3_d', 'O5_p', 'Si8_p'])

        Args:
            orbitals (typing.List): List of orbitals in "AtomIndex_{s,p,d}" format
            as given in the examples above

        Returns:
            typing.Dict: A dictionary of {orbital:projection matrices}
        """
        result = dict()
        if orbitals != None:
            for orb in orbitals:
                proj = self.axml.get_partial_proj([orb])
                result[orb] = proj
        else:
            result[self.orbital] = self.axml.proj
        return result
    
    def get_mat(self, 
                proj_dict : typing.Dict, 
                ksi : float = 1.0, 
                min_band : int = 0,
                max_band : int = -1,
                deltaE = 10) -> typing.Dict:
        """Main routine to calculate the MAE.
        Also returns a dictionary where energy (lx_e/lz_e) and band (lx_mat/lz_mat) 
        resolved MAE are stored for Lx and Lz angular momentum components. 
        Energy resolved MAE data is later can be used to plot energy resolved 
        MAE densities as in https://arxiv.org/abs/2205.00300. 

        Args:
            proj_dict (typing.Dict): {"orbital": atomic_wavefunction} dict. 
            Use "get_proj" function to produce this input.
            ksi (float, optional): Spin orbit coupling constant. Defaults to 1.0. Units cm^-1. 
            deltaE (int, optional): A scaling factor to calculate numerically significant MAE 
            contributions. For an insulating system, this can be considered as the energy scale in 
            eV. For example, a deltaE value of 10 in an insulating system, interband couplings with 
            more than 10 eV difference between the bands are not calculated. For metallic systems, 
            would also include the multiplications of the occupancies. For larger 
            deltaE, results are expected to converge.  Defaults to 10. 

        Returns:
            typing.Dict: a dictionary of energy and band resolved Lx and Lz data for each orbitals
        """
        nk = self.axml.num_kpts
        nbands = self.axml.nbnd_up
        nspin = 2
        ksi =  ksi * 1.239 * 1e-4
        eig_kn = self.axml.eig # eigenvalues
        results = dict()
        if max_band == -1:
            max_band = self.axml.nbnd_up
        if min_band != 0 or max_band != self.axml.nbnd_up:
            nbands = max_band - min_band
            print("Calculating bands between {}-{}".format(min_band, max_band))
            
        for orbital, proj in proj_dict.items():
            lx_mat = np.zeros((nbands,nbands)) + 0j
            lz_mat = np.zeros((nbands,nbands)) + 0j
            lx_e = []
            lz_e = []
            
            if orbital[-1] == 'p':
                lx = self.lxp12
                lz = self.lzp12
                num_orb = 3
            elif orbital[-1] == 'd':
                lx = self.lxd12
                lz = self.lzd12
                num_orb = 5
            else:
                print("Orbital undefined for MAE calculation")
                print("Currently can only use complete sets of p and d orbitals!")
                exit(1)
            indices = np.zeros((nspin * nbands, 2), dtype=int)
            ind = 0
            for i in range(nspin):
                for j in range(min_band, max_band, 1):
                    indices[ind] = [i,j]
                    ind += 1

            # Use some threshold to save computing time
            # if e_n - e_n' is larger than 10 eV, or occupations are too small, do not calculate Lx or Lz 
            # results should converge at large deltaE values
            threshold = 1./deltaE
            occ = self.axml.occ
            lx_e = []
            lz_e = []
            for ik in range(nk):
                for sv, v in indices:
                    for sc, c in indices:
                        factor = (occ[ik,sv, v]) * (1-occ[ik,sc,c]) / (eig_kn[ik, sc, c] - eig_kn[ik, sv, v]+1E-8)
                        if factor > threshold:
                            # print(ik, sv, sc, v, c, occ[ik,sv, v], 1-occ[ik,sc,c])
                            spin_sign = (2.0 * (sv == sc)) - 1.0
                            factor *= self.axml.kw[ik] * spin_sign * ksi**2 * 1000  # eV to meV
                            # <v, s|Lx|c, sc> = \sum_d1,d2 <v, sv|d1> <d1|Lx|d2> <d2|c, sc>
                            lx_vc = np.zeros((num_orb,num_orb)) + 0j 
                            lz_vc = np.zeros((num_orb,num_orb)) + 0j
                            psiv_phii_conj = proj[ik, sv, :, v].conj()
                            psic_phij = proj[ik, sc, :, c]
                            
                            lx_vc = psiv_phii_conj @ lx @ psic_phij # <v, sv|d1> <d1|Lx|d2> <d2|c, sc>
                            lz_vc = psiv_phii_conj @ lz @ psic_phij
                        
                            slx_vc = np.sum(lx_vc)
                            slz_vc = np.sum(lz_vc)
                            cx = factor * slx_vc * slx_vc.conj()
                            cz = factor * slz_vc * slz_vc.conj()
                            lx_mat[v-min_band,c-min_band] += cx
                            lz_mat[v-min_band,c-min_band] += cz
                            lx_e.append([sv, sc, eig_kn[ik, sv, v], eig_kn[ik, sc, c], cx.real])
                            lz_e.append([sv, sc, eig_kn[ik, sv, v], eig_kn[ik, sc, c], cz.real])
                    #end for sc
                #end for sv
            #end for k
            print("LZ :{0:.3f} meV".format(np.sum(lz_mat).real))
            print("LX :{0:.3f} meV".format(np.sum(lx_mat).real))
            print("Total (LZ - LX) : {0:.3f} meV".format((np.sum(lz_mat) - np.sum(lx_mat)).real))
            results[orbital] = {'lx_e':np.array(lx_e, dtype=object), 'lz_e':np.array(lz_e, dtype=object), 'lx_mat':lx_mat, 'lz_mat':lz_mat}
        return results
    
    # @staticmethod
    # def plot_dos_1D(mat : typing.Dict, emin = -6, emax = 4, prefix = 'mat', line_density = 100, sigma = 0.05, show=False):
    #     """Plots 1D MAE density
    #     Since MAE is a pair density, to plot electronic DOS like figure,
    #     conduction and valence band regions are calculated as "marginal" densities

    #     Args:
    #         mat (typing.Dict): mat dictionary returned from get_mat function
    #         emin (int, optional): Minimum valence band energy. 
    #                               Units in eV, with respect to Fermi energy. Defaults to -6.
    #         emax (int, optional): Maximum conduction band energy. Defaults to 4.
    #         prefix (str, optional): Saved figure prefix. Defaults to 'mat'.
    #         line_density (int, optional): Line density per eV. Defaults to 100.
    #         show (bool, optional): If True, plot only do not save figure. Defaults to False.
    #     """
    #     assert(emin < 0)
    #     assert(emax > 0)
        
    #     lx_e = mat['lx_e']
    #     lz_e = mat['lz_e']
        
    #     lx_v_plus   =  lx_e[lx_e[:,0] == 0] # Lx contribution from spin up valence bands 
    #     lz_v_plus   = lz_e[lz_e[:,0] == 0] # Lz contribution from spin up conduction bands
    #     v_lx_plus   = lx_v_plus[:,4]
    #     v_lz_plus   = lz_v_plus[:,4]
    #     v_mae_plus  = v_lz_plus - v_lx_plus
    #     e_v_plus    = lx_v_plus[:,2]
        
    #     lx_v_minus  = lx_e[lx_e[:,0] == 1]
    #     lz_v_minus  = lz_e[lz_e[:,0] == 1]
    #     v_lx_minus  = lx_v_minus[:,4]
    #     v_lz_minus  = lz_v_minus[:,4]
    #     v_mae_minus = v_lz_minus - v_lx_minus
    #     e_v_minus   = lx_v_minus[:,2]
        
    #     lx_c_plus   =  lx_e[lx_e[:,1] == 0]
    #     lz_c_plus   = lz_e[lz_e[:,1] == 0]
    #     c_lx_plus   = lx_c_plus[:,4]
    #     c_lz_plus   = lz_c_plus[:,4]
    #     c_mae_plus  = c_lz_plus - c_lx_plus
    #     e_c_plus    = lx_c_plus[:,3]
        
    #     lx_c_minus  = lx_e[lx_e[:,1] == 1]
    #     lz_c_minus  = lz_e[lz_e[:,1] == 1]
    #     c_lx_minus  = lx_c_minus[:,4]
    #     c_lz_minus  = lz_c_minus[:,4]
    #     c_mae_minus = c_lz_minus - c_lx_minus
    #     e_c_minus   = lx_c_plus[:,3]
        
    #     intervals = [[emin, 0], # valence
    #                  [0, emax]] # conduction band
    #     vals = [[e_v_plus, e_v_minus, v_mae_plus, v_mae_minus], # valence
    #             [e_c_plus, e_c_minus, c_mae_plus, c_mae_minus]] # conduction band
        
    #     plt.figure()
    #     for ind, interval in enumerate(intervals):
    #         i_min, i_max = interval
    #         e_plus, e_minus, mae_plus, mae_minus = vals[ind]
    #         x = np.linspace(i_min, i_max, (i_max-i_min)*line_density+1)
            
    #         # Initialize zero array
    #         y = 0 * normal(0, x)
    #         for i, ei in enumerate(e_plus):
    #             if ei > i_min and ei < i_max:
    #                 y += mae_plus[i] * normal(ei, x, sig=sigma)

    #         for i, ei in enumerate(e_minus):
    #             if ei > i_min and ei < i_max:
    #                 y += mae_minus[i] * normal(ei, x, sig=sigma)
    #         plt.plot(x, y)
        
    #     plt.axhline(y=0, linestyle='dashed', color='k')
    #     plt.axvline(x=0, linestyle='dashed', color='k')
    #     plt.xlabel('Energy (eV)')
    #     plt.ylabel('MAE (meV)')
    #     plt.xlim((emin, emax))
    #     plt.grid(linestyle='-')
    #     plt.grid(which='minor', linestyle='--', axis='x', zorder=-100)
    #     if show:
    #         plt.show()
    #     else:
    #         plt.savefig('./'+prefix+'-mae-dos-1D.png', dpi=300, pad_inches=0) 

    @staticmethod
    def plot_dos_1D(mat : typing.Dict, emin = None, emax = None, prefix = 'mat', line_density = 100, sigma = 0.05, show=False):
        """Plots 1D MAE density
        Since MAE is a pair density, to plot electronic DOS like figure,
        conduction and valence band regions are calculated as "marginal" densities

        Args:
            mat (typing.Dict): mat dictionary returned from get_mat function
            emin (int, optional): Minimum valence band energy. 
                                  Units in eV, with respect to Fermi energy. Defaults to -6.
            emax (int, optional): Maximum conduction band energy. Defaults to 4.
            prefix (str, optional): Saved figure prefix. Defaults to 'mat'.
            line_density (int, optional): Line density per eV. Defaults to 100.
            show (bool, optional): If True, plot only do not save figure. Defaults to False.
        """
        
        lx_e = mat['lx_e']
        lz_e = mat['lz_e']
        
        lx_v = lx_e[:,[2,4]]
        lx_c = lx_e[:,[3,4]]

        lz_v = lz_e[:,[2,4]]
        lz_c = lz_e[:,[3,4]]

        l_v = lx_v.copy()
        l_v[:,1] = lz_v[:,1] - lx_v[:,1]
        l_c = lx_c.copy()
        l_c[:,1] = lz_c[:,1] - lx_c[:,1]

        val_min = np.min(l_v[:,0]) - 3 * sigma
        val_max = np.max(l_v[:,0]) + 3 * sigma

        cond_min = np.min(l_c[:,0]) - 3 * sigma
        cond_max = np.max(l_c[:,0]) + 3 * sigma

        if emin != None:
            val_min = emin
        if emax != None:
            cond_max = emax

        intervals = [[val_min, val_max], # valence
                     [cond_min, cond_max]] # conduction band

        plt.figure()
        for ind, interval in enumerate(intervals):
            i_min, i_max = interval
            if ind == 0:
                val = l_v
            else:
                val = l_c

            x = np.linspace(i_min, i_max, int((i_max-i_min)*line_density+1))
            
            # Initialize zero array
            y = 0 * normal(0, x)
            for ei in val:
                y += ei[1] * normal(ei[0], x, sig=sigma)

            plt.plot(x, y)
        
        plt.axhline(y=0, linestyle='dashed', color='k')
        plt.axvline(x=0, linestyle='dashed', color='k')
        plt.xlabel('Energy (eV)')
        plt.ylabel('MAE (meV)')
        plt.xlim((val_min, cond_max))
        plt.grid(linestyle='-')
        plt.grid(which='minor', linestyle='--', axis='x', zorder=-100)
        if show:
            plt.show()
        else:
            plt.savefig('./'+prefix+'-mae-dos-1D.png', dpi=300, pad_inches=0) 


    @staticmethod
    def plot_dos_2D(mat : typing.Dict, emin = -6, emax = 4, prefix = 'mat', line_density = 20, sigma = 0.1, show=False):
        """Plots 2D MAE full pair density

        Args:
            mat (typing.Dict): mat dictionary returned from get_mat function
            emin (int, optional): Minimum valence band energy. 
                                  Units in eV, with respect to Fermi energy. Defaults to -6.
            emax (int, optional): Maximum conduction band energy. Defaults to 4.
            prefix (str, optional): Saved figure prefix. Defaults to 'mat'.
            line_density (int, optional): Line density per eV. Defaults to 20.
            show (bool, optional): If True, plot only do not save figure. Defaults to False.
        """
        assert(emin < 0)
        assert(emax > 0)
        
        lx_e = mat['lx_e']
        lz_e = mat['lz_e']
        
        x_density = (0-emin)*line_density+1
        y_density = emax*line_density+1
        x, y = np.meshgrid(np.linspace(emin, 0, x_density), np.linspace(0,emax,y_density))
        
        # Initialize 0 array
        z = 0 * normal2d(0,0,0,0)
        for lx in lx_e:
            mx = lx[2]
            my = lx[3]
            mz = lx[4]
            z -= mz*normal2d(x, y, mx, my, sig=sigma)
        
        for lz in lz_e:
            mx = lz[2]
            my = lz[3]
            mz = lz[4]
            z += mz*normal2d(x, y, mx, my, sig=sigma)
        
        fig1, ax1 = plt.subplots(constrained_layout=True)
        mae_max = max(abs(np.min(z)), np.max(z))
        divnorm = mcolors.TwoSlopeNorm(vmin=-mae_max,
                                  vcenter=0., vmax=mae_max)
        CS = ax1.contourf(x, y, z, 20, norm=divnorm, cmap='PuOr')
        fig1.colorbar(CS)
        plt.xlabel('Valence Energies (eV)')
        plt.ylabel('Conduction Energies (eV)')
        plt.grid(linestyle='-')
        plt.grid(which='minor', linestyle='--', axis='x', zorder=-100)
        
        if show:
            plt.show()
        else:
            plt.savefig('./'+prefix+'-mae-dos-2D.png', dpi=300, pad_inches=0) 

class QE_PDOS(QE_XML):
    def __init__(self,
                 filename_data,
                 location = './',
                 wfc_dict = None) -> None:
        super().__init__(filename_data=filename_data, filename_proj=None, data_only=True, wfc_dict=wfc_dict)
        self._collect_atom_wfc(location)
    
    def _collect_atom_wfc(self, location):
        self.pdos = []
        files = glob.glob(location+'/*pdos_atm*')
        files = sorted(files)
        for f in files:
            a = re.split('\(|\)', f)
            num = re.split('#', a[0])[1]
            self.pdos.append([a[1]+'_'+a[3], a[1]+num+'_'+a[3], f])
    
    @staticmethod
    def _read(filename, efermi):
        print(filename)
        pp = np.genfromtxt(filename) 
        num_orb = int((pp.shape[1] - 3) / 2)
        pp_new = np.zeros((num_orb + 2 , 2, pp.shape[0]))
        pp_new[0 , 0, :] = pp[:, 0] - efermi
        pp_new[0 , 1, :] = pp[:, 0] - efermi
        pp_new[-1 , 0, :] = pp[:, 1]
        pp_new[-1 , 1, :] = pp[:, 2]
        for orb in range(num_orb):
            pp_new[orb+1 , 0, :] = pp[:, 2*orb+3]
            pp_new[orb+1 , 1, :] = -pp[:, 2*orb+4]
        # returns 
        # 0 index x values in spin up and down
        # 1..num_orb indices: pdos
        # last index is ldos
        return pp_new
        
    def plot_total(self, orbitals, emin = -6, emax = 4, prefix = 'mat', show = False):
        total = dict()
        x = None
        for pdos in self.pdos:
            wfc, f = pdos
            if wfc in orbitals:
                print(wfc)
                pf = self._read(f, self.e_fermi)
                if wfc in total.keys():
                    total[pdos[0]] += np.sum(pf[1:-1,:,:], axis=0) 
                    x = pf[0,0,:]
                else:
                    total[pdos[0]] = np.sum(pf[1:-1,:,:], axis=0) 

        colors = ['r', 'b', 'k', 'g']
        i = 0
        for key, value in total.items():
            lw = 3
            next_color = True
            if next_color:
                i += 1
            plt.plot(x, value[0], colors[i], linewidth=lw, label=r"$\rm "+key+"$")
            plt.plot(x, value[1], colors[i], linewidth=lw, label=r"$\rm "+key+"$")

        plt.xlim((emin, emax))
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.axvline(x=0, color = 'k', linestyle='--')
        plt.axhline(y=0, color = 'k', linestyle='--')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Density of States (DOS)')
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])
        plt.grid(linestyle='-')
        plt.grid(which='minor', linestyle='--', axis='x')
        
        if show:
            plt.show()
        else:
            plt.savefig(prefix+'-dos-simple.png', dpi=300, bbox_inches="tight", pad_inches=0)

    def _plotd(self, pp):
        lw = 1
        plt.plot(pp[0, 0], pp[1, 0, :], 'k', label=r'$d_{z^2}$', linewidth=lw)
        plt.plot(pp[0, 0], pp[1, 1, :], 'k', linewidth=lw)
        plt.plot(pp[0, 0], pp[2, 0, :], 'r', label=r'$d_{xz}$', linewidth=lw)
        plt.plot(pp[0, 0], pp[2, 1, :], 'r', linewidth=lw)
        plt.plot(pp[0, 0], pp[3, 0, :], 'g', label=r'$d_{yz}$', linewidth=lw)    
        plt.plot(pp[0, 0], pp[3, 1, :], 'g', linewidth=lw)
        plt.plot(pp[0, 0], pp[4, 0, :], 'magenta', label=r'$d_{x^2-y^2}$', linewidth=lw)
        plt.plot(pp[0, 0], pp[4, 1, :], 'magenta', linewidth=lw)
        plt.plot(pp[0, 0], pp[5, 0, :], 'b', label=r'$d_{xy}$', linewidth=lw)
        plt.plot(pp[0, 0], pp[5, 1, :], 'b', linewidth=lw)

    def _plotp(self, pp):
        lw = 1
        plt.plot(pp[0, 0], pp[1, 0, :], 'r', label=r'$p_z$', linewidth=lw)
        plt.plot(pp[0, 0], pp[1, 1, :], 'r', linewidth=lw)
        plt.plot(pp[0, 0], pp[2, 0, :], 'g', label=r'$p_x$', linewidth=lw)
        plt.plot(pp[0, 0], pp[2, 1, :], 'g', linewidth=lw)
        plt.plot(pp[0, 0], pp[3, 0, :], 'b', label=r'$p_y$', linewidth=lw)
        plt.plot(pp[0, 0], pp[3, 1, :], 'b', linewidth=lw)

    def plot_decomp(self, orbitals, emin = -6, emax = 4, prefix = 'mat', show = False):
        assert(len(orbitals) == 1)
        total = dict()
        x = None
        plt.figure(figsize=(6,4))
        for pdos in self.pdos:
            wfc, wfc_index, f = pdos
            if wfc in orbitals or wfc_index in orbitals:
                print(wfc, wfc_index)
                pf = self._read(f, self.e_fermi)
                if wfc[-1] == 'd':
                    self._plotd(pf)
                elif wfc[-1] == 'p':
                    self._plotp(pf)

        plt.xlim((emin, emax))
        plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.axvline(x=0, color = 'k', linestyle='--')
        plt.axhline(y=0, color = 'k', linestyle='--')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Density of States (DOS)')
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])
        plt.grid(linestyle='-')
        plt.grid(which='minor', linestyle='--', axis='x')
        
        if show:
            plt.show()
        else:
            plt.savefig(prefix+'-dos-simple.png', dpi=300, bbox_inches="tight", pad_inches=0)            
