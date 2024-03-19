try:
    import os
    import warnings
    import numpy as np
    from scipy import signal
    from astropy.io import fits
    from astropy import constants
    from pyuvdata import UVData
    from pyuvdata.data import DATA_PATH
    from astropy import units as u
    from astropy.cosmology import Planck18
    from astropy import cosmology
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from SSINS import SS, INS
    from itertools import combinations
    
    np.seterr(invalid='ignore')
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['xtick.major.size'] = 8
    mpl.rcParams['xtick.direction']= 'in'
    mpl.rcParams['ytick.direction']= 'in'
    mpl.rcParams['xtick.minor.size'] = 4
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['xtick.minor.width'] = 1.3
    mpl.rcParams['ytick.major.size'] = 8
    mpl.rcParams['ytick.minor.size'] = 4
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['ytick.minor.width'] = 1.3
    mpl.rcParams['patch.linewidth']=1.3
    mpl.rcParams['axes.linewidth']=2
    
except ImportError:
    raise ImportError

class Get_Antenna_Info(object):
    
    '''
    Basic radio data handling;
    Read uvfits to get antennae releted information, antenna triads and baseline vectors.
    available functions: count_antennae(), get_triplets_blvectors()

    '''
    
    def __init__(self):
        
        pass
    
    def count_antennae(self, uv):
        
        '''

        Check for the MWA antennae in the data, 
        This function can help get visibilites from specific MWA antenne/configurations,
        e.g. Redundant Hexagon, non-redundant Tiles
        
        Params:
       
        uv: UVDATA object
        
        Return: MWA {Hexagon-East (HexE), Hexagon-South (HexS), Tile,}
        configuration as antenna numbers, antenna names. (dtype: ndarray)
        
        Example: X = count_antennae(uv_object)
        X[0] : antanna indicies of HexE 
        X[1] : antanna indicies of HexS
        X[2] : antanna indicies of HexE + HexS combined
        X[3] : antanna indicies of Tiles 
        X[4] : antanna names of HexE
        X[5] : antanna names of HexS
        X[6] : antanna names of HexE + HexS combined
        X[7] : antanna names of Tiles
        X[8] : count of any extra antenna in the data, if any
        
        '''
    
        ant_indices = np.unique(uv.ant_1_array.tolist() + uv.ant_2_array.tolist()) - 1
        antenna_names = np.array(uv.antenna_names)
        antenna_names = antenna_names[ant_indices]
        
        Tile_index = np.where(np.char.find(antenna_names, 'Tile')==0)[0]
        Tile_names = antenna_names[Tile_index]
        
        HexE_index = np.where(np.char.find(antenna_names, 'HexE')==0)[0]
        HexE_names = antenna_names[HexE_index]
        
        HexS_index = np.where(np.char.find(antenna_names, 'HexS')==0)[0]
        HexS_names = antenna_names[HexS_index]
        
        Hex_combine_index = np.concatenate((HexE_index, HexS_index))
        Hex_combine_names = np.concatenate((HexE_names, HexS_names))
        
        Extra_index = np.setdiff1d(ant_indices, np.concatenate([Tile_index, Hex_combine_index]))
        Extra_names = antenna_names[Extra_index]

        return HexE_index, HexS_index, Hex_combine_index, Tile_index, Extra_index,\
               HexE_names, HexS_names, Hex_combine_names, Tile_names, Extra_names

    def get_triplets_blvectors(self, ant_labels, ant_positions, baseline_length, triad_angles, if_equilateral_triad=True):
        """
        Get the three antenna pairs forming a close triangle for given baseline length,
        
        Args:
            ant_labels (list of str): list or array of antenna names,
            e.g. numbers correspond to HexE, HexS, HexE+HexS combined, Tiles antennae.
            ant_positions (ndarray[float], optional): list or array of location of antennae of your choices.
            e.g. HexE, HexS, HexE+S obtained from count_antennae()
            baseline_length (int): basline length in meters.
            triad_angles (int): set of antenna triad angles forming a closed triangle in physical baseline
            e.g. for antenna pairs {ab ,bc, ca}, triad angles would be defined as: extenal angles (pi - internal angle) of {abc, bca, cab}
            if_equilateral_triad (bool, optional): if using equilateral triads in the analysis. Defaults to True.
            
        Raises:
            TypeError: not a bool type

        Returns:
            ndarray, ndarray: antenna triplets, corresponding baselines
        """

        bltriplets = []
        blvecttriplets = []
        anttriplets = []
        
        for aind1, albl1 in enumerate(ant_labels):
            for aind2, albl2 in enumerate(ant_labels):

                bl12 = ant_positions[aind2] - ant_positions[aind1]
                bl12_len = np.sqrt(np.sum(bl12**2))
                if np.around(bl12_len)!=baseline_length:
                    continue
                elif bl12_len > 0.0 and bl12[1]<0.:
                    bl12str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl12)
                    for aind3,albl3 in enumerate(ant_labels):
                        if aind1!= aind2 != aind3:
                            bl23 = ant_positions[aind3] - ant_positions[aind2]
                            bl31 = ant_positions[aind1] - ant_positions[aind3]

                            bl23_len = np.sqrt(np.sum(bl23**2))
                            bl31_len = np.sqrt(np.sum(bl31**2))
                            ang1 = (180./np.pi)*np.arccos(np.dot((bl23/np.linalg.norm(bl23)), (bl12/np.linalg.norm(bl12))))
                            ang2 = (180/np.pi)*np.arccos((np.dot((bl23/np.linalg.norm(bl23)),(bl31/np.linalg.norm(bl31)))))
                            ang3 = (180/np.pi)*np.arccos((np.dot((bl31/np.linalg.norm(bl31)),(bl12/np.linalg.norm(bl12)))))

                            if if_equilateral_triad == True:
                                if np.around(bl31_len)==np.around(bl23_len)==np.around(bl12_len) == baseline_length :
                                    if np.around(ang1)== np.around(ang2) ==np.around(ang3) == triad_angles :
                                        if bl12[2]>0. and bl31[1]>0. and bl31[2]>0. :
                                            bl23str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl23)
                                            bl31str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl31)
                                            list123_str = [bl12str, bl23str, bl31str]
                                            bltriplets += [list123_str]
                                            blvecttriplets += [[bl12, bl23, bl31]]
                                            anttriplets += [[albl1, albl2, albl3]]
                                         
                                else:

                                    if {np.around(bl31_len),np.around(bl23_len), np.around(bl12_len)} == baseline_length :
                                        if {np.around(ang1), np.around(ang2),np.around(ang3)} == triad_angles :
                                            if bl12[2]>0. and bl31[1]>0. and bl31[2]>0. :
                                                bl23str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl23)
                                                bl31str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl31)
                                                list123_str = [bl12str, bl23str, bl31str]
                                                bltriplets += [list123_str]
                                                blvecttriplets += [[bl12, bl23, bl31]]
                                                anttriplets += [[albl1, albl2, albl3]]
                            else:
                                continue

        return np.array(anttriplets), np.array(blvecttriplets)  
class Cosmology(object):
    
    '''
    Cosmology functions
    
    available functions: redshift estimator, Energy density estimator
    
    '''

    def __init__(self, delays, freq):
        
        self.delays = delays
        self.freq   = freq

    def Z(self, f_obs=None):

        '''

        redshift estimate

        params:

        f_obs: observed frequency in Hz

        returns: redshift

        '''

        f_em = (1420*1e6)* u.Hz

        return ((f_em - f_obs)/f_obs) 

    def E(self, z=None):

        '''

        cosmology, Energy density parameter

        params:

        z: redshift

        return: energy density parameter

        '''

        O_m = 0.3
        O_k = 0
        O_l = 0.7
        
        return (O_m*((1+z)**3) + O_k*((1+z)**2) + O_l)**(1/2)
    
    def get_k_par(self):
        
        '''
        get k|| modes in h_inv Mpc units
        
        '''
        
        kB = constants.k_B
        f_em = 1420*10**6*u.Hz #in Hz
        H0 = cosmology.Planck18.H0
        c = constants.c
        
        def Z(f):
            f_em = 1420*10**6*u.Hz #in Hz
            return abs(f_em.value - f.value)/f.value
        
        def E(z):
            O_m = cosmology.Planck18.Om0
            O_k = cosmology.Planck18.Ok0
            O_l = 1-(O_m+O_k)
            return (O_m*((1+z)**3) + O_k*((1+z)**2) + O_l)**(1/2)
        
        k_par = (2*np.pi*(self.delays)*f_em*H0*E(z=Z(self.freq)))/(c*(1+Z(self.freq))**2)
        
        return (k_par.to(1/u.Mpc))/cosmology.Planck18.h # in the units of hMpc-1
    
    def get_k_perp(self, baseline_length, use_fmid=True):
        
        z = self.Z(f_obs=self.freq)
        if use_fmid==True:
            k_perp = 2*np.pi*(abs(baseline_length)*u.meter*(self.freq[int(self.freq.size/2)]/constants.c))\
                /cosmology.Planck18.comoving_distance(z)
        else:
            k_perp = 2*np.pi*(abs(baseline_length)*u.meter*(self.freq/constants.c))\
                /cosmology.Planck18.comoving_distance(z)
        return k_perp
class Vis(object):

    '''

    visibility and bispectrum phase.

    available functions: vis(), phase(), vis_all_triads(), 
    
    '''
    
    def __init__(self,):
        
        pass

    def vis(self, ant1=None, ant2=None, ant3=None, uv=None): 
        
        '''
        
        Get individual antenna pair visibilities, bispectrum both unnormalised, normalised
        
        Params:
        
        ant1, ant2, ant3: antenna numbers
        uv: UVData object
        
        Return: visibilites, visibility triple-product
        
        e.g.: v = vis(n1,n2,n3)
              v[0] : visibilities correspond to {ant1, ant2}
              v[1] : visibilities correspond to {ant2, ant3}
              v[2] : visibilities correspond to {ant3, ant1}
              v[3] : bispectrum/visibility triple product correspond to {ant1, ant2, ant3}
        
        '''
        
        v1 = uv.get_data(ant1, ant2)
        v2 = uv.get_data(ant2, ant3)
        v3 = uv.get_data(ant3, ant1)
        
        # bispectrum
        vp = v1*v2*v3
        
        return v1, v2, v3, vp

    def phase(self, vis=None):
        
        '''
        
        Get the phase of visibilites
        
        Params:
        
        vis : input complex visibility
        
        Return: visibility phase in radian units
        
        '''
        
        ang = np.angle(vis)
        
        return ang *u.rad
    
    def vis_all_triads(self, anttriplets=None, timestamps=None, dsize=None, uv=None, index=None):
        
        '''
        
        Get bispectrum for all antenna triads
        
        Params:
        
        anttriplets: {n1, n2, n3} antenna-pairs, can be generated using Get_antenna_triplets class
        timestamps: time_stemps in the data, default: 14
        dsize: size of the data set, default: 768, (size of nfreq)
        uv: UVData object
        index: index is to choose the visibilites from either pairs 
               between ({ant1, ant2}, {ant2, ant3}, {ant3, ant1}) antennae.
               index = [0, 1, 2] correspond to  visibilites at corresponding {ant_i, ant_j} baseline.
               index = [3] correspond to  bisepctrum
               
        Return: visibilities
        
        '''
        
        Vis_arr = np.empty(shape=(len(anttriplets), timestamps, dsize), dtype=np.complex64) 
        #data size N_triplets x N_timestamps x N_freq

        if __name__ ==  "__main__":

            for i in range(len(anttriplets)):
                Vis_arr[i] = self.vis(*anttriplets[i], uv)[index] 
                ## gives vis at antenna triplets (a,b), (b,c), (c,a)
        
        else:

            for i in range(len(anttriplets)):
                Vis_arr[i] = Vis.vis(self, *anttriplets[i], uv)[index] 
                ## gives vis at antenna triplets (a,b), (b,c), (c,a)

        return Vis_arr
class Data_Creation(object):
    
    '''
    
    Create visibility datasets:
    
    available functions: create_vis_data(), fix_visfile_triads_check(),
    get_triad_counts(), 
    
    '''
    
    def __init__(self,): 
        self.Aeff_MWA = np.loadtxt('MWA_Aeff_167_197MHz')

    def create_vis_data(self, obsID, baseline_length, \
        triad_angles, timestamps, Hex_index, polarizations, \
            datapath, modelpath, HIpath, loadfilenames, savefilename, \
                savefilepath, woden_coarseband=True, if_equilateral_triad=True, \
                only_model=False, skip_model=False):
        
        """
        Extracting the visiblities correspond to the antenna triplets {a, b, c} from .uvfits and stored as .npy file.
        
        The datatype (dtype) store the following information: once the vis.npy file is loaded then check the dtype using 'dtype.names'
        current dataset stores following dtypes:
        {'vis_data:Jy/uncal',
        'vis_FG:Jy',
        'vis_FGI:Jy',
        'vis_Noise:Jy',
        'vis_HI:Jy',
        'vis_FG+Noise:Jy',
        'vis_FGI+Noise:Jy',
        'vis_FG+Noise+HI:Jy',
        'vis_FGI+Noise+HI:Jy'}
        access them using above strings.

        By default only a single realisation of the HI is used for data creation.
        The datasets for an observations stores [[V_ab, V_bc, V_ca], N_triads, N_timestamps, N_freq_channels] 
        e.g. for 14 meter baseline_length, dataset has a shape of (3,47,14,768)

        Args:
            
            obsIDs (int): GPS time of the observation
            baseline_length (int): set of baseline vectors forming closed triangles in cyclic manner {ab ,bc, ca}.
            baseline length in meteres for which the data needs to be extracted 
            (usual equilateral baselines 14m, 24m, 28m, 42m for MWAII Hex config)
            triad_angles (int): set of antenna triad angles forming a closed triangle in physical baseline 
            e.g. for antenna pairs {ab ,bc, ca}, triad angles would be defined as:
            extenal angles (pi - internal angle) of {abc, bca, cab} for equilateral triangle all angles are equal it is {120, 120, 120} degrees.
            timestamps (int): total time stamps in the data to process.
            Hex_index (int): MWA antenna config index type int, 
                            choose amongst following [0: HexE, 1: HexS, 2: HexE+S, 3: Other Tiles]
                            e.g: from MWA_config = count_antennae(uv_object) class
                            MWA_config[0] : antanna indicies of HexE 
                            MWA_config[1] : antanna indicies of HexS
                            MWA_config[2] : antanna indicies of HexE + HexS combined
                            MWA_config[3] : antanna indicies of rest of the Tiles 
            polarizations: polarizations, available options: 'xx', 'yy'.
            woden_coarseband: HI simulations generated using Jack Line's WODEN. simulations have 24 different bands .uvifts for 1.28MHz coarse band.
            Set to True if woden simulations are at coarseband. Defaults to True.
            if_equilateral_triad (bool):  if using equilateral triads in the analysis, defaults to True. Future developments..
            datapath (str): path to the observation data .uvfits
            modelpath (str): path to the foreground model .uvfits
            HIpath (str): path to the HI simulation .uvfits
            loadfilenames (str of [datafilename, modelfilename, HIfilename]): loadfilenames. 
            By default only a single realisation of the HI is used for data creation, therefore the loadfilename is hard coded as defualt.
            Default names ['%obsID', '%obsID_Puma_25000', '%obsID_Puma_Unity25000', 'HI_%obsID_band%bandnum'] + '.uvfits'
            savefilename (str): save as the given name. Default savefilename: 'vis_%baseline_length_%polarisation_%obsID.npy'
            savefilepath (str): path to save the file.
            skip_model (bool, optional): True if do not want model visibilities along with the data. 
            This option will only avoid dtypes having FG, FGI, however, the dtypes with HI will be present. Defaults to False.
            only_model (bool, optional): True if only model visibilites is required. 
            This option will avoid dtype containing data, however, the dtypes with HI will be present. Defaults to False.
            
            
        Raises:
        
            IOError: file not present
            
        Returns:
            saves visibilities and antenna config files
            
        """
       
        #try:
        if loadfilenames == None:
            #By default only a single realisation of the HI is used for data creation, obsID 1162989040 used for HI.
            loadfilenames = ['%d'%obsID, '%d_25000'%obsID, '%d_25000_unity_gains'%obsID, 'HI_1162989040_band']
        
        if only_model!=True:
            
            try:
                fd = os.path.join(DATA_PATH, datapath + loadfilenames[0] + '.uvfits')
                print('processing... %s.uvfits'%loadfilenames[0])
                

            except IOError:
                raise IOError('file not found')

            uvd = UVData()
            get_int_info=Get_Antenna_Info()
            VIS = Vis()
            
            uvd.read(fd, read_data=False)
            Hex=get_int_info.count_antennae(uv=uvd)[Hex_index] ## Hex_index 0 is HexE 1 is HexS, 2 HexE+S
            ant_loc= uvd.antenna_positions[Hex-1]
            freq = uvd.freq_array[0]
            dsize = len(freq)

            uvd.read(fd, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S

            anttriplets = get_int_info.get_triplets_blvectors(ant_labels=Hex, ant_positions=ant_loc,\
                            baseline_length=baseline_length,\
                            triad_angles=triad_angles, unique=True,\
                            if_equilateral_triad=if_equilateral_triad)[0]
            
            ### (data, model, ideal model)
            v1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvd, index=0)
            v2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvd, index=1)
            v3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvd, index=2)
            v_d = np.array([v1, v2, v3])
            print('DATA visibility %d processed!'%obsID)
            
            
            #if woden_coarseband==True:
            #    vis_HI=[]
            #    for band in range(1, 25):
            #
            #        try:
            #            if band<10:
            #                HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[3] + '0%d.uvfits'%band)
            #                print('processing... %s.uvfits, %d'%(loadfilenames[3], band))
            #            else:
            #                HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[3] + '%d.uvfits'%band)
            #                print('processing... %s.uvfits, %d'%(loadfilenames[3], band))
            #
            #        except IOError:
            #            raise IOError
            #
            #        uv_HI = UVData()
            #        get_int_info=Get_Antenna_Info()
            #        VIS = Vis()
            #        
            #        uv_HI.read(HI_f, read_data=False)
            #        Hex=get_int_info.count_antennae(uv=uv_HI)[Hex_index] ## index 0 is HexE 1 is HexS, 2 HexE+S
            #        ant_loc= uv_HI.antenna_positions[Hex-1]
            #        freq = uv_HI.freq_array[0]
            #        dsize = len(freq)
            #        uv_HI.read(HI_f, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S
            #        anttriplets = get_int_info.get_triplets_blvectors(ant_labels=Hex, ant_positions=ant_loc,\
            #                        baseline_length=baseline_length,\
            #                        triad_angles=triad_angles, unique=True, \
            #                        if_equilateral_triad=if_equilateral_triad)[0]
            #
            #        vH1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=0)
            #        vH2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=1)
            #        vH3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=2)
            #        vis_HI.append(np.array([vH1, vH2, vH3])) 
            #
            #    vis_HI = np.array(vis_HI)
            #    vis_HI = np.concatenate(vis_HI, axis=-1)
            #    vis_HI = np.array(vis_HI)
            #    print('HI visibility %d processed!'%obsID)
            #
            #if woden_coarseband==False:
            #    print('woden generated .uvfits are not at 1.28MHz')
            #    
            #    try:
            #        HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[3] + '.uvfits')
            #        print('processing... %s.uvfits'%loadfilenames[3])
            #
            #    except IOError:
            #        raise IOError
            #
            #    uv_HI = UVData()
            #    get_int_info=Get_Antenna_Info()
            #    VIS = Vis()
            #
            #    uv_HI.read(HI_f, read_data=False)
            #    Hex=get_int_info.count_antennae(uv=uv_HI)[Hex_index] ## index 0 is HexE 1 is HexS, 2 HexE+S
            #    ant_loc= uv_HI.antenna_positions[Hex-1]
            #    freq = uv_HI.freq_array[0]
            #    dsize = len(freq)
            #    uv_HI.read(HI_f, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S
            #    
            #    anttriplets = get_int_info.get_triplets_blvectors(ant_labels=Hex, ant_positions=ant_loc,\
            #                    baseline_length=baseline_length,\
            #                    triad_angles=triad_angles, unique=True, \
            #                    if_equilateral_triad=if_equilateral_triad)[0]
            #
            #    vH1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=0)
            #    vH2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=1)
            #    vH3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=2)
            #    vis_HI = np.array([vH1, vH2, vH3])
            #    print('HI visibility %d processed!'%obsID)
    
        if skip_model != True:

            try:
                model_FG = os.path.join(DATA_PATH, modelpath + loadfilenames[1] + '.uvfits')
                print('processing... %s.uvfits'%loadfilenames[1])
                model_FGI = os.path.join(DATA_PATH, modelpath + loadfilenames[2] + '.uvfits')
                print('processing... %s.uvfits'%loadfilenames[2])
                
            except IOError:
                raise IOError 
            
            get_int_info=Get_Antenna_Info()
            VIS = Vis()
            uvFG = UVData()
            uvFGI = UVData()
            uvFG.read(model_FG, read_data=False)
            uvFGI.read(model_FGI, read_data=False)
            freq = uvFG.freq_array[0]
            dsize = len(freq)
            Hex=get_int_info.count_antennae(uv=uvFG)[Hex_index] ## index 0 is HexE 1 is HexS, 2 HexE+S
            ant_loc= uvFG.antenna_positions[Hex-1]
            
            uvFG.read(model_FG, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S
            uvFGI.read(model_FGI, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S
            
            vFG1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=0)
            vFG2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=1)
            vFG3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=2)

            vFGI1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFGI, index=0)
            vFGI2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFGI, index=1)
            vFGI3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFGI, index=2)

            v_FG = np.array([vFG1, vFG2, vFG3])
            print('FG visibility %d processed!'%obsID)
            v_FGI = np.array([vFGI1, vFGI2, vFGI3])
            print('FGI visibility %d processed!'%obsID)
            
            Noise = MWA_Noise_Analysis()
            SEFD = Noise.get_SEFD(Aeff_MWA=self.Aeff_MWA, freq=freq*1e-6, dsize=freq.size, integration_time=112,\
                    frequency_resolution=abs(freq[1]-freq[0]), if_rand=False, in_Jy=True)
            vis_Noise=Noise.get_obs_compatible_SEFD(SEFD=SEFD[0], SEFD_rms=SEFD[1], if_use_rms=True, \
                    N_timestamps=timestamps, N_triads=v_FG.shape[1],\
                        if_save_arr=False, savefilename=None, savepath=None)[1]
            print('Noise visibility %d processed!'%obsID)   

            if only_model==True:
                
                data_keys=('FG:Jy', 'FG-Ideal:Jy')
            
                vis_arr = np.empty(shape=v_FG.shape,\
                    dtype=[(data_keys[key_index], complex) for key_index in range(len(data_keys))])
                
                vis_arr['FG:Jy']=v_FG
                vis_arr['FG-Ideal:Jy']=v_FGI
            
            if only_model==False:

                data_keys=('data:Jy/uncal', 'FG:Jy', 'FG-Ideal:Jy', 'Noise:Jy')#, 'HI:Jy')
                
                vis_arr = np.empty(shape=v_d.shape,\
                    dtype=[(data_keys[key_index], complex) for key_index in range(len(data_keys))])
                
                vis_arr['data:Jy/uncal']=v_d
                vis_arr['FG:Jy']=v_FG
                vis_arr['FG-Ideal:Jy']=v_FGI
                vis_arr['Noise:Jy']=vis_Noise
                #vis_arr['HI:Jy']=vis_HI      
        
        if skip_model == True:
            
            data_keys=('data:Jy/uncal', 'Noise:Jy', 'HI:Jy')
            
            vis_arr = np.empty(shape=v_d.shape,\
                dtype=[(data_keys[key_index], complex) for key_index in range(len(data_keys))])
            
            vis_arr['data:Jy/uncal']=v_d
            vis_arr['Noise:Jy']=vis_Noise
            #vis_arr['HI:Jy']=vis_HI

        if savefilename == None:
            print('savefilename set to None; using default savefilename as:')
            
            if only_model==True:
                savefilename = 'vis_only_model%d_%s_%d'%(baseline_length, polarizations, obsID)

            if skip_model==True:
                savefilename = 'vis_data%d_%s_%d'%(baseline_length, polarizations, obsID)

            else:
                savefilename = 'vis_%d_%s_%d'%(baseline_length, polarizations, obsID)
                
        print(savefilename)
        np.save(savefilepath + savefilename, vis_arr)
        np.save(savefilepath + 'antT' + savefilename, anttriplets)
     
    def fix_visfile_triads_check(self, obsID_ref=None, ref_ant_path=None, ant_ref_name=None, antfilename=None,\
                            data_path=None, ant_path=None, filename=None, obsIDs=None,\
                            baseline_length=None, timestamps=14, freq_channels=768, \
                            path_to_save=None, if_HI=False):
        
        '''
        
        This function will use the triad counts for the given baseline
        and append NaN visibilities at the missing triads, so that all of the processed 
        data has same number of triads
        
        params:
        ref_ant_path: reference antenna path, reference obsID: 1160570752
        has all antenna tiles working 
        data_path: path to the processed data, type: str
        obsIDs: GPS time array of observations, type: int, array(int)
        baseline_length: baseline length of triads, need to fix/modify, type: int, 
        available baseline_lengths: {14,24,28,42}
        timestamps: time stamps in the data or want to process, type: int, default 14.
        freq_channels: frequency channels in the data or want to process, type: int, default 768, correspond to 40kHz.
        path_to_save: path to save the data
        if_HI: bool, True if processing HI simulation file, default: False
        returns: None

        save: numpy array of visiblities, append NaNs at the missing triads
        
        '''
        
        ##reference_info
        ## for baseline_length=14, N_triads_max = 47
        ## for baseline_length=24, N_triads_max = 32
        ## for baseline_length=28, N_triads_max = 29
        ## for baseline_length=42, N_triads_max = 14
        
        ## reference obsIDs having all triads working 1160570752
        if obsID_ref==None:
            obsID_ref = 1160570752
        
        if ant_ref_name == None: #antTv14_1160764792.npy
            
            ant_ref_name = 'antTv%d_%d.npy'%(baseline_length, obsID_ref)
            
        ant_ref = np.load(ref_ant_path + ant_ref_name, allow_pickle=True)
    
        N_type = 3 # no. of data type 3 data, model, ideal_model
        N_vis = 3 # no. of visibilities 3, v1, v2, v3 for ants {ab, bc, ca}
        N_triads = len(ant_ref)

        #for k in range(len(obsIDs)):
            
        if if_HI != True:
                
            try:
                if filename == None:
                        
                    filename = 'v%d_%d.npy'%(baseline_length, obsIDs) #v14_1160595880.npy
                    
                    antfilename = 'antTv%d_%d.npy'%(baseline_length, obsIDs)
                    
                else:
                        
                    filename = filename
                    antfilename = antfilename
                    
                    
                ants_triplets = np.load(ant_path + antfilename, allow_pickle=True)
                data_vis = np.load(data_path + filename)
                    
                
            except IOError:
                    
                raise IOError('file absent')
                
        else:
                
            try:
                
                filename = 'HI_v%d_%d.npy'%(baseline_length, obsIDs)
                ants_triplets = np.load(ant_path + 'antTv%d_%d.npy'%(baseline_length, obsIDs), allow_pickle=True)
                data_vis = np.load(data_path + filename)

            except IOError:

                raise IOError('file absent')
            
                
        match_indicies = []
        missing_indicies = []

        total_indicies = np.arange(len(ant_ref), dtype=np.int32)

        for i in range(len(ants_triplets)):

            match_indicies.append(np.where(ant_ref==ants_triplets[i])[0][0])

        match_indicies = np.sort(np.array(list(set(match_indicies))))

        missing_indicies = np.sort(np.array(list(set(total_indicies) - set(match_indicies))))   

        if len(missing_indicies) == 0:

            print('all triads present')

        else:
            
            if data_vis.ndim == 5:
                
                dummy_data_vis = np.empty(shape=(N_type, N_vis, N_triads, timestamps, freq_channels), dtype=np.complex64)
                    
                nan_insert= np.empty(shape=(N_type, N_vis, timestamps, freq_channels), dtype=np.complex64)
                nan_insert.real[:] = np.nan
                nan_insert.imag[:] = np.nan
                    
                for i in range(len(match_indicies)):

                    dummy_data_vis[:, :, match_indicies[i], :, :] = data_vis[:, :, i, :, :]

                for i in range(len(missing_indicies)):

                    dummy_data_vis[:, :, missing_indicies[i], :, :] = nan_insert *u.Jy
                
            else:

                dummy_data_vis = np.empty(shape=(N_vis, N_triads, timestamps, freq_channels), dtype=np.complex64)
                    
                nan_insert= np.empty(shape=(N_vis, timestamps, freq_channels), dtype=np.complex64)
                nan_insert.real[:] = np.nan
                nan_insert.imag[:] = np.nan
                    
                for i in range(len(match_indicies)):

                    dummy_data_vis[:, match_indicies[i], :, :] = data_vis[:, i, :, :]

                for i in range(len(missing_indicies)):

                    dummy_data_vis[:, missing_indicies[i], :, :] = nan_insert *u.Jy


            # another way of doing, laborious
            #data_vis = np.insert(arr=data_vis, obj=28, values=nan_insert, axis=2)

            np.save(path_to_save + filename, dummy_data_vis,)

            print('obsIDs, done!')
                
    def get_triad_counts(self, path=None, obsIDs=None, baseline_length=None):
        
        '''
        
        Get the number of working triads stored in the processed data, **specific function**
        This function will use the triad counts for the given baseline
        and append NaN visibilities at the missing triads, so that all of the processed 
        data has same number of triads
        
        params:
        
        path: path to the processed data numpy array, type: str
        obsIDs: GPS time array of observations, type: array(int)
        baseline_length: baseline length of triads, type: int, available baseline_lengths: {14, 24, 28, 42}
        
        return: triad count array 
        
        '''
        
        triad_counts = np.empty(len(obsIDs))

        for i in range(len(obsIDs)):
            
            try: 
                if filename==None:
                    filename='v%dH_%d.npy'%(baseline_length, obsIDs[i])
                data_vis = np.load(path + filename) 
                ## data is stored in specific name

                triad_counts[i] = len(data_vis[0,0,:])
                
            except:
                
                triad_counts[i] = np.nan
        
        return triad_counts
    
    def get_vis_n_bphase(self, loadfilepath=None, loadfilename=None, obsID=None,\
                        baseline_length=None, polarizations=None, savefilename=None,\
                             savefilepath=None, return_data=False):
        
    
        '''
        Designed to get the visibility data and the bispectrum phase from dataset of specific .npy format
        
        params:
        filename: vis file name, type: str, default name used: 'v%dH_%d.npy'%(baseline_length, obsIDs) format
        path: path to the processed numpy file, type: str
        obsIDs: GPS time of the observation, type: int
        baseline_length: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int
        if_use_extreme_channels: inspect the bispectrum phase at 3 extreme channels,
        lowest, middle, highest frequency channels,
        type: bool, default: False
        if_use_all_channels: if get the data at all triads, timestamps, and channels,
        type: bool, default: True 

        Noise Vis add parameters:

        freq: frequencies in Hz, dtype: array
        Aeff_MWA: type: float or ndarray, effective collecting area of MWA for the frequency range.
        dsize: typically number of frequencies samples, len(freq), type: int
        integration_time: integration time of observation
        freq_res: frequency resolution of the observation
        in_Jy: True if want noise in Jansky units, default : True

        if_return_vis: True if want visibilities {v1, v2, v3} at antenna pairs {ab, bc, ca} 
        along with bispectrum phase

        return:
        Bispectrum_phase : measured data, Foreground model, Foreground Ideal model, Noise, 
        Foreground_model_with_Noise, Ideal_foreground_with_Noise

        if_return_vis: True

        {vis_data*}, {vis_FG_model*}, {vis_FG_ideal_model*}, {vis_Noise*}, 

        * v1, v2, v3 
        '''

        if loadfilename == None:

            loadfilename = ['vis_all_%d_%s_%d.npy'%(baseline_length, polarizations, obsID),\
                            'vis_HI_%d_%s_1162989040.npy'%(baseline_length, polarizations)]

        try:

            vis = np.load(loadfilepath + loadfilename[0]) 
            vis_HI = np.load(loadfilepath + loadfilename[1])
            keys = vis.dtype.names + vis_HI.dtype.names
            print('vis data and HI shapes', vis.shape, vis_HI.shape)
            print('data processing started....')
            print('for obsID: %s, for %s meter baseline length and %s polarizations'%(obsID, baseline_length, polarizations))
        except IOError:

            raise IOError('file absent')
        keys_vis = ('vis_data:Jy/uncal',\
                    'vis_FG:Jy',\
                    'vis_FGI:Jy',\
                    'vis_Noise:Jy',\
                    'vis_HI:Jy',\
                    'vis_FG+Noise:Jy',\
                    'vis_FGI+Noise:Jy',\
                    'vis_FG+Noise+HI:Jy',\
                    'vis_FGI+Noise+HI:Jy')
        vis_arr = np.empty(shape=vis.shape,\
                              dtype=[(keys_vis[key_index], complex) for key_index in range(len(keys_vis))])
        
        bphase_arr = np.empty(shape=vis.shape,\
                              dtype=[(keys_vis[key_index], np.float64) for key_index in range(len(keys_vis))])
        
        v1_arr, v2_arr, v3_arr = [], [], []
        
        for key_index in range(len(keys)): # data, model, ideal_model, Noise, HI
            if key_index < 4:
                v1, v2, v3 = vis[keys[key_index]]
                
                print('processing...%s at index %d'%(keys_vis[key_index], key_index))
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
            else:
                v1, v2, v3 = vis_HI[keys[key_index]]
                
                print('processing...%s at index %d'%(keys_vis[key_index], key_index))
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
            bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
            vis_arr[keys_vis[key_index]] = v1, v2, v3 
       
        for model in range(2):
            
            key_index = 5
            
            v1, v2, v3 = v1_arr[model+1] + v1_arr[3],\
                            v2_arr[model+1] + v2_arr[3],\
                                v3_arr[model+1] + v3_arr[3]
            
            print('processing...%s at index %d'%(keys_vis[key_index+model], key_index+model))
            
            bphase_arr[keys_vis[key_index+model]] = np.angle(v1 * v2 * v3)
            vis_arr[keys_vis[key_index+model]] = v1, v2, v3

            v1, v2, v3 = v1_arr[model+1] + v1_arr[3] + v1_arr[4],\
                            v2_arr[model+1] + v2_arr[3] + v2_arr[4],\
                                v3_arr[model+1] + v3_arr[3] + v3_arr[4]
            
            print('processing...%s at index %d'%(keys_vis[key_index+model+2], key_index+model+2))
            bphase_arr[keys_vis[key_index+2+model]] = np.angle(v1 * v2 * v3)
            vis_arr[keys_vis[key_index+2+model]] = v1, v2, v3
        print('done!')
        
        if return_data==True:
            return vis_arr, bphase_arr
        
        else:
            if savefilename==None:
                
                savefilename = ['vis_all_cases_%d_%s_%d'%(baseline_length, polarizations, obsID), \
                               'bphase_all_cases_%d_%s_%d'%(baseline_length, polarizations, obsID)]
                               
                np.save(savefilepath + savefilename[0], vis_arr)
                np.save(savefilepath + savefilename[1], bphase_arr)
            else:
                print('either savefilename or savefilepath not given!')
                
    def create_correlation_data(self, loadfilepath=None, loadfilename=None, obsID=None,\
                        baseline_length=None, polarizations=None, savefilename=None,\
                             savefilepath=None, return_data=False):
        
    
        '''
        Designed to get the bispectrum phase from dataset in specific format
        
        params:
        
        filename: vis file name, type: str, default name used: 'v%dH_%d.npy'%(baseline_length, obsIDs) format
        path: path to the processed numpy file, type: str
        obsIDs: GPS time of the observation, type: int
        baseline_length: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int
        if_use_extreme_channels: inspect the bispectrum phase at 3 extreme channels,
        lowest, middle, highest frequency channels,
        type: bool, default: False
        if_use_all_channels: if get the data at all triads, timestamps, and channels,
        type: bool, default: True 

        Noise Vis add parameters:

        freq: frequencies in Hz, dtype: array
        Aeff_MWA: type: float or ndarray, effective collecting area of MWA for the frequency range.
        dsize: typically number of frequencies samples, len(freq), type: int
        integration_time: integration time of observation
        freq_res: frequency resolution of the observation
        in_Jy: True if want noise in Jansky units, default : True

        if_return_vis: True if want visibilities {v1, v2, v3} at antenna pairs {ab, bc, ca} 
        along with bispectrum phase
        
        

        return:
        
        Bispectrum_phase : measured data, Foreground model, Foreground Ideal model, Noise, 
        Foreground_model_with_Noise, Ideal_foreground_with_Noise

        if_return_vis: True

        {vis_data*}, {vis_FG_model*}, {vis_FG_ideal_model*}, {vis_Noise*}, 

        * v1, v2, v3 
        
        '''

        if loadfilename == None: # these are the default names of the files

            loadfilename = ['vis_coherent_HI_n_Model_%d_%s_%d.npy'%(baseline_length, polarizations, obsID),
                            'vis_Noise_Height0_%d_%s_%d.npy'%(baseline_length, polarizations, obsID) ]
        
        try:

            vis = np.load(loadfilepath + loadfilename[0]) # data contains FG, HI
            vis_Noise = np.load(loadfilepath + loadfilename[1]) # data contains Noise
            keys = vis.dtype.names
            keys_Noise = vis_Noise.dtype.names
            
            print('vis and noise has data shapes', vis.shape, vis_Noise.shape)
            print('data processing started....')
            print('available data as ', keys, keys_Noise)
            print('for obsID: %s, for %s meter baseline length and %s polarizations'%(obsID, baseline_length, polarizations))
            
        except IOError:

            raise IOError('file absent')
        
        # complex-visibilities will be stored as FG, HI, Noise, FG+HI, FG+Noise, FG+HI+Noise
        
        keys_vis = ('HI:Jy', 
                    'FG:Jy',
                    'Noise:Jy',
                    'FG+HI:Jy',
                    'FG+Noise:Jy',
                    'FG+Noise+HI:Jy')
        
        vis_arr = np.empty(shape=vis.shape,\
                              dtype=[(keys_vis[key_index], complex) for key_index in range(len(keys_vis))])
        
        # closure phase (bispectrum phase) will be stored as FG, HI, Noise, FG+HI, FG+Noise, FG+HI+Noise,
        # difference of the bispectrum phases of ({FG+HI} - FG), ({FG+HI+Noise} - FG)
        
        bphase_arr = np.empty(shape=vis.shape,\
                              dtype=[(keys_vis[key_index], np.float64) for key_index in range(len(keys_vis))])
        
        v1_arr, v2_arr, v3_arr = [], [], [] # temp vis store
        
        for key_index in range(len(keys_vis)): # FG model, HI, Noise
            
            if key_index < 2:
                
                v1, v2, v3 = vis[keys[key_index]] # data from vis
                
                print('processing... %s at index %d'%(keys_vis[key_index], key_index))
                
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
                bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
                vis_arr[keys_vis[key_index]] = v1, v2, v3 
                
            elif key_index == 2:
                
                v1, v2, v3 = vis_Noise[keys_Noise[key_index-key_index]] # data from noise vis

                print('processing... %s at index %d'%(keys_vis[key_index], key_index))
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
                bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
                vis_arr[keys_vis[key_index]] = v1, v2, v3 
            
            elif key_index == 3: # adding vis + HI
                
                v1, v2, v3 = vis[keys[key_index-key_index]]
                v1H, v2H, v3H = vis[keys[key_index-key_index+1]]
                
                v1, v2, v3 = v1+v1H, v2+v2H, v3+v3H
                
                print('processing... %s at index %d'%(keys_vis[key_index], key_index))
                
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
                bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
                vis_arr[keys_vis[key_index]] = v1, v2, v3 
                
            elif key_index == 4: # adding vis + noise vis
                
                v1, v2, v3 = vis[keys[key_index-key_index]]
                
                v1N, v2N, v3N = vis_Noise[keys_Noise[key_index-key_index]]

                v1, v2, v3 = v1+v1N, v2+v2N, v3+v3N
                
                print('processing... %s at index %d'%(keys_vis[key_index], key_index))
                
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
                bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
                vis_arr[keys_vis[key_index]] = v1, v2, v3 
                
            elif key_index == 5: # adding vis + HI + noise vis 
                
                v1, v2, v3 = vis[keys[key_index-key_index]]
                v1H, v2H, v3H = vis[keys[key_index-key_index+1]]
                v1N, v2N, v3N = vis_Noise[keys_Noise[key_index-key_index]]

                v1, v2, v3 = v1+v1H+v1N, v2+v2H+v2N, v3+v3H+v3N
                
                print('processing... %s at index %d'%(keys_vis[key_index], key_index))
                
                v1_arr.append(v1)
                v2_arr.append(v2)
                v3_arr.append(v3)
                
                bphase_arr[keys_vis[key_index]] = np.angle(v1 * v2 * v3)
                vis_arr[keys_vis[key_index]] = v1, v2, v3 
                

        print('done!')
        if return_data==True:
            return vis_arr, bphase_arr
        
        else:
            if savefilename==None:
                
                savefilename = ['vis_coherent_%d_%s_%d'%(baseline_length, polarizations, obsID), \
                               'bphase_coherent_%d_%s_%d'%(baseline_length, polarizations, obsID)]
                               
                np.save(savefilepath + savefilename[0], vis_arr)
                np.save(savefilepath + savefilename[1], bphase_arr)
            else:
                print('either savefilename or savefilepath not given!')
   
    def get_V_eff(self, loadfilepath=None, loadfilename=None, obsID=None,\
            baseline_length=None, polarizations=None, data_key=None, N_chan=None,\
                freq_res=None, use_window_sq=None, if_save=False, savefilepath=None, savefilename=None):
            
            """
            Get effective visibilities

            Args:
                loadfilepath (str): path to visiblility .npy file. Defaults to None.
                loadfilename (str): Name of the visibility .npy file, the files are saved with some default names
                e.g. 'vis_all_cases_%basline_length_%polarisation_%obsID.npy'. Defaults to None.
                obsID (int) : obsID, GPS time of the observation. Defaults to None.
                baseline_length (int): baseline length, available {14, 24, 28, 42} meter baselines. Defaults to None.
                polarizations (str): polarizations, either 'xx' or 'yy'. Defaults to None.
                data_key (str): dataset has keys, choose amongst following keys:
                'vis_data:Jy/uncal',
                'vis_FG:Jy',
                'vis_FGI:Jy',
                'vis_Noise:Jy',
                'vis_HI:Jy',
                'vis_FG+Noise:Jy',
                'vis_FGI+Noise:Jy',
                'vis_FG+Noise+HI:Jy',
                'vis_FGI+Noise+HI:Jy'.
                'all_keys'. Defaults to None.
                N_chan (int): usual size of the data. Number of frequency channels in the dataset.
                e.g. N_channels=768 for data @ 40kHz, 384 for data @ 80kHz, 24 for data  @ 1.28MHz. Defaults to None.
                freq_res (int): frequency resolution of the data. Defaults to None.
                use_window_sq (bool, optional): set to True if use window squared function to estimate the effective visibilities. Defaults to False.
                if_save (bool, optional): set to True if save the effective visibilites as .npy arrays. Defaults to False.
                savefilepath (str, optional): if above is True, then provide the path to save the file. Defaults to None.
                savefilename (str, optional): if above is True, then provide the filename. Defaults to None.

            Raises:
                TypeError: if put incorrect name of the loading file.
                IOError: if the file is absent from the given directory.

            Returns:
                ndarray: numpy array of the efffective visibilities
            """
            
            # getting the window function to normalise the visibilities
            
            delays = np.fft.fftshift(np.fft.fftfreq(N_chan, freq_res*u.Hz))
            dtau = abs(delays[0] - delays[1])
            
            window = signal.blackmanharris(delays.size, True)
            window_sq = signal.convolve(window, window, mode='full', method='fft')[::2]

            windowft = (np.abs(np.fft.fft(window))**2)
            windowft2 = (np.abs(np.fft.fft(window_sq))**2)
            
            # normalising the windows and window**2 function
            ## using window**2 so that we have the compatible dynamic range of 1e-5 of the EoR.
            
            area_windows = np.trapz(windowft, x= delays, dx=dtau)
            area_windows2 = np.trapz(windowft2, x= delays, dx=dtau)
            norm = area_windows2/area_windows
            
            windowft2 = windowft2/norm
            window_sq = window_sq/np.sqrt(norm)
            
            if use_window_sq == True:
                window = window_sq
                
            if loadfilename == None:
                loadfilename = 'vis_all_cases_%s_%s_%s.npy'%(baseline_length, polarizations, obsID)
            
            else:
                raise TypeError('filename not given!')
            
            try:
                data_vis = np.load(loadfilepath + loadfilename)
                keys = data_vis.dtype.names
            
            except IOError:
                raise IOError('file absent from given directory: %s'%loadfilepath)
            
            if data_key != 'all_keys':

                v1, v2, v3 = data_vis[data_key]

                V_inv1 = np.sum(np.abs(v1) * window * freq_res, axis=2) / np.sum(window * freq_res)
                V_inv2 = np.sum(np.abs(v2) * window * freq_res, axis=2) / np.sum(window * freq_res)
                V_inv3 = np.sum(np.abs(v3) * window * freq_res, axis=2) / np.sum(window * freq_res)
                    
                V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
                                        + (1./V_inv3)**2
                                        
                V_eff = np.array(np.sqrt(1./V_eff_inv), dtype=[(data_key, np.float64)])
            
            else:
                
                V_eff = np.empty(shape=(data_vis.shape[1:3]), \
                    dtype=[(data_key[key_index], np.float64) for key_index in range(len(data_key))])
                
                for key_indx in range(len(keys)):
                    
                    v1, v2, v3 = data_vis[keys[key_indx]]

                    V_inv1 = np.sum(np.abs(v1) * window * freq_res, axis=2) / np.sum(window * freq_res)
                    V_inv2 = np.sum(np.abs(v2) * window * freq_res, axis=2) / np.sum(window * freq_res)
                    V_inv3 = np.sum(np.abs(v3) * window * freq_res, axis=2) / np.sum(window * freq_res)
                        
                    V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
                                            + (1./V_inv3)**2
                    V_eff[data_key[key_indx]] = np.sqrt(1./V_eff_inv)
            
            if if_save == True:
                if savefilename==None:
                    if data_key != 'all_cases':
                        savefilename='v_eff_%s_%s_%s_%s.npy'%(data_key, baseline_length, polarizations, obsID)
                    else:
                        savefilename='v_eff_%s_%s_%s.npy'%(baseline_length, polarizations, obsID)
                np.save(savefilepath+savefilename, V_eff)
                
            else:
                return V_eff
    
    def get_full_bphase(self, path=None, obsIDs=None, baseline_length=None,\
                                           keep_dims=True, N_triads=None, N_timestamps=None):
        
        '''
        
        Get incoherrent pbhase from the data and FG model
        
        params:
        
        path: path to the visibility file
        obsIDs: GPS time of the observation (full array)
        baseline_length: baseline length, dataset available in 14m , 28m, 42m baselines
        include_downsample: bool, if True then include downsample indicies with unflagged, 
        otherwise returns only unflagged data
        keep_dims: bool, if keep the dimension of the dataset the same, 
        i.e. shape: (N_obs, N_triads, N_timestamps, N_channels),
        otherwise, flattens the shape to (N_obs x N_triads x N_timestamps, N_channels)

        return:
        
        bispectrum phase data array
        
        '''
        
        B_phase_m = []
        B_phase_FG = []
        B_phase_FGI = []
        absent_obs = []

        for i in range(len(obsIDs)):
            
            try:
                    
                data_vis = np.load(path +'v%dH_%d.npy'%(baseline_length, obsIDs[i]))

            except IOError:

                absent_obs.append(obsIDs[i])

                raise IOError('vis file absent, obsIDs, %d'%obsIDs[i])
            if keep_dims != True:
                
                for N_triad in range(len(data_vis[0,0])):

                    for N_timestamp in range(len(data_vis[0,0,0])):

                        v1_m, v2_m, v3_m = data_vis[0][:, N_triads, N_timestamps, :]

                        v1_FG, v2_FG, v3_FG = data_vis[1][:, N_triads, N_timestamps, :]

                        v1_FGI, v2_FGI, v3_FGI = data_vis[2][:, N_triads, N_timestamps, :]
                            
                        B_phase_m.append(np.angle(v1_m * v2_m * v3_m))
                        B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))
                        B_phase_FGI.append(np.angle(v1_FGI * v2_FGI * v3_FGI))
                    
            else :
                
                data_ang = np.angle(data_vis[0][0] * data_vis[0][1] * data_vis[0][2])
                FG_ang = np.angle(data_vis[1][0] * data_vis[1][1] * data_vis[1][2])
                FG_angI = np.angle(data_vis[2][0] * data_vis[2][1] * data_vis[2][2])
                    
                B_phase_m.append(data_ang)
                B_phase_FG.append(FG_ang)
                B_phase_FGI.append(FG_angI)
            
        B_phase_m = np.array(B_phase_m)
        B_phase_FG = np.array(B_phase_FG)
        B_phase_FGI = np.array(B_phase_FGI)
        absent_obs  = np.array(absent_obs, dtype=np.int32)

        return B_phase_m, B_phase_FG, B_phase_FGI, absent_obs
    
    def get_full_bphase_with_noise(self, filename=None, path=None, obsIDs=None,\
                                    baseline_length=None, N_triads=None,\
                    N_timestamps=None, freq=None, Aeff_MWA=None,\
                        dsize=None, integration_time=None, freq_res=None):
        
        '''
        
        Get incoherrent pbhase from the data and FG model
        
        params:
        
        filename: vis file name, type: str, default name used: 'v%dH_%d.npy'%(baseline_length, obsIDs) format
        path: path to the processed numpy file, type: str
        obsIDs: GPS time of the observation, type: int
        baseline_length: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int

        Noise Vis add parameters:

        freq: frequencies in Hz, dtype: array
        Aeff_MWA: type: float or ndarray, effective collecting area of MWA for the frequency range.
        dsize: typically number of frequencies samples, len(freq), type: int
        integration_time: integration time of observation
        freq_res: frequency resolution of the observation
        in_Jy: True if want noise in Jansky units, default : True


        return:
        
        bispectrum phase data array
        
        '''
        B_phase_m_arr = []
        B_phase_FG_arr = []
        B_phase_FGI_arr = []
        B_phase_N_arr = []
        B_phase_FG_N_arr = []
        B_phase_FGI_N_arr = []

        for i in range(len(obsIDs)):

            B_phase_m, B_phase_FG, B_phase_FGI, B_phase_N, B_phase_FG_n_N, B_phase_FGI_n_N = \
                self.get_data_bphase(filename=filename, path=path, obsIDs=obsIDs[i],\
                                baseline_length=baseline_length, N_triads=N_triads,\
                                N_timestamps=N_timestamps, if_use_extreme_channels=False,\
                                if_use_all_channels=True, freq=freq, Aeff_MWA=Aeff_MWA,\
                                dsize=dsize, integration_time=integration_time, freq_res=freq_res, \
                                in_Jy=True, if_return_vis=False)
        
            B_phase_m_arr.append(B_phase_m)
            B_phase_FG_arr.append(B_phase_FG)
            B_phase_FGI_arr.append(B_phase_FGI)
            B_phase_N_arr.append(B_phase_N)
            B_phase_FG_N_arr.append(B_phase_FG_n_N)
            B_phase_FGI_N_arr.append(B_phase_FGI_n_N)
        
        B_phase_m_arr = np.array(B_phase_m_arr)
        B_phase_FG_arr = np.array(B_phase_FG_arr)
        B_phase_FGI_arr = np.array(B_phase_FGI_arr)
        B_phase_N_arr = np.array(B_phase_N_arr)
        B_phase_FG_N_arr = np.array(B_phase_FG_N_arr)
        B_phase_FGI_N_arr = np.array(B_phase_FGI_N_arr)

        return B_phase_m_arr, B_phase_FG_arr, B_phase_FGI_arr,\
              B_phase_N_arr, B_phase_FG_N_arr, B_phase_FGI_N_arr
    
    def create_coherencetime_data(self, obsID, baseline_length,\
        triad_angles, timestamps, Hex_index, woden_coarseband, polarizations,\
            HIpath, modelpath, loadfilenames, savefilename, \
                savefilepath, if_equilateral_triad=True, only_HI=False, skip_HI=False):
        
        """
        Create coherence dataset for FG, HI, Noise from .uvfits and simulated noise and store as .npy files

        Args:
            obsIDs (int, ndarray): GPS time of the observation
            baseline_length (int): baseline length in meters. available baselines 14, 24, 28, 42
            triad_angles (int): triad angle.
            timestamps (int): time stamps in the data or want to process.
            Hex_index (int): MWA antenna config index type int, 
                            choose amongst following [0: HexE, 1: HexS, 2: HexE+S, 3: Tile]
                            e.g: from MWA_config = count_antennae(uv_object) class
                            MWA_config[0] : antanna indicies of HexE 
                            MWA_config[1] : antanna indicies of HexS
                            MWA_config[2] : antanna indicies of HexE + HexS combined
                            MWA_config[3] : antanna indicies of Tiles 
            polarizations: polarizations, available options: 'xx', 'yy'.
            HIpath (str): path to the observation HI uvfits file
            modelpath (str): path to the model uvfits, foreground
            loadfilenames (str of [datafilename, modelfilename]): loadfilenames. 
            Default names ['obsID', 'obsID_Puma_25000', 'obsID_Puma_Unity25000']
            savefilename (str): save filename.
            savefilepath (str): path to save the file.
            if_equilateral_triad (bool):  if using equilateral triads in the analysis, defaults to True.
            only_HI (bool): True if creating the data for the HI file.
        """
        
        if loadfilenames == None: # defaults names define your names
            loadfilenames = ['%d_25000_Height0'%obsID, 'HI_%d_band'%obsID]
        
        if skip_HI==False:
            
            if woden_coarseband==True:
                
                vis_HI=[]
                for band in range(1, 25):

                    try:
                        if band<10:
                            HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[1] + '0%d.uvfits'%band)
                            print('processing... %s.uvfits, %d'%(loadfilenames[1], band))
                        else:
                            HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[1] + '%d.uvfits'%band)
                            print('processing... %s.uvfits, %d'%(loadfilenames[1], band))

                    except IOError:
                        raise IOError

                    uv_HI = UVData()
                    get_int_info=Get_Antenna_Info()
                    VIS = Vis()
                    
                    uv_HI.read(HI_f, read_data=False)
                    HexHI=get_int_info.count_antennae(uv=uv_HI)[4] ## index 0 is HexE 1 is HexS, 2 HexE+S
                    ant_loc= uv_HI.antenna_positions[HexHI-1]
                    
                    
                    freq = uv_HI.freq_array[0]
                    dsize = len(freq)
                    uv_HI.read(HI_f, polarizations=[polarizations], antenna_nums=HexHI) #Hex E + Hex S
                    anttriplets = get_int_info.get_triplets_blvectors(ant_labels=HexHI, ant_positions=ant_loc,\
                                    baseline_length=baseline_length,\
                                    triad_angles=triad_angles, unique=True, \
                                    if_equilateral_triad=if_equilateral_triad)[0]
                    if baseline_length==24:
                        anttriplets = np.array([[1,7,5]])
                    
                    vH1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=0)
                    vH2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=1)
                    vH3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=2)
                    vis_HI.append(np.array([vH1, vH2, vH3])) 

                vis_HI = np.array(vis_HI)
                vis_HI = np.concatenate(vis_HI, axis=-1)
                vis_HI = np.array(vis_HI)
                print(vis_HI, vis, 'HI visibility %d processed!'%obsID)
            
            if woden_coarseband==False:
                print('woden generated .uvfits are not at 1.28MHz')
                
                try:
                    HI_f = os.path.join(DATA_PATH, HIpath + loadfilenames[1] + '.uvfits')
                    print('processing... %s.uvfits'%loadfilenames[1])

                except IOError:
                    raise IOError

                uv_HI = UVData()
                get_int_info=Get_Antenna_Info()
                VIS = Vis()

                uv_HI.read(HI_f, read_data=False)
                HexHI=get_int_info.count_antennae(uv=uv_HI)[4] ## index 0 is HexE 1 is HexS, 2 HexE+S
                ant_loc= uv_HI.antenna_positions[HexHI-1]
                
                freq = uv_HI.freq_array[0]
                dsize = len(freq)
                uv_HI.read(HI_f, polarizations=[polarizations], antenna_nums=HexHI) #Hex E + Hex S
                
                anttriplets = get_int_info.get_triplets_blvectors(ant_labels=HexHI, ant_positions=ant_loc,\
                                baseline_length=baseline_length,\
                                triad_angles=triad_angles, unique=True, \
                                if_equilateral_triad=if_equilateral_triad)[0]
                if baseline_length==24:
                        anttriplets = np.array([[1,7,5]])
                        
                vH1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=0)
                vH2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=1)
                vH3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uv_HI, index=2)
                vis_HI = np.array([vH1, vH2, vH3])
                print(vis_HI, 'HI visibility %d processed!'%obsID)
            
        if only_HI==False:

            try:
                model_FG = os.path.join(DATA_PATH, modelpath + loadfilenames[0] + '.uvfits')
                print('processing... %s.uvfits'%loadfilenames[0])
  
            except IOError:
                raise IOError 
            
            get_int_info=Get_Antenna_Info()
            VIS = Vis()
            uvFG = UVData()
            uvFG.read(model_FG, read_data=False)
            freq = uvFG.freq_array[0]
            dsize = len(freq)
            Hex=get_int_info.count_antennae(uv=uvFG)[Hex_index] ## index 0 is HexE 1 is HexS, 2 HexE+S
            Hex=np.array([67, 68, 69, 70, 74, 75, 80, 86]) + 1
            
            ant_loc= uvFG.antenna_positions[Hex-1]
            uvFG.read(model_FG, polarizations=[polarizations], antenna_nums=Hex) #Hex E + Hex S
            anttriplets = get_int_info.get_triplets_blvectors(ant_labels=Hex, ant_positions=ant_loc,\
                                baseline_length=baseline_length,\
                                triad_angles=triad_angles, unique=True, \
                                if_equilateral_triad=if_equilateral_triad)[0]
            if baseline_length==24: #68, 80, 75
                anttriplets=np.array([[69, 81, 76]])
                
            vFG1 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=0)
            vFG2 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=1)
            vFG3 = VIS.vis_all_triads(anttriplets=anttriplets, timestamps=timestamps, dsize=dsize, uv=uvFG, index=2)

            v_FG = np.array([vFG1, vFG2, vFG3])
            print(v_FG, 'FG visibility %d processed!'%obsID)
            
            Noise = MWA_Noise_Analysis()
            SEFD = Noise.get_SEFD(Aeff_MWA=self.Aeff_MWA, freq=freq*1e-6, dsize=freq.size, integration_time=112,\
                    frequency_resolution=abs(freq[1]-freq[0]), if_rand=False, in_Jy=True)
            vis_Noise=Noise.get_obs_compatible_SEFD(SEFD=SEFD[0], SEFD_rms=SEFD[1], if_use_rms=True, \
                    N_timestamps=timestamps, N_triads=v_FG.shape[1],\
                        if_save_arr=False, savefilename=None, savepath=None)[1]
            print(vis_Noise, 'Noise visibility %d processed!'%obsID)
            
        data_keys=('vis_model:Jy', 'vis_HI:Jy', 'vis_Noise:Jy')
        vis = np.empty(shape=v_FG.shape,\
            dtype=[(data_keys[key_index], complex) for key_index in range(len(data_keys))])
            
        if vis_HI.shape != v_FG.shape:
            raise TypeError('HI and FG shape mismatch at: ', obsID, vis_HI.shape, v_FG.shape)
        
        if only_HI==True:
            vis['vis_HI:Jy']= vis_HI

        if skip_HI==True:
            vis['vis_model:Jy'] = v_FG
            vis['vis_Noise:Jy'] = vis_Noise

        else:
            vis['vis_HI:Jy'] = vis_HI
            vis['vis_model:Jy'] = v_FG
            vis['vis_Noise:Jy'] = vis_Noise

        if savefilename == None:
            print('savefilename set to None; using default savefilename as:')
            
            if only_HI==True:
                savefilename = 'HI_Coh_only_%d_%s_%d'%(baseline_length, polarizations, obsID)
                
            if skip_HI==True:
                savefilename = 'FG_Coh_only_%d_%s_%d'%(baseline_length, polarizations, obsID)

            else:
                savefilename = 'FG_HI_Coh_%d_%s_%d'%(baseline_length, polarizations, obsID)
                
        print(savefilename)
        np.save(savefilepath + savefilename, vis)
class MWA_Noise_Analysis(object):
    '''
    
    Get the sky, noise and system temperature of MWA 
    for EoR frequencies
    
    '''
    
    def __init__(self):
        
        pass
    
    def sky_temp(self, freq = None):
        
        '''
        
        Get sky temperature for EoR freuqencies
        params:

        freq: frequency in MHz

        returns:

        Cold sky temperature for MWA
        
        '''

        alpha = -2.5 # power-law index
        T_scale = 180.*u.Kelvin # in K units
        freq_scale = 180*u.MHz # in MHz units
        T0 = T_scale/(freq_scale ** alpha) # scaling factor
        T_sky = T0 * (freq*u.MHz) ** alpha

        return T_sky
    
    def antenna_noise_temp(self, size=None, if_rand=None):
        
        '''
        
        Function creates a random antenna noise temperature
        with an assumption of antenna temperature varies between
        40K - 50K in the EoR frequencies.
        
        params:
        
        size: number of samples required
        if_rand: if generate random temp. from 40-50K
        returns:
        
        Noise Temperature
        
        '''
        
        mean = 50 # in K units
        scale = 5 # +- 5 K variation

        if if_rand == True:
            T_noise = np.random.normal(loc=mean, scale=scale, size=size)
        else:
            T_noise = mean*np.ones(size)
        return T_noise*u.Kelvin
    
    def system_temp(self, freq=None, size=None, if_rand=None):
        
        '''
        
        Get the sky temperature in EoR frequencies
        
        T_sys = T_sky + T_noise
        
        params:
        
        freq: frequency in MHz units
        size: number of samples for which noise temperature is calculated
        (e.g. length of the frequency array).
        if_rand: if generate random temp. from 40-50K

        returns:
        
        System Temperature of MWA at the EoR frequencies in Kelvin
        
        '''
        if __name__ ==  "__main__":
            
            T_sys = self.sky_temp(freq) + self.antenna_noise_temp(size, if_rand)
        else:
            T_sys = MWA_Noise_Analysis.sky_temp(self, freq) + MWA_Noise_Analysis.antenna_noise_temp(self, size, if_rand)
        
        return T_sys
    
    def get_SEFD(self, Aeff_MWA=None, freq=None, dsize=None, integration_time=None,
                frequency_resolution=None, if_rand=None, in_Jy=None):
        
        '''
        
        Get the radiometric noise in Jy (SEFD) for MWA at the EoR frequencies
        
        params:
        
        A_eff_MWA: effective area of MWA in m^2 units.
        T_sys: system temperature in K units.
        freq: frequency in MHz units.
        size: number of samples for which noise temperature is calculated
        (e.g. length of the frequency array).
        integration_time: integration time of the data in seconds (or required)
        frequency_resolution: frequency resolution of the data in Hz (or assumed)
        if_rand: if generate random temp. from 40-50K
        in_Jy: if SEFD required in Jansky units
        returns:
        
        System equivalent flux density (SEFD) using Radiometric Noise equation in Jy units
        and RMS flux density of SEFD
        
        '''
        kB = constants.k_B # Boltzmann's constant
        if __name__ ==  "__main__":
            T_sys = self.system_temp(freq, dsize, if_rand)
        else:
            T_sys = MWA_Noise_Analysis.system_temp(self, freq, dsize, if_rand)
        SEFD = (2.0 * kB * T_sys) / (Aeff_MWA *u.m *u.m)
        SEFD_rms = SEFD / np.sqrt(frequency_resolution *u.Hz * integration_time *u.second)
    
        if in_Jy == True:
            SEFD = SEFD.to(u.Jy)
            SEFD_rms = SEFD_rms.to(u.Jy)
            return SEFD, SEFD_rms

        else:
            return SEFD, SEFD_rms
    
    def get_obs_compatible_SEFD(self, SEFD, SEFD_rms, N_timestamps, N_triads, drawn_same=False,\
        if_save_arr=False, savefilename=None, savepath=None):
        
        '''
        
        Get the SEFD noise estimate of N number of observations
        
        params:
        
        N_obs: number of observations
        SEFD: system noise in Jy units
        SEFD_rms: system noise rms in Jy units
        drawn_same (bool): set to True if drawing real and imaginary part in the noise estimation. Defaults to False.
        N_timestamps: number of time stamps in the data (or required); default = 14
        N_triads: given the baseline the no of triads vary, T14: 47, T24: 32, T28:29, T42:14
        default T14: 47
        if_save_arr: True if save as array, Defaults to False


        returns:
        
        complex noise SEFD of numpy array of the shape: (N_obs, N_timestamps, N_freq_channels)
        compatible with the conventional visibility dataset.
        default N_timestamps = 14, and N_freq_channels = len(SEFD) provided.
        3get_V_eff
        '''
        N_vis = 3
        
        rand_noise_vis_real = np.random.normal(loc=0, scale=SEFD_rms,\
                                                size=(int(N_vis * N_timestamps * N_triads), len(SEFD)))
        
        rand_noise_vis_imag = np.random.normal(loc=0, scale=SEFD_rms,\
                                                size=(int(N_vis * N_timestamps * N_triads), len(SEFD)))
        
        rand_noise_vis_real = rand_noise_vis_real.reshape(N_vis,  N_triads,  N_timestamps, len(SEFD))/np.sqrt(2)
        
        if drawn_same==False:
            rand_noise_vis_imag = rand_noise_vis_imag.reshape(N_vis, N_triads,  N_timestamps, len(SEFD))/np.sqrt(2)
            
        else:
            rand_noise_vis_imag = rand_noise_vis_real

        noise = rand_noise_vis_real + 1j*rand_noise_vis_imag
        noise = noise*u.Jy
        SEFD = SEFD + 1j*SEFD
        SEFD = SEFD.reshape(1, len(SEFD))
        
        SEFD_with_noise = np.array([SEFD + noise[0], SEFD + noise[1], SEFD + noise[2]]) # 0, 1, 2 for v1, v2, v3
        
        if if_save_arr==True:
            Noise_arr = np.array(noise, dtype=(np.record, [('Noise_SEFD(units:Jy)', np.complex64)]))
            np.save(savepath + savefilename, Noise_arr)
            
        else:
            return SEFD_with_noise, noise  
class RFI_class(object):

    '''
    RFI investigation in the bisepctrum phase data of specific format
    '''
    def __init__(self) -> None:
        pass

    def get_RFI_info(self, data_path, loadfilename, obsID,\
                      baseline_length, assume_same_traids, mode='median', use_numpy=None, along_freq=None):

        '''
        get the RFI metric for the bispectrum phase data for given baseline triads
        
        params: 
        
        data_path: path to the bispectrum data file, currently support .npy files generated via Data_Creation class,
          see all info above in
        loadfilename: name of the bispectrum phase dataset file. default: 'v%baseline_length_%obsID.npy'
        obsID: GPSTIME of the observation, dtype: int
        baseline_length: baseline length of the triads, available options: 14, 24, 28, 42
        assume_same_traids: bool, assuming all triads have similar bispectrum phase, default: True
        mode: mode of metric calculation, default: uses median statistics, other option, 'mean'
        use_numpy (bool): True if use numpy for deviation,
        along_freq (bool): True if deviation along frequency axis.
        return:

        bispectrum phase at all triad baselines and timestamps, deviation in the bispectrum phase at all triad baselines and timestamps
        type: ndarray

         '''
        
        if loadfilename == None:

            loadfilename = 'v%d_%d.npy'%(baseline_length, obsID)

        try:

            vis = np.load(data_path + loadfilename)
            
            if vis.ndim < 5:
                vis = vis
                
            else:
                vis = vis[0]
        
        except IOError:

            raise IOError('loadfile absent')
        
        # 0, 1, 2 stands for v1, v2, v3

        bphase = np.angle((vis[0] * vis[1] * vis[2])) # typical shape (47, 14, 768) for 14 meter triads (include nans as well as missing triads)
        
        if along_freq != True:
            
            if assume_same_traids != True:
                
                if use_numpy == True:
                    
                    deviation_metric = np.diff(bphase, axis=1, n=1)
                    
                else:
                    
                    deviation_metric = np.empty(shape=bphase.shape,)   
                    bphase_metric = bphase

                    for triads in range(len(bphase[:,0])): # over triads e.g. 47

                        for timestamps in range(1, len(bphase[0])): # over timestamps e.g. 14

                            prev_timestamp = int(timestamps - 1)
                            dbphase = bphase[triads, timestamps] - bphase[triads, prev_timestamp]

                            deviation_metric[triads, timestamps] = dbphase
                    
                    deviation_metric = np.delete(deviation_metric, 0, axis=1)
            
            elif assume_same_traids == True:
                
                if use_numpy == True:
                    
                    if mode == 'median':
                        bphase_metric = np.nanmedian(bphase, axis=0)
                        deviation_metric = np.diff(bphase_metric, axis=0, n=1)
                        
                    elif mode == 'mean':
                        
                        bphase_metric = np.nanmean(bphase, axis=0)
                        deviation_metric = np.diff(bphase_metric, axis=0, n=1)

                    
                else:
                    
                    deviation_metric = np.empty(shape=bphase[0,:].shape) # shape = 14, 768 
                    bphase_metric = bphase
                    
                    if mode == 'median':
                        
                            bphase_m = np.nanmedian(bphase, axis=0)

                    elif mode == 'mean':
                        
                            bphase_m = np.nanmean(bphase, axis=0)
                            
                    for timestamps in range(1, len(bphase[0])): # timestamps e.g .14

                        prev_timestamp = int(timestamps - 1)
                        
                        dbphase = bphase_m[timestamps] - bphase_m[prev_timestamp]

                        deviation_metric[timestamps] = dbphase

                    deviation_metric = np.delete(deviation_metric, 0, axis=0)
        else:
            
            if use_numpy == True:
                
                    if assume_same_traids!=True:
                        
                        deviation_metric = np.diff(bphase, axis=2, n=1)
                        
                    else:
                        
                        if mode == 'median':
                            bphase_metric = np.nanmedian(bphase, axis=0)
                            deviation_metric = np.diff(bphase_metric, axis=1, n=1)
                            
                        elif mode == 'mean':
                            
                            bphase_metric = np.nanmean(bphase, axis=0)
                            deviation_metric = np.diff(bphase_metric, axis=0, n=1)
            else:
                raise TypeError('only numpy div available')

        return deviation_metric, bphase_metric
    
    def get_SSINS_metric(self, datapath=None, loadfilename=None, obsID=None,\
            niter=None, uniq_XX=None, uniq_YY=None, savepath=None, savefilename=None):
        
        """
        
        SSINS metric for zscore, mean subtracted visibilities

        Args:
            datapath (str): path to the data file, e.g. uvfits file
            loadfilename (str, optional): name of the file, e.g. uvfits file
            obsID (int): GPSTIME of the observation.
            savepath (str): path to save the file.
            savefilename (tuple): save file name ordering as autos, cross baselines 
            [zscore_autos, vis_sub_autos, zscore_cross, vis_sub_cross]
            
        """
        
        if loadfilename == None:
            
            loadfilename = '%d.uvfits'%obsID
            
        if savefilename == None:
                
            savefilename = ['zscore_autos_%d'%obsID, 'vis_sub_autos_%d'%obsID, 'zscore_cross_%d'%obsID, 'vis_sub_cross_%d'%obsID]
        
        ss = SS()  
        ss.read(filename = datapath + loadfilename, read_data=False)  
        times = np.unique(ss.time_array)[1:-1]
        
        if niter != None:
            
            times_XX = times[uniq_XX]
            times_YY = times[uniq_YY]
            
            ss.read(filename = datapath + loadfilename, read_data=True, times=times_XX, diff=True, polarizations=['xx'])
            ins_autosXX = INS(ss, spectrum_type="auto")
            ins_crossXX = INS(ss, spectrum_type='cross')
            
            ss.read(filename = datapath + loadfilename, read_data=True, times=times_YY, diff=True, polarizations=['yy'])
            ins_autosYY = INS(ss, spectrum_type="auto")
            ins_crossYY = INS(ss, spectrum_type='cross')
            
            np.save(savepath + savefilename[0] + 'XX', ins_autosXX.metric_ms)
            np.save(savepath + savefilename[1] + 'XX', ins_autosXX.metric_array)
            np.save(savepath + savefilename[2] + 'XX', ins_crossXX.metric_ms)
            np.save(savepath + savefilename[3] + 'XX', ins_crossXX.metric_array)
            
            np.save(savepath + savefilename[0] + 'YY', ins_autosYY.metric_ms)
            np.save(savepath + savefilename[1] + 'YY', ins_autosYY.metric_array)
            np.save(savepath + savefilename[2] + 'YY', ins_crossYY.metric_ms)
            np.save(savepath + savefilename[3] + 'YY', ins_crossYY.metric_array)
            
            print('obsID: %d done!'%obsID)

        elif niter == None:
            
            ss.read(filename = datapath + loadfilename, read_data=True, times=times, diff=True, polarizations=['xx', 'yy'])
            ins_autos = INS(ss, spectrum_type="auto", )
            ins_cross = INS(ss, spectrum_type='cross',)
            
            np.save(savepath + savefilename[0], ins_autos.metric_ms.data)
            np.save(savepath + savefilename[1], ins_autos.metric_array.data)
            np.save(savepath + savefilename[2], ins_cross.metric_ms.data)
            np.save(savepath + savefilename[3], ins_cross.metric_array.data)
            
            print('obsID: %d done!'%obsID)
    
    def plot_deviation_metric(self, obsID, bphase_XX=None, bphase_YY=None, dbphase_XX=None, dbphase_YY=None,\
                              savepath=None, savefilename=None, if_pdf=True, c='coolwarm', origin='lower', subtitle=None, extent=None):

        '''
        plot deviation metric

        params:
        bphase_XX, bphase_YY, dbphase_XX, dbphase_YY
        savepath: path to save the file, type: str
        savefilename: savefilename, type: str, default: 'obsID_bphase_RFI.pdf'
        if_pdf (bool): True if figure is in pdf. Defaults to True
        c (str): colormap style. Defaults to 'coolwarm'
        origin (str): origin of imshow image. Defaults to 'lower'.
        suptitle (str): figure suptitle
        extent (tuple): [xmin, xmax, ymin, ymax] figure imshow extents.
        
        '''
        fig = plt.figure(figsize=(30, 14))
        
        if subtitle==None:
    
            subtitle = r'$\rm %d$'%obsID
            
        plt.suptitle(subtitle, size=25)
        plt.subplot(221)
        plt.title(r'$\rm XX$', size=25)
        plt.ylabel(r'$\rm Timestamps~(\Delta T = 8.sec)$', size=26)
        plt.xlabel(r'$\rm Freq (MHz)$', size=26)
        plt.xticks(size=22)
        plt.yticks(size=22)
        plt.imshow(bphase_XX, aspect='auto', extent=extent, origin=origin, cmap=c, interpolation='none')
        cmap = plt.colorbar()
        cmap.set_label(label = r'$\Delta_B$', size=29)
        cmap.set_ticks(ticks=[-3.14,-2,-1,0,1,2,3.14], size=22)
        cmap.set_ticklabels(ticklabels=[r'$-3.14$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3.14$'], size=22)

        plt.subplot(223)
        plt.title(r'$\rm YY$', size=25)
        plt.ylabel(r'$\rm Timestamps~(\Delta T = 8.sec)$', size=26)
        plt.xlabel(r'$\rm Freq (MHz)$', size=26)
        plt.xticks(size=22)
        plt.yticks(size=22)
        plt.imshow(bphase_YY, aspect='auto', extent=extent, origin=origin, cmap=c, interpolation='none')
        cmap = plt.colorbar()
        cmap.set_label(label = r'$\Delta_B$', size=29)
        cmap.set_ticks(ticks=[-3.14,-2,-1,0,1,2,3.14], size=22)
        cmap.set_ticklabels(ticklabels=[r'$-3.14$',r'$-2$',r'$-1$',r'$0$',r'$1$',r'$2$',r'$3.14$'], size=22)

        plt.subplot(222)
        plt.title(r'$\rm XX$', size=25)
        plt.xlabel(r'$\rm Freq (MHz)$', size=26)
        plt.xticks(size=22)
        plt.yticks(size=22)
        plt.imshow(dbphase_XX, aspect='auto', extent = extent, origin=origin, cmap=c, interpolation='none')
        cmap = plt.colorbar()
        cmap.set_label(label = r'$\delta \Delta_B$', size=29)
        cmap.ax.tick_params(labelsize=22)

        plt.subplot(224)
        plt.title(r'$\rm YY$', size=25)
        plt.xlabel(r'$\rm Freq (MHz)$', size=26)
        plt.xticks(size=22)
        plt.yticks(size=22)
        plt.imshow(dbphase_YY, aspect='auto', extent=extent, origin=origin, cmap=c, interpolation='none')
        cmap = plt.colorbar()
        cmap.set_label(label = r'$\delta \Delta_B$', size=29)
        cmap.ax.tick_params(labelsize=22)
        fig.tight_layout()

        if savefilename == None:
            
            savefilename = '%d_bphase_RFI'%obsID

        if if_pdf == True:

            plt.savefig(savepath + savefilename + '.pdf', dpi=120, bbox_inches='tight')
        
        else:
            
            plt.savefig(savepath + savefilename + '.jpg', dpi=120, bbox_inches='tight')

        plt.show()

    def analyse_Z_score(self, z_score_path, obsIDs, z_thres, occupancy, bl_type, n_timestamps):
        """
        
            Accessing Z-score data

            Args:
                Z_score_path (str): path to z-score data files
                obsIDs (int): GPSTIME of the observation
                z_thres (float): z-score threshold
                occupancy (float): RFI occupancy
                n_timestamps (int): number of timestamps used in the z-score 
                bl_type (str): baseline type, available: 'autos', 'cross'

            Raises:
                IOError: file absent

            Returns:
                occXX (float): XX occupancy percentage
                occYY (float): YY occupancy percentage
                uniq_sumXXYY (array): unique timestamps where z-score is good
                use_XX (bool): True means good data
                use_YY (bool): True means good data

        """
        n_channels=768
        use_XX = False
        use_YY = False
        try:
            if bl_type == 'autos':
                z_score_filename = 'zscore_autos_%d.npy'%obsIDs
                z_score_arr = np.load(z_score_path + z_score_filename) # index 0 -XX index 1 -YY pol

            elif bl_type == 'cross':
                z_score_filename = 'zscore_cross_%d.npy'%obsIDs
                z_score_arr = np.load(z_score_path + z_score_filename)

        except IOError:
            raise IOError('file absent')
            print(obsID_index,'error')

        mask = np.abs(z_score_arr) > z_thres
        occp = np.nansum(mask, axis=1) / n_channels

        ####### XX ##############
        bad_XX = np.where(occp[:,0] > occupancy)[0]
        addXX = bad_XX + 1
        subXX = bad_XX - 1   
        bad_XX = np.unique(np.concatenate([bad_XX, addXX, subXX]))
        bad_XX = bad_XX[np.where((bad_XX >=0) & (bad_XX <= occp[:,0].size))[0]]
        good_XX = np.unique(np.setdiff1d(np.arange(occp[:,0].size), bad_XX))
        
        ########## YY ###########
        bad_YY = np.where(occp[:,1] > occupancy)[0]
        addYY = bad_YY + 1
        subYY = bad_YY - 1   
        bad_YY = np.unique(np.concatenate([bad_YY, addYY, subYY]))
        bad_YY = bad_YY[np.where((bad_YY >=0) & (bad_YY <= occp[:,1].size))[0]]
        good_YY = np.unique(np.setdiff1d(np.arange(occp[:,1].size), bad_YY))

        if n_timestamps == 'None':
            n_timestamps = int(len(z_score_arr)/2)

        return occp, good_XX, good_YY, use_XX, use_YY

    def get_good_timestamps(self, z_score_path, obsIDs, z_thres, occupancy, bl_type, n_timestamps, use_common=False):
        
        """
        Get Good Timestamps

            Args:
                Z_score_path (str): path to z-score data files
                obsIDs (int): GPSTIME of the observation
                z_thres (float): z-score threshold
                occupancy (float): RFI occupancy
                n_timestamps (int): number of timestamps used in the z-score 
                bl_type (str): baseline type, available: 'autos', 'cross'

            Returns:
                occXX (float): XX occupancy percentage
                occYY (float): YY occupancy percentage
                uniq_sumXXYY (array): unique timestamps where z-score is good
                use_XX (bool): True means good data
                use_YY (bool): True means good data
        """

        good_XX_arr = []
        good_YY_arr = []
        occp_arr=[]

        for obsID_index in range(len(obsIDs)):

            occp, good_XX, good_YY, use_XX, use_YY = self.analyse_Z_score(z_score_path=z_score_path,
                                                                     obsIDs=obsIDs[obsID_index],
                                                                     z_thres=z_thres,
                                                                     occupancy=occupancy,
                                                                     bl_type=bl_type,
                                                                     n_timestamps=n_timestamps)

            uniq_XX_index = np.unique(np.concatenate([good_XX, good_XX + 1]))
            uniq_YY_index = np.unique(np.concatenate([good_YY, good_YY + 1]))

            if use_common == True:

                if use_XX == use_YY == True:

                    good_XX_arr.append([obsID_index, uniq_XX_index, good_XX])
                    good_YY_arr.append([obsID_index, uniq_YY_index, good_YY])
                    occp_arr.append(occp)

                else:
                    pass

            else:
                good_XX_arr.append([obsID_index, uniq_XX_index, good_XX])
                good_YY_arr.append([obsID_index, uniq_YY_index, good_YY])
                occp_arr.append(occp)

        good_XX_arr = np.array(good_XX_arr, dtype=object)
        good_YY_arr = np.array(good_YY_arr, dtype=object)
        occp_arr = np.array(occp_arr, dtype=object)

        return good_XX_arr, good_YY_arr, occp_arr
        
    def get_good_data(self, path, obsIDs, z_thres, occupancy, n_timestamps, bl_type, use_common):
        
        """
        
        Get good data after RFI removal
        
        Args:
            path (str): path to z-score data files
            obsIDs (ndarray): GPSTIME array 
            z_thres (float): z-score threshold
            occupancy (float): RFI occupancy
            n_timestamps (int): number of timestamps used in the z-score 
            bl_type (str): baseline type, available: 'autos', 'cross'
            use_common (bool): if True then uses the obsIDs only if both XX and YY are good.

        Returns:
            uniq_index_arr (ndarray): good data timestamps
            use_XX_arr (ndarray): good data indicies array for XX polarizations.
            use_YY_arr (ndarray): good data indicies array for YY polarizations.
            occpXX_arr (float, ndarray): occupancy percentage array for XX polarizations.
            occpYY_arr (float, ndarray): occupancy percentage array for YY polarizations.
            
        """
        
        uniq_XX_index_arr = []
        uniq_YY_index_arr = []
        use_XX_arr = []
        use_YY_arr = []
        occpXX_arr = []
        occpYY_arr = []
        mask_arr = []
        for obsID_index in range(len(obsIDs)):
            try:
                if __name__ == '__main__':
                    occpXX, occpYY, uniq_XX, uniq_YY, use_XX, use_YY, mask_XX = self.analyse_Z_score(path, obsID=obsIDs[obsID_index],\
                                                                            z_thres=z_thres,\
                                                                            occupancy=occupancy,\
                                                                            bl_type=bl_type,\
                                                                            n_timestamps=n_timestamps)
                else:
                    occpXX, occpYY, uniq_XX, uniq_YY, use_XX, use_YY, mask_YY = RFI_class().analyse_Z_score(path, obsID=obsIDs[obsID_index],\
                                                                            z_thres=z_thres,\
                                                                            occupancy=occupancy,\
                                                                            bl_type=bl_type,\
                                                                            n_timestamps=n_timestamps)

                uniq_XX_index = np.unique(np.concatenate([uniq_XX, uniq_XX + 1]))
                uniq_YY_index = np.unique(np.concatenate([uniq_YY, uniq_YY + 1]))
                
                if use_common == True:

                    if use_XX == use_YY == True:
                        use_XX_arr.append([obsID_index, use_XX])
                        use_YY_arr.append([obsID_index, use_YY])
                        occpXX_arr.append(occpXX)
                        occpYY_arr.append(occpYY)
                        uniq_XX_index_arr.append(uniq_XX_index)
                        uniq_YY_index_arr.append(uniq_YY_index)
                        mask_arr.append([mask_XX, mask_YY])
                    else:
                        continue

                else:
                    use_XX_arr.append([obsID_index, use_XX])
                    use_YY_arr.append([obsID_index, use_YY])
                    occpXX_arr.append(occpXX)
                    occpYY_arr.append(occpYY)
                    uniq_XX_index_arr.append(uniq_XX_index)
                    uniq_YY_index_arr.append(uniq_YY_index)
                    mask_arr.append([mask_XX, mask_YY])
            except:
                pass
        
        uniq_XX_index_arr = np.array(uniq_XX_index_arr , dtype=object)#, dtype=(np.record, [('good_XX_pol_timestamps(int)', 'object')]))
        uniq_YY_index_arr = np.array(uniq_YY_index_arr , dtype=object)#, dtype=(np.record, [('good_YY_pol_timestamps(int)', 'object')]))
        use_XX_arr = np.array(use_XX_arr , dtype=object)
        use_YY_arr = np.array(use_YY_arr, dtype=object)
        occpXX_arr = np.array(occpXX_arr,  dtype=object)#dtype=(np.record, [('RFI_occupencyYY(percentage)', object)]))
        mask_arr = np.array(mask_arr, dtype=object)
        return uniq_XX_index_arr, uniq_YY_index_arr, use_XX_arr, use_YY_arr, occpXX_arr, occpYY_arr, mask_arr
    
    def get_masked_arrays(self, z_score_path, npy_path, obsIDs, baseline_length, pol, occupancy=0.05, z_thres=2.0,\
                     bl_type='cross', n_timestamps=12, field=None, savepath=None):
        """_summary_

        Args:
            z_score_path (str): zscore data file path
            npy_path (str): visibility data path 
            obsIDs (ndarray): obsIDs
            baseline_length (int): baseline length
            pol (str): polarization, either 'xx' or 'yy'
            occupancy (float, optional): RFI occupancy level. Defaults to 0.05.
            z_thres (float, optional): z score threshold. Defaults to 2.0.
            n_timestamps (int, optional): number of timestamps used in the zscore evaluation. Defaults to 12.
            field (str, optional): EoR field of interest, choose either EoR0 or EoR1. Defaults to None.
            savepath (str, optional): path to save the full masked array, masks=True where the RFI is greater than given threshold and occupancy level. Defaults to None.
        """
        good_XX_arr, good_YY_arr, occp_arr = RFI_class.get_good_timestamps(z_score_path=z_score_path,\
                                                                    obsIDs=obsIDs,\
                                                                    z_thres=z_thres,\
                                                                    occupancy=occupancy,\
                                                                    bl_type=bl_type,\
                                                                    n_timestamps=n_timestamps)
        if pol=='xx':
            good_arr = good_XX_arr
            
        elif pol=='yy':
            good_arr = good_YY_arr
            
        else:
            print('choose either xx or yy')
            
        
        vis_test = np.load(npy_path+'vis_%d_%s_%d.npy'%(baseline_length, pol, obsIDs[0]))
        data_keys= vis_test.dtype.names
        
        vis_masked_full = np.ma.empty(shape=(obsIDs.shape+vis_test.shape),\
                        dtype=[(data_keys[key_index], complex) for key_index in range(len(data_keys))],\
                                    fill_value=(np.nan+1j*np.nan))
        print(vis_test.shape, vis_masked_full.shape)
        
        bad_obsIDs=[]
        for obsID_index in range(len(obsIDs)):
            try:
                vis = np.load(npy_path+'vis_%d_%s_%d.npy'%(baseline_length, pol, obsIDs[obsID_index]))
                
                mask = np.ones(vis.shape, dtype=bool)
                mask[:, :, good_arr[obsID_index, 1], :] = False
                
                for model_keys in range(len(data_keys)):
                    vis_temp = np.ma.masked_array(vis[data_keys[model_keys]], mask=mask, fill_value=(np.nan+1j*np.nan))
                    vis_masked_full[data_keys[model_keys]][obsID_index] = vis_temp
                    print(data_keys[model_keys], obsID_index)
                
            except:
                print('error', obsID_index) # bad indicies(all timestamps are RFI affected) will be removed in later steps
                bad_obsIDs.append([obsIDs[obsID_index], obsID_index])

        bad_obsIDs= np.array(bad_obsIDs)
        vis_masked_full.dump(savepath+'vis_masked_full_%s_%s_%s_occp_%.2f_zscore_%s'%(field, baseline_length,\
                                                                                    pol, occupancy, z_thres),\
                            protocol=4)
        
        np.save(savepath+'bad_obsIDs_%s_%s_%s_occp_%.2f_zscore_%s'%(field, baseline_length,\
                                                                    pol, occupancy, z_thres), bad_obsIDs)
        
        print('saved vis_masked_full_%s_%s_%s_occp_%.2f_zscore_%s'%(field, baseline_length,\
                                                                    pol, occupancy, z_thres))
class Obs_Info(object):
    
    def __init__(self):
        pass
    def get_LSTs_n_obsIDs(self, mfitspath: str, obsIDs: list, field: str, if_only_zenith=True): 
        
        """_summary_
        Get the EoR field specific LSTs and corresponding obsIDs

        Args:
            mfitspath (str): path to the observation's metafits file
            obsIDs (list): GPSTIME of the observation, dtype: ndarray, list
            field (str): name of the EoR field, type=str, e.g. EoR0, EoR1, EoR2, all
            if field='all', returns LST0,1,2, obsIDs0,1,2 format
            if_only_zenith (bool, optional): True if only zenith pointed observations required. Defaults to True.

        Returns:
            ndarray: LSTs and corresponding obsIDs in array format
        """
    
        
        deg_to_hour = 24./360.
        
        LST_EoR0 = []
        LST_EoR1 = []
        LST_EoR2 = []
        
        obsIDs_EoR0 = []
        obsIDs_EoR1 = []
        obsIDs_EoR2 = []
        
        if if_only_zenith == True:
            
            grid_N=[]
            
            for i in range(len(obsIDs)):
                mfits = fits.open(mfitspath+ '%d.metafits'%obsIDs[i])[0]
                grid_N.append(mfits.header['GRIDNUM'])
            grid_N = np.array(grid_N)
            zenith_indx = np.where(grid_N==0)[0]
            obsIDs = obsIDs[zenith_indx]
        
        
        for i in range(len(obsIDs)):
            
            mfits = fits.open(mfitspath+ '%d.metafits'%obsIDs[i])
            
            RA = mfits[0].header['RA']
                
            if (RA*deg_to_hour < 2. or RA*deg_to_hour > 22.): ## EoR0

                LST_EoR0.append(mfits[0].header['LST'])
                obsIDs_EoR0.append(obsIDs[i])
                
            elif (RA*deg_to_hour < 6. and RA*deg_to_hour > 2.):  ## EoR1
                
                LST_EoR1.append(mfits[0].header['LST'])
                obsIDs_EoR1.append(obsIDs[i])
                
            elif (RA*deg_to_hour < 12. and RA*deg_to_hour > 8.):  ## EoR2
                
                LST_EoR2.append(mfits[0].header['LST'])
                obsIDs_EoR2.append(obsIDs[i])

        LST_EoR0= np.array(LST_EoR0)
        LST_EoR1= np.array(LST_EoR1)
        LST_EoR2= np.array(LST_EoR2)
        
        obsIDs_EoR0 = np.array(obsIDs_EoR0)
        obsIDs_EoR1 = np.array(obsIDs_EoR1)
        obsIDs_EoR2 = np.array(obsIDs_EoR2)
        
        if field == 'EoR0':
            
            return LST_EoR0, obsIDs_EoR0
        
        elif field == 'EoR1':
            
            return LST_EoR1, obsIDs_EoR1
        
        elif field == 'EoR2':
            
            return LST_EoR2, obsIDs_EoR2
        
        elif field == 'all':
            
            return LST_EoR0, LST_EoR1, LST_EoR2, obsIDs_EoR0, obsIDs_EoR1, obsIDs_EoR2

    def get_same_LSTs(self, LSTs=None, obsIDs=None, tolerence=1):
        
        '''
        
        Given the LSTs of the observations, select the same LSTs given the tolerence
        
        params:
        
        LSTs: Local Sidereal Time of the observaiton in degrees, dtype: array
        obsIDs: gpstime of observations, dtype: array(int)
        tolerence: time tolerence between the LSTs in mintues; default: 1 min
        
        
        return: identical LST counts, mean LST [hour] and corresponding obsIDs
        
        '''
        
        LSTs_argsort = np.argsort(LSTs)
        LSTs_minutes_sort = LSTs[LSTs_argsort] * 4.0 ## 4 min. in one degree
        obsIDs_sort = obsIDs[LSTs_argsort]
        
        true_matrix = []
        avg_indices = [0]
        
        for i in range(1, len(LSTs_minutes_sort)):
            
            a = LSTs_minutes_sort[i-1]
            b = LSTs_minutes_sort[i]
            true_matrix.append(np.isclose(a, b, atol=tolerence))
            
            if np.isclose(a, b, atol=tolerence) == False:
                avg_indices.append(i)
                
        
        true_matrix = np.array(true_matrix)
        avg_indices = np.array(avg_indices)
        
        LST_obsIDs_metric = []
        
        for i in range(1, len(avg_indices)):
            
            LST_obsIDs_metric.append([avg_indices[i]-avg_indices[i-1],\
                                    obsIDs_sort[avg_indices[i-1]: avg_indices[i]],\
                                    np.mean(LSTs_minutes_sort[avg_indices[i-1]: avg_indices[i]])/4.0])
        
        LST_obsIDs_metric = np.array(LST_obsIDs_metric, dtype=object)
        
        return LST_obsIDs_metric
class Spectrum(object):

    """
    Main class for delay analysis. 
    
    Available functions: bin_to_bin_split(), get_spectrum(), get_delay_powerspectrum(), 

    Returns:
        _type_: _description_
    """
    
    def __init__(self, freq, A_eff, B_eff):
        """
        Args:
            freq (ndarray): frequency array in Hz units
            A_eff(ndarray): effective collecting area of the telescope with frequency, in m^2 units.
            B_eff (float):  effective bandwidth for the spectral window function, in Hz units
        """
        
        self.freq       = freq*u.Hz
        self.A_eff      = A_eff*u.m**2
        self.B_eff      = B_eff*u.Hz
        self.pol        = ['xx', 'yy']
        self.BW         = abs(self.freq[-1] - self.freq[0])
        self.freq_res   = abs(self.freq[1] - self.freq[0])
        self.delays     = np.fft.fftshift(np.fft.fftfreq(len(self.freq), self.freq_res))
        window          = signal.blackmanharris(self.freq.size, True)
        window_sq       = signal.convolve(window, window, mode='full', method='fft')[::2]
        area_windows_sq = np.trapz(window_sq, x= self.freq, dx=self.freq_res)
        norm_factor     = area_windows_sq/self.B_eff
        window_sq       = window_sq/norm_factor
        windowft        = np.abs(np.fft.fftshift(np.fft.fft(window)))
        window_sqft     = np.abs(np.fft.fftshift(np.fft.fft(window_sq)))
        self.window     = window
        self.window_sq  = window_sq
        self.windowft   = windowft
        self.window_sqft= window_sqft
        
    def bin_to_bin_split(self, data, LST_bins, sections, mean=True):
        
        """
        Splitting the data into two Julian Date (bins) axis, this is to estimate the delay spectra in individual JD bins first,
        then estimating the cross power between the JD bins, since they share the same LST. 
        Please note, the input data is masked, and might have different number of observations, so the data is split happens
        on the masked data.

        Takes the data (closure phase, spectrum etc.) and perform at LST x JD split in the data.
        Individual LST bins are then summed together in the mean, and the corresponding split size (weights) are returned.

        Args:
            data (ndarray): data to be splitted
            LST_bins (ndarray):LST bin indicies array.
            sections (int): number of split
            mean (bool): set to True if take mean otherwise it takes median, defaults to True

        Returns:
            data_split_mean_array, split_size_array : mean splitted data, and the corresponding split size (weights) are returned
            data_split_mean_array of shape (N_LSTs, N_JDs, N_Triads, N_channels)
            
        
        """
        
        data_split = np.zeros(shape = (LST_bins.shape[0], sections,)+ data.shape[1:], \
                          dtype = data.dtype)
        split_size= np.zeros(shape = (LST_bins.shape[0], sections,) + data.shape[1:-1], \
                            dtype=[(dtype, int) for dtype in data.dtype.names])
        print('splitting the data of shape: ', data.shape, 'to: ', data_split.shape)
        
        for pols in data.dtype.names:

            for LST_ind in range(data_split.shape[0]):

                for triad_ind in range(data_split.shape[2]):

                    good_data_ind = np.where(data[pols][LST_bins[LST_ind], triad_ind, 0].mask == False)[0]

                    if good_data_ind.size % sections == 0:

                        split_size[pols][LST_ind, :, triad_ind] = good_data_ind.size/sections
                        split_ = np.split(good_data_ind, indices_or_sections=sections)
                        
                        if mean==True:
                            data_split[pols] [LST_ind, :, triad_ind] = np.nanmean(data[pols][LST_bins[LST_ind],\
                                                                            triad_ind][split_], axis=1)
                        else:
                            data_split[pols] [LST_ind, :, triad_ind] = np.nanmedian(data[pols][LST_bins[LST_ind],\
                                                                            triad_ind][split_], axis=1)
                    else:

                        split_size[pols][LST_ind, :, triad_ind] = good_data_ind.size/sections
                        split_ = np.split(good_data_ind[:int(good_data_ind.size/sections) * sections],\
                                        indices_or_sections=sections)
                        if mean==True:
                            data_split[pols] [LST_ind, :, triad_ind] = np.nanmean(data[pols][LST_bins[LST_ind],\
                                                                            triad_ind][split_], axis=1)
                        else:
                            data_split[pols] [LST_ind, :, triad_ind] = np.nanmedian(data[pols][LST_bins[LST_ind],\
                                                                            triad_ind][split_], axis=1)
        return data_split, split_size

    def get_spectra(self, veff, bphase, use_window_sq=True):
        
        """
        Function accepts ndimensional numpy array to directly get the spectrum

        Args:
            veff (ndarray): effective foreground visibilities, in Jy units
            bphase (ndarray): closure phase, in radian units
            use_window_sq (bool): True if using squared window function. Defaults to True.

        Returns:
            ndarray: closure phase spectrum in Jy units
        """
        
        psi = np.zeros(shape=bphase.shape, dtype=[(ind, complex) for ind in veff.dtype.names])
        
        if use_window_sq==True:
            window = self.window_sq
            print('using window sq')
            
        else:
            window = self.window
            print('using window')
            
        for dtype in veff.dtype.names:
            window = np.expand_dims(window, axis=list(np.arange(bphase.ndim-1)))
            psi[dtype] = veff[dtype]*np.exp(1j*bphase[dtype])*window*self.freq_res
            #delay_spectrum_arr[ind] = np.tensordot(psi[ind], ds_factor, axes=([2,1]))
        return psi
    
    def get_delay_powerspectra(self, delay_spectrum_0=None, delay_spectrum_1=None):
        
        """
        Get delay powerspectrum

        Args:
            delay_spectrum_0 (ndarray, optional): Delay spectrum 1
            delay_spectrum_1 (ndarray, optional): Delay spectrum 2

        Returns:
            ndarray: delay powerspectrum in psudo mk^2Mpc^3/ h^3 units.  
        """

        delay_spectrum_0 = (delay_spectrum_0)*u.Jy*u.Hz
        delay_spectrum_1 = (delay_spectrum_1)*u.Jy*u.Hz
        
        c = constants.c
        kB = constants.k_B
        freq_mid = self.freq[int(len(self.freq)/2)]
        lambda_mid = (c/freq_mid)
        
        z0 = Cosmology.Z(self, f_obs=self.freq[0])
        z_mid = Cosmology.Z(self, f_obs=freq_mid)
        z1 = Cosmology.Z(self, f_obs=self.freq[-1])
        
        Relative_comoving_distance = cosmology.Planck18.comoving_distance(z0)-\
                                        cosmology.Planck18.comoving_distance(z1)
        Mean_comoving_distance = cosmology.Planck18.comoving_distance(z_mid)

        C_terms = (self.A_eff / (lambda_mid**2 * self.BW)) * \
            (((Mean_comoving_distance **2) * Relative_comoving_distance) / self.BW) * \
            ((lambda_mid**2) / (2 * kB))**2
        power = (2/3) * np.conjugate(delay_spectrum_0)*delay_spectrum_1 * C_terms
        power = power.si.to(u.Mpc**3*u.milliKelvin**2)*(cosmology.Planck18.h**3)
        return power # in units of millkelvin**2 Mpc**3/h**3
    
    def get_delay_powerspectrum(self, data, split_size):
        
        """
        Get the full delay cross powerspectrum of the bin-averaged datasets

        Args:
            data (ndarray): bin splitted data

        Returns:
            cross_power: bin averaged cross power
        """
        
        JD_bins      = np.array(list(combinations(range(data.shape[1]), 2))) # along the JD axis
        triad_comb   = np.array(list(combinations(range(data.shape[2]), 2))) # along the Triad axis

        if JD_bins.shape[0] != 1:
            power = np.zeros(shape=(data.shape[0], JD_bins.shape[0], triad_comb.shape[0], data.shape[-1]), dtype=data.dtype)
            triad_wts = np.zeros(shape=(data.shape[0], JD_bins.shape[0], triad_comb.shape[0], 1), dtype=[(ind, float) for ind in data.dtype.names]) 
            
        else:
            power = np.zeros(shape=(data.shape[0], triad_comb.shape[0], data.shape[-1]), dtype=data.dtype)
            triad_wts = np.zeros(shape=(data.shape[0], triad_comb.shape[0], 1), dtype=[(ind, float) for ind in data.dtype.names])
            
        for dtype in data.dtype.names:
            for LST_ind in range(data.shape[0]):
                for JD_ind in range(JD_bins.shape[0]):
                    for triad_ind in range(triad_comb.shape[0]):
                        if JD_bins.shape[0] != 1:
                            power[LST_ind, JD_ind, triad_ind] = self.get_delay_powerspectra(
                                                                        delay_spectrum_0=data[dtype][LST_ind, JD_bins[JD_ind,0],\
                                                                            triad_comb[triad_ind, 0]],\
                                                                        delay_spectrum_1=data[dtype][LST_ind, JD_bins[JD_ind,1],\
                                                                            triad_comb[triad_ind, 1]],)

                            triad_wts[dtype][LST_ind, JD_ind, triad_ind] = (split_size[dtype][LST_ind, JD_bins[JD_ind,0], triad_comb[triad_ind, 0]] \
                                                                    + split_size[dtype][LST_ind, JD_bins[JD_ind,1], triad_comb[triad_ind, 1]])/2
                        else:
                            
                            power[LST_ind, triad_ind] = self.get_delay_powerspectra(
                                                                        delay_spectrum_0=data[dtype][LST_ind, JD_bins[JD_ind,0],\
                                                                            triad_comb[triad_ind, 0]],\
                                                                        delay_spectrum_1=data[dtype][LST_ind, JD_bins[JD_ind,1],\
                                                                            triad_comb[triad_ind, 1]],)

                            triad_wts[dtype][LST_ind, triad_ind] = (split_size[dtype][LST_ind, JD_bins[JD_ind,0], triad_comb[triad_ind, 0]] \
                                                                    + split_size[dtype][LST_ind, JD_bins[JD_ind,1], triad_comb[triad_ind, 1]])/2
        if JD_bins.shape[0] != 1:
            mean_power = np.zeros(shape=power.shape[0:2]+power.shape[-1:], dtype=power.dtype)
            wts = np.zeros(shape=power.shape[0:2]+(1,), dtype=triad_wts.dtype)
            for dtype in mean_power.dtype.names:
                wts[dtype] = np.nansum(triad_wts[dtype], axis=2)
                mean_power[dtype] = np.sum(power[dtype]*triad_wts[dtype], axis=2)/np.sum(triad_wts[dtype], axis=2)
        else:
            mean_power = np.zeros(shape=power.shape[0:1]+power.shape[-1:], dtype=power.dtype)
            wts = np.zeros(shape=power.shape[0:1]+(1,), dtype=triad_wts.dtype)
            for dtype in mean_power.dtype.names:
                wts[dtype] = np.nansum(triad_wts[dtype], axis=1)
                mean_power[dtype] = np.sum(power[dtype]*triad_wts[dtype], axis=1)/np.sum(triad_wts[dtype], axis=1)
                
        print('processed x power')
        return mean_power, wts
    
    def get_error(self, data, wts, real=True, norm_factor=2):
        
        """
        Get normalised uncertainity in the data/cross-power bin to bin
        

        Args:
            data (ndarray): dataset for the error estimation.
            wts (ndarray): weights array.
            norm_factor (int, optional): normalisation factor, based on the N_JD split this factor will change.
            Defaults to 2 for N_JD = 4.

        Returns:
            wtd_sigma, error_wts: weighted std, weights
        """
        norm_var_LST = np.zeros(shape=data.shape[0:1]+data.shape[-1:], dtype=data.dtype)
        wtd_var      = np.zeros(shape=data.shape[-1], dtype=data.dtype)

        for dtype in wtd_var.dtype.names:
            if real == True:
                data[dtype] = abs(data[dtype])
            else:
                data[dtype] = data[dtype]
            #### taking varaince
            n_eff = np.nansum(wts[dtype], axis=1)**2 / np.nansum( wts[dtype]**2, axis=1)
            
            wtd_mean  = np.nansum(data[dtype] * wts[dtype], axis=1)/\
                        np.nansum(wts[dtype], axis=1)
            wtd_mean  = np.expand_dims(wtd_mean, axis=1)
            variance = np.nansum((data[dtype]- wtd_mean)**2 * wts[dtype], axis=1)/\
                            (np.nansum(wts[dtype], axis=1) * ((n_eff-1) / n_eff))
            variance = variance / norm_factor

            wts_ = np.nansum(wts[dtype], axis=1)
            n_eff = np.nansum(wts_)**2 / np.nansum(wts_**2)
            norm_var_LST[dtype] = np.nansum(variance * wts_, axis=0)/\
                            (np.nansum(wts_) * ((n_eff-1) / n_eff))
        return norm_var_LST, wtd_var
    
    def SEM(self, weights, data, axis):
    
        n = weights.size
        
        weights_mean = np.nanmean(weights, axis=axis)
        
        data_wtd_mean    = np.nansum(data * weights, axis=axis)/ np.nansum(weights, axis=axis)
        
        factor = n/((n-1) * np.nansum(weights, axis=axis)**2)
        
        F1 = np.nansum((weights * data - weights_mean * data_wtd_mean)**2, axis=axis)

        F2 = -2 * data_wtd_mean * np.nansum((weights - weights_mean) * (weights * data - weights_mean*data_wtd_mean), axis=axis)

        F3 = (data_wtd_mean ** 2) * np.nansum((weights - weights_mean)** 2, axis=axis)
        
        SEM_sq = factor * (F1 + F2 + F3)
    
        return SEM_sq
    
    def get_inv_var_wtd_power(self, power=None, var=None):
        """
        The powerspectrum is estimated at each LST bins separately, since the sky changes with LST
        we are required to normalise the power between the different LST bins, therefore, a inverse variance 
        weighting is required to correctly estimate the final power.
        
        Args:
            power (ndarray): LST binned powerspectrum array of shape (N_LST, N_channels)
            var (ndarray): LST binned variance array of shape (N_LST, N_channels)

        Returns:
            wtd_power: Inverse variance weighted power
            
        """
        
        power_wtd = np.empty(power.shape[-1], dtype=power.dtype)

        for dtype in power.dtype.names:
            
            power_wtd[dtype] = np.nansum(power[dtype] * (1/var[dtype]), axis=0)/np.nansum(1/var[dtype], axis=0)
        print('estimated inverse variance weighted x power')
        return power_wtd
    
    def comb_pol(self, data=None, var=None, wts=None):
        
        """
        combining polarisations

        Args:
            data (ndarray, optional): powerspectrum array
            var (ndarray, optional):  variance array
            wts (ndarray, optional):  weights array

        Returns:
            wtd_data, wtd_var: weighted average
        """
        N = np.array(data.dtype.names)
        data_ = np.zeros(shape=(N.size, data.shape[0]), dtype=complex)
        var_  = np.zeros(shape=(N.size, data.shape[0]), dtype=complex)
        wts_  = np.zeros(shape=(N.size, 1),  dtype=float)
        
        for i in range(N.size):
            data_[i]  = data[N[i]]
            var_ [i]  = var[N[i]]
            wts_ [i]  = np.nansum(wts[N[i]][:,0])
        
        n_eff = np.nansum(wts_)**2 / np.nansum( wts_**2 )
        wtd_data = np.nansum(data_ * wts_, axis=0)/  np.nansum(wts_, axis=0)
        wtd_var  = np.nansum(wts_  * var_, axis=0) / (np.nansum(wts_) * ((n_eff-1) / n_eff))
        
        print('combined XX, YY polarisation, estimated weighted power, variance')
        return wtd_data, wtd_var

class Data_Flagging(object):
    
    def __init__(self) -> None:
        
        pass 
    
    def triad_filter(self, data, pol, k=1.4862):

        """
        Removing bad triads based on Median Absolute Deviation
        

        Args:
            data (ndarray): data array, usually operated in the closure phases
            pol (str): polarisation
            threshold (float): threshold for discarding bad data, it is a multple of standard deviation on 
            the Median Absolute Deviation.

        Returns:
            MAD_array, MAD_mask: MAD mask for good data, full MAD array
        """
        
        data_med = np.nanmedian(data[pol], axis=1) # axis 1 is triad axis
        data_med = np.expand_dims(data_med, axis=1)
        MAD      = np.abs(data[pol] - data_med) # median absolute deviation

        #finding the mean, median of the MAD along the frequency axis
        MAD_mean   = np.nanmean(MAD, axis=-1) # -1 axis is frequency
        MAD_median = np.nanmedian(MAD_mean, axis=-1)
        MAD_median = np.expand_dims(MAD_median, axis=-1)
        
        MAD2 = abs(MAD_mean  - MAD_median) # estimating the MAD of MAD_mean
        sigma = MAD2*k
        mask  = (MAD_mean-sigma) < 0
        mask = np.reshape(np.repeat(mask, repeats=data.shape[-1], axis=1), data.shape)
        
        return MAD2, mask

class Get_Coherence_Time(object):
    
    def __init__(self, freq):
        """_summary_

        Args:
            freq (_type_): _description_
        
        create following instenses:
           
        N_chan (int): usual size of the data. Number of frequency channels in the dataset.
                e.g. N_channels=768 for data @ 40kHz, 384 for data @ 80kHz, 24 for data  @ 1.28MHz. Defaults to 768.
        freq_res (int): frequency resolution of the data. Defaults to 40kHz.    
        """
        self.pol = ['xx', 'yy']
        self.freq = freq*u.Hz
        self.freq_res = abs(self.freq[1] - self.freq[0])
        
        delays = np.fft.fftfreq(len(self.freq), self.freq_res)
        dtau = abs(delays[1] - delays[0])
        self.delays = delays
    
    def frac_loss(self, DS, step, triad=0):
        
        """_summary_

        Args:
            DS (ndarray): Delay Spectrum continuous along time axis (timeseries of delayspectrum)
            step (int): steps upto the data is coherently averaged; steps have a cadence of 8sec.
            triad (int): mention the triad number usually have 1 or 2 triads only for coherence time calculations, defaults to 0

        Returns:
            _type_: _description_
        """
    
        '''
        params:
        
        DS: Delay Spectrum
        step: Integration Time in terms of multiples of 8 sec.
        
        return:
        
        fractional loss in HI power, or coherence time
        
        '''
        delays = self.delays.to(u.microsecond)
        index = np.where(abs(delays) > 2*u.microsecond)[0]
        #DS = DS[:,triad,:,:].value
        DS_arr=[]
        for obs_indx in range(DS.shape[0]):
            for timestamps in range(DS.shape[1]):
                DS_arr.append(DS[obs_indx][timestamps])

        DS_arr = np.array(DS_arr)
        DS_arr = DS_arr[:, index]

        PS_sq_mean=[]
        PS_mean_sq=[]

        PS_DATA = np.abs(DS_arr)**2
        for i in range(DS_arr.shape[0]):
            A = np.nanmean(PS_DATA[i:(i+1+step)], axis=0)
            B = (np.abs(np.nanmean(DS_arr[i:(i+1+step)], axis=0))**2)
            PS_sq_mean.append(A)
            PS_mean_sq.append(B)

        PS_sq_mean = np.array(PS_sq_mean)
        PS_mean_sq = np.array(PS_mean_sq)
        eta = 1 - ((PS_sq_mean - PS_mean_sq)/PS_sq_mean)
        return 1 - np.nanmean(eta)

    def frac_loss_ARR_model(self, DS, triad, total_samples): 
        '''
        params:
        
        DS: delay spectrum array
        total_time: total integration time in seconds (will be in the multiples of 8)
        upto which the signal loss will be evaluated, e.g. 2,3,4 etc.
        '''
        
        intg_time = 8#sec
        time = np.arange(0, total_samples*intg_time, intg_time)

        frac_loss_arr = []
        for i in range(total_samples):
            frac_loss_arr.append(self.frac_loss(DS=DS, triad=triad, step=i))
        frac_loss_arr = np.array(frac_loss_arr)
        return time, frac_loss_arr       
   

#class Spectrum(object):
#
#    '''
#    Estimate the delayspectrum and delaypowerspectrum.
#    
#    available functions: get_delay_spectrum(), get_delay_powerspectrum()
#    
#    '''
#
#    def __init__(self, freq, B_eff):
#        """_summary_
#
#        Args:
#            freq (_type_): _description_
#        
#        create following instenses:
#           
#        N_chan (int): usual size of the data. Number of frequency channels in the dataset.
#                e.g. N_channels=768 for data @ 40kHz, 384 for data @ 80kHz, 24 for data  @ 1.28MHz. Defaults to 768.
#        freq_res (int): frequency resolution of the data. Defaults to 40kHz.    
#        """
#        self.pol = ['xx', 'yy']
#        self.freq = freq*u.Hz
#        self.freq_res = abs(self.freq[1] - self.freq[0])
#        
#        delays = np.fft.fftfreq(len(self.freq), self.freq_res)
#        dtau = abs(delays[1] - delays[0])
#        df   = abs(freq[1] - freq[0])
#        B_eff = B_eff*u.Hz
#        
#        window = signal.blackmanharris(self.delays.size, True)
#        window_sq = signal.convolve(window, window, mode='full', method='fft')[::2]
#        area_windows_sq = np.trapz(window_sq, x= freq, dx=df)
#        
#        norm_factor = area_windows_sq/B_eff
#        window_sq = window_sq/norm_factor
#        
#        windowft = np.abs(np.fft.fft(window))
#        window_sqft=np.abs(np.fft.fft(window_sq))
#        
#        self.delays = delays
#        self.window = window
#        self.window_sq = window_sq
#        self.windowft = windowft
#        self.window_sqft = window_sqft
#    
#    def get_delay_spectrum(self, V_eff=None, bphase=None, ds_factor=None, use_window_sq=True):
#        
#        """Get delay spectrum
#
#        Args:
#            V_eff (ndarray, optional): Effective Visibility array. Defaults to None.
#            bphase (ndarray, optional): bispectrum phase array. Defaults to None.
#            use_window_sq (bool, optional): if use window^2 function. Defaults to True.
#
#        Returns:
#            ndarray: delay spectrum for given bispectrum phase model and effective visibilities
#        """
#        if use_window_sq==True:
#            window=self.window_sq
#        else:
#            window=self.window
#        
#        D_p = np.expand_dims(V_eff, axis=-1)*np.exp(1j*bphase)*window
#        for i in range(len(self.delays)):
#            delay_spectrum_arr.append(self.freq_res*np.sum(D_p*ds_factor[i], axis=-1))
#            print('done, ch: %d'%i)
#        delay_spectrum_arr = np.ma.array(delay_spectrum_arr)
#        delay_spectrum_arr = np.moveaxis(delay_spectrum_arr, 0, -1)
#        return delay_spectrum_arr
#        
#    def get_delay_powerspectrum(self, delay_spectrum_0=None, delay_spectrum_1=None,\
#                                A_eff=None):
#        """Get delay powerspectrum
#
#        Args:
#            delay_spectrum_0 (ndarray, optional): Delay spectrum 1. Defaults to None.
#            delay_spectrum_1 (ndarray, optional): Delay spectrum 2. Defaults to None.
#
#        Returns:
#            ndarray: Delay powerspectrum of the bispectrum phase.
#        """
#        
#        f_em = 1420*10**6 * u.Hz
#        kB = constants.k_B # Boltzmann's constant
#        c = constants.c
#        A_eff = A_eff * u.m *u.m
#        wavelength = (c/f_em)
#        bandwidth = self.freq[-1] - self.freq[0]
#        freq_m = self.freq[int(len(self.freq)/2)]
#
#        z0 = Cosmology.Z(self, f_obs=self.freq[0])
#        z_mid = Cosmology.Z(self, f_obs=freq_m)
#        z1 = Cosmology.Z(self, f_obs=self.freq[-1])
#        
#        Relative_comoving_distance = Planck18.comoving_distance(z0)-\
#                                        Planck18.comoving_distance(z1)
#        Mean_comoving_distance = Planck18.comoving_distance(z_mid)
#
#        CC = (A_eff / (wavelength**2 * bandwidth)) * \
#        (((Mean_comoving_distance **2) * Relative_comoving_distance) / bandwidth) * \
#            ((wavelength**2) / (2 * kB))**2
#        power = np.conjugate(delay_spectrum_0)*delay_spectrum_1 * CC
#        power = power.si.to(u.Mpc*u.Mpc*u.Mpc*u.milliKelvin*u.milliKelvin*u.Hz*u.Hz)
#        return power
#    
#    def get_delay_spectrum(self, obsIDs=None, loadfilepath=None, loadfilename=None, baseline_length=None, N_vis=None, N_chan=None, N_triads=None, vis_shape=None,\
#                    freq_res=None, use_window_sq=False, savefilename=None, savefilepath=None, field=None):
#        
#        vis_data = np.load(loadfilepath+ loadfilename, allow_pickle=True)
#        vis_data = vis_data.swapaxes(3, 1)
#        vis_data = vis_data.reshape(-1, N_vis, N_triads, N_chan)
#
#        bphase_data = np.empty(shape=(2772, N_triads, N_chan), dtype=vis_data.dtype)
#        bphase_data['xx'] = np.angle(np.prod(vis_data['xx'], axis=1, ))
#        bphase_data['yy'] = np.angle(np.prod(vis_data['yy'], axis=1, ))
#        #bphase_data.dump(npy_path+'bphase_new_FGIN_%s_%s_%s_occp_%.2f_zscore_%s'%(field, baseline_length, pol, occupancy, z_thres), protocol=4)
#   def get_weighted_power(self, PS=None, weights=None):
#           """
#           weighted averaged power
#   
#           Args:
#               PS (ndarray, optional): powerspectrum array. Defaults to None.
#               weights (ndarray, optional): weights array. Defaults to None.
#   
#           Returns:
#               power: weighted averaged power
#           """
#           
#           weights = np.expand_dims(weights, axis=-1)
#           power = np.zeros(shape=(PS.shape[0], PS.shape[-1]), dtype=PS.dtype)
#   
#           for dtype in PS.dtype.names: # weighted avergaing ## for triads
#               
#               power[dtype] = np.nansum(PS[dtype] * weights[dtype], axis=1)/np.nansum(weights[dtype], axis=1)
#   
#           return power
#       
#       def get_incoherent_power(self, DS_binned=None, A_eff=None, weights=None):
#           
#           """ Incoherently averaging the power
#   
#           Args:
#               DS_binned (ndarray, optional): binned delayspectrum. Defaults to None.
#               A_eff (ndarray, optional): effective collecting area of the telescope.. Defaults to None.
#               weights (ndarray, optional): weights corresponding to the binned spectrums. Defaults to None.
#   
#           Returns:
#               Auto_PS, Cross_PS, weights_Auto, weights_Cross: powerspectrum auto, cross, and corresponding weights
#           """
#           
#           Cross_PS = np.zeros(shape=(int(DS_binned.shape[0]/2),\
#                                      int((DS_binned.shape[1]*(DS_binned.shape[1]-1))/2), \
#                                    DS_binned.shape[2]), dtype=DS_binned.dtype)
#           Auto_PS=np.zeros(shape=(int(DS_binned.shape[0]/2),)+ DS_binned.shape[1:], dtype=DS_binned.dtype)
#           
#           weights_Auto = np.zeros(shape=(int(weights.shape[0]/2),)+ weights.shape[1:], dtype=weights.dtype)
#           weights_Cross = np.zeros(shape=(int(DS_binned.shape[0]/2),\
#                                          int((DS_binned.shape[1]*(DS_binned.shape[1]-1))/2)),\
#                                   dtype=weights.dtype)
#           
#           for dtype in DS_binned.dtype.names:
#               for indx in range(0, DS_binned.shape[0], 2):
#                   int_auto = 0
#                   int_cross = 0
#                   for triad1 in range(DS_binned.shape[1]):
#                       for triad2 in range(DS_binned.shape[1]):
#                           if triad1==triad2:
#                               int_auto+=1
#                               Auto_PS[dtype][int(indx/2)][int(int_auto-1)] = self.get_delay_powerspectrum(delay_spectrum_0=DS_binned[dtype][indx][triad1],\
#                                                                           delay_spectrum_1=DS_binned[dtype][indx+1][triad2],\
#                                                                           A_eff=A_eff)
#                               weights_Auto[dtype][int(indx/2)][int(int_auto-1)] = (weights[dtype][indx][triad1] + weights[dtype][indx+1][triad2])/2
#                               
#                           else:
#                               if triad1>triad2:
#                                   int_cross+=1
#                                   Cross_PS[dtype][int(indx/2)][int(int_cross-1)] = self.get_delay_powerspectrum(delay_spectrum_0=DS_binned[dtype][indx][triad1],\
#                                                                           delay_spectrum_1=DS_binned[dtype][indx+1][triad2],\
#                                                                           A_eff=A_eff)
#                                   weights_Cross[dtype][int(indx/2)][int(int_cross-1)] = (weights[dtype][indx][triad1] + weights[dtype][indx+1][triad2])/2
#                               else:
#                                   pass
#                               
#           print('processed + and x power')
#           
#           return Auto_PS, Cross_PS, weights_Auto, weights_Cross
#   
#       def get_incoherent_power2(self, DS_binned=None, A_eff=None, weights=None):
#           """
#           After coherently averaging the closure phases, we estimate the delay spectrum of the closure phases.
#           In the second step we require to estimate the power from the delay spectrum.
#           The delay spectrum are LST binned, however they are produced from different number of observations, 
#           which requires weighted averaging.
#           The binned delay specturm data has a typical shape of (2*N_LST, N_triads, N_channels), where 2 belongs to Julian Date axis (JD)
#           
#           Please note that, we estimating the power between the delay spectrum two JD axis.
#   
#           Args:
#               DS_binned (ndarray): LST binned delay spectrum of shape (2*N_LST, N_triads, N_chans)
#               A_eff (ndarray): Effective collecting are of the telescope.
#               weights (_type_, optional): _description_. Defaults to None.
#   
#           Returns:
#               _type_: _description_
#           """
#           
#   
#           Cross_PS=np.zeros(shape=(int(DS_binned.shape[0]/2), int((DS_binned.shape[1]*(DS_binned.shape[1]-1))/2), \
#                                    DS_binned.shape[2]), dtype=DS_binned.dtype)
#           Auto_PS=np.zeros(shape=DS_binned.shape, dtype=DS_binned.dtype)
#           
#           weights_Auto = np.zeros(shape=weights.shape, dtype=weights.dtype)
#           weights_Cross = np.zeros(shape=(int(DS_binned.shape[0]/2),\
#                                          int((DS_binned.shape[1]*(DS_binned.shape[1]-1))/2)),\
#                                   dtype=weights.dtype)
#           
#           for dtype in DS_binned.dtype.names:
#               for indx in range(0, DS_binned.shape[0], 2):
#                   int_auto = 0
#                   int_cross = 0
#                   for triad1 in range(DS_binned.shape[1]):
#                       for triad2 in range(DS_binned.shape[1]):
#                           if triad1==triad2:
#                               int_auto+=1
#                               #Auto_PS[dtype][indx][int(int_auto-1)] = self.get_delay_powerspectrum(delay_spectrum_0=DS_binned[dtype][indx][triad1],\
#                               #                                            delay_spectrum_1=DS_binned[dtype][indx][triad2],\
#                               #                                            A_eff=A_eff)
#                               #weights_Auto[dtype][indx][int(int_auto-1)] = weights[dtype][indx][triad1] + weights[dtype][indx][triad2]
#                               
#                           else:
#                               if triad1>triad2:
#                                   int_cross+=1
#                                   Cross_PS[dtype][int(indx/2)][int(int_cross-1)] = self.get_delay_powerspectrum(delay_spectrum_0=DS_binned[dtype][indx][triad1],\
#                                                                           delay_spectrum_1=DS_binned[dtype][indx+1][triad2],\
#                                                                           A_eff=A_eff)
#                                   weights_Cross[dtype][int(indx/2)][int(int_cross-1)] = (weights[dtype][indx][triad1] + weights[dtype][indx+1][triad2])/2
#                               else:
#                                   pass
#                               
#           print('processed + and x power')
#           
#           return Auto_PS, Cross_PS, weights_Auto, weights_Cross

#class Spectrum2(object):
#
#    '''
#    Estimate the delayspectrum and delaypowerspectrum.
#    
#    available functions: get_delay_spectrum(), get_delay_powerspectrum()
#    
#    '''
#
#    def __init__(self, freq, B_eff):
#        """_summary_
#
#        Args:
#            freq (_type_): _description_
#        
#        create following instenses:
#           
#        N_chan (int): usual size of the data. Number of frequency channels in the dataset.
#                e.g. N_channels=768 for data @ 40kHz, 384 for data @ 80kHz, 24 for data  @ 1.28MHz. Defaults to 768.
#        freq_res (int): frequency resolution of the data. Defaults to 40kHz.    
#        """
#        self.pol = ['xx', 'yy']
#        self.freq = freq*u.Hz
#        self.freq_res = abs(self.freq[1] - self.freq[0])
#        
#        delays = np.fft.fftfreq(len(self.freq), self.freq_res)
#        dtau = abs(delays[1] - delays[0])
#        df   = abs(freq[1] - freq[0])
#        B_eff = B_eff*u.Hz
#        
#        window = signal.blackmanharris(delays.size, True)
#        window_sq = signal.convolve(window, window, mode='full', method='fft')[::2]
#        area_windows_sq = np.trapz(window_sq, x= freq, dx=df)
#        
#        norm_factor = area_windows_sq/B_eff
#        window_sq = window_sq/norm_factor
#        
#        windowft = np.abs(np.fft.fft(window))
#        window_sqft=np.abs(np.fft.fft(window_sq))
#        
#        self.delays = delays
#        self.window = window
#        self.window_sq = window_sq
#        self.windowft = windowft
#        self.window_sqft = window_sqft
#        
#        
#    def get_delay_spectrum(self, V_eff=None, bphase=None, ds_factor=None, use_window_sq=True):
#        
#        """Get delay spectrum
#
#        Args:
#            V_eff (ndarray, optional): Effective Visibility array. Defaults to None.
#            bphase (ndarray, optional): bispectrum phase array. Defaults to None.
#            use_window_sq (bool, optional): if use window^2 function. Defaults to True.
#
#        Returns:
#            ndarray: delay spectrum for given bispectrum phase model and effective visibilities
#        """
#        if use_window_sq==True:
#            window=self.window_sq
#        else:
#            window=self.window
#        #print(V_eff.shape)
#        delay_spectrum_arr=[]
#        D_p = np.expand_dims(V_eff, axis=-1)*np.exp(1j*bphase)*window
#        #print(D_p.shape, ds_factor.shape)
#        mem = np.arange(768)
#        for i in range(mem.shape[0]):
#            delay_spectrum_arr.append(self.freq_res*np.sum(D_p*ds_factor[i], axis=-1))
#            #print('done, ch: %d'%i)
#        delay_spectrum_arr = np.ma.array(delay_spectrum_arr)
#        delay_spectrum_arr = np.moveaxis(delay_spectrum_arr, 0, -1)
#        return delay_spectrum_arr
#        
#    def get_delay_powerspectrum(self, delay_spectrum_0=None, delay_spectrum_1=None,\
#                                A_eff=None):
#        """Get delay powerspectrum
#
#        Args:
#            delay_spectrum_0 (ndarray, optional): Delay spectrum 1. Defaults to None.
#            delay_spectrum_1 (ndarray, optional): Delay spectrum 2. Defaults to None.
#
#        Returns:
#            ndarray: Delay powerspectrum of the bispectrum phase.
#        """
#        
#        f_em = 1420*10**6 * u.Hz
#        kB = constants.k_B # Boltzmann's constant
#        c = constants.c
#        A_eff = A_eff *u.m *u.m
#        wavelength = (c/f_em)
#        bandwidth = self.freq[-1] - self.freq[0]
#        freq_m = self.freq[int(len(self.freq)/2)]
#
#        z0 = Cosmology.Z(self, f_obs=self.freq[0])
#        z_mid = Cosmology.Z(self, f_obs=freq_m)
#        z1 = Cosmology.Z(self, f_obs=self.freq[-1])
#        
#        Relative_comoving_distance = Planck18.comoving_distance(z0)-\
#                                        Planck18.comoving_distance(z1)
#        Mean_comoving_distance = Planck18.comoving_distance(z_mid)
#
#        CC = (A_eff / (wavelength**2 * bandwidth)) * \
#        (((Mean_comoving_distance **2) * Relative_comoving_distance) / bandwidth) * \
#            ((wavelength**2) / (2 * kB))**2
#        power = np.conjugate(delay_spectrum_0)*delay_spectrum_1 * CC
#        power = power.si.to(u.Mpc*u.Mpc*u.Mpc*u.milliKelvin*u.milliKelvin*u.Hz*u.Hz)
#        return power
#    
#    def create_autocorr_data(self, loadfilepath=None, loadfilename=None,\
#            obsIDs=None, baseline_length=None, pol=None, model='FG+HI:Jy', ds_factor=None,\
#                             timestamps=None, A_eff=None, use_window_sq=True):
#            """
#            Predefined filename: 'FG_HI_Coh_%baseline_length_%pol_%obsID.npy'
#            Predefined datatypes
#            'vis_model:Jy', 'vis_HI:Jy', 'vis_Noise:Jy'
#            Args:
#                loadfilepath (str): data path
#                loadfilename (str): filename
#                obsIDs (ndarray): obsIDs
#                baseline_length (int): baseline length
#                pol (str): polarization, choose either 'xx' or 'yy            
#                timestamps (int): number of timestamps
#                A_eff (ndarray): effective collecting area of the telescope.
#                use_window_sq (bool, optional): use window square function in delayspectrum estimation. Defaults to True.
#
#            Returns:
#                _type_: _description_
#            """
#            ## loading the continuous datasets
#            Bphase_FGHI = []
#            V_eff_arr=[]
#            for indx in range(len(obsIDs)):
#                if loadfilename==None:
#                    loadfilename='FG_HI_Coh_%s_%s_%d.npy'%(baseline_length, pol, obsIDs[indx])
#
#                vis_coh = np.load(loadfilepath+loadfilename)
#
#                if model=='FG+HI:Jy':
#                    vFG1, vFG2, vFG3 = vis_coh['vis_model:Jy']
#                    vHI1, vHI2, vHI3 = vis_coh['vis_HI:Jy']
#                    vFGHI1, vFGHI2, vFGHI3 = vFG1+vHI1, vFG2+vHI2, vFG3+vHI3
#                    
#                    V_inv1 = np.sum(np.abs(vFGHI1) * self.window * self.freq_res, axis=-1) / np.sum(self.window * self.freq_res)
#                    V_inv2 = np.sum(np.abs(vFGHI2) * self.window * self.freq_res, axis=-1) / np.sum(self.window * self.freq_res)
#                    V_inv3 = np.sum(np.abs(vFGHI3) * self.window * self.freq_res, axis=-1) / np.sum(self.window * self.freq_res)
#
#                    V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
#                                            + (1./V_inv3)**2
#                    V_eff_arr.append(np.sqrt(1./V_eff_inv))
#                    Bphase_FGHI.append(np.angle(vFGHI1 * vFGHI2 * vFGHI3))
#
#            V_eff_arr=np.array(V_eff_arr)[:,0,:]
#            Bphase_FGHI = np.array(Bphase_FGHI)[:,0,:,:]
#            print(V_eff_arr.shape, Bphase_FGHI.shape)
#            
#            DS_FGHI=self.get_delay_spectrum(V_eff=V_eff_arr, bphase=Bphase_FGHI, \
#                                  ds_factor=ds_factor, use_window_sq=True)* u.Jy * u.Hz * u.Hz
#            print(DS_FGHI.shape)
#            PS_FGHI_arr = self.get_delay_powerspectrum(delay_spectrum_0=DS_FGHI,\
#                                                       delay_spectrum_1=DS_FGHI, A_eff=A_eff)
#
#                        
#            return V_eff_arr, Bphase_FGHI, DS_FGHI, PS_FGHI_arr  

#class Get_Spectrum(object):
#
#    '''
#    Estimate the delayspectrum and delaypowerspectrum.
#    
#    available functions: get_delay_spectrum(), get_delay_powerspectrum()
#    
#    '''
#
#    def __init__(self, freq, B_eff):
#        """_summary_
#
#        Args:
#            freq (_type_): _description_
#        
#        create following instenses:
#           
#        N_chan (int): usual size of the data. Number of frequency channels in the dataset.
#                e.g. N_channels=768 for data @ 40kHz, 384 for data @ 80kHz, 24 for data  @ 1.28MHz. Defaults to 768.
#        freq_res (int): frequency resolution of the data. Defaults to 40kHz.    
#        """
#        self.pol = ['xx', 'yy']
#        self.freq = freq*u.Hz
#        self.freq_res = abs(self.freq[1] - self.freq[0])
#        
#        delays = np.fft.fftfreq(len(self.freq), self.freq_res)
#        dtau = abs(delays[1] - delays[0])
#        df   = abs(freq[1] - freq[0])
#        B_eff = B_eff*u.Hz
#        
#        window = signal.blackmanharris(delays.size, True)
#        window_sq = signal.convolve(window, window, mode='full', method='fft')[::2]
#        area_windows_sq = np.trapz(window_sq, x= freq, dx=df)
#        
#        norm_factor = area_windows_sq/B_eff
#        window_sq = window_sq/norm_factor
#        
#        windowft = np.abs(np.fft.fft(window))
#        window_sqft=np.abs(np.fft.fft(window_sq))
#        
#        self.delays = delays
#        self.window = window
#        self.window_sq = window_sq
#        self.windowft = windowft
#        self.window_sqft = window_sqft
#    
#    def get_V_eff(self, vis=None, loadfilepath=None, loadfilename=None,\
#        obsID=None, baseline_length=None, polarizations=None,\
#            data_key=None, use_window_sq=None, from_file=None, data_shape=None, if_save=False,\
#                savefilepath=None, savefilename=None):
#            
#            """
#            Get effective visibilities
#
#            Args:
#                vis (ndarray): if from_file==False, then provide the vis array. Defaults to None.
#                loadfilepath (str): path to visiblility .npy file. Defaults to None.
#                loadfilename (str): Name of the visibility .npy file, the files are saved with some default names
#                e.g. 'vis_all_cases_%basline_length_%polarisation_%obsID.npy'. Defaults to None.
#                obsID (int) : obsID, GPS time of the observation. Defaults to None.
#                baseline_length (int): baseline length, available {14, 24, 28, 42} meter baselines. Defaults to None.
#                polarizations (str): polarizations, either 'xx' or 'yy'. Defaults to None.
#                data_key (str): dataset has keys, choose amongst following keys:
#                'vis_data:Jy/uncal',
#                'vis_FG:Jy',
#                'vis_FGI:Jy',
#                'vis_Noise:Jy',
#                'vis_HI:Jy',
#                'vis_FG+Noise:Jy',
#                'vis_FGI+Noise:Jy',
#                'vis_FG+Noise+HI:Jy',
#                'vis_FGI+Noise+HI:Jy'.
#                'all_keys'. Defaults to None.
#                
#                use_window_sq (bool, optional): set to True if use window squared function to estimate the effective visibilities. Defaults to False.
#                from_file (bool, ): set to True, if loading a visibility file to estimate the V_effective. Defaults to None.
#                data_shape (tuple): if from_file is set to True, then provide the expected shape of the final Veff dataset.
#                if_save (bool, optional): set to True if save the effective visibilites as .npy arrays. Defaults to False.
#                savefilepath (str, optional): if above is True, then provide the path to save the file. Defaults to None.
#                savefilename (str, optional): if above is True, then provide the filename. Defaults to None.
#
#            Raises:
#                TypeError: if put incorrect name of the loading file.
#                IOError: if the file is absent from the given directory.
#
#            Returns:
#                ndarray: numpy array of the efffective visibilities
#            """
#            
#            # getting the window function to normalise the visibilities
#
#            if use_window_sq == True:
#                window = self.window_sq
#            else:
#                window = self.window
#            
#            if from_file == True:
#                if loadfilename == None:
#                    loadfilename = 'vis_all_cases_%s_%s_%s.npy'%(baseline_length, polarizations, obsID)
#
#                try:
#                    data_vis = np.load(loadfilepath + loadfilename)
#                    keys = data_vis.dtype.names
#                except IOError:
#                    raise IOError('file absent from given directory: %s'%loadfilepath)
#                
#                if data_key != 'all_keys':
#
#                    v1, v2, v3 = data_vis[data_key]
#
#                    V_inv1 = np.sum(np.abs(v1) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#                    V_inv2 = np.sum(np.abs(v2) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#                    V_inv3 = np.sum(np.abs(v3) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#
#                    V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
#                                            + (1./V_inv3)**2
#                                            
#                    V_eff = np.array(np.sqrt(1./V_eff_inv), dtype=[(data_key, np.float64)])
#                
#                else:
#                    
#                    V_eff = np.ma.empty(shape=(data_vis.shape[1:3]), \
#                        dtype=[(data_key[key_index], np.float64) for key_index in range(len(data_key))])
#                    
#                    for key_indx in range(len(keys)):
#                        
#                        v1, v2, v3 = data_vis[keys[key_indx]]
#
#                        V_inv1 = np.sum(np.abs(v1) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#                        V_inv2 = np.sum(np.abs(v2) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#                        V_inv3 = np.sum(np.abs(v3) * window * self.freq_res, axis=2) / np.sum(window * self.freq_res)
#                            
#                        V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
#                                                + (1./V_inv3)**2
#                        V_eff[data_key[key_indx]] = np.sqrt(1./V_eff_inv)
#            else:
#                data_key = vis.dtype.names
#                V_eff = np.ma.empty(shape=data_shape, \
#                        dtype=[(data_key[key_index], np.float64) for key_index in range(len(data_key))])
#                for indx in range(vis.shape[0]): 
#                    for key_indx in range(len(data_key)):
#                        
#                        v1, v2, v3 = vis[data_key[key_indx]][indx]
#
#                        V_inv1 = np.sum(np.abs(v1) * window * self.freq_res, axis=-1) / np.sum(window * self.freq_res)
#                        V_inv2 = np.sum(np.abs(v2) * window * self.freq_res, axis=-1) / np.sum(window * self.freq_res)
#                        V_inv3 = np.sum(np.abs(v3) * window * self.freq_res, axis=-1) / np.sum(window * self.freq_res)
#                            
#                        V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
#                                                + (1./V_inv3)**2
#                        V_eff[data_key[key_indx]][indx] = np.sqrt(1./V_eff_inv)
#            
#            if if_save == True:
#                if savefilename==None:
#                    if data_key != 'all_cases':
#                        savefilename='v_eff_%s_%s_%s_%s.npy'%(data_key, baseline_length, polarizations, obsID)
#                    else:
#                        savefilename='v_eff_%s_%s_%s.npy'%(baseline_length, polarizations, obsID)
#                np.save(savefilepath+savefilename, V_eff)
#                
#            else:
#                return V_eff  
#    
#    def get_delay_spectrum(self, V_eff=None, bphase=None, ds_factor=None, use_window_sq=True):
#        
#        """Get delay spectrum
#
#        Args:
#            V_eff (ndarray, optional): Effective Visibility array. Defaults to None.
#            bphase (ndarray, optional): bispectrum phase array. Defaults to None.
#            use_window_sq (bool, optional): if use window^2 function. Defaults to True.
#
#        Returns:
#            ndarray: delay spectrum for given bispectrum phase model and effective visibilities
#        """
#        if use_window_sq==True:
#            window=self.window_sq
#        else:
#            window=self.window
#        
#        delay_spectrum_arr=[]
#        D_p = np.expand_dims(V_eff, axis=-1)*np.exp(1j*bphase)*window
#        for i in range(len(self.delays)):
#            delay_spectrum_arr.append(self.freq_res*np.sum(D_p*ds_factor[i], axis=-1))
#            print('done, ch: %d'%i)
#        delay_spectrum_arr = np.ma.array(delay_spectrum_arr)
#        delay_spectrum_arr = np.moveaxis(delay_spectrum_arr, 0, -1)
#        return delay_spectrum_arr
#        
#    def get_delay_powerspectrum(self, delay_spectrum_0=None, delay_spectrum_1=None,\
#                                A_eff=None):
#        """Get delay powerspectrum
#
#        Args:
#            delay_spectrum_0 (ndarray, optional): Delay spectrum 1. Defaults to None.
#            delay_spectrum_1 (ndarray, optional): Delay spectrum 2. Defaults to None.
#
#        Returns:
#            ndarray: Delay powerspectrum of the bispectrum phase.
#        """
#        
#        f_em = 1420*10**6 * u.Hz
#        kB = constants.k_B # Boltzmann's constant
#        c = constants.c
#        A_eff = A_eff * u.m *u.m
#        wavelength = (c/f_em)
#        bandwidth = self.freq[-1] - self.freq[0]
#        freq_m = self.freq[int(len(self.freq)/2)]
#
#        z0 = Cosmology.Z(self, f_obs=self.freq[0])
#        z_mid = Cosmology.Z(self, f_obs=freq_m)
#        z1 = Cosmology.Z(self, f_obs=self.freq[-1])
#        
#        Relative_comoving_distance = Planck18.comoving_distance(z0)-\
#                                        Planck18.comoving_distance(z1)
#        Mean_comoving_distance = Planck18.comoving_distance(z_mid)
#
#        CC = (A_eff / (wavelength**2 * bandwidth)) * \
#        (((Mean_comoving_distance **2) * Relative_comoving_distance) / bandwidth) * \
#            ((wavelength**2) / (2 * kB))**2
#        power = np.conjugate(delay_spectrum_0)*delay_spectrum_1 * CC
#        power = power.si.to(u.Mpc*u.Mpc*u.Mpc*u.milliKelvin*u.milliKelvin*u.Hz*u.Hz)
#        return power  
#   
  
