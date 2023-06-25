try:
    
    import os
    import pyuvdata
    import numpy as np
    from pyuvdata import UVData
    from pyuvdata import utils
    from pyuvdata import utils as uvutils
    from pyuvdata.data import DATA_PATH
    

except:
    
    print('some repos missing')
        

class Vis_Phase(object):
    
    '''
    Get visiblity and phase from the uvfits data
    '''
    
    def __init__(self,):
        
        pass

    def vis(self, n1=None, n2=None, n3=None, uv=None): 
        
        '''
        Get individual antenna pair visibilities, bispectrum both unnormalised, normalised
        
        Params:
        
        n1, n2, n3: antenna numbers
        uv: UVData object
        
        Return: visibilites, visibility triple-product
        
        e.g.: v = vis(n1,n2,n3)
              v[0] : visibilities correspond to {n1, n2}
              v[1] : visibilities correspond to {n2, n3}
              v[2] : visibilities correspond to {n3, n1}
              v[3] : bispectrum/visibility triple product correspond to {n1, n2, n3}
        
        '''
        
        v1 = uv.get_data(n1, n2)
        v2 = uv.get_data(n2, n3)
        v3 = uv.get_data(n3, n1)
        
        # bispectrum
        vp = v1*v2*v3
        
        return v1, v2, v3, vp

    def phase(self, vis=None):
        
        '''
        Get the phase of visibilites
        
        Params:
        
        vis : input complex visibility
        
        Return: visibility phase
        '''
        
        ang = np.angle(vis)
        
        return ang
    
    def vis_phase(self, anttriplets=None, time_stamps=None, dsize=None, uv=None, index=None):
        
        '''
        Get visibilites and phase for antenna-pairs
        
        Params:
        
        anttriplets: {n1, n2, n3} antenna-pairs, can be generated using Get_antenna_triplets class
        time_stamps: time_stemps in the data, default: 14
        dsize: size of the data set, default: 768, (size of nfreq)
        uv: UVData object
        index: index is to choose the visibilites from either pairs 
               between ({n1, n2}, {n2, n3}, {n3, n1}) antennae.
               index = [0, 1, 2, 4, 5, 6] correspond to  visibilites/normalised_visibilities 
               of above three pairs.
               index = [3,7] correspond to  bisepctrum/normalised_bispectrum 
               of above three pairs.
               
        Return: visibilities
        '''
        
        vis = np.empty(shape = (len(anttriplets),time_stamps, dsize),\
                      dtype=np.complex64)        #data size N_triplets x N_time_stamps x N_freq
        
        for i in range(len(anttriplets)):
            vis[i] = self.vis(*anttriplets[i], uv)[index] ## gives vis at antenna triplets (a,b), (b,c), (c,a)
            
        
        return vis
    

    def create_3min_vis_data(self, path=None, obsID=None, time=None, time_offset=None, bll=None,\
        get_ants=None, ant_comp=None, path_to_save=None, loadfilename=None, savefilename=None, ):
    
        '''
        Create managable 3min foreground, HI simulations datasets
        
        params:
        
        path: path to the visiblity uvfits file (given that the default loadfile has specific name)
        obsID: GPS time of the observation
        time: 3min time UTC, if not loading HI file use time_offset array instead of time UTC
        time_offset: time offset array
        bll: baseline length in meters. available baselines 14, 24, 28, 42
        get_ants: Get antennas
        ant_comp: Get antenna components
        path_to_save: path to save the file, upgarding to .h5 format
        loadfilename: input file name, type: str, defaultname: 'HI_%obsID_%time_utc__' for HI and
        '%obsID_Puma_25000_offset%time_offset' for FG
        savefilename: output file name, type: str, defaultname: 'HI_v%bll_%time_utc_%obsID' for HI and
        'FG_v%bll_%time_utc_%obsID' for FG
        
        return: None
        save the visibilities in a file
        
        '''
        
        time_stamps = 14
        triad_angle = 120
        bl_str = np.loadtxt('bl_str')
        
        Vis = []
        
        if loadfilename == None:
            
            if ifHI == True:
                
                loadfilename = 'HI_%d_%s__'%(obsID, time_utc[:-1])
            
                for ind in range(1, 25):

                    if ind<10:

                        uvf_name = os.path.join(DATA_PATH, path + \
                                                loadfilename + '0%d.uvfits'%ind)


                    else:

                        uvf_name = os.path.join(DATA_PATH, path +\
                                                loadfilename + '%d.uvfits'%ind)
                        
                    uvinfo = UVData()
                    uvinfo.read(uvf_name, read_data=False)
                    
                    freq = uvinfo.freq_array[0]

                    Hex=Get_antenna_triplets.count_antennae(self, uv=uvd)[2] ## index 0 is HexE 1 is HexS, 2 HexE+S

                    uvinfo.read(uvf_name, polarizations=['xx'], antenna_nums=Hex) #Hex E + Hex S

                    ant_loc= uvinfo.antenna_positions[Hex]

                    anttriplets, blvecttriplets = Get_antenna_triplets.getThreePointCombinations(self, baselines=bl_str,\
                                             labels=Hex, positions=ant_loc, length=bll, angle=triad_angle)

                    dsize = len(freq)

                    v1 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, \
                                        uv=uvinfo, index=0)
                    v2 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, \
                                        uv=uvinfo, index=1)
                    v3 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, \
                                        uv=uvinfo, index=2)
                    
                    Vis.append(np.array([v1, v2, v3]))
                    
                Vis = np.array(Vis)
                Vis = np.concatenate(Vis, axis=-1)
                        
            else:
                
                loadfilename = '%d_Puma_25000_offset%d'%(obsID, time_offset)
                
                
        uvf_name = os.path.join(DATA_PATH, path + \
                                                loadfilename + '.uvfits')
        uvinfo = UVData()
        uvinfo.read(uvf_name, read_data=False)

        freq = uvinfo.freq_array[0]
        
        Hex=Get_antenna_triplets.count_antennae(self, uv=uvd)[2] ## index 0 is HexE 1 is HexS, 2 HexE+S

        uvinfo.read(uvf_name, polarizations=['xx'], antenna_nums=Hex) #Hex E + Hex S

        ant_loc= uvinfo.antenna_positions[Hex]

        anttriplets, blvecttriplets = Get_antenna_triplets.getThreePointCombinations(self, baselines=bl_str,\
                                             labels=Hex, positions=ant_loc, length=bll, angle=triad_angle)
        dsize = len(freq)
    
        v1 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps,\
                            dsize=dsize, uv=uvinfo, index=0)
        v2 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps,\
                            dsize=dsize, uv=uvinfo, index=1)
        v3 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps,\
                            dsize=dsize, uv=uvinfo, index=2)

        Vis.append(np.array([v1, v2, v3]))
        Vis = np.array(Vis)
        
        if savefilename == None:
            
            if ifHI == True:
                savefilename = 'HI_v%d_%s_%d'%(bll, time[:-1], obsID)

            else:
                savefilename = 'FG_v%d_%s_%d'%(bll, time[:-1], obsID)
                
        np.save(path_to_save + savefilename + '.npy', Vis)
        
        print('done', obsID)
    
    def create_vis_data(self, data_path=None, model_path=None, obsID=None, bll=None,\
               get_ants=None, ant_comp=None, path_to_save=None, loadfilenames=None, savefilename=None):
        '''
        Stores numpy array in the following format

        [[data, model, model_ideal_case,], [corresponding V1, V2, V3] , N_triads, N_timestamps, N_freq_channels ]
        e.g. shape (3,3,47,14,768)

        params:

        data_path: path to the observation data uvfits file
        model_path: path to the model uvfits, foreground
        obsID: GPS time of the observation
        bll: baseline length in meters. available baselines 14, 24, 28, 42
        get_ants: Get antennas
        ant_comp: Get antenna components
        path_to_save: path to save the file, upgarding to .h5 format
        loadfilenames: input file name, type: str, defaultname: '%obsID' for data and
        '%obsID_Puma_25000' for FG, '%obsID_Puma_Unity25000' for FG ideal model
        savefilename: output file name, type: str, defaultname: 'v%bll_%obsID'

        saves Visiblitites and antenna files
        '''
        try:
            if loadfilenames == None:
                ## names are data, model, ideal model default
                loadfilenames = ['%d'%obsID, '%d_Puma_25000'%obsID, '%d_Puma_Unity25000'%obsID]

            fd = os.path.join(DATA_PATH, data_path + loadfilenames[0] + '.uvfits')
            model_P = os.path.join(DATA_PATH, model_path + loadfilenames[1] + '.uvfits')
            model_PU = os.path.join(DATA_PATH, model_path + loadfilenames[2] + '.uvfits')

            bl_str = np.loadtxt('bl_str')
            triad_angle = 120.
            time_stamps = 14

            uvd = UVData()

            uvd.read(fd, read_data=False)

            uvm = UVData()
            uvm.read(model_P, read_data=False)

            uvmu = UVData()
            uvmu.read(model_PU, read_data=False)

            Hex=Get_antenna_triplets.count_antennae(self, uv=uvd)[2] ## index 0 is HexE 1 is HexS, 2 HexE+S
            ant_loc= uvd.antenna_positions[Hex]
            freq = uvd.freq_array[0]
            dsize = len(freq)

            uvd.read(fd,polarizations=['xx'], antenna_nums=Hex) #Hex E + Hex S
            uvm.read(model_P,polarizations=['xx'], antenna_nums=Hex) #Hex E + Hex S
            uvmu.read(model_PU,polarizations=['xx'], antenna_nums=Hex) #Hex E + Hex S

            anttriplets, blvecttriplets = Get_antenna_triplets.getThreePointCombinations(self, baselines=bl_str,\
                                             labels=Hex, positions=ant_loc, length=bll, angle=triad_angle)
            
            
            ######################### 14 meter baselines #################### (data, model, ideal model)
            v1 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvd,index=0)
            v2 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvd,index=1)
            v3 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvd,index=2)

            vm1 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvm,index=0)
            vm2 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvm,index=1)
            vm3 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvm,index=2)

            vmu1 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvmu,index=0)
            vmu2 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvmu,index=1)
            vmu3 = self.vis_phase(anttriplets=anttriplets, time_stamps=time_stamps, dsize=dsize, uv=uvmu,index=2)

            v_d = np.array([v1, v2, v3])
            v_m = np.array([vm1, vm2, vm3])
            v_mu = np.array([vmu1, vmu2, vmu3])
                        
            Vis = np.array([v_d, v_m, v_mu]) #######  (data, model, ideal model)

            if savefilename == None:
                savefilename = 'v%dH_%d'%(bll, obsID)

            np.save(path_to_save + savefilename, Vis)
            
            np.save(path_to_save + 'antT' + savefilename, anttriplets)
            
            print(obsID, 'done')

        except:

            #error.append(obsID)
            print(obsID, 'error')

class Read_Numpy_Arr(object):
    '''
    Class to read visiblity and bispectrum phase numpy array files

    '''

    def __init__(self):

        pass
    
    def get_3min_FG_HI(self, FG_path=None, HI_path=None, obsID=None, time=None, bll=None, HI_already_corrected=None):
    
        '''
        Designed to read visibility dataset in specific format
        Full numpy data set
        
        params:
        
        FG_path: foreground simulation path
        HI_path: HI simulation path
        obsID: GPS time of the observation (single obsID)
        time: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        HI_already_corrected: if the HI vis shape already corrected, bool

        return:
        
        FG_vis in array format datashape (v1, v2, v3, N_triad, N_timestamps, N_channels)
        HI_vis in array format datashape (N_coarse_channels, v1, v2, v3, N_triad, N_timestamps, N_fine_ch)
        reutrns in format                (v1, v2, v3, N_triad, N_timestamps, N_channels)
        '''
        
        FG_vis = np.load(FG_path +'FG_v%d_%s_%d.npy'%(bll, time, obsID))
        HI_vis = np.load(HI_path +'HI_v%d_%s_%d.npy'%(bll, time, obsID))

        if HI_already_corrected != True:
            HI_vis = np.concatenate(HI_vis, axis=-1)

        return FG_vis, HI_vis

    def get_3min_FG_HI_bphase(self, FG_path=None, HI_path=None, obsID=None, time=None, bll=None, N_triad=None,\
                        N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False):
        
        '''
        Designed to get the bispectrum phase from dataset in specific format
        
        params:
        
        FG_path: foreground simulation path
        HI_path: HI simulation path
        obsID: GPS time of the observation (single obsID)
        time: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int
        N_channel: get data at a fixed frequency channel, type: int
        freq: alternatively specify frequency, type: float or frequencies of the dataset
        if_check: inspect the bispectrum phase at 3 extreme channels, lowest, middle, highest frequency,
        type: bool, default: False
        if_all: if get the data at all triads, timestamps, and channels type: bool, default: False 
        
        return:
        
        Bispectrum_phase Foreground, HI at given input parameters
        
        '''
        
        FG_vis, HI_vis = self.get_3min_FG_HI(FG_path, HI_path, obsID, time, bll)
        
        if if_check==True:
            
            B_phase_FG = []
            B_phase_HI = []
            N_channel = np.linspace(0,len(FG_vis[0,0,0])-1, 3, dtype=np.int32)

            for i in N_channel:

                v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp, i]

                v1_HI, v2_HI, v3_HI = HI_vis[:, N_triad, N_timestamp, i]

                B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))

                B_phase_HI.append(np.angle(v1_HI * v2_HI * v3_HI))
            
            B_phase_FG = np.array(B_phase_FG)
            B_phase_HI = np.array(B_phase_HI)
            
        elif if_all==True:
            
            v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp,:]

            v1_HI, v2_HI, v3_HI = HI_vis[:, N_triad, N_timestamp,:]

            B_phase_FG = np.angle(v1_FG * v2_FG * v3_FG)
            B_phase_HI = np.angle(v1_HI * v2_HI * v3_HI)


        return B_phase_FG, B_phase_HI


    def get_3min_indi_FG_HI_bphase(self, FG_path=None, HI_path=None, obsID=None, time=None, bll=None, N_triad=None,\
                    N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False):
    
        '''
        Designed to get the bispectrum phase from dataset in specific format
        (full frequency range) or single frequency
        
        params:
        
        FG_path: foreground simulation path
        HI_path: HI simulation path
        obsID: GPS time of the observation (single obsID)
        time: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int
        N_channel: get data at a fixed frequency channel, type: int
        freq: alternatively specify frequency, type: float or frequencies of the dataset
        if_check: inspect the bispectrum phase at 3 extreme channels, lowest, middle, highest frequency,
        type: bool, default: False
        if_all: if get the data at all triads, timestamps, and channels type: bool, default: False 
        
        return:
        
        Bispectrum_phase Foreground, HI at given input parameters
        
        '''
        
        FG_vis, HI_vis = self.get_3min_FG_HI(FG_path, HI_path, obsID, time, bll)
        
        if if_check==True:

            B_phase_FG = []
            B_phase_HI = []
            N_channel = np.linspace(0,len(FG_vis[0,0,0])-1,3, dtype=np.int32)
            for i in N_channel:
                v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp, i]

                v1_HI, v2_HI, v3_HI = HI_vis[:, N_triad, N_timestamp, i]

                B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))

                B_phase_HI.append(np.angle(v1_HI * v2_HI * v3_HI))
            
            B_phase_FG = np.array(B_phase_FG)
            B_phase_HI = np.array(B_phase_HI)
            
        elif if_all==True:
            
            v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp,:]

            v1_HI, v2_HI, v3_HI = HI_vis[:, N_triad, N_timestamp,:]

            B_phase_FG = np.angle(v1_FG * v2_FG * v3_FG)
            B_phase_HI = np.angle(v1_HI * v2_HI * v3_HI)


        return B_phase_FG, B_phase_HI

    def get_data_bphase(self, path=None, obsID=None, bll=None, N_triad=None,\
                    N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False):
    
        '''
        Designed to get the bispectrum phase from dataset in specific format
        
        params:
        
        path: path to the visibility file
        obsID: GPS time of the observation
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        N_triad : get data for a fixed antenna triad, type: int
        N_timestamp: get data at a given observation timestamp, type: int
        N_channel: get data at a fixed frequency channel, type: int
        freq: alternatively specify frequency, type: float or frequencies of the dataset
        if_check: inspect the bispectrum phase at 3 extreme channels, lowest, middle, highest frequency,
        type: bool, default: False
        if_all: if get the data at all triads, timestamps, and channels type: bool, default: False 
        
        return:
        
        Bispectrum_phase measured, Foreground model, Ideal model at given input parameters
        
        '''
        
        data_vis = np.load(path +'v%dH_%d.npy'%(bll,obsID))
        
        if if_check==True:

            B_phase_m = []
            B_phase_FG = []
            B_phase_FGI = []
            N_channel = np.linspace(0,len(data_vis[0,0,0,0])-1,3, dtype=np.int32)
            
            for i in N_channel:
                v1_m, v2_m, v3_m = data_vis[0, :, N_triad, N_timestamp, i] ## measured vis
                v1_FG, v2_FG, v3_FG = data_vis[1, :, N_triad, N_timestamp, i] ## model vis
                v1_FGI, v2_FGI, v3_FGI = data_vis[2, :, N_triad, N_timestamp, i] ## ideal model vis

                B_phase_m.append(np.angle(v1_m * v2_m * v3_m))
                B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))
                B_phase_FGI.append(np.angle(v1_FGI * v2_FGI * v3_FGI))
            
            B_phase_m = np.array(B_phase_m)
            B_phase_FG = np.array(B_phase_FG)
            B_phase_FGI = np.array(B_phase_FGI)
            
        elif if_all==True:
            
            v1_m, v2_m, v3_m = data_vis[0, :, N_triad, N_timestamp, :]
            
            v1_FG, v2_FG, v3_FG = data_vis[1,:, N_triad, N_timestamp,:]

            v1_FGI, v2_FGI, v3_FGI = data_vis[2,:, N_triad, N_timestamp,:]

            B_phase_m = np.angle(v1_m * v2_m * v3_m)
            B_phase_FG = np.angle(v1_FG * v2_FG * v3_FG)
            B_phase_FGI = np.angle(v1_FGI * v2_FGI * v3_FGI)


        return B_phase_m, B_phase_FG, B_phase_FGIdef 

    def get_3min_V_eff_model(self, obsID_arr=None, time_arr=None, bll=None, window=None):
        '''
        Get Veff from model
        
        params:
        
        obsID_arr: GPS time of the observation
        time_arr: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        
        return:
        
        effective visibility
        
        '''
        V_eff = []

        for i in range(14, len(obsID_arr)):

            FG_vis, HI_vis = self.get_3min_FG_HI(obsID=obsID_arr[i], time=time_arr[i][:-1], bll=bll)
            
            for N_triad in range(len(HI_vis[1])):
                
                for N_timestamp in range(len(HI_vis[0][0])):
                    
                    v1, v2, v3 = FG_vis[:, N_triad, N_timestamp, :]

                    V_inv1 = np.sum(np.abs(v1) * window * df) / np.sum(window * df)
                    V_inv2 = np.sum(np.abs(v2) * window * df) / np.sum(window * df)
                    V_inv3 = np.sum(np.abs(v3) * window * df) / np.sum(window * df)
                    
                    V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
                                + (1./V_inv3)**2

                    V_eff.append(np.sqrt(1./V_eff_inv))
                    
        V_eff = np.array(V_eff)           
        
        return V_eff

    def get_V_eff_data_n_model(self, obsID_arr=None, bll=None, window=None, type=None):
        '''
        Get Veff
        params:
        
        obsID_arr: GPS time of the observation
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        window: window function
        type: put from these options 'data', 'real_model', 'ideal_model', type: str

        return:
        
        effective visibility
        
        '''
        V_eff = []
        
        for i in range(14, len(obsID_arr)):

            data_vis = np.load(arr_path_local_data+'v%dH_%d.npy'%(bll, obsID_arr[i]))
            
            for N_triad in range(len(data_vis[0,0])):
                
                for N_timestamp in range(len(data_vis[0,0,0])):

                    if type == 'data':

                        v1, v2, v3 = data_vis[0,:, N_triad, N_timestamp,:]
                    
                    elif type == 'real_model':

                        v1, v2, v3 = data_vis[1,:, N_triad, N_timestamp,:]
                    
                    elif type == 'ideal_model':

                        v1, v2, v3 = data_vis[2,:, N_triad, N_timestamp,:]
                    
                    
                    V_inv1 = np.sum(np.abs(v1) * window * df) / np.sum(window * df)
                    V_inv2 = np.sum(np.abs(v2) * window * df) / np.sum(window * df)
                    V_inv3 = np.sum(np.abs(v3) * window * df) / np.sum(window * df)
                    
                    V_eff_inv = (1./V_inv1)**2 + (1./V_inv2)**2 \
                                + (1./V_inv3)**2

                    V_eff.append(np.sqrt(1./V_eff_inv))
                    
        V_eff = np.array(V_eff)           
    
        return V_eff

    def get_triad_counts(self, path=None, obsIDs=None, bll=None):
        '''
        Get the number of working triads stored in the processed data, **specific function**
        This function will use the triad counts for the given baseline
        and append NaN visibilities at the missing triads, so that all of the processed 
        data has same number of triads
        
        params:
        
        path: path to the processed data, type: str
        obsIDs: GPS time array of observations, type: array(int)
        bll: baseline length of triads, type: int, available blls: {14, 24, 28, 42}
        
        return 
        
        triad count array '''
        
        triad_counts = np.empty(len(obsIDs))

        for i in range(len(obsIDs)):
            
            try: 
                data_vis = np.load(path + 'v%dH_%d.npy'%(bll, obsIDs[i])) ## data is stored in specific name

                triad_counts[i] = len(data_vis[0,0,:])
                
            except:
                
                triad_counts[i] = np.nan
        
        return triad_counts

    def vis_modify(self, path=None, obsIDs=None, bll=None, path_to_save=None):
        
        '''
        This function will use the triad counts for the given baseline
        and append NaN visibilities at the missing triads, so that all of the processed 
        data has same number of triads
        
        params:
        
        path: path to the processed data, type: str
        obsIDs: GPS time array of observations, type: array(int)
        bll: baseline length of triads, need to fix/modify, type: int, available blls: {14,24,28,42}
        path_to_save: path to save the data
        return 
        
        save: numpy array of visiblities, append NaNs at the missing triads
        
        '''
        
        ##reference_info
        ## for bll=14, N_triads_max = 47
        ## for bll=24, N_triads_max = 32
        ## for bll=28, N_triads_max = 29
        ## for bll=42, N_triads_max = 14
        
        ## reference obsID having all triads working 1160570752
        obsID_ref = 1160570752
        ant_ref = np.load(path + 'antT%dH_%d.npy'%(bll, obsID_ref), allow_pickle=True)
        
        N = 3
        N_vis = 3
        N_triads = len(ant_ref)
        N_timestamps = 14
        N_freq = 768
        
        for k in range(len(obsIDs)):
            try:
                
                ants_triplets = np.load(data_path + 'antT%dH_%d.npy'%(bll, obsIDs[k]), allow_pickle=True)
                data_vis = np.load(data_path + 'v%dH_%d.npy'%(bll, obsIDs[k]))

                match_indicies = []
                missing_indicies = []

                total_indicies = np.arange(len(ant_ref), dtype=np.int32)
                
                for i in range(len(ants_triplets)):

                    match_indicies.append(np.where(ant_ref==ants_triplets[i])[0][0])


                match_indicies = np.sort(np.array(list(set(match_indicies))))

                missing_indicies = np.sort(np.array(list(set(total_indicies) - set(match_indicies))))   
                
                if len(missing_indicies) == 0:
                    
                    print('all triads present', k)
                    
                else:
                    
                    dummy_data_vis = np.empty(shape=(N, N_vis, N_triads, N_timestamps, N_freq), dtype=np.complex64)

                    nan_insert= np.empty(shape=(N, N_vis, N_timestamps, N_freq), dtype=np.complex64)
                    nan_insert.real[:] = np.nan
                    nan_insert.imag[:] = np.nan

                    for i in range(len(match_indicies)):

                        dummy_data_vis[:, :, match_indicies[i], :, :] = data_vis[:, :, i, :, :]

                    for i in range(len(missing_indicies)):

                        dummy_data_vis[:, :, missing_indicies[i], :, :] = nan_insert


                    # another way of doing, laborious
                    #data_vis = np.insert(arr=data_vis, obj=28, values=nan_insert, axis=2)

                    np.save(data_path + 'v%dH_%d.npy'%(bll, obsIDs[k]), dummy_data_vis)

                    print('obsID, done!', k)
                
            except:
                print('data missing', k)

    def get_LSTs_n_obsIDs(self, path=None, obsIDs=None, field=None):    
        '''
        Get the EoR field specific LSTs and corresponding obsIDs
        
        params:
        
        path: path to the observation's metafits file, type: str
        obsIDs: GPSTIME of the observation, dtype: arr[int]
        field: name of the EoR field, type=str, e.g. EoR0, EoR1, EoR2, all
        if field='all', returns LST0,1,2, obsIDs0,1,2 format
        
        return: 
        
        LSTs and corresponding obsIDs in array format
        
        '''
        
        deg_to_hour = 24./360.
        
        LST_EoR0 = []
        LST_EoR1 = []
        LST_EoR2 = []
        
        obsIDs_EoR0 = []
        obsIDs_EoR1 = []
        obsIDs_EoR2 = []
        
        for i in range(len(obsIDs)):
            
            mfits = fits.open(path+ '%d.metafits'%obsIDs[i])
            
            LSTs.append(mfits[0].header['LST'])
            
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

    def get_same_LSTs(self, LSTs=None, obsIDs=None, tolerence=1,):
        '''
        Given the LSTs of the observations, select the same LSTs given the tolerence
        
        params:
        
        LSTs: Local Sidereal Time of the observaiton in degrees, dtype: array
        obsIDs: gpstime of observations, dtype: array(int)
        tolerence: time tolerence between the LSTs in mintues; default: 1 min
        
        
        return:
        
        identical LST counts, mean LST [hour] and corresponding obsIDs
        
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
        
        LST_obsID_metric = []
        
        for i in range(1, len(avg_indices)):
            
            LST_obsID_metric.append([avg_indices[i]-avg_indices[i-1],\
                                    obsIDs_sort[avg_indices[i-1]: avg_indices[i]],\
                                    np.mean(LSTs_minutes_sort[avg_indices[i-1]: avg_indices[i]])/4.0])
        
        LST_obsID_metric = np.array(LST_obsID_metric, dtype=object)
        
        return LST_obsID_metric

    def get_3min_incoherrent_bphase(self, FG_path=None, HI_path=None, obsIDs=None, time=None, bll=None):
    
        '''
        Get incoherrent phase from 
        
        params:
        FG_path: foreground simulation path
        HI_path: HI simulation path
        obsIDs: GPS time of the observation (fulll array)
        time: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        
        return:
        
        bispectrum phase array, FG model, HI model
        
        '''
        B_phase_FG = []
        B_phase_HI = []
        
        for i in range(14, len(obsIDs)):
            FG_vis, HI_vis = get_FG_HI(FG_path=FG_path, HI_path=HI_path, obsIDs=obsIDs[i], time=time[i][:-1], bll=bll)
            
            for N_triad in range(len(HI_vis[1])):
                
                for N_timestamp in range(len(HI_vis[0][0])):
                    
                    v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp, :]
                    v1_HI, v2_HI, v3_HI = HI_vis[:, N_triad, N_timestamp, :]
                    
                    B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))
                    B_phase_HI.append(np.angle(v1_HI * v2_HI * v3_HI))
                    
                    
            
        B_phase_FG = np.array(B_phase_FG)
        B_phase_HI = np.array(B_phase_HI)
        
        return B_phase_FG, B_phase_HI

    def get_incoherrent_bphase_data(self, path=None, obsIDs=None, bll=None):
        '''
        Get incoherrent pbhase
        params:
        
        path: path to the visibility file
        obsIDs: GPS time of the observation (full array)
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        
        return:
        
        bispectrum phase data array
        
        '''
        
        B_phase_m = []
        B_phase_FG = []
        B_phase_FGI = []
        for i in range(14, len(obsIDs)):

            data_vis = np.load(path +'v%dH_%d.npy'%(bll,obsIDs[i]))
            
            for N_triad in range(len(data_vis[0,0])):
                
                for N_timestamp in range(len(data_vis[0,0,0])):
            
                    v1_m, v2_m, v3_m = data_vis[0, :, N_triad, N_timestamp, :]

                    v1_FG, v2_FG, v3_FG = data_vis[1,:, N_triad, N_timestamp,:]

                    v1_FGI, v2_FGI, v3_FGI = data_vis[2,:, N_triad, N_timestamp,:]

                    B_phase_m.append(np.angle(v1_m * v2_m * v3_m))
                    B_phase_FG.append(np.angle(v1_FG * v2_FG * v3_FG))
                    B_phase_FGI.append(np.angle(v1_FGI * v2_FGI * v3_FGI))
                    
                    
            
        B_phase_m = np.array(B_phase_m)
        B_phase_FG = np.array(B_phase_FG)
        B_phase_FGI = np.array(B_phase_FGI)
        
        return B_phase_m, B_phase_FG, B_phase_FGI

    

class Get_antenna_triplets(object):
    
    def __init__(self):
        
        pass
    
    def count_antennae(self, uv=None):
        '''
        Check for the MWA antennae in the data, 
        This function can help get visibilites from specific MWA antenne/configurations,
        e.g. Redundant Hexagon, non-redudant Tiles
        
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
    
        ant_indices = np.unique(uv.ant_1_array.tolist() + uv.ant_2_array.tolist())-1
        antenna_names = np.array(uv.antenna_names)
 
        antenna_names = antenna_names[ant_indices]


        Tile_names = []
        Tile_index = []

        HexE_names = []
        HexE_index = []

        HexS_names = []
        HexS_index = []

        Extra = 0

        for i in range(len(antenna_names)):

            if antenna_names[i][:4] == 'Tile':
                Tile_names.append(antenna_names[i])
                Tile_index.append(ant_indices[i])

            elif antenna_names[i][:4] == 'HexE':
                HexE_names.append(antenna_names[i])
                HexE_index.append(ant_indices[i])


            elif antenna_names[i][:4] == 'HexS':
                HexS_names.append(antenna_names[i])
                HexS_index.append(ant_indices[i])

            else: 
                Extra += 1

        Tile_index = np.array(Tile_index)
        Tile_names = np.array(Tile_names)
        
        HexE_index = np.array(HexE_index)
        HexE_names = np.array(HexE_names)
        
        HexS_index = np.array(HexS_index)
        HexS_names = np.array(HexS_names)
        
        Hex_combine_index = np.concatenate((HexE_index, HexS_index))
        Hex_combine_names = np.concatenate((HexE_names, HexS_names))

        return HexE_index, HexS_index, Hex_combine_index, Tile_index,\
               HexE_names, HexS_names, Hex_combine_names, Tile_names, Extra

    def getThreePointCombinations(self, baselines=None, labels=None, positions=None, length=None, angle=None, unique=True):
        '''
        Get the three antenna pairs forming a close equilateral triangle for given baseline length,
        
        Params:
        
        baselines: list of all baseline vectors. dtype: list, ndarray
        labels: list of antenna names, e.g. HexE, HexS, HexE+HexS combined, Tiles. dtype: list, ndarray
        positions: location of antennae
        length: baseline length in meteres to be extracted
        angle: angle between baseline vectors, for equilateral triangle it is 120 degrees.
        unique: only do the unique baselines. type: bool
        
        Return: antenna triplets, corresponding baselines
   
        '''
        if not isinstance(unique, bool):
            raise TypeError('Input unique must be boolean')
        bl = baselines + 0.0 
        blstr = np.unique(['{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(lo) for lo in bl])
        bltriplets = []
        blvecttriplets = []
        anttriplets = []
        cross=[]
        np.seterr(invalid='ignore')
        for aind1,albl1 in enumerate(labels):
            for aind2,albl2 in enumerate(labels):
                bl12 = positions[aind2] - positions[aind1]

                bl12_len = np.sqrt(np.sum(bl12**2))

                if np.around(bl12_len)!=length:
                    continue
                elif bl12_len > 0.0 and bl12[1]<0.:
                    bl12str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl12)
                    for aind3,albl3 in enumerate(labels):
                        if aind1!= aind2 != aind3:
                            bl23 = positions[aind3] - positions[aind2]
                            bl31 = positions[aind1] - positions[aind3]
                            bl23_len = np.sqrt(np.sum(bl23**2))
                            bl31_len = np.sqrt(np.sum(bl31**2))

                            ang1 = (180./np.pi)*np.arccos(np.dot((bl23/np.linalg.norm(bl23)), (bl12/np.linalg.norm(bl12))))

                            ang2 = (180/np.pi)*np.arccos((np.dot((bl23/np.linalg.norm(bl23)),(bl31/np.linalg.norm(bl31)))))
                            ang3 = (180/np.pi)*np.arccos((np.dot((bl31/np.linalg.norm(bl31)),(bl12/np.linalg.norm(bl12)))))
                            #cross1 = np.cross((bl12/np.linalg.norm(bl12)), (bl23/np.linalg.norm(bl23)))
                            #cross2 = np.cross((bl23/np.linalg.norm(bl23)), (bl31/np.linalg.norm(bl31)))
                            #cross3 = np.cross((bl31/np.linalg.norm(bl31)), (bl12/np.linalg.norm(bl12)))
                            if np.around(bl31_len)==np.around(bl23_len)==np.around(bl12_len)==length :
                                if np.around(ang1)== np.around(ang2) ==np.around(ang3) == angle :
                                    if bl12[2]>0. and bl31[1]>0. and bl31[2]>0. :

                                        bl23str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl23)

                                        bl31str = '{0[0]:.2f}_{0[1]:.2f}_{0[2]:.2f}'.format(bl31)
                                        list123_str = [bl12str, bl23str, bl31str]

                                        bltriplets += [list123_str]
                                        blvecttriplets += [[bl12, bl23, bl31]]
                                        anttriplets += [{albl1, albl2, albl3}]
                                        #cross+=[cross1]

                            else:
                                continue
        return anttriplets, blvecttriplets
   
class Cosmology(object):

    def __init__(self):

        pass

    def Z(self, f_obs=None):
        '''
        redshift estimate

        params:

        f_obs: observed frequency
        '''

        f_em = 1420*10**6 # in Hz
        return (f_em - f_obs)/f_obs

    def E(self, z=None):
        '''
        cosmology

        params:

        z: redshift
        '''
        O_m = 0.3
        O_k = 0
        O_l = 0.7
        
        return (O_m*((1+z)**3) + O_k*((1+z)**2) + O_l)**(1/2)

class Spectrum(object):

    '''
    Class to estimate the delayspectrum and delaypowerspectrum
    '''

    def __init__(self):

        pass
    
    def get_delay_spectrum(self, V_eff=None, bphase=None, window=None, if_incoherrent=None):

        '''
        Get delay spectrum
        params:
        
        bphase: bispectrum phase
        window: window function
        if_incoherrent: average all, if True. type: bool, default: True
        
        return:
        
        delay_spectrum_arr
        '''
        
        delay_spectrum_arr = []

        if if_incoherrent == True:
            
            V_bphase_ft = np.fft.fft(V_eff.reshape(len(V_eff), 1)*np.exp(1j*bphase))
            window_ft = np.fft.fft(window)
            
            for i in range(len(V_bphase_ft)):

                delay_spectrum_arr.append(np.convolve(V_bphase_ft[i], window_ft, mode='same'))

            delay_spectrum_arr = np.array(delay_spectrum_arr)
            
            return delay_spectrum_arr
        
        else:
            return None
        
    def get_delay_powerspectrum(self, delay_spectrum=None, if_incoherrent=None):
        
        '''
        Get delay powerspectrum
        
        params:
        
        delay_spectrum: delay spectrum
        V_eff: effective foreground visiblities
        if_incoherrent: average all, if True. type: bool, default: True
        
        return:
        
        delay powerspectrum
        '''
        
        df = 40000
        f_em = 1420*10**6
        kB = 1.380649*1e-23
        c = 3*10**8
        Tot_A_eff = 19.921281147451342#MWA_Aeff_167_197MHz
        dD = 8879.5 - 8721.9 #8988.7-8470.8 #in Mpc
        D =  8800.5 #in Mpc
        dB = 10e6
        freq_m = freq[383]
        
        CC = ((Tot_A_eff*freq_m)/(c*dB))*(D**2*dD/dB)*((((c/freq_m)**2)/2*kB)**2)*(10**41)
        
        
        if if_incoherrent == True:
            
            power = delay_spectrum**2 * CC
            
            return power
            
        elif if_incoherrent == False:
            
            return None


class Noise_n_median_analysis(object):

    '''
    Get Noise and Analysis
    '''

    def __init__(self):

        pass

    def get_noise_FG_model(self, obsIDs=None, time=None, bll=None):

        '''
        Function takes visiblity difference from N, N-1 timestamp, makes bispectrum phase
        
        params:
        
        obsIDs: GPS time of the observation (full array)
        time: UTC Time array inforamtion
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        
        return:
        
        bispectrum phase array, FG model, HI model
        
        '''
        
        dB_phase_FG = []
        
        for i in range(14, len(obsIDs)):
            
            FG_vis, HI_vis = get_FG_HI(obsIDs=obsIDs[i], time=time[i][:-1], bll=bll)
            
            for N_triad in range(len(HI_vis[1])):
                
                for N_timestamp in range(1, len(HI_vis[0][0])):
                    
                    v1_FG, v2_FG, v3_FG = FG_vis[:, N_triad, N_timestamp-1, :]
                    v4_FG, v5_FG, v6_FG = FG_vis[:, N_triad, N_timestamp, :]
                    
                    dv1_FG = v4_FG - v1_FG
                    dv2_FG = v5_FG - v2_FG
                    dv3_FG = v6_FG - v3_FG
                    
                    dB_phase_FG.append(np.angle(dv1_FG * dv2_FG * dv3_FG))
                      
        dB_phase_FG = np.array(dB_phase_FG)

        return dB_phase_FG

    def get_noise_data(self, obsIDs=None, bll=None):

        '''
        Function takes visiblity difference from N,
        N-1 timestamp, makes bispectrum phase
        
        params:
        
        obsIDs: GPS time of the observation (full array)
        bll: baseline length, dataset available in 14m , 28m, 42m baselines
        
        return:
        
        bispectrum phase data array
        
        '''
        
        dB_phase_m = []
        
        for i in range(14, len(obsIDs)):
            
            data_vis = np.load(arr_path_local_data+'v%dH_%d.npy'%(bll, obsIDs[i]))
            
            for N_triad in range(len(data_vis[0,0])):
                
                for N_timestamp in range(1, len(data_vis[0,0,0])):
            
                    v1_m, v2_m, v3_m = data_vis[0,:, N_triad, N_timestamp-1, :]
                    v4_m, v5_m, v6_m = data_vis[0,:, N_triad, N_timestamp, :]
                    dv1_m = v4_m - v1_m
                    dv2_m = v5_m - v2_m
                    dv3_m = v6_m - v3_m

                    dB_phase_m.append(np.angle(dv1_m * dv2_m * dv3_m))
                    
        dB_phase_m = np.array(dB_phase_m)
        
        return dB_phase_m
        
    def get_3min_median_bphase(self, obsIDs=None, time=None, bll=None): 

        '''
        Get median averaged Bispectrum phase from FG, HI simulation
        
        params:
        
        obsIDs: GPS time of observation (all obsIDs)
        time: TIME UTC array
        bll: baseline length
        
        return:
        
        Bispectrum phase, mean Bispectrum phase, B_phase_median, B_phase_median_absolute_deviation
        '''
        try:
            FG_vis, HI_vis, data_vis = Read_Numpy_Arr.get_3min_FG_HI(obsIDs, time, bll)
            v1FG, v2FG, v3FG = FG_vis
            v1HI, v2HI, v3HI = HI_vis
            v1m, v2m, v3m = data_vis[0]
            B_FG = v1FG * v2FG * v3FG
            B_FG_phase = np.angle(B_FG)

            B_HI = v1HI * v2HI * v3HI
            B_HI_phase = np.angle(B_HI)

            B_m = v1m * v2m * v3m
            B_m_phase = np.angle(B_m)

            B_m_phase = np.nanmean(B_m_phase, axis=1)
            B_HI_phase = np.nanmean(B_HI_phase, axis=1)
            B_FG_phase = np.nanmean(B_FG_phase, axis=1)

            B_m_phase_mean = np.nanmean(B_m_phase, axis=0)
            B_m_phase_median = np.nanmedian(B_m_phase, axis=0)
            B_m_phase_median_absolute_deviation = np.median(np.abs(B_m_phase - B_m_phase_median), axis=0)

            B_FG_phase_mean = np.nanmean(B_FG_phase, axis=0)
            B_FG_phase_median = np.nanmedian(B_FG_phase, axis=0)
            B_FG_phase_median_absolute_deviation = np.median(np.abs(B_FG_phase - B_FG_phase_median), axis=0)

            B_HI_phase_mean = np.nanmean(B_HI_phase, axis=0)
            B_HI_phase_median = np.nanmedian(B_HI_phase, axis=0)
            B_HI_phase_median_absolute_deviation = np.median(np.abs(B_HI_phase - B_HI_phase_median), axis=0)
        
        except:
            
            data_vis = np.load(arr_path_local_data+'v%dH_%d.npy'%(bll,obsIDs))
            
            
            v1m, v2m, v3m = data_vis[0]
            
            B_m = v1m * v2m * v3m
            B_m_phase = np.angle(B_m)

            B_m_phase = np.nanmean(B_m_phase, axis=1)
            

            B_m_phase_mean = np.nanmean(B_m_phase, axis=0)
            B_m_phase_median = np.nanmedian(B_m_phase, axis=0)
            B_m_phase_median_absolute_deviation = np.median(np.abs(B_m_phase - B_m_phase_median), axis=0)

        
        return B_m_phase, B_m_phase_mean, B_m_phase_median, B_m_phase_median_absolute_deviation

    def get_median_bphase_data(self, obsIDs=None, index_close=None, index=None, bll=None): 

        '''
        Get median averaged Bispectrum phase from data
        
        params:
        
        obsIDs: GPS time of observation (all obsIDs)
        time: TIME UTC array
        bll: baseline length
        
        return:
        
        Bispectrum phase, mean Bispectrum phase, B_phase_median, B_phase_median_absolute_deviation
        '''
        
        B_m_full =[]
        B_m_rms_full = []
        if index ==0:
            for i in range(0, index_close[index]):

                data_vis = np.load(arr_path_local_data + 'v%dH_%d.npy'%(bll, obsIDs[i]))

                v1m, v2m, v3m = data_vis[0]

                B_m_rms_arr = np.array([v1m, v2m, v3m])
                B_m_rms_full.append(B_m_rms_arr)
                B_m = v1m * v2m * v3m
                B_m_phase = np.angle(B_m)
                B_m_full.append(B_m_phase)
        
        else:
            for i in range(index_close[index-1], index_close[index]):

                data_vis = np.load(arr_path_local_data + 'v%dH_%d.npy'%(bll, obsIDs[i]))

                v1m, v2m, v3m = data_vis[0]

                B_m_rms_arr = np.array([v1m, v2m, v3m])
                B_m_rms_full.append(B_m_rms_arr)
                B_m = v1m * v2m * v3m
                B_m_phase = np.angle(B_m)
                B_m_full.append(B_m_phase)

        B_m_full = np.array(B_m_full)
        B_m_full = np.concatenate(B_m_full, axis=1)
        B_m_full_avg = np.nanmean(B_m_full, axis=1)
        B_m_phase_mean = np.nanmean(B_m_full_avg, axis=0)
        B_m_phase_median = np.nanmedian(B_m_full_avg, axis=0)
        B_m_phase_median_absolute_deviation = np.median(np.abs(B_m_full_avg - B_m_phase_median), axis=0)

        B_m_rms_full = np.array(B_m_rms_full)
        B_m_rms_full = np.concatenate(B_m_rms_full, axis=2)
        
        B_m_rms_full = np.angle(B_m_rms_full)**2
        B_m_rms_full = np.nanmean(B_m_rms_full, axis=2)
        B_m_rms_full = np.sqrt(np.sum(B_m_rms_full, axis=0))/np.sqrt(3)
        
        return B_m_full_avg, B_m_phase_mean, B_m_phase_median, B_m_phase_median_absolute_deviation, B_m_rms_full