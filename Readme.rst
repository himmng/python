======================
Closure phase analysis 
======================

.. image:: http://ForTheBadge.com/images/badges/made-with-python.svg
   :target: https://www.python.org/

::

    Specific functions for MWA data analysis

::



Requirements
------------

::

    numpy
    pyuvdata


::


How to use
----------

::

    available classes Vis_Phase, Read_Numpy_Arr, Get_antenna_triplets, Cosmology,
    Spectrum, Noise_n_median_analysis


    Using the Vis_Phase class in python:



::



    ## using Vis_phase 

    from bphase import Vis_Phase

    Vis = Vis_Phase()

    Vis.vis(n1=None, n2=None, n3=None, uv=None)

    Vis.phase(v=None)

    Vis.vis_phase(anttriplets=None, time_stamps=None, dsize=None, uv=None, index=None)
    
    # create 3min visibility data

    Vis.create_3min_vis_data(path=None, obsID=None, time=None, time_offset=None, bll=None,\
    get_ants=None, ant_comp=None, path_to_save=None, loadfilename=None, savefilename=None,)

    # create visibility data

    Vis.create_vis_data(data_path=None, model_path=None, obsID=None, bll=None,\
    get_ants=None, ant_comp=None, path_to_save=None, loadfilenames=None, savefilename=None)
   
   

::



    ## using Read_Numpy_Arr

    from bphase import Read_Numpy_Arr

    # reading numpy array files

    Get_arr = Read_Numpy_Arr()

    # get 3min foreground and HI simulated visibilities

    Get_arr.get_3min_FG_HI(FG_path=None, HI_path=None, obsID=None, time=None, bll=None, HI_already_corrected=None)
    
    # get 3min foreground and HI bispectrum phase

    Get_arr.get_3min_FG_HI_bphase(FG_path=None, HI_path=None, obsID=None, time=None, bll=None, N_triad=None,\
    N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False)

    # get 3min foreground and HI bispectrum individual timestamps

    Get_arr.get_3min_indi_FG_HI_bphase(G_path=None, HI_path=None, obsID=None, time=None, bll=None, N_triad=None,\
    N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False)

    # get bispectrum phase from data

    Get_arr.get_data_bphase(path=None, obsID=None, bll=None, N_triad=None,\
    N_timestamp=None, N_channel=None, freq=None, if_check=False, if_all=False)
    
    # get 3min effective visibilities

    Get_arr.get_3min_V_eff_model(obsID_arr=None, time_arr=None, bll=None, window=None)

    # get effective visibilities from data and model

    Get_arr.get_V_eff_data_n_model(obsID_arr=None, bll=None, window=None, type=None)

    # get triad count

    Get_arr.get_triad_counts(path=None, obsIDs=None, bll=None)

    # modify visibility files (append NaNs in missing triads)

    Get_arr.vis_modify(path=None, obsIDs=None, bll=None, path_to_save=None)

    # get LSTs and obsIDs from metafits

    Get_arr.get_LSTs_n_obsIDs(path=None, obsIDs=None, field=None)

    # get same LSTs 

    Get_arr.get_same_LSTs(LSTs=None, obsIDs=None, tolerence=1,)

    # get 3min incoherrent bispectrum

    Get_arr.get_3min_incoherrent_bphase(FG_path=None, HI_path=None, obsIDs=None, time=None, bll=None)

    # get incoherrent bispectrum phase from data

    Get_arr.get_incoherrent_bphase_data(path=None, obsIDs=None, bll=None)
    


::



    ## using Get_antenna_triplets

    from bphase import Get_antenna_triplets

    ant_info = Get_antenna_triplets()

    # count antennae

    ant_info.count_antennae(uv=None)

    # get antenna triplets, baseline triplets

    ant_info.getThreePointCombinations(self, baselines=None, labels=None, positions=None, length=None, angle=None, unique=True)



::



    ## using Cosmology class

    from bphase import Cosmology

    cosmo = Cosmology()

    # get redshift

    cosmo.Z(f_obs=None)

    # get Energy-density

    cosmo.E(z=None)


::



    ## using Spectrum class

    from bphase import Spectrum

    spec = Spectrum()

    # get delay spectrum

    spec.get_delay_spectrum(V_eff=None, bphase=None, window=None, if_incoherrent=None)

    # get delay bipsectrum

    spec.get_delay_powerspectrum(delay_spectrum=None, if_incoherrent=None)



::



    ## using Noise_n_median_analysis class

    from bphase import Noise_n_median_analysis

    N_analysis = Noise_n_median_analysis()

    # get foreground noise

    N_analysis.get_noise_FG_model(obsIDs=None, time=None, bll=None)

    # get data noise

    N_analysis.get_noise_data(obsIDs=None, bll=None)

    # get mean and median of the bispectrum phase of 3min foreground and HI simulation

    N_analysis.get_3min_median_bphase(obsIDs=None, time=None, bll=None)

    # get mean and median statistics of bispectrum phase data and foreground simulation

    N_analysis.get_median_bphase_data(obsIDs=None, index_close=None, index=None, bll=None)


