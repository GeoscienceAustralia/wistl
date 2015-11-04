# script to set data for sim_towers_v13-2
import os

pdir = os.getcwd()
shape_file_tower = os.path.join(
    pdir, 'Shapefile_2015_01/Towers_with_extra_strainers_WGS84.shp')
shape_file_line = os.path.join(
    pdir, 'Shapefile_2015_01/Lines_NGCP_with_synthetic_attributes_WGS84.shp')

file_frag = os.path.join(pdir, 'input/fragility_GA.csv')
file_cond_pc = os.path.join(pdir, 'input/cond_collapse_prob_NGCP.csv')
file_terrain_height = os.path.join(pdir, 'input/terrain_height_multiplier.csv')
flag_strainer = ['Strainer', 'dummy'] # consider strainer 

file_design_value = os.path.join(pdir, 'input/design_value_current.csv')
#file_topo_value = os.path.join(pdir, 'input/topo_value_scenario_50yr.csv')
file_topo_value = None

dir_wind_timeseries = os.path.join(pdir, 'glenda_reduced')
dir_output = os.path.join(pdir, 'output_current_glenda')

if not os.path.exists(dir_output):
    os.makedirs(dir_output)

# number of simulations for MC
nsims = 100

# flag for save
flag_save = 1

if __name__ == '__main__':
    from sim_towers_v13_2 import main
    main(shape_file_tower, shape_file_line, dir_wind_timeseries, 
    file_frag, file_cond_pc, file_design_value, file_terrain_height, 
    file_topo_value, flag_strainer, flag_save, dir_output, nsims)
