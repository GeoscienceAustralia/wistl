# k: 0.33 for a single, 0.5 for double circuit
# no_circuit can be defined by line
K_FACTOR = {1: 0.33, 2: 0.5}  # hard-coded
NO_CIRCUIT = 2

FIELDS_TOWER = ['name',
                'type',
                'latitude',
                'longitude',
                'function',
                'devangle',
                'axisaz',
                'height',
                'lineroute',
                'design_span',
                'design_speed',
                'terrain_cat',
                'height_z',
                'shape',
                'design_level',
               ]

params_tower = FIELDS_TOWER + [
                'coord',
                'coord_lat_lon',
                'point',
                'frag_dic',
                'file_wind_base_name',
                'ratio_z_to_10',
                'actual_span',
                'u_factor',
                'collapse_capacity',
                'cond_pc',
                'max_no_adj_towers',
                'id_adj', 'idl',
                'cond_pc_adj',
                'cond_pc_adj_sim_idx',
                'cond_pc_adj_sim_prob'
               ]

params_line = ['linename',
               'type',
               'owner',
               'operator',
               'capacity',
               'typeconduc',
               'numcircuit',
               'current',
               'yrbuilt',
               'shapes',
               'coord',
               'coord_lat_lon',
               'line_string',
               'name_output',
               'no_towers',
               'actual_span',
               'seed',
               'id2name',
               'ids',
               'names',
               'name2id',
              ]

params_event = ['id',
                'path_wind_event',
                'name',
                'scale',
                'seed',
                ]

