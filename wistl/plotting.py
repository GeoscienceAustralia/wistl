import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import numpy as np


def plot_tower_line(line):

    coord_towers = []
    for i in range(line.no_towers):
        coord_towers.append(line.df_towers.loc[i, 'coord'])
    coord_towers = np.array(coord_towers)

    plt.figure()
    plt.plot(line.coord[:, 0],
             line.coord[:, 1], 'ro',
             coord_towers[:, 0],
             coord_towers[:, 1], 'b-')
    plt.title(line.name)

    png_file = os.path.join(line.path_output,
                            'line_{}.png'.format(line.name))

    if not os.path.exists(line.path_output):
        os.makedirs(line.path_output)

    plt.savefig(png_file)
    print('{} is created'.format(png_file))
    plt.close()


def plot_line_interaction(network, cfg):

    for line_name, line in network.lines.items():

        plt.figure()
        plt.plot(line.coord[:, 0],
                 line.coord[:, 1], '-', label=line_name)

        for target_line in cfg.line_interaction[line_name]:

            plt.plot(network.lines[target_line].coord[:, 0],
                     network.lines[target_line].coord[:, 1],
                     '--', label=target_line)

            for _, tower in line.towers.items():
                try:
                    id_pt = tower.id_on_target_line[target_line]['id']
                except KeyError:
                    plt.plot(tower.coord[0], tower.coord[1], 'ko')
                else:
                    target_tower_name = network.lines[
                        target_line].name_by_line[id_pt]
                    target_tower = network.lines[target_line].towers[
                        target_tower_name]

                    plt.plot([tower.coord[0], target_tower.coord[0]],
                             [tower.coord[1], target_tower.coord[1]],
                             'ro-',
                             label='{}->{}'.format(tower.name,
                                                   target_tower_name))

        plt.title(line_name)
        plt.legend(loc=0)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        png_file = os.path.join(cfg.path_output,
                                'line_interaction_{}.png'.format(line_name))

        if not os.path.exists(cfg.path_output):
            os.makedirs(cfg.path_output)

        plt.savefig(png_file)
        print('{} is created'.format(png_file))
        plt.close()


