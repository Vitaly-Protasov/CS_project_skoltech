import numpy as np
import pandas as pd
import os
import ast
from parse_data import *

def get_frame(path):
    columns = ['Player', 'Side', 'Round', 'Round len', 'Gun shot accuracy', 'Knife shot accuracy', 'Items picked up', 'Playtime',
               'Steps number', 'Distance', 'Avg time btw gun fires', 'Std time btw gun fires', 'Avg gun hit distance', 
               'Std gun hit distance', 'Avg time btw knife fires', 'Std time btw knife fires', 'Avg knife hit distance',
               'Std knife hit distance','Kills', 'Deaths', 'Assists', 'KD Ratio', 'KDA Ratio', 'Team velocity', 'Gun fire freq', 
               'Grenade fire freq', 'Knife fire freq', 'Jumps freq', 'ADR', 'Headshot',
               'Smoke coords', 'Smoke_ticks', 'Molotov coords', 'Flash blinded', 'Flash killed', 'Entry frag', 'Vest', 'Vesthelm', 'Wounded',
               'Distance between killer']

    data = []

    for csv_file in os.listdir(path):
        if csv_file[-3:] == 'csv':
            print(f'From {csv_file} was parsed')
            df = pd.read_csv('{}'.format(csv_file))
            stats = get_round_stat(df)
            for round_num in [0, 15]:

                curr_players = list(stats[round_num]['players'].keys())

                for player_id in curr_players:
                    gun_shot_accuracy = 0.
                    knife_shot_accuracy = 0.
                    items_num = 0.
                    playtime = 0.
                    steps_num = 0.
                    distance = 0.
                    avg_gun_fire_time = 0.
                    std_gun_fire_time = 0.
                    avg_gun_hit_dist = 0.
                    std_gun_hit_dist = 0.
                    avg_knife_fire_time = 0.
                    std_knife_fire_time = 0.
                    avg_knife_hit_dist = 0.
                    std_knife_hit_dist = 0.
                    velocity = 0.
                    gun_fire_freq = 0.
                    grenade_fire_freq = 0.
                    knife_fire_freq = 0.
                    jumps_freq = 0.
                    dmg_health = 0.
                    kills_round = 0.
                    deaths_round = 0.
                    assists_round = 0.
                    hs = 0.
                    smoke_coords = []
                    smoke_ticks = []
                    molotov_coords = []
                    flash_blinded = []
                    flash_killed = []
                    vest = 0
                    vesthelm = 0
                    wounded = 0
                    if 'gun_hit' in stats[round_num]['players'][player_id].keys() and stats[round_num]['players'][player_id]['gun_weapon_fire'] != 0:
                        gun_shot_accuracy = float(stats[round_num]['players'][player_id]['gun_hit']) / float(stats[round_num]['players'][player_id]['gun_weapon_fire'])
                        if len(stats[round_num]['players'][player_id]['gun_weapon_fire_ticks']) > 1:
                            avg_gun_fire_time = np.mean(stats[round_num]['players'][player_id]['gun_weapon_fire_ticks'])
                            std_gun_fire_time = np.std(stats[round_num]['players'][player_id]['gun_weapon_fire_ticks'])
                        if len(stats[round_num]['players'][player_id]['gun_hit_distance']) > 1:
                            avg_gun_hit_dist = np.mean(stats[round_num]['players'][player_id]['gun_hit_distance'])
                            std_gun_hit_dist = np.std(stats[round_num]['players'][player_id]['gun_hit_distance'])
                    if 'knife_hit' in stats[round_num]['players'][player_id].keys() and stats[round_num]['players'][player_id]['knife_weapon_fire'] != 0:
                        knife_shot_accuracy = float(stats[round_num]['players'][player_id]['knife_hit']) / float(stats[round_num]['players'][player_id]['knife_weapon_fire'])
                        if len(stats[round_num]['players'][player_id]['knife_weapon_fire_ticks']) > 1:
                            avg_knife_fire_time = np.mean(stats[round_num]['players'][player_id]['knife_weapon_fire_ticks'])
                            std_knife_fire_time = np.std(stats[round_num]['players'][player_id]['knife_weapon_fire_ticks'])
                        if len(stats[round_num]['players'][player_id]['knife_hit_distance']) > 1:
                            avg_knife_hit_dist = np.mean(stats[round_num]['players'][player_id]['knife_hit_distance'])
                            std_knife_hit_dist = np.std(stats[round_num]['players'][player_id]['knife_hit_distance'])


                    if 'item_pickup' in stats[round_num]['players'][player_id].keys():
                        items_num = stats[round_num]['players'][player_id]['item_pickup']
                    if 'playtime' in stats[round_num]['players'][player_id].keys():
                        playtime = stats[round_num]['players'][player_id]['playtime']
                    steps_num = stats[round_num]['players'][player_id]['footsteps']

                    for i in range(1, len(stats[round_num]['players'][player_id]['step position'])):
                        curr_pos = stats[round_num]['players'][player_id]['step position'][i - 1]
                        next_pos = stats[round_num]['players'][player_id]['step position'][i]
                        distance += np.linalg.norm(curr_pos - next_pos)
                    velocity = float(stats[round_num]['players'][player_id]['footsteps']) / float(stats[round_num]['round_len'])
                    if 'gun_weapon_fire' in stats[round_num]['players'][player_id].keys(): 
                        gun_fire_freq = stats[round_num]['players'][player_id]['gun_weapon_fire'] / float(stats[round_num]['round_len'])
                    if 'grenade_weapon_fire' in stats[round_num]['players'][player_id].keys(): 
                        grenade_fire_freq = stats[round_num]['players'][player_id]['grenade_weapon_fire'] / float(stats[round_num]['round_len'])
                    if 'knife_weapon_fire' in stats[round_num]['players'][player_id].keys(): 
                        knife_fire_freq = stats[round_num]['players'][player_id]['knife_weapon_fire'] / float(stats[round_num]['round_len'])
                    if 'jumps' in stats[round_num]['players'][player_id].keys(): 
                        jumps_freq = stats[round_num]['players'][player_id]['jumps'] / float(stats[round_num]['round_len'])
                    if 'dmg_health' in stats[round_num]['players'][player_id].keys():
                        dmg_health = stats[round_num]['players'][player_id]['dmg_health'] / float(stats[round_num]['round_len'])
                    if 'smoke_coords' in stats[round_num]['players'][player_id].keys():
                        smoke_coords = stats[round_num]['players'][player_id]['smoke_coords']
                    if 'smoke_ticks' in stats[round_num]['players'][player_id].keys():
                        smoke_ticks = stats[round_num]['players'][player_id]['smoke_ticks']

                    if 'molotov_coords' in stats[round_num]['players'][player_id].keys():
                        molotov_coords = stats[round_num]['players'][player_id]['molotov_coords']

                    if 'flash_blinded' in stats[round_num]['players'][player_id].keys():
                        flash_blinded = stats[round_num]['players'][player_id]['flash_blinded']
                    if 'flash_killed' in stats[round_num]['players'][player_id].keys():
                        flash_killed = stats[round_num]['players'][player_id]['flash_killed']
                    if 'entry_frag' in stats[round_num]['players'][player_id].keys():
                        entry_frag = stats[round_num]['players'][player_id]['entry_frag']
                    if 'vest' in stats[round_num]['players'][player_id].keys():
                        vest = stats[round_num]['players'][player_id]['vest']
                    if 'vesthelm' in stats[round_num]['players'][player_id].keys():
                        vesthelm = stats[round_num]['players'][player_id]['vesthelm']
                    if 'wounded' in stats[round_num]['players'][player_id].keys():
                        wounded = stats[round_num]['players'][player_id]['wounded']


                    kills_round = kills_deaths_assists_per_round(stats, player_id, round_num)[0]
                    deaths_round = kills_deaths_assists_per_round(stats, player_id, round_num)[1]
                    assists_round = kills_deaths_assists_per_round(stats, player_id, round_num)[2]
                    hs = kills_deaths_assists_per_round(stats, player_id, round_num)[3]
                    round_len = stats[round_num]['round_len']

                    if deaths_round != 0.:
                        kd = kills_round / deaths_round
                    else:
                        kd = kills_round 
                    kda = (kills_round + assists_round) / max(1., deaths_round)



                    side = stats[round_num]['players'][player_id]['side']
                    dist_btw_killer = stats[round_num]['players'][player_id]['distance between killer']
                    
                    row = [player_id, side, round_num, round_len, gun_shot_accuracy, knife_shot_accuracy, items_num, playtime, 
                           steps_num, distance, avg_gun_fire_time, std_gun_fire_time, avg_gun_hit_dist, std_gun_hit_dist,
                           avg_knife_fire_time, std_knife_fire_time, avg_knife_hit_dist, std_knife_hit_dist, kills_round, 
                           deaths_round, assists_round, kd, kda, velocity, gun_fire_freq, grenade_fire_freq, 
                           knife_fire_freq, jumps_freq, dmg_health, hs,
                            smoke_coords,smoke_ticks, molotov_coords, flash_blinded, 
                           flash_killed, entry_frag, vest, vesthelm, wounded, dist_btw_killer]

                    data.append(dict(zip(columns, row)))

    return pd.DataFrame(data, columns = columns)