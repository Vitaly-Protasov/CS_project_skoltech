import numpy as np
import pandas as pd
import os
import ast

def fill_dict(round_len):
    d = {'playtime': round_len, 'was_killed_by': set([]), 'killed': set([]), 
                                 'assisted_to_kill': set([]), 'headshot': 0, 'footsteps': 0, 'step position': [], 
                                 'jumps': 0, 'gun_weapon_fire': 0,  'gun_weapon_fire_ticks': [], 'grenade_weapon_fire': 0, 
                                 'knife_weapon_fire': 0,  'knife_weapon_fire_ticks': [],
                                 'gun_hit': 0, 'gun_hit_distance': [], 'knife_hit': 0, 'knife_hit_distance': [], 
                                 'dmg_health': 0, 'item_pickup': 0, 'side': str(), 'smoke_coords': [], 
                                 'molotov_coords': [], 'flash_blinded': [], 'flashed_to_kill': [],
                                 'entry_frag': False, 'vest': 0, 'vesthelm': 0, 'wounded': 0}
    return d
   
def rounds_start_end(df, start, end):
    start_idx = 0
    start_end_indices = []
   
    for end_idx in df['event'][df['event'] == end].index:
        df_tmp = df.iloc[start_idx:end_idx]
        if start in df_tmp['event'].values:
            start_end_indices.append((start_idx, df_tmp[df_tmp['event'] == start].index[-1], end_idx))
        start_idx = end_idx
    return start_end_indices

def get_round_stat(df):
    rounds = []
    pstart_oended = rounds_start_end(df, 'round_freeze_end', 'round_officially_ended')
    
    for fs, rs, re in pstart_oended:
        df_buy_round = df.loc[fs:rs]
        df_round = df.loc[rs:re]
        players = {}
        
        round_len = df_round['tick'].iloc[len(df_round) - 1] - df_round['tick'].iloc[0]
        T_entry_frag = False
        CT_entry_frag = False
        df_round_player_death = df_round[df_round['event'] == 'player_death']
        for item in df_round_player_death['parameters']:
            item_d = ast.literal_eval(item) 
            user = item_d['userid'].split(' (id:')[0]
            attacker = item_d['attacker'].split(' (id:')[0]
            assister = item_d['assister'].split(' (id:')[0]
            headshot = item_d['headshot']
            if user not in players:
                players[user] = fill_dict(round_len)
            if attacker not in players:
                players[attacker] = fill_dict(round_len)
            
            if assister != '0' and assister not in players:
                players[assister] = fill_dict(round_len)

            players[user]['was_killed_by'].add(attacker)
            players[user]['playtime'] = df_round_player_death[df_round_player_death['parameters'] == item]['tick'].values[0] - df_round['tick'].iloc[0]
            players[attacker]['killed'].add(user)
            if headshot != '0':
                players[attacker]['headshot'] += 1
            if assister != '0':
                players[assister]['assisted_to_kill'].add(user)
                
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side
            if (T_entry_frag == False) or (CT_entry_frag == False):
                if players[attacker]['side'] == 'T':
                    if T_entry_frag == False:
                        players[attacker]['entry_frag'] = True
                        T_entry_frag = True
                if players[attacker]['side'] == 'CT':
                    if CT_entry_frag == False:
                        players[attacker]['entry_frag'] = True
                        CT_entry_frag = True
                        
            players[attacker]['wounded'] += 1

       
        df_round_player_footstep = df_round[df_round['event'] == 'player_footstep']
        for item in df_round_player_footstep['parameters']:
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            
            if user not in players:
                players[user] = fill_dict(round_len)

            players[user]['footsteps'] += 1
            if 'userid position' in item_d.keys():
                players[user]['step position'].append(np.array(list(map(float, item_d['userid position'].split(', ')))))
            
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side
            
        df_round_player_jump = df_round[df_round['event'] == 'player_jump']
        for item in df_round_player_jump['parameters']:
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            if user not in players:
                players[user] = fill_dict(round_len)

            players[user]['jumps'] += 1
            
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side
            
        df_round_player_weapon_fire = df_round[df_round['event'] == 'weapon_fire']
        for tick, item in zip(df_round_player_weapon_fire['tick'], df_round_player_weapon_fire['parameters']):
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            if user not in players:
                players[user] = fill_dict(round_len)
  
            if 'knife' not in item_d['weapon'] and 'grenade' not in item_d['weapon']:    
                players[user]['gun_weapon_fire'] += 1
                players[user]['gun_weapon_fire_ticks'].append(df_round_player_weapon_fire[df_round_player_weapon_fire['parameters'] == item]['tick'].values[0])
            if 'knife' in item_d['weapon']:
                players[user]['knife_weapon_fire'] += 1
                players[user]['knife_weapon_fire_ticks'].append(df_round_player_weapon_fire[df_round_player_weapon_fire['parameters'] == item]['tick'].values[0])
            if 'grenade' in item_d['weapon']:
                players[user]['grenade_weapon_fire'] += 1

            if item_d['weapon'] == 'weapon_molotov':
                players[user]['molotov_coords'].append(np.array(list(map(float, item_d['userid position'].split(', ')))))
                
            
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side

        df_round_player_hurt = df_round[df_round['event'] == 'player_hurt']
        for item in df_round_player_hurt['parameters']:
            item_d = ast.literal_eval(item)
            attacker = item_d['attacker'].split(' (id:')[0]
            if 'attacker position' in item_d.keys():
                attacker_position = np.array(list(map(float, item_d['attacker position'].split(', '))))
            else:
                attacker_position = np.array([0., 0., 0.])
                
            if 'userid position' in item_d.keys():
                victim_position = np.array(list(map(float, item_d['userid position'].split(', '))))
            else:
                victim_position = np.array([0., 0., 0.])
            if attacker not in players:
                players[attacker] = fill_dict(round_len)
  
            players[attacker]['dmg_health'] += int(item_d['dmg_health'])
            
            if 'knife' not in item_d['weapon'] and 'grenade' not in item_d['weapon']:
                players[attacker]['gun_hit'] += 1
                players[attacker]['gun_hit_distance'].append(np.linalg.norm(attacker_position-victim_position))
            if 'knife' in item_d['weapon']:
                players[attacker]['knife_hit'] += 1
                players[attacker]['knife_hit_distance'].append(np.linalg.norm(attacker_position-victim_position))
                
            if 'attacker team' in item_d.keys() and players[attacker]['side'] == '':
                players[attacker]['side'] = item_d['attacker team']

            players[attacker]['wounded'] += 1

            
        df_round_item_pickup = df_round[df_round['event'] == 'item_pickup']
        for item in df_round_item_pickup['parameters']:
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            if user not in players:
                players[user] = fill_dict(round_len)

            players[user]['item_pickup'] += 1
            
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side

        df_round_player_smoke_deton = df_round[df_round['event'] == 'smokegrenade_detonate']
        for tick, item in zip(df_round_player_smoke_deton['tick'], df_round_player_smoke_deton['parameters']):
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            if user not in players:
                players[user] = fill_dict(round_len)
            
            players[user]['smoke_coords'].append(np.array([float(item_d['x']), float(item_d['y']), float(item_d['z'])]))

            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side
                  
        df_round_flash_detonate = df_round[df_round['event'] == 'flashbang_detonate']
        indices = df_round_flash_detonate.index.tolist()
        for i, item in enumerate(df_round_flash_detonate['parameters']):
            item_d = ast.literal_eval(item)
            user = item_d['userid'].split(' (id:')[0]
            if user not in players:
                players[user] = fill_dict(round_len)

            if i < len(indices) - 1:
                curr_flash_df = df_round.loc[indices[i]:indices[i + 1]]
            else:
                curr_flash_df = df_round.loc[indices[i]:]

            curr_blinded_df = curr_flash_df[curr_flash_df['event'] == 'player_blind']
            curr_killed_df = curr_flash_df[curr_flash_df['event'] == 'player_death']
            
            blinded_list = []
            for blinded_item in curr_blinded_df['parameters']:
                blinded_item_d = ast.literal_eval(blinded_item)
                if 'userid team' in blinded_item_d.keys(): 
                    blinded_list.append(blinded_item_d['userid'].split(' (id:')[0])
            
            killed_list = []
            for killed_item in curr_killed_df['parameters']:
                killed_item_d = ast.literal_eval(killed_item)
                killed_list.append(killed_item_d['userid'].split(' (id:')[0])
            
            if 'userid' in item_d.keys():
                players[user]['flash_blinded'] += blinded_list
            if 'userid' in item_d.keys():
                players[user]['flashed_to_kill'] += killed_list
            if 'userid team' in item_d.keys():
                user_side = item_d['userid team']
                players[user]['side'] = user_side

        
        for event, item in zip(df_buy_round['event'], df_buy_round['parameters']):
            if event == 'item_pickup':                
                item_d = ast.literal_eval(item)
                user = item_d['userid'].split(' (id:')[0]
                if user not in players:
                    players[user] = fill_dict(round_len)

                if item_d['item'] == 'vest':
                    players[user]['vest'] += 1
                if item_d['item'] == 'vesthelm':
                    players[user]['vesthelm'] += 1
                if 'userid team' in item_d.keys():
                    user_side = item_d['userid team']
                    players[user]['side'] = user_side

                    
            if event == 'item_equip':                
                item_d = ast.literal_eval(item)
                user = item_d['userid'].split(' (id:')[0]
                if user not in players:
                    players[user] = fill_dict(round_len)

                if item_d['item'] == 'vest':
                    players[user]['vest'] += 1
                if item_d['item'] == 'vesthelm':
                    players[user]['vesthelm'] += 1
                if 'userid team' in item_d.keys():
                    user_side = item_d['userid team']
                    players[user]['side'] = user_side

            
            if event == 'item_remove':                
                item_d = ast.literal_eval(item)
                user = item_d['userid'].split(' (id:')[0]
                if user not in players:
                    players[user] = fill_dict(round_len)
                
                if item_d['item'] == 'vest':
                    players[user]['vest'] -= 1
                if item_d['item'] == 'vesthelm':
                    players[user]['vesthelm'] -= 1
                if 'userid team' in item_d.keys():
                    user_side = item_d['userid team']
                    players[user]['side'] = user_side

        
        cround = {'round_len': round_len,
                  'round_start': rs,
                  'round_officially_ended': re,
                  'round_start_tick': df['tick'].loc[rs],
                  'round_officially_ended_tick': df['tick'].loc[re],
                  'player_death_num': len(df_round_player_death),
                  'players': players}
        

        rounds.append(cround)
    return rounds
    
def kills_deaths_assists_per_round(stats, player, round_num):
    kills = 0.
    deaths = 0.
    assists = 0.
    hs = 0.
    player_info = stats[round_num]['players'][player]
    if 'killed' in player_info.keys():
        kills = len(player_info['killed'])
    if 'was_killed_by' in player_info.keys():
        deaths = len(player_info['was_killed_by'])
    if 'assisted_to_kill' in player_info.keys():
        assists = len(player_info['assisted_to_kill'])
    if 'headshot' in player_info.keys():
        hs = player_info['headshot']

    return kills, deaths, assists, hs
