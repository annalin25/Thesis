# packages

import torch
import math
import pandas as pd
import numpy as np
import json
import random


from score_new import *

# load data
# train-dataset
rd = open('dataset/train/route_data.json')
tt = open('dataset/train/travel_times.json')
aseq = open('dataset/train/actual_sequences.json')

train_route = json.load(rd)
train_time = json.load(tt)
train_seq = json.load(aseq)

# load model

def group_in_zone(route):
    """
    the zone time matrix in ONE route
    """
    # route = data_route['RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77']

    # set a dict, res={stop_id: zone_id}
    res = {}
    for k in route['stops']:
        res.setdefault(k,)

    for i in res:
        x = route['stops'][i]['zone_id']
        if str(x) == "nan":
            res[i] = 'depot'
        else:
            res[i] = x
    
    # the stop with same zone_id is listed together
    res_new = {} 
    for i,j in res.items():
        if j not in res_new:
            res_new[j] = [i]
        else:
            res_new[j].append(i)
    del res_new['depot'] 
    return res_new

def one_pair_zone_score(m,n,t_matrix):
    score = 0
    count = 0
    if m != n:
        for i in m:
            for j in n:
                score += t_matrix[i][j]
                count += 1

        for i in n:
            for j in m:
                score += t_matrix[i][j]
                count += 1
        score = np.round(score/count,4)
    
    return score

def find_pairs_score(df, group, t_matrix):
    for i in df.columns:
        for j in df.index:
            s = one_pair_zone_score(group[i],group[j],t_matrix)
            df[i][j] = s
    return df

def zone_travel_time(route, route_time):
    # sample_route = data_time['RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77']
    sample_route_matrix = pd.DataFrame(route_time)

    res_new = group_in_zone(route)
    
    tt = {}
    for i in res_new:
        tt.setdefault(i,)
    tt = {k: 0 if not v else v for k, v in tt.items() }
    tt2 = tt.copy()

    for j in tt2:
        tt2[j] = tt

    tt3 = pd.DataFrame(tt2)

    return find_pairs_score(tt3, res_new, sample_route_matrix)

def remove_duplicates(input_list):
    """input a list of (zone_id) order, return the remove duplicates & nan list"""
    unique_list = []
    seen = set()

    for item in input_list:
        if item not in seen:
            unique_list.append(item)
            seen.add(item)
    
    unique_list = [item for item in unique_list if not (isinstance(item, float) and math.isnan(item))]

    return unique_list

def zone_mat_order_pad2(route_id, mode, max=48): # stop_id_2index, cost_mat_2index
    """ input:
            route_id = 'RouteID_64cb7ba5-342d-46db-9e04-962248c6f667'
            mode = train or test set
        return: 
            zone time_matrix, order (index format)
    """

    # single route    
    if mode == 'train':
        sample_route = train_route[route_id]
        sample_time = train_time[route_id]
        sample_order = train_seq[route_id]
    elif mode == 'test':
        sample_route = test_route[route_id]
        sample_time = test_time[route_id]
        sample_order = test_seq[route_id]

    # stop order & time_matrix (index format) 
    # sample_order_index = stop_id_2index(sample_order['actual'])
    # sample_time_index = cost_mat_2index(sample_order['actual'], sample_time)

    # for zone
    sorted_list = sorted(sample_order['actual'].items(), key=lambda x: x[1])
    stop_order_in_id = [i[0] for i in sorted_list]

    zone_order_in_id = []
    for i in stop_order_in_id:
        zone_order_in_id.append(sample_route['stops'][i]['zone_id'])
        
    # remove duplicates in zone_order_in_id 
    unique_list = remove_duplicates(zone_order_in_id)

    # {zone_id: index}
    zone_time = zone_travel_time(sample_route, sample_time)

    list_len = [i for i in range(len(zone_time))]
    list_value = list(zone_time.index.values)

    route_index = {k:v for k,v in zip(list_value, list_len)} 

    # {order: zone_index} = {order : (zone_id's index)2,11,5,1,7,0,8,4,3,10,9,6}
    list_len = [i for i in range(len(zone_time))]
    list_zone = [route_index[i] for i in unique_list] # 2,11,5,1,7,0,8,4,3,10,9,6

    zone_order_index = {k:v for k,v in zip(list_len, list_zone)} 

    # zone_time_matrix (index, dataframe)
    zone_time = zone_time.rename(columns=route_index)
    zone_time = zone_time.rename(index=route_index)

    # padding
    zone_time2 = torch.tensor(zone_time.values, dtype=torch.float32)

    n = len(zone_time)
    new_zone_mat = torch.zeros(max,max)
    new_zone_mat[:n, :n] = zone_time2

    dummy = [i for i in range(n,max)]
    actual_seq = list_zone + dummy
    
    # return zone_time, list_zone # before padding
    return new_zone_mat, actual_seq


# 2-stage

def calculate_time_in_zone(stops_in_1zone, route_id, mode): # stops_in_1zone=stop_find_in_zone
    # already get the route, time, order, etc in a certain route (no need to input a route id)
    """
    input: 
        stops_in_1zone: list. stops, eg.['KM', 'CO', 'IA', 'UI', 'EO', 'KG', 'BG'] in one zone (from a single route)
    output:
        res: int. the total time of these stops, with the input order.
    """
    if mode == 'train':
        sample_time = train_time[route_id]
    elif mode == 'test':
        sample_time = test_time[route_id]
    res = 0
    for i in range(len(stops_in_1zone)-1):
        res += sample_time[stops_in_1zone[i]][stops_in_1zone[i+1]] 

    return res

def three_opt_swap(tour, i, j, k):
    """Perform the 3-opt swap on the tour."""
    if i < k:
        return tour[:i + 1] + tour[j:k + 1] + tour[i + 1:j][::-1] + tour[k + 1:]
    else:
        return tour[:k + 1] + tour[j:i + 1][::-1] + tour[k + 1:j + 1][::-1] + tour[i + 1:]

def three_opt(route, distance_matrix, route_id, mode):
    """Apply the 3-opt algorithm to improve the tour."""
    n = len(route)
    best_distance = calculate_time_in_zone(route, route_id, mode) # , distance_matrix
    improvement = True

    while improvement:
        improvement = False

        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    # Perform the 3-opt swap
                    new_route = three_opt_swap(route, i, j, k)
                    new_distance = calculate_time_in_zone(new_route, route_id, mode) # , distance_matrix

                    # Check if the new tour is shorter
                    if new_distance < best_distance:
                        route = new_route
                        best_distance = new_distance
                        improvement = True

    return route, best_distance

def random_seq(route, route_id, mode):
    temp = random.sample(route, len(route))
    time = calculate_time_in_zone(temp, route_id, mode)

    return temp, time

def second_stage(tour_zone, route_id, method, mode): # method = '2opt' ,'3opt', 'random', 'brute_force'

    id = route_id
    if mode == 'train':    
        sample_route = train_route[id]
        sample_time = train_time[id]
        sample_order = train_seq[id]
    elif mode == 'test':
        sample_route = test_route[id]
        sample_time = test_time[id]
        sample_order = test_seq[id]


    ### ---- for zone ----
    sorted_list = sorted(sample_order['actual'].items(), key=lambda x: x[1])
    stop_order_in_id = [i[0] for i in sorted_list]

    zone_order_in_id = []
    for i in stop_order_in_id:
        zone_order_in_id.append(sample_route['stops'][i]['zone_id'])

    # remove duplicates in zone_order_in_id 
    # unique_list = remove_duplicates(zone_order_in_id)

    # {zone_id: index}
    zone_time = zone_travel_time(sample_route, sample_time)

    list_len = [i for i in range(len(zone_time))]
    list_value = list(zone_time.index.values)

    route_index = {k:v for k,v in zip(list_len, list_value)} 

    # get the zone seq in ids
    zone_seq_ids = []
    for i in tour_zone:
        if i.item() in route_index.keys():
            zone_seq_ids.append(route_index[i.item()])
    # print(zone_seq_ids)

    # find stop_ids in each zone
    initial_seq = []
    for zone_id in zone_seq_ids:
        stop_find_in_zone = []
        for i in (stop_order_in_id):
            if sample_route['stops'][i]['zone_id'] == zone_id:
                stop_find_in_zone.append(i)
        initial_seq.append(stop_find_in_zone)
    # print(initial_seq)
    
    # return initial_seq


    #------------------------ different methods ---------------------------
    final_no_depot = []
    for zone in initial_seq:
        if method == '2opt':
            temp_stop, temp_time = two_opt(zone, sample_time, route_id, mode)
        elif method == '3opt':
            temp_stop, temp_time = three_opt(zone, sample_time, route_id, mode)
        elif method == 'random':
            temp_stop, temp_time = random_seq(zone, route_id, mode)
        elif method == 'brute_force':
            temp_stop, temp_time = tsp_brute_force(zone, route_id, mode)
        
        # print(temp_stop, temp_time)
        final_no_depot.append(temp_stop)

    flat_list = [item for sublist in final_no_depot for item in sublist]

    # find the id of depot
    for id in sample_route['stops']:
        if sample_route['stops'][id]['type'] == 'Station':
            # print(id)
            depot = id
    final_route = [depot] + flat_list
    # print(len(final_route))

    # check the length, some stop has zone_id 'nan'
    if len(final_route) != len(sample_route['stops']):
        for i in sample_route['stops']:
            if type(sample_route['stops'][i]['zone_id']) != str and sample_route['stops'][i]['type'] != 'Station':
                missing = i
                final_route = final_route + [missing]

    return final_route

def name_later(tour_indices, route_id, method, mode): # '3opt', 'train'
    res = []
    for i in range(len(tour_indices)):
        if mode == 'train':
            sub = second_stage(tour_indices[i], route_id[i], method, mode)

            x = train_seq[route_id[i]]['actual']
            x2 = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
            act = list(x2.keys())

            sample_time = train_time[route_id[i]]
            s = score(act, sub, sample_time)
            res.append(s)
        elif mode == 'test':
            sub = second_stage(tour_indices[i], route_id[i], method, mode)

            x = test_seq[route_id[i]]['actual']
            x2 = {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
            act = list(x2.keys())

            sample_time = test_time[route_id[i]]
            s = score(act, sub, sample_time)
            res.append(s)
    return res

def zone_mat_temp(route_id, mode): # stop_id_2index, cost_mat_2index
    """ input:
            route_id = 'RouteID_64cb7ba5-342d-46db-9e04-962248c6f667'
            mode = train or test set
        return: 
            zone time_matrix, order (index format)
    """

    # single route    
    if mode == 'train':
        sample_route = train_route[route_id]
        sample_time = train_time[route_id]
        sample_order = train_seq[route_id]
    elif mode == 'test':
        sample_route = test_route[route_id]
        sample_time = test_time[route_id]
        sample_order = test_seq[route_id]

    # stop order & time_matrix (index format) 
    # sample_order_index = stop_id_2index(sample_order['actual'])
    # sample_time_index = cost_mat_2index(sample_order['actual'], sample_time)

    # for zone
    sorted_list = sorted(sample_order['actual'].items(), key=lambda x: x[1])
    stop_order_in_id = [i[0] for i in sorted_list]

    zone_order_in_id = []
    for i in stop_order_in_id:
        zone_order_in_id.append(sample_route['stops'][i]['zone_id'])
        
    # remove duplicates in zone_order_in_id 
    unique_list = remove_duplicates(zone_order_in_id)

    # {zone_id: index}
    zone_time = zone_travel_time(sample_route, sample_time)

    list_len = [i for i in range(len(zone_time))]
    list_value = list(zone_time.index.values)

    route_index = {k:v for k,v in zip(list_value, list_len)} 

    # {order: zone_index} = {oder : (zone_id's index)2,11,5,1,7,0,8,4,3,10,9,6}
    list_len = [i for i in range(len(zone_time))]
    list_zone = [route_index[i] for i in unique_list] # 2,11,5,1,7,0,8,4,3,10,9,6

    zone_order_index = {k:v for k,v in zip(list_len, list_zone)} 

    # zone_time_matrix (index, dataframe)
    zone_time = zone_time.rename(columns=route_index)
    zone_time = zone_time.rename(index=route_index)

    return zone_time

def cal_time(tour_indices, route_id, mode):
    res_fin = []
    for t in range(tour_indices.size()[0]):
        time_mat = zone_mat_temp(route_id[t], mode)

        true_zones = [i for i in range(len(time_mat))]

        sub = tour_indices.tolist()[t]
        sub2 = []
        for i in sub:
            if i in true_zones:
                sub2.append(i)
        # print(sub2)

        res = 0
        for i in range(len(sub2)-1):
            res += time_mat[sub2[i]][sub2[i+1]]

        res_fin.append(res)

    return res_fin

# train
def zoneid_weight2(zone_label1, zone_label2):
    """
    input: two zone_id name
    output: the difference between the two
    """

    # x = unique_list[13] # E-24.1E
    # print(x)
    x = zone_label1
    macro = x.split('.')[0] # E-24
    micro = x.split('.')[1] # 1E
    macro_0 = macro.split('-')[0]
    macro_1 = macro.split('-')[1]

    # x2 = unique_list[14]
    # print(x2)
    x2 = zone_label2
    macro2 = x2.split('.')[0]
    micro2 = x2.split('.')[1]
    macro2_0 = macro2.split('-')[0]
    macro2_1 = macro2.split('-')[1]

    # macro
    if macro_0 == macro2_0:
      w0 = 0
    else:
      w0 = 1
    
    w1 = w0 + abs(int(macro_1)-int(macro2_1))
    # w1 = abs(ord(macro_0)-ord(macro2_0)) + abs(int(macro_1)-int(macro2_1))

    # micro
    w2 = abs(int(micro[0])-int(micro2[0])) + abs(ord(micro[1])-ord(micro2[1]))

    # print('weight:', w1+w2)
    return w1+w2


def zone_mat_temp2(route_id, mode): # stop_id_2index, cost_mat_2index
    """ input:
            route_id = 'RouteID_64cb7ba5-342d-46db-9e04-962248c6f667'
            mode = train or test set
        return: 
            zone time_matrix, order (index format)
    """

    # single route    
    if mode == 'train':
        sample_route = train_route[route_id]
        sample_time = train_time[route_id]
        sample_order = train_seq[route_id]
    elif mode == 'test':
        sample_route = test_route[route_id]
        sample_time = test_time[route_id]
        sample_order = test_seq[route_id]

    # stop order & time_matrix (index format) 
    # sample_order_index = stop_id_2index(sample_order['actual'])
    # sample_time_index = cost_mat_2index(sample_order['actual'], sample_time)

    # for zone
    sorted_list = sorted(sample_order['actual'].items(), key=lambda x: x[1])
    stop_order_in_id = [i[0] for i in sorted_list]

    zone_order_in_id = []
    for i in stop_order_in_id:
        zone_order_in_id.append(sample_route['stops'][i]['zone_id'])
        
    # remove duplicates in zone_order_in_id 
    unique_list = remove_duplicates(zone_order_in_id)

    # {zone_id: index}
    zone_time = zone_travel_time(sample_route, sample_time)
    zone_time2 = zone_time.copy()
    for i in zone_time2.columns:
        for j in zone_time2.index:
            zone_time2[i][j] = zone_time2[i][j] * zoneid_weight2(i,j)

    list_len = [i for i in range(len(zone_time2))]
    list_value = list(zone_time2.index.values)

    route_index = {k:v for k,v in zip(list_value, list_len)} 

    # {order: zone_index} = {order : (zone_id's index)2,11,5,1,7,0,8,4,3,10,9,6}
    list_len = [i for i in range(len(zone_time2))]
    list_zone = [route_index[i] for i in unique_list] # 2,11,5,1,7,0,8,4,3,10,9,6

    zone_order_index = {k:v for k,v in zip(list_len, list_zone)} 

    # zone_time_matrix (index, dataframe)
    zone_time2 = zone_time2.rename(columns=route_index)
    zone_time2 = zone_time2.rename(index=route_index)

    return zone_time2

def customize_time_mat(tour_indices, route_id, mode):

    res_fin = []
    for t in range(tour_indices.size()[0]):
        # time_mat = zone_mat_temp(route_id[t], mode)
        time_mat = zone_mat_temp2(route_id[t], mode)

        true_zones = [i for i in range(len(time_mat))]

        sub = tour_indices.tolist()[t]
        sub2 = []
        for i in sub:
            if i in true_zones:
                sub2.append(i)
        # print(sub2)

        res = 0
        for i in range(len(sub2)-1):
            res += time_mat[sub2[i]][sub2[i+1]]

        res_fin.append(res)

    return res_fin





