#from sklearn.metrics.pairwise import euclidean_distances as distance_fn
from sklearn.metrics import pairwise_distances as distance_fn
from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np

def get_cost_matrix(v1, v2, max_distance_limit, weight):
    assert len(v2)>0 and len(v1)>0, "len(v2)==0 or len(v1)==0"
    cost_matrix = distance_fn(v1, v2)
    cost_matrix[cost_matrix>max_distance_limit] = 10*5 # Assign a large number
    return cost_matrix*weight
    
def munkres_algo(cost_matrix):
    matches = linear_assignment(cost_matrix)  
    return matches

def munkres_algo_w_match(v1_list, v2_list, 
                         max_distance_limit_list, weight_list,
                         map_role_for_v2, num_role_register): #欲求得 v1 的 role
    assert len(v1_list)==len(v2_list) 
    assert len(v1_list)==len(max_distance_limit_list) 
    # get_cost_matrix
    cost_matrix_list = []
    for v1, v2, mdl, w in zip(v1_list, v2_list, max_distance_limit_list, weight_list):
        cost_matrix_list.append(get_cost_matrix(v1, v2, mdl, w))# list of shape (n,m)
    cost_matrix = np.sum(cost_matrix_list,0)#(n,m)
    
    n, m = v1.shape[0], v2.shape[0]
    assert len(v2)==len(map_role_for_v2), "len(v2)!=len(map_role_for_v2)"     
    matches = munkres_algo(cost_matrix) 
    map_role_for_v1 = np.empty((n,))
    map_role_for_v1[matches[:,0]] = map_role_for_v2[matches[:,1]]#舊角色出現的位置 補上值
    if n>m: #若有新角色
        complement = list(set(range(n)).difference(set(matches[:,0])))#新角色出現的位置
        n_c = len(complement) # same as (n-m)
        map_role_for_v1[complement] = np.array(range(num_role_register,num_role_register+n_c))
        num_role_register += n_c
    
    return map_role_for_v1, num_role_register, cost_matrix


    