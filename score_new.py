# https://github.com/MIT-CAVE/rc-cli/blob/main/scoring/score.py
import numpy as np


def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''
    norm_mat=normalize_matrix(cost_mat)
    if isinvalid(actual,sub):
        actual = valid_route(actual)
        sub = valid_route(sub)

    return seq_dev(actual,sub)*erp_per_edit(actual,sub,norm_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : dict
        Cost matrix.

    Returns
    -------
    new_mat : dict
        Normalized cost matrix.

    '''
    new_mat=mat.copy()
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat

def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.

    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.

    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.

    '''
    res=0
    for p in path:
        res+=g
    return res

def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : str
        ID of point.
    p_2 : str
        ID of other point.
    mat : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''
    # print("p_1", p_1)
    # print("p_2", p_2)
    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist = mat[p_1][p_2]
    return dist

def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    actual=actual[1:-1] 
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum

def isinvalid0(actual,sub):
    '''
    Checks if submitted route is invalid.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    bool
        True if route is invalid. False otherwise.

    '''
    if len(actual)!=len(sub) or set(actual)!=set(sub):
        return True
    elif actual[0]!=sub[0]:
        return True
    else:
        return False
    
def isinvalid(actual,sub):
    '''
    Checks if submitted route is invalid.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    bool
        True if route is invalid. False otherwise.

    '''
    if len(actual) != len(sub):
        print("the two routes have different length.")
    elif set(actual)!=set(sub):
        print("the stop_id in two routes are different.")
    elif actual[0] != sub[0]:
        print("the predict route has differ depot !!!")
    else:
        return True

def valid_route(input_list):
    temp = input_list[0]
    input_list.append(temp)
    return input_list
