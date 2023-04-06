import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.special import comb, perm
from itertools import combinations, permutations
import sys
import random
import argparse
import numpy as np

def torch_to_list(input):
    if torch.is_tensor(input): 
        input = input.cpu().numpy().tolist()   
    return input

def get_order_list(input):
    '''
    input: a list of float 
    eg: [7.5, -3.5, 2.5]

    output: the order of the float
    eg: [2, 0, 1]
    '''
    if torch.is_tensor(input): input = input.cpu().numpy().tolist()
    ordered = sorted(input)
    output = []
    for i in input:
        for idx, j in enumerate(ordered):
            if i == j:
                output.append(idx)
                break

    return output

def get_order_index(input):
    '''
    input: a order_idx list 
    eg: [2, 0, 1]

    ouput: the position of 0st 1st 2nd 3rd ...
    eg: [1, 2, 0]
    '''
    sorted_nums = sorted(enumerate(input), key=lambda x:x[1])
    output = [i[0] for i in sorted_nums]

    return output

def rSublist(arr, length):
    '''
    return list of all subsets of length r
    '''
    return list(combinations(arr, length))

def is_sublist(shortlist, longlist):
    '''
    Whether shortlist is a subset of longlist, the elements should have same order
    '''

    assert False, ''

    jdx = 0
    for i in range(len(shortlist)):
        flag = False
        for j in range(jdx, len(longlist)):
            if shortlist[i] == longlist[j]:
                jdx = j+1
                flag = True
                continue
        if not flag:
            return False
    return True

def ArbitraryLengthMatching(Pred, GT=[]):
    '''
    input: Pred list is from get_order_list(), normally GT is [0, 1, 2, ...]
    output: the score give to the Pred, score is from 0 to 1
    eg: [0, 1, 3, 2]
    for len(subset) = 1 , subscore = 4 / 4
    for len(subset) = 2 , subscore = 5 / 6
    for len(subset) = 3 , subscore = 2 / 4
    for len(subset) = 4 , subscore = 0 / 1
    If we do not calculate len=1, 
        we have the score (5*2 + 2*3 + 0*4) / (6*2 + 4*3 + 1*4) = 16 / 28 = 4 / 7 = 0.5714
        # or score = (5+2+0) / (6+4+1) = 7 / 11
    '''
    N = len(Pred)
    score_num = 0 #numerator
    score_deno = 0 #denominator
    if GT == [] : GT = [i for i in range(N)]

    assert False, 'A bug have not finished' # TODO

    for length in range(2, N+1):
        for i in rSublist(Pred, length):
            if is_sublist(i, GT):
                score_num += length
            score_deno += length 

    score = score_num / score_deno
    return score

def TripleLengthMatching(Pred, GT=[]):
    '''
    Only consider len(subset) = 3
    eg: [0, 1, 3, 2]
    for len(subset) = 3 , subscore = 2 / 4
    we have score = 0.5
    '''
    N = len(Pred)
    score_num = 0 #numerator
    score_deno = 0 #denominator   
    if GT == [] : GT = [i for i in range(N)]
    if torch.is_tensor(Pred): Pred = Pred.cpu().numpy().tolist()
    if torch.is_tensor(GT): GT = GT.cpu().numpy().tolist()

    assert(N >= 3)


    Pred_Sublist = rSublist(Pred, 3)
    GT_Sublist = rSublist(GT, 3)
    
    score_num = len(set(Pred_Sublist) & set(GT_Sublist))
    score_deno = len(Pred_Sublist)

    score = score_num / score_deno
    return score

def DoubleLengthMatching(Pred, GT=[]):
    '''
    Only consider len(subset) = 2
    eg: [0, 1, 3, 2]
    for len(subset) = 2 , subscore = 5 / 6
    we have score = 0.83
    '''
    N = len(Pred)
    score_num = 0 #numerator
    score_deno = 0 #denominator   
    if GT == [] : GT = [i for i in range(N)]
    if torch.is_tensor(Pred): Pred = Pred.cpu().numpy().tolist()
    if torch.is_tensor(GT): GT = GT.cpu().numpy().tolist()

    assert(N >= 2)

    Pred_Sublist = rSublist(Pred, 2)
    GT_Sublist = rSublist(GT, 2)
    
    score_num = len(set(Pred_Sublist) & set(GT_Sublist))
    score_deno = len(Pred_Sublist)

    score = score_num / score_deno
    return score

def StrictLengthMatching(Pred):
    '''
    Only all the order is correct gain 1 score, else 0
    '''
    for i in range(len(Pred)):
        if Pred[i] != i:
            return 0
    
    return 1

def LengthOfLIS(nums):
    torch_to_list(nums)
    if not nums:
        return 0
    dp = [1 for _ in range(len(nums))]
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    return max(dp)

if __name__ == '__main__':
    # TODO change input type
    '''
    $python metrics.py -M <str: metric type> -G <list: GT order> -P <list: Predict order>
    $python metrics.py -M <str: metric type> -R <int: random length>
    '''
    metric_type = sys.argv[1]
    input = sys.argv[2:]
    metric_func = StrictLengthMatching
    output = 0.0
    GT = []

    if metric_type == 'arbitrary':
        metric_func = ArbitraryLengthMatching
    elif metric_type == 'triple':
        metric_func = TripleLengthMatching
    elif metric_type == 'double':
        metric_func = DoubleLengthMatching
    elif metric_type == 'strict':
        metric_func = StrictLengthMatching
    else:
        assert False , "no such metric type"

    if len(input) > 1:
        input_ordered = get_order_list(input)
        output = metric_func(input_ordered, GT)
    elif False and len(input) == 1 and metric_type == 'arbitrary':
        times = 10000
        input_list = list(range(int(input[0])))
        for i in range(times):
            random.shuffle(input_list)
            # print(input_list)
            output += metric_func(input_list)
        output /= times
    elif len(input) == 1 or metric_type == 'triple':
        input_list = list(range(int(input[0])))
        times = 0
        for i in permutations(input_list):
            output += metric_func(i)
            times += 1
        output /= times
    else:
        assert False, "no input list"
        
    
    print(output)