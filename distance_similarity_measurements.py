from math import *
from decimal import *
import numpy as np
from scipy.spatial import distance

def euclidean_distance(x,y):
    distance = sqrt(sum(pow(a-b, 2) for a, b in zip(x,y)))
    return distance

def manhattan_distance(x,y):
     distance = sum(abs(a-b) for a, b in zip(x,y))
     return distance

def __nth_root(value, n_root):
    root_value = 1/float(n_root)
    root = round((Decimal(value) ** Decimal(root_value)), 3)
    return root

def minkowski_distance(x,y,p_value):
    distance = __nth_root(sum(pow(abs(a-b), p_value) for a,b in zip(x,y)), p_value)
    return distance

def __squared_rooted(x):
    root = round(sqrt(sum([a*a for a in x])),3)
    return root

def cosine_similarity(x,y):
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = __squared_rooted(x)*__squared_rooted(y)
    distance = round(numerator/float(denominator),3)
    return distance

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    similarity =  intersection_cardinality/float(union_cardinality)
    return similarity
