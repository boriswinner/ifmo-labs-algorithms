import random
import numpy as np
CONST_OUTPUT = 1

def constantFunction(v):
    return CONST_OUTPUT

def sumFunction(v):
    return np.sum(v)

def productFunction(v):
    return np.prod(v)

def polynomialFunction(v, x = 1.5):
    p = []
    for k, v_k in enumerate(v):
        p.append(v_k * x**k )
    return np.sum(p)

def polynomialFunctionHorner(v, x = 1.5):
    t = 1
    for k in range( len(v) - 1, -1, -1):
        t = v[k] + x*t
    return t

def bubbleSort(nlist):
    for passnum in range(len(nlist)-1,0,-1):
        for i in range(passnum):
            if nlist[i]>nlist[i+1]:
                temp = nlist[i]
                nlist[i] = nlist[i+1]
                nlist[i+1] = temp

def qsort(nums):
   if len(nums) <= 1:
       return nums
   else:
       q = random.choice(nums)
       s_nums = []
       m_nums = []
       e_nums = []
       for n in nums:
           if n < q:
               s_nums.append(n)
           elif n > q:
               m_nums.append(n)
           else:
               e_nums.append(n)
       return qsort(s_nums) + e_nums + qsort(m_nums)

def timSort(array):
    array.sort()

def matrixMultiplication(m1, m2):
    return m1.dot(m2)