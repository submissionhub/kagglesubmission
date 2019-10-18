import numpy as np
import math

def mean(X):
    i = 0
    j = 0
    (t,n) = X.shape
    m = np.zeros(n)
    for row in X :
        flg = True
        for col in row : 
            if (math.isnan(col)):
                flg = False
        if (flg):
            j=j+1
            m = m + row
        i = i+1

    return m/j

def addmean(X,M):
    filled = +X
    j = 0
    for row in X: 
        i = 0
        for n in row :
            if (math.isnan(n)):
                #print ('ADDMEAN: replacing ', n , ' with ', M[i])
                row[i] = M[i]
                filled[j] = row
            i = 1+i
        j = 1+j

    return filled

def cleanNUM(X,C):
    #perform numerical cleaning required for training data
    cleandata = +X
    if (len(C) < 1) : 
        print('CLEAN: no columns specified for cleaning')
        return  X
    (m,n) = X.shape
    dataNUM = X[:,0:2]
    for c in C :
        dataNUM = np.append(dataNUM,X[:,c:c+1],axis = 1)

    i = 0
    for row in dataNUM[:,2:] :
        #print(row)
        flg = False
        for col in row : 
            if (math.isnan(col)):
                flg = True
        if (flg):
            #print('CLEAN: row ', i ,' deleted')
            cleandata = np.delete(cleandata,i,0)
        else : 
            i = i+1
    
    return  cleandata

def cleanOLD(X,Y):
    #outdated ver
    #perform numerical cleaning required for training data
    cleandata = +X
    cleanANS = +Y
    i = 0
    for row in X :
        #print(row)
        flg = False
        for col in row : 
            if (math.isnan(col)):
                flg = True
        if (flg):
            #print('CLEAN: row ', i ,' deleted')
            cleandata = np.delete(cleandata,i,0)
            cleanANS = np.delete(cleanANS,i,0)
        else : 
            i = i+1
    
    return  cleandata ,cleanANS

def cleanstr(X):
    #perform string cleaning required for training data
    cleandata = +X
    i = 0
    for row in X :
        #print(row)
        flg = False
        for col in row :
            if (type(col) != type('string')):
                flg = True
        if (flg):
            #print('CLEANBIN: row ', i ,' deleted')
            cleandata = np.delete(cleandata,i,0)
        else : 
            i = i+1
    
    return  cleandata

def split(X,T):
    (m,n) = X.shape
    train = []
    test = []
    if (T <= 1) :
        split = int(m*T)
        train = X[0:split,:]
        test = X[split:m,:]
    else :
        print('SPLIT: T must be less than 1')

    return train,test

def tr_onehot(X):
    #take data that requires transformation to one-hot encoding
    #used scikit functions instead in main
    onehot = [] 
    return onehot

def gen_float(X):
    #generate a table of transformations for custom float
    CT = []
    return CT

def tr_float(X,CT):
    c = []
    return c

