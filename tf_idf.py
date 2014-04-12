import numpy as np
import math

def getidf(arr):
    k,l = arr.shape
    idf = []
    j = 0
    for i in arr:
        idf.append(math.log10( l/sum([1 for p in i if p>0])))
    return idf

def tf_idf(txt,k):
    f = open(txt,'r')
    l = sum(1 for line in f)
    nparr = np.zeros((l,k))
    f = open(txt,'r')
    # calculate tf
    j = 0
    for line in f:
        value = [int(i.split(":")[1]) for i in line.split(" ")[1:-1]]
        value = [ float(i)/sum(value) for i in value]
        nparr[j] = np.asarray(value)
        j += 1
        
    idf = getidf(nparr.T )
    
    #calculate tf-idf
    for i in range(l):
        nparr[i] = nparr[i]*idf 

    return idf,nparr.T

idf,ndarr = tf_idf("/Users/minhuigu/Desktop/tfidf/Multiclass_samples.txt",20)
#print idf,arr
