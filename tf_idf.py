import numpy as np
import math
import vq

def getidf(arr):
    k,l = arr.shape
    idf = []
    j = 0
    for i in arr:
        idf.append(math.log10( l/sum([1 for p in i if p>0])))
        
    #return idf
    
    #change for small K !!!!!!
    return [1 for i in range(k)]

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

def tfidf_rank(img,code_book,tfidf_arr,idf):
    codebook = vq.code_book("nomeaning", "Mixed",code_book,K=20,save=True, read_from_txt=True)
    hist = vq.quatization(img, codebook, soft=False).values()
    tf = [ float(i)/sum(hist) for i in hist]
    tfidf = [ x*y for (x,y) in zip(tf,idf)]
    k,l = tfidf_arr.shape
    for i in range(k):
        tfidf_arr[i] = [ j*tfidf[i] for j in tfidf_arr[i]]
    r = []
    for i in range(l):
        r.append(sum(tfidf_arr.T[i]))
    rank = {x:y for x,y in zip([z for z in range(l)],r)}
    rank_imgs = [i for (i,j) in sorted(rank.items(), key=lambda x: x[1])][::-1]

    
idf,ndarr = tf_idf("/Users/minhuigu/Desktop/tfidf/Multiclass_samples.txt",20)
print tfidf_rank("/Users/minhuigu/Desktop/598/pic.jpeg","/Users/minhuigu/Desktop/Multi_class_codebook.txt",ndarr,idf)

