"""
Module for TF-IDF
"""

import numpy as np
import math
import operator as opt
from itertools import imap, ifilter

def ham_dist(str1, str2):
    """
    Calculate hamming distance between two binary digits string.
    """
    return sum(imap(opt.ne, str1, str2))

def hmatchs(u_iter, v, dist):
    """
    Get a iterator of the list of hamming distances smaller than
    the given hamming distance.

    Parameters
    ----------
    u_iter: list of binary string.
    v: input binary string.
    dist: the largest length of hamming distance.

    Returns
    -------
    A hamming distance iterator.
    [(dist, binary string from original iterator), ....]
    """
    def _get_dist(target):
        return ham_dist(target, v), target
    return ifilter(lambda t: t[0] < dist, imap(_get_dist, u_iter))

def getidf(arr):
    """
    Calculate bag of words idf from all images
    
    Parameters
    ----------
    arr: k*l ndarray containing bag of word representations of all images
    
    Returns
    -------
    a list of k idf 
    """
    k,l = arr.shape
    idf = []
    j = 0
    threshold = 0.01
    for i in arr:
        if max(i) <= 0:
            idf.append(0)
        else:
            idf.append(math.log10( l/sum([1 for p in i if p>0])))  
    return idf

def gettf(txt,k):
    """
    Calculate bag of word tf for all images
    
    Parameters
    ----------
    txt: images bag of word file
    k: bag of word number
    
    Returns
    -------
    l: total image number
    nparr: ndarray containing all images bag of word representation
    """
    f = open(txt,'r')
    l = sum(1 for line in f)
    nparr = np.zeros((l,k))
    f = open(txt,'r')
    # calculate tf
    j = 0
    for line in f:
        print "line: ",j+1
        for p in line.split(" ")[2:-1]:
            k,v = int(p.split(":")[0]),float(p.split(":")[1])
            nparr[j][k-1] = v
        s = sum(nparr[j])
        nparr[j] = [ i/s for i in nparr[j]]
        j += 1
    return l,nparr
    
def tf_idf(txt,k):
    """
    Calculate tf-idf for whole images
    
    Parameters
    ----------
    txt: images bag of word file
    k: bag of word number
    
    Returns
    -------
    idf: images idf
    nparr.T: k*l ndarray with tf-idf weighted bag of words
    
    """
    #calculate tf
    l,nparr = gettf(txt,k)
    
    #calculate idf       
    idf = getidf(nparr.T )
    
    #calculate tf-idf
    for i in range(l):
        nparr[i] = nparr[i]*idf 
        #normalization
        s = math.sqrt(sum([ j*j for j in nparr[i]]))
        nparr[i] = [j/s for j in nparr[i]]
        
    return idf,nparr.T

def tfidf_rank(img,code_book,tfidf_arr,idf):
    """
    Give back top 10 similar images for the incoming one
    
    Parameters
    ----------
    img: target image path
    code_book: txt that contain codebook detail
    tfidf_arr: whole image tf-idf matrix
    idf: word idf list
    
    Returns
    -------
    top 10 similar images indexes
    
    """
    #
    #codebook = vq.code_book("nomeaning", "Mixed",code_book,K=20,save=True, read_from_txt=True)
    #hist = vq.quatization(img, codebook, soft=False).values()
    
    test = "1:0.00315789 2:0.00210526 3:0.00315789 5:0.00526316 6:0.00105263 7:0.00105263 8:0.00105263 10:0.0147368 11:0.00421053 12:0.00105263 13:0.00105263 14:0.00105263 15:0.00210526 17:0.00421053 18:0.0115789 19:0.00210526 20:0.00315789 22:0.00526316 25:0.00315789 30:0.00421053 31:0.00526316 33:0.00210526 34:0.00526316 35:0.00842105 36:0.00421053 37:0.00210526 38:0.00105263 40:0.00526316 42:0.00105263 43:0.00526316 44:0.00105263 45:0.00421053 46:0.00105263 47:0.00105263 49:0.00105263 50:0.00105263 53:0.00421053 54:0.00210526 55:0.00631579 56:0.00105263 59:0.00736842 60:0.00105263 61:0.00105263 62:0.00210526 63:0.00105263 65:0.00842105 67:0.00210526 69:0.00315789 70:0.00210526 71:0.00947368 72:0.00736842 73:0.00210526 75:0.00315789 76:0.00526316 78:0.00210526 79:0.00315789 80:0.00526316 81:0.00105263 82:0.00315789 83:0.00210526 84:0.00421053 85:0.00315789 87:0.00210526 88:0.00105263 89:0.00105263 90:0.00315789 91:0.00105263 93:0.00631579 94:0.00842105 96:0.00947368 97:0.00210526 101:0.00631579 102:0.00105263 103:0.00526316 105:0.00315789 109:0.0115789 110:0.00210526 113:0.00315789 114:0.00210526 117:0.00315789 118:0.00105263 119:0.00421053 122:0.00105263 123:0.00210526 126:0.00105263 131:0.00315789 132:0.00210526 133:0.00842105 134:0.00315789 135:0.00105263 137:0.00105263 139:0.00315789 140:0.00105263 141:0.00526316 142:0.00526316 144:0.00526316 145:0.00526316 146:0.00421053 149:0.00421053 150:0.00210526 152:0.00947368 153:0.00526316 154:0.00210526 155:0.00210526 156:0.00421053 158:0.00315789 159:0.00631579 161:0.0189474 163:0.00210526 165:0.00210526 166:0.00421053 167:0.00210526 171:0.0105263 172:0.00105263 174:0.00105263 175:0.00421053 177:0.00105263 178:0.00210526 179:0.00526316 182:0.00210526 184:0.00105263 186:0.00210526 187:0.0115789 188:0.00105263 190:0.00105263 193:0.00315789 195:0.00210526 196:0.00526316 197:0.00105263 198:0.00736842 199:0.00842105 201:0.00105263 205:0.00736842 206:0.0157895 208:0.00736842 210:0.00842105 211:0.00421053 212:0.00105263 214:0.00105263 216:0.00210526 218:0.00210526 221:0.00105263 222:0.00210526 223:0.00105263 224:0.00210526 225:0.00105263 228:0.00421053 229:0.00315789 231:0.00105263 232:0.00210526 233:0.00105263 234:0.0136842 235:0.0105263 236:0.00105263 237:0.00421053 240:0.00842105 241:0.00421053 242:0.00210526 244:0.00105263 246:0.00421053 254:0.00315789 255:0.00210526 256:0.00526316 258:0.00421053 260:0.00421053 261:0.00421053 262:0.00105263 263:0.00736842 266:0.00210526 267:0.00526316 268:0.00526316 269:0.00526316 270:0.00210526 271:0.0115789 272:0.00105263 274:0.00105263 275:0.00105263 276:0.00526316 278:0.00631579 279:0.00210526 280:0.0136842 281:0.00315789 283:0.00315789 284:0.00631579 285:0.00315789 286:0.00210526 288:0.00210526 289:0.00736842 290:0.00421053 291:0.00842105 292:0.00105263 296:0.00421053 297:0.00421053 300:0.00421053 302:0.00421053 304:0.00842105 305:0.00315789 306:0.00210526 308:0.00842105 309:0.00210526 311:0.00526316 312:0.00210526 315:0.00315789 316:0.00421053 318:0.00736842 321:0.00421053 323:0.0136842 324:0.00315789 325:0.00105263 326:0.00105263 328:0.00315789 329:0.00210526 330:0.00210526 331:0.00105263 332:0.00105263 334:0.00315789 335:0.00210526 338:0.00526316 341:0.00210526 342:0.00736842 343:0.00210526 344:0.00105263 345:0.00105263 346:0.00105263 347:0.00210526 348:0.00105263 349:0.00105263 350:0.00210526 351:0.00842105 352:0.00315789 353:0.00105263 356:0.00210526 357:0.00210526 359:0.00315789 361:0.00105263 362:0.0105263 364:0.00105263 365:0.00526316 368:0.00210526 369:0.00315789 370:0.00631579 371:0.00315789 372:0.0105263 373:0.00210526 374:0.00210526 377:0.00315789 378:0.00526316 379:0.00421053 380:0.00210526 381:0.00210526 382:0.00315789 384:0.00421053 386:0.00315789 390:0.00105263 391:0.00105263 392:0.00421053 393:0.0126316 394:0.00105263 395:0.00421053 396:0.00210526 397:0.00421053 398:0.00105263 399:0.00105263 400:0.00315789"
    hist = [0 for i in range(400)]
    for i in test.split(" "):
        hist[int(i.split(":")[0])-1] = float(i.split(":")[1])
    tf = [ float(i)/sum(hist) for i in hist]
    tfidf = [ x*y for (x,y) in zip(tf,idf)]
    s = math.sqrt(sum([ j*j for j in tfidf]))
    tfidf = [j/s for j in tfidf]
    k,l = tfidf_arr.shape
    print np.asarray(tfidf).shape,tfidf_arr.shape
    rank =  np.dot(np.asarray(tfidf),tfidf_arr)
    
    rank = zip([z for z in range(l)], rank)
    sorted_rank = sorted((r for r in rank if not math.isnan(r[1])), key=lambda r: r[1])
    return sorted_rank
    # imgs = sorted(t for t in rank.items() if not math.isnan(t))
    # print imgs
    # return rank_imgs 
    

#idf,ndarr = tf_idf("/Users/minhuigu/Desktop/tfidf/Multiclass_samples.txt",20)
idf,ndarr = tf_idf("/Users/minhuigu/Desktop/new.txt",400)
np.savetxt('./array.txt', ndarr)
f2 = open("./idf.txt",'w')
f2.write(" ".join(str(i) for i in idf))
f2.close()

print tfidf_rank("/Users/minhuigu/Downloads/archive/ayahoo_test_images/bag_9.jpg","/Users/minhuigu/Desktop/Multi_class_codebook.txt",ndarr,idf)[-10:]
