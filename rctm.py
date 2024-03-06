import random 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file_path = r"Housing.csv"

dataset = pd.read_csv(file_path)
ycolumn = "price"


def makespecial(d):
    axes = d.axes[1]
    n = len(axes)
    for column in axes : 
        unique  = d[column].unique()
        nu = len(unique)
        if(d[column].dtype != "object"):
            continue
        mapping = {unique[i]:i for i in range(nu) }
         
        d[column] = d[column].replace(mapping)

def traintestspliter(y,data):
    trainper = 0.8
    split_index = int(len(data) * trainper)
    
    splited = data.head(split_index)
    under = data.tail(len(data)  - split_index)
    return np.array(splited.T.values),np.array(under.values),np.array(y.head(split_index)),np.array(y.tail(len(data)  - split_index))


def derror(value,estimaded):
    s = 0
    for e1,e2 in zip(value,estimaded):
        s = (e1 - e2)**2
    return s**0.5


def errorsquared(a):
    if len(a) == 0 :
        return float("inf") 
    s=0
    mean = np.mean(a)
    s+=sum((a-mean)**2)
    return s

def slice(y,data,slicer):
    lslice = np.where(data<=slicer)
    rslice = np.where(data>slicer)
    ly,ry= None,None
    if lslice : 
        ly = y[lslice]
    if rslice :
        ry = y[rslice]
    return ly,ry


def findthebestsplit(y,data,uniquepoints):

    emini =  float("inf")
    optimalslice = None 
    for i in uniquepoints : 
        ly,ry = slice(y,data,i)
        error = errorsquared(ly) + errorsquared(ry)
        if error < emini : 
            emini = error
            optimalslice = i
    return optimalslice,emini


def train(y,data):
    uniquepoints =  np.unique(data)
    uniquepoints = [(uniquepoints[i] + uniquepoints[i+1])/2 for i in range(len(uniquepoints)-1)  ]
    D = [data]
    splited = [y]  
    size = 20
    readyones = []
    while any(list(part) not in readyones for part in  splited):
        minierror = float("inf")
        bestsplit = None
        for i,part in enumerate(splited) :
            best,error  = findthebestsplit(part,D[i],uniquepoints)
            if(len(part) <= size ):
                readyones.append(list(part))
                continue
            if(list(part) in readyones  ):
                continue
            if (error <minierror ):
                minierror = error
                bestsplit = (best,i)
        if bestsplit  :
            x = splited.pop(bestsplit[1])
            d = D.pop(bestsplit[1])
            ly,ry = slice(x,d,bestsplit[0])
            ld,rd = slice(d,d,bestsplit[0])
            D.append(ld)
            D.append(rd)
            splited.append(ly)
            splited.append(ry)
            if(minierror == 0) : 
                readyones.append(list(ly))
                readyones.append(list(ry))
        if (not bestsplit):
            break 

    return [np.mean(s) for s in splited],[np.mean(d) for d in D]

def dt(y,datas):
    D = []
    for data in datas :  
        ys,ds =train(y,data)
        D.append((ys,ds,errorsquared(ys)))
    sorted(D,key= lambda x : x[2],reverse=True)
    plt.scatter(D[0][1] ,D[0][0], color='red', label='reggretion tree',zorder=2)
    return lambda xs : round(sum((1/(i+1)) *sorted(zip(ys,[1/(xs[i]- d)**2 for d in D[i][0] ]),key= lambda x : x[1],reverse=True)[0][0] for i in range(len(D)))/len(D))

makespecial(dataset)

y = dataset.pop(ycolumn)
trainx,testx,trainy,testy =traintestspliter(y,dataset)


rctmodel= dt(np.array(y),np.array(dataset.T.values))

print("a randon test : ")
indexchoosen = random.randint(0,len(testx))
res = rctmodel(testx[indexchoosen])
print("result : " , res)
print("real value : " , testy[indexchoosen])
print("error : " , abs(testy[indexchoosen] - res))


plt.scatter(trainx[0],trainy, color='blue', label='d', zorder=1)
plt.xlabel("b")
plt.ylabel("a")
plt.legend()
plt.show()


