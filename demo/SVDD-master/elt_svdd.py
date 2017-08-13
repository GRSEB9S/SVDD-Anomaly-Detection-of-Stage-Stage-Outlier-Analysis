from __future__ import division
import sys
import numpy as np
import scipy.optimize
from pyspark import  SparkContext, SparkConf
from numpy import vectorize
import gc

#conf = SparkConf().setAppName("svdd").setMaster("spark://192.168.44.145:7077")
             
#sc = SparkContext(conf=conf)
sc = SparkContext(appName="Svdd")	

def func(x):
     if x>0:
          return -1
     else:
          return 1
vfunc = vectorize(func)

def accuracy(x,y,n):
    count=0
    for i in range(0,n):     
        if x[i]==y[i]:
            count=count+1       
    return count/n*100

ncenter=4
C=1
s=0.06911

sig=-1/(2*s)
file_name = sys.argv[1]   
a = np.loadtxt(file_name)
s=a.shape
ntrain=s[0]
d=s[1]-1

tr_label = np.zeros(shape=(ntrain,1))
tr_label=a[:,d]

a=np.delete(a,np.s_[d::], axis=1)
file_center = sys.argv[2]
b = np.loadtxt(file_center)

r2 = np.empty(ntrain)
zero1 = np.empty(ntrain)
zero1[:]=0
rdd_array=sc.parallelize(zero1,1)
#rdd_array1=sc.parallelize(r,ntrain)

smsq=np.zeros(shape=(ncenter,ntrain))

k=0
i=0  
for k in range(0,ncenter): 
    rdd_array=sc.parallelize(zero1,1)
    for i in range(0,d):
        r1=a[:,i]
        r2[:]=b[k,i]
        rdd1 = sc.parallelize(r1,1)
        rdd2 = sc.parallelize(r2,1)
        rdd3=rdd1.zip(rdd2)
        rdd3=rdd3.map(lambda (x,y):(x-y)**2)        
        rdd_array=rdd_array.zip(rdd3)        
        rdd_array=rdd_array.map(lambda (x,y): x+y)
    rdd_array1=rdd_array    
    smsq[k,:]=rdd_array1.collect()

#print "sum of squares array \n",smsq

dec=np.zeros(shape=(ntrain,1)).ravel()
coef=np.zeros(shape=(ncenter+1,1)).ravel()
smsq1=np.array(smsq[0]).reshape(ntrain,1).ravel()
smsq2=np.array(smsq[1]).reshape(ntrain,1).ravel()
smsq3=np.array(smsq[2]).reshape(ntrain,1).ravel()
smsq4=np.array(smsq[3]).reshape(ntrain,1).ravel()
tr_label.ravel()
coef.ravel()

def cons1(coef,smsq1,smsq2,smsq3,smsq4,tr_label,c=C) :   
        return np.maximum(np.multiply(tr_label,2-2*(coef[1]*np.exp(sig*smsq1)+coef[2]*np.exp(sig*smsq2)+coef[3]*np.exp(sig*smsq3)+coef[4]*np.exp(sig*smsq4)).T).T-coef[0]**2,0)
def cons2(coef,smsq1,smsq2,smsq3,smsq4,tr_label,c=C) :
        return np.multiply(np.maximum(np.multiply(tr_label,2-2*(coef[1]*np.exp(sig*smsq1)+coef[2]*np.exp(sig*smsq2)+coef[3]*np.exp(sig*smsq3)+coef[4]*np.exp(sig*smsq4)).T).T-coef[0]**2,0),-1).T
def cons3(coef,smsq1,smsq2,smsq3,smsq4,tr_label,c=C):
    return coef[0]
def fun(coef,smsq1,smsq2,smsq3,smsq4,tr_label,c=C):     
        f = coef[0]**2+c*(np.sum(np.maximum(np.multiply(tr_label,2-2*(coef[1]*np.exp(sig*smsq1)+coef[2]*np.exp(sig*smsq2)+coef[3]*np.exp(sig*smsq3)+coef[4]*np.exp(sig*smsq4)).T).T-coef[0]**2,0)))       
        return f
        
#res=scipy.optimize.fmin_cobyla(fun,[.5,.5,.5,.5],[cons1,cons2,cons3],args=(dec,smsq1,smsq2,smsq3,tr_label,),rhobeg=1.0, rhoend=0.0001, iprint=1, maxfun=5000, disp=None, catol=0.0002)

res=scipy.optimize.fmin_cobyla(fun,[.5,.5,.5,.5,.5],[cons1,cons2,cons3],args=(smsq1,smsq2,smsq3,smsq4,tr_label,),rhobeg=1, rhoend=0.001, iprint=1, maxfun=5000, disp=None, catol=0.05)

#print "\n",res[0],res[1],res[2],res[3]

dec=np.maximum(np.multiply(tr_label,2-2*(res[1]*np.exp(sig*smsq1)+res[2]*np.exp(sig*smsq2)+res[3]*np.exp(sig*smsq3)+res[4]*np.exp(sig*smsq4)).T).T-res[0]**2,0)
#print "decision fn \n",dec

tr_prediction=vfunc(dec)
print tr_label
print tr_prediction


tr_accuracy =accuracy(tr_label,tr_prediction,ntrain)         
print "Training Accouracy:",tr_accuracy,"%\n"

#Testing
file_test = sys.argv[3]
c = np.loadtxt(file_test)
s=c.shape
ntest=s[0]
ts_label = np.zeros(shape=(ntest,1))
ts_label=c[:,d]
c=np.delete(c,np.s_[d::], axis=1)

r4 = np.empty(ntest)
zero2 = np.empty(ntest)
zero2[:]=0
rdd_array_test=sc.parallelize(zero2,ntest)
#rdd_array2=sc.parallelize(r,ntest)
smsqt=np.zeros(shape=(ncenter,ntest))
#rdd_array1=rdd_array
k=0
i=0  
for k in range(0,ncenter): 
    rdd_array_test=sc.parallelize(zero2,ntest)
    for i in range(0,d):
        r3=c[:,i]
        #print "r3\n", r3
        r4[:]=b[k,i]
        #print "r4\n", r4
        rdd1_test = sc.parallelize(r3,ntest)
        
        rdd2_test = sc.parallelize(r4,ntest)
        rdd3_test=rdd1_test.zip(rdd2_test)

        rdd3_test=rdd3_test.map(lambda (x,y):(x-y)**2)        
        rdd_array_test=rdd_array_test.zip(rdd3_test)        
        rdd_array_test=rdd_array_test.map(lambda (x,y): x+y)
    rdd_array2_test=rdd_array_test    
    smsqt[k,:]=rdd_array2_test.collect()
dec2=np.zeros(shape=(ntest,1)).ravel()
#coef=np.zeros(shape=(ncenter+1,1)).ravel()
smsqt1=np.array(smsqt[0]).reshape(ntest,1).ravel()
smsqt2=np.array(smsqt[1]).reshape(ntest,1).ravel()
smsqt3=np.array(smsqt[2]).reshape(ntest,1).ravel()
smsqt4=np.array(smsqt[3]).reshape(ntest,1).ravel()
print "smsq test\n",smsqt
g=2-2*(res[1]*np.exp(sig*smsqt1)+res[2]*np.exp(sig*smsqt2)+res[3]*np.exp(sig*smsqt3)+res[4]*np.exp(sig*smsqt4)).T-res[0]**2
print"G:\n", g,"\n"
dec2=np.maximum(2-2*(res[1]*np.exp(sig*smsqt1)+res[2]*np.exp(sig*smsqt2)+res[3]*np.exp(sig*smsqt3)+res[4]*np.exp(sig*smsqt4)).T-res[0]**2,0)
print "decision fn test \n",dec2

ts_prediction=vfunc(dec2)
print ts_label
print ts_prediction


ts_accuracy =accuracy(ts_label,ts_prediction,ntest)         
print "Testing Accouracy:",ts_accuracy,"%\n"
gc.collect()
