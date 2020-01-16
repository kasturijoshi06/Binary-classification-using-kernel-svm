#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from cvxopt import matrix
from cvxopt import solvers
from numpy.linalg import norm


# In[4]:


x1 = np.random.uniform(0,1,(1,100))
x2 = np.random.uniform(0,1,(1,100))
x = np.concatenate((x1,x2),axis=0)
print (x)


# In[5]:


d = np.empty([100,1])
positive = []
negative = []
for i in range(0,100):
    if(x[1,i]<((1/5)*np.sin(10*x[0,i])+0.3) or ((x[1,i]-0.8,2)**2 +(x[0,i]-0.5,2)**2)<(0.15*0.15)):
        d[i]=1.0
        pos.append(x[:,i])
    else:
        d[i]=-1.0
        neg.append(x[:,i])
positive = np.asarray(positive)
negative = np.asarray(negative)


# In[17]:


plt.scatter(positive[:,0], positive[:,1], label= "cross", color= "blue",  
            marker= "x", s= 40)
plt.scatter(negative[:,0], negative[:,1], label= "dot", color= "red",  
            marker= ".", s= 10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[7]:


P = np.zeros([100,100])
for i in range(0,100):
    for j in range(0,100):
        P[i][j] = d[i]*d[j]*math.exp(-((x[0,i]-x[0,j])*(x[0,i]-x[0,j])+((x[1,i]-x[1,j])*(x[1,i]-x[1,j])))/2)
print(P)


# In[8]:


P = matrix(P)
q = matrix(-np.ones([100,1]),(100,1),'d')
G = matrix(-np.identity(100))
h = matrix(np.zeros([100,1]),(100,1),'d')
d2 = np.reshape(d,(1,100))
A = matrix(d2)
b = matrix(0,(1,1),'d')
sol = solvers.qp(P,q,G,h,A,b)
print(sol['x'])


# In[18]:


alpha = np.array(sol['x'])
s_v = []
d_sv = []
sv_a = []
for i in range(0,100):
    if(alpha[i]>=math.pow(10,3)):
        sv_a.append(alpha[i])
        s_v.append(x[:,i])
        d_sv.append(d[i])
s_v = np.asarray(s_v)
d_sv = np.asarray(d_sv)
sv_a = np.asarray(sv_a)
print(sv_a)
plt.scatter(s_v[:,0], s_v[:,1], label= "cross", color= "blue", marker= "d", s= 40)
plt.scatter(positive[:,0], positive[:,1], label= "star", color= "black", marker= "*", s= 40)
plt.scatter(negative[:,0], negative[:,1], label= "dot", color= "red",  marker= ".", s= 10)
plt.show()


# In[10]:


theta = 0
for i in range(0,len(s_v)):
    theta = theta + (sv_a[i]*d_sv[i]*math.exp(-(math.pow(norm(s_v[i]-s_v[1]),2))/2))                     
theta = d_sv[1]-theta
print(theta)


# In[11]:


w = np.zeros([len(s_v),1]) 
gx = []
for i in range(0,len(s_v)):
    for j in range(0,len(s_v)):
        w[i] = w[i]+(sv_a[j]*d_sv[j]*math.exp(-(math.pow(norm(s_v[j]-s_v[i]),2))/2))
    gx.append(w[i]+theta)
gx = np.asarray(gx)
print(gx)


# In[12]:


gx1 = np.zeros([100,100])
x1 = np.linspace(0,1,100)
x2 = np.linspace(0,1,100)
for i in range(len(x1)):
    for j in range(len(x2)):
        temp = 0
        for k in range(100):
            temp = temp + (alpha[k]*d[k]*math.exp(-(math.pow(norm(x[:,k]-np.asarray([x1[i],x2[j]])),2))/2))
        gx1[i][j] = temp + theta
print(gx1)           


# In[13]:


p = []
q = []
r = []
for i in range(100):
    for j in range(100):
        if(gx1[i][j]>-1.1 and gx1[i][j]<-0.9):
            #print("-1")
            p.append(np.asarray([x1[i],x2[j]]))
        if(gx1[i][j]>-0.1 and gx1[i][j]<0.1):
            #print("0")
            q.append(np.asarray([x1[i],x2[j]]))
        if(gx1[i][j]>0.9 and gx1[i][j]<1.1):
            #print("1")
            r.append(np.asarray([x1[i],x2[j]]))
print (p)
print (q)
print (r)


# In[15]:


p = np.array(p)
q = np.array(q)
r = np.array(r)

plt.scatter(p[:,0], p[:,1], label= "hline", color= "black",  marker= "_", s= 4)
plt.scatter(q[:,0], q[:,1], label= "hline", color= "blue",  marker= "_", s= 4)
plt.scatter(r[:,0],r[:,1], label= "hline", color= "red", marker= "_", s= 4)
plt.scatter(s_v[:,0], s_v[:,1], label= "cross", color= "blue",  
            marker= ".", s= 40)
plt.scatter(pos[:,0], pos[:,1], label= "cross", color= "red",  
            marker= ".", s= 40)
plt.scatter(neg[:,0], neg[:,1], label= "dot", color= "black",  
            marker= ".", s= 10)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


# In[ ]:




