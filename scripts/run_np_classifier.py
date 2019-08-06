import os
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import metrics
from collections import Counter

import time
start_time = time.time()

topdir='LDA-jet-substructure'
if os.getcwd().split('/')[-1]==topdir:
    print('... running the Neyman-Pearson classifier'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
if os.getcwd().split('/')[-1]!=topdir:
    print('... the script needs to be run from the top level directory, the default is '+topdir+'/.',flush=True)
    print('... if you have renamed the directory you need to edit the "run_np_classifier.py" script.',flush=True)
    sys.exit(0)

# opening pure samples
open_bfile=open('parsed-data/'+sys.argv[1]+'/'+sys.argv[1]+'.txt','r')
open_sfile=open('parsed-data/'+sys.argv[2]+'/'+sys.argv[2]+'.txt','r')

# making directory
tagger_name=sys.argv[5]
tagger_dir='taggers/'+tagger_name
os.system('mkdir '+tagger_dir)

with open(tagger_dir+'/'+'command_used.txt','a+') as cmdfout:
    cmdfout.write(' '.join(sys.argv[0:]))

print('... a directory has been set up in '+tagger_dir+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

print('... building the probability distributions (%s seconds)' % round(time.time() - start_time,4),flush=True)

# building probability distributions for pure samples over feature-space
words_bfile=[]
for line in open_bfile:
    words_bfile=words_bfile+line.rstrip().split()
words_sfile=[]
for line in open_sfile:
    words_sfile=words_sfile+line.rstrip().split()
prob_bfile=Counter(words_bfile)
total_bfile=0
for i in prob_bfile.values():
    total_bfile+=i
for i in prob_bfile.keys():
    prob_bfile[i]=float(prob_bfile[i])/float(total_bfile)
prob_sfile=Counter(words_sfile)
total_sfile=0
for i in prob_sfile.values():
    total_sfile+=i
for i in prob_sfile.keys():
    prob_sfile[i]=float(prob_sfile[i])/float(total_sfile)
open_bfile.close()
open_sfile.close()
with open(tagger_dir+'/'+sys.argv[5]+'_prob_b.txt','a+') as fout:
    for w,p in prob_bfile.items():
        fout.write(str(w)+' '+str(p)+'\n')
with open(tagger_dir+'/'+sys.argv[5]+'_prob_s.txt','a+') as fout:
    for w,p in prob_sfile.items():
        fout.write(str(w)+' '+str(p)+'\n')

print('... probability distributions built and saved in '+tagger_dir+'/'+sys.argv[5]+'/',flush=True)

# calculating the Neyman-Pearson log-likelihood ratios

open_bfile=open('parsed-data/'+sys.argv[3]+'/'+sys.argv[3]+'.txt','r')
open_sfile=open('parsed-data/'+sys.argv[4]+'/'+sys.argv[4]+'.txt','r')
blr=[]
for line in open_bfile:
    line2=line.rstrip().split()
    pwords_b=[]
    pwords_s=[]
    for word in line2:
        if word in prob_bfile.keys():
            pwords_b.append(prob_bfile[word])
        if word not in prob_bfile.keys():
            pwords_b.append(0.00000001)
        if word in prob_sfile.keys():
            pwords_s.append(prob_sfile[word])
        if word not in prob_sfile.keys():
            pwords_s.append(0.00000001)
    blr_temp=0
    for k in range(len(pwords_s)):
        blr_temp+=(np.log(pwords_s[k])-np.log(pwords_b[k]))
    blr.append(blr_temp)

slr=[]
for line in open_sfile:
    line2=line.rstrip().split()
    pwords_b=[]
    pwords_s=[]
    for word in line2:
        if word in prob_bfile.keys():
            pwords_b.append(prob_bfile[word])
        if word not in prob_bfile.keys():
            pwords_b.append(0.00000001)
        if word in prob_sfile.keys():
            pwords_s.append(prob_sfile[word])
        if word not in prob_sfile.keys():
            pwords_s.append(0.00000001)
    if np.prod(pwords_b)!=0:
#        slr.append(np.log(np.prod(pwords_s)/np.prod(pwords_b)))
        slr_temp=0
        for k in range(len(pwords_s)):
            slr_temp+=(np.log(pwords_s[k])-np.log(pwords_b[k]))
        slr.append(slr_temp)
#    if np.prod(pwords_b)==0:
#        slr.append(10000)

open_bfile.close()
open_sfile.close()
estvec=blr+slr
estvec=np.asarray(estvec)
truvec=np.asarray([0.0]*len(blr)+[1.0]*len(slr))
openROC_tru=open(tagger_dir+'/'+tagger_name+'_lr_tru.txt',"a+")
openROC_est=open(tagger_dir+'/'+tagger_name+'_lr_est.txt',"a+")
for i in estvec:
    openROC_est.write(str(i)+'\n')
openROC_est.close()
for i in truvec:
    openROC_tru.write(str(int(i))+'\n')
openROC_tru.close()
print('... calculating log-likelihood ratios done, results stored in '+tagger_dir+'/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# constructing the ROC curves using log-likelihood ratio
print('... putting together the ROC curves using log-likelihood ratio',flush=True)
fpr_rand=np.array([i/100 for i in range(100)])
tpr_rand=np.array([i/100 for i in range(100)])
fpr, tpr, _ = metrics.roc_curve(truvec,estvec)
auc = round(metrics.roc_auc_score(truvec,estvec,sample_weight=None),3)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.plot(fpr_rand,tpr_rand,linestyle='dashed',color='red')
plt.xlabel('mis-tag')
plt.ylabel('efficiency')
plt.legend(loc='best')
plt.savefig(tagger_dir+'/'+tagger_name+'_roc_lr1.png')
plt.clf()
#plt.show()
fpr_rand_inv=1/(fpr_rand+0.000001)
fpr_inv=1/(fpr+0.000001)
plt.plot(tpr,fpr_inv,label='auc='+str(auc))
plt.plot(tpr_rand,fpr_rand_inv,linestyle='dashed',color='red')
plt.xlabel('efficiency')
plt.ylabel('inverse mis-tag')
plt.legend(loc='best')
plt.yscale('log')
plt.ylim((1,10000))
plt.savefig(tagger_dir+'/'+tagger_name+'_roc_lr2.png')
plt.clf()
#plt.show()
print('... ROC curves saved in '+tagger_dir+'/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

