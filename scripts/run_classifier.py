import os
import gensim
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",category=RuntimeWarning)

import time
start_time = time.time()

topdir='LDA-jet-substructure'
if os.getcwd().split('/')[-1]==topdir:
    print('... running the classifier'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
if os.getcwd().split('/')[-1]!=topdir:
    print('... the script needs to be run from the top level directory, the default is '+topdir+'/.',flush=True)
    print('... if you have renamed the directory you need to edit the "run_classifier.py" script.',flush=True)
    sys.exit(0)

cdir=os.getcwd()

configFile=sys.argv[1]
bfile='parsed-data'+'/'+sys.argv[2]+'/'+sys.argv[2]+'.txt'
sfile='parsed-data'+'/'+sys.argv[3]+'/'+sys.argv[3]+'.txt'
tagger_name=sys.argv[6]

# making directories
tagger_dir='taggers/'+tagger_name
os.system('mkdir '+tagger_dir)
os.system('mkdir '+tagger_dir+'/modelFiles')
os.system('mkdir '+tagger_dir+'/modelLogs')
os.system('mkdir '+tagger_dir+'/docFiles')
os.system('mkdir '+tagger_dir+'/rocCurves')
os.system('mkdir '+tagger_dir+'/topicDistributions')

with open(tagger_dir+'/'+'command_used.txt','a+') as cmdfout:
    cmdfout.write(' '.join(sys.argv[0:]))

print('... a directory tree has been set up in '+tagger_dir+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# copying the config file to the tagger directory, adding the directory to the path, and importing configs
os.system('cp '+configFile+' '+'taggers/'+tagger_name+'/taggerConfigFile.py')
sys.path.insert(0,cdir+'/taggers/'+tagger_name+'/')
from taggerConfigFile import *

# constucting a mixed sample from the signal and background samples
# NOTE:  here I use gshuf to shuffle the events randomly, linux machines will use shuf
os.system('gshuf -r -n '+str(Nb)+' '+bfile+' >> '+'taggers/'+tagger_name+'/docFiles/'+'tmp_allevents'+tagger_name+'.txt')
if Ns!=0:
    os.system('gshuf -r -n '+str(Ns)+' '+sfile+' >> '+'taggers/'+tagger_name+'/docFiles/'+'tmp_allevents'+tagger_name+'.txt')
os.system('gshuf '+'taggers/'+tagger_name+'/docFiles/'+'tmp_allevents'+tagger_name+'.txt >> '+'taggers/'+tagger_name+'/docFiles/'+'all_events'+tagger_name+'.txt')
os.system('rm '+'taggers/'+tagger_name+'/docFiles/'+'tmp_allevents'+tagger_name+'.txt')
print('... an unlabelled mixed sample with the specified S/B has been saved in '+'taggers/'+tagger_name+'/docFiles/'+'tmp_allevents'+tagger_name+'.txt'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# setting up logging
logging.basicConfig(filename='taggers/'+tagger_name+'/modelLogs/'+tagger_name+'.log',format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print('... logs will be output to '+'taggers/'+tagger_name+'/modelLogs/'+tagger_name+'.log',flush=True)

# setting up the corpus feed
class LineCorpus(gensim.corpora.textcorpus.TextCorpus):
    def get_texts(self):
        with open(self.input) as f:
            for l in f:
                yield l.split()
corpus = LineCorpus('taggers/'+tagger_name+'/docFiles/'+'all_events'+tagger_name+'.txt')

# extracting the topics
print('... extracting topics',flush=True)
model=eval('gensim.models.LdaModel(corpus, id2word=corpus.dictionary,'+gensim_parameters+')')
model.save('taggers/'+tagger_name+'/modelFiles/'+tagger_name)
print('... topics extracted and stored in '+'taggers/'+tagger_name+'/modelFiles/'+tagger_name+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# infer topic proportions on the unshuffled documents
print('... inferring topic proportions on the documents for classification',flush=True)
bfile='parsed-data'+'/'+sys.argv[4]+'/'+sys.argv[4]+'.txt'
sfile='parsed-data'+'/'+sys.argv[5]+'/'+sys.argv[5]+'.txt'
open_bfile=open(bfile,"r")
open_sfile=open(sfile,"r")
openROC_tru=open('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tru_tp.txt',"a+")
openROC_est=open('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_est_tp.txt',"a+")
estvec=np.array([])
truvec=np.array([])
# background first
for line in open_bfile:
    line2=line.rstrip().split()
    lineID = model.id2word.doc2bow(line2)
    l=model[lineID]
    if len(l)<2:
        l1=[[i][0] for i in range(len(l))]
        l2=[(i,0) for i in range(2)]
        for i in range(2):
            if l2[i][0] in l1:
                l2[i]=(i,l[l1.index(i)][1])
    if len(l)==2:
        l2=l
    openROC_tru.write(str(0))
    openROC_est.write(str(l2[1][1]))
    openROC_tru.write('\n')
    openROC_est.write('\n')
    truvec=np.append(truvec,0.0)
    estvec=np.append(estvec,l2[1][1])
# now signal
for line in open_sfile:
    line2=line.rstrip().split()
    lineID = model.id2word.doc2bow(line2)
    l=model[lineID]
    if len(l)<2:
        l1=[[i][0] for i in range(len(l))]
        l2=[(i,0) for i in range(2)]
        for i in range(2):
            if l2[i][0] in l1:
                l2[i]=(i,l[l1.index(i)][1])
    if len(l)==2:
        l2=l
    openROC_tru.write(str(1))
    openROC_est.write(str(l2[1][1]))
    openROC_tru.write('\n')
    openROC_est.write('\n')
    truvec=np.append(truvec,1.0)
    estvec=np.append(estvec,l2[1][1])
open_bfile.close()
open_sfile.close()
openROC_tru.close()
openROC_est.close()
print('... inferring topic proportions done, results stored in '+'taggers/'+tagger_name+'/rocCurves/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# constructing the ROC curves using topic proportions
print('... putting together the ROC curves using topic proportions',flush=True)
fpr_rand=np.array([i/100 for i in range(100)])
tpr_rand=np.array([i/100 for i in range(100)])
fpr, tpr, _ = metrics.roc_curve(truvec,estvec)
auc = round(metrics.roc_auc_score(truvec,estvec,sample_weight=None),3)
plt.plot(fpr,tpr,label='auc='+str(auc))
plt.plot(fpr_rand,tpr_rand,linestyle='dashed',color='red')
plt.xlabel('mis-tag')
plt.ylabel('efficiency')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tp1.png')
plt.clf()
#plt.show()
print('... ROC curve saved in '+'taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tp1.png'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
fpr_rand_inv=1/(fpr_rand+0.000001)
fpr_inv=1/(fpr+0.000001)
plt.plot(tpr,fpr_inv,label='auc='+str(auc))
plt.plot(tpr_rand,fpr_rand_inv,linestyle='dashed',color='red')
plt.xlabel('efficiency')
plt.ylabel('inverse mis-tag')
plt.legend(loc='best')
plt.yscale('log')
plt.ylim((1,10000))
plt.savefig('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tp2.png')
plt.clf()
#plt.show()
print('... ROC curve saved in '+'taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tp2.png'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# calculating the log-likelihood ratio classifier
open_bfile=open(bfile,"r")
open_sfile=open(sfile,"r")
# defining a word:weight dictionary
btopic={}
for (word,weight) in model.show_topic(0,topn=1000000):
    btopic.update({word:weight})
stopic={}
for (word,weight) in model.show_topic(1,topn=1000000):
    stopic.update({word:weight})
# calculating log-likelihood ratios
blr=[]
for line in open_bfile:
    line2=line.rstrip().split()
    pwords_b=[]
    pwords_s=[]
    for word in line2:
        if word in btopic.keys():
            pwords_b.append(btopic[word])
        if word not in btopic.keys():
            pwords_b.append(0.00000001)
        if word in stopic.keys():
            pwords_s.append(stopic[word])
        if word not in stopic.keys():
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
        if word in btopic.keys():
            pwords_b.append(btopic[word])
        if word not in btopic.keys():
            pwords_b.append(0.00000001)
        if word in stopic.keys():
            pwords_s.append(stopic[word])
        if word not in stopic.keys():
            pwords_s.append(0.00000001)
    slr_temp=0
    for k in range(len(pwords_s)):
        slr_temp+=(np.log(pwords_s[k])-np.log(pwords_b[k]))
    slr.append(slr_temp)
open_bfile.close()
open_sfile.close()
estvec=blr+slr
estvec=np.asarray(estvec)
truvec=np.asarray([0.0]*len(blr)+[1.0]*len(slr))
openROC_tru=open('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_tru_lr.txt',"a+")
openROC_est=open('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_est_lr.txt',"a+")
for i in estvec:
    openROC_est.write(str(i)+'\n')
openROC_est.close()
for i in truvec:
    openROC_tru.write(str(int(i))+'\n')
openROC_tru.close()
print('... calculating log-likelihood ratios done, results stored in '+'taggers/'+tagger_name+'/rocCurves/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)


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
plt.savefig('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_lr1.png')
plt.clf()
#plt.show()
print('... ROC curve saved in '+'taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_lr1.png'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
fpr_rand_inv=1/(fpr_rand+0.000001)
fpr_inv=1/(fpr+0.000001)
plt.plot(tpr,fpr_inv,label='auc='+str(auc))
plt.plot(tpr_rand,fpr_rand_inv,linestyle='dashed',color='red')
plt.xlabel('efficiency')
plt.ylabel('inverse mis-tag')
plt.legend(loc='best')
plt.yscale('log')
plt.ylim((1,10000))
plt.savefig('taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_lr2.png')
plt.clf()
#plt.show()
print('... ROC curve saved in '+'taggers/'+tagger_name+'/rocCurves/'+tagger_name+'_roc_lr2.png'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# visualising the topics
print('... generating topic visualisations',flush=True)
# everything below is just plotting topics...

# extracting the weights for each feature in each topic
btopic=[]
for (word,weight) in model.show_topic(0,topn=1000000):
    btopic.append([weight,[float(i) for i in word.split('_')]])
stopic=[]
for (word,weight) in model.show_topic(1,topn=1000000):
    stopic.append([weight,[float(i) for i in word.split('_')]])

# the topic data is written out below in a parsed format that will be easy to manipulate after training
pdat=open('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_topicData.txt',"a+")
pdat.write(str([btopic,stopic]))
pdat.close()
print('... topic data saved out in '+'taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_topicData.txt')

# deriving the list of bins for each observable
sj_bins=[i*sj_bin_size for i in range(round(sj_min_bin/sj_bin_size),round(sj_max_bin/sj_bin_size+1))]
md_bins=[i*md_bin_size for i in range(0,round(1/md_bin_size+1))]
kt_bins=[i*kt_bin_size for i in range(0,round(1/kt_bin_size+1))]
pmr_bins=[i*pmr_bin_size for i in range(0,round(1/pmr_bin_size+1))]
ha_bins=[i*ha_bin_size for i in range(-round(1/ha_bin_size),round(1/ha_bin_size+1))]

# plotting 2D histograms of subjet mass vs mass drop
# for background
tx=[]
ty=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[0])
    ty.append(vector[1])
    tw.append(weight)
plt.hist2d(tx,ty,weights=tw,cmap=plt.cm.Reds,bins=[sj_bins,md_bins])
plt.xlim(sj_min_bin,sj_max_bin)
plt.ylim((0,1))
plt.xlabel('subjet mass (GeV)')
plt.ylabel('mass drop')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_B_sj_vs_md.png')
# for signal
tx=[]
ty=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[0])
    ty.append(vector[1])
    tw.append(weight)
plt.hist2d(tx,ty,weights=tw,cmap=plt.cm.Reds,bins=[sj_bins,md_bins])
plt.xlim(sj_min_bin,sj_max_bin)
plt.ylim((0,1))
plt.xlabel('subjet mass (GeV)')
plt.ylabel('mass drop')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_S_sj_vs_md.png')
plt.clf()
print('... 2D histograms of subjet mass vs mass drop for topics saved to '+'taggers/'+tagger_name+'/topicDistributions/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# plotting 1D histograms for all of the observables
# subjet mass
tx=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[0])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=sj_bins,color='red',alpha=0.5,label='background',density=True)
tx=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[0])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=sj_bins,color='green',alpha=0.5,label='signal',density=True)
plt.xlim(sj_min_bin,sj_max_bin)
plt.xlabel('subjet mass (GeV)')
plt.ylabel('weight')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_sj.png')
plt.clf()
# mass drop
tx=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[1])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=md_bins,color='red',alpha=0.5,label='background',density=True)
tx=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[1])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=md_bins,color='green',alpha=0.5,label='signal',density=True)
plt.xlim((0,1))
plt.xlabel('mass drop')
plt.ylabel('weight')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_md.png')
plt.clf()
# parent mass ratio
tx=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[2])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=pmr_bins,color='red',alpha=0.5,label='background',density=True)
tx=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[2])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=pmr_bins,color='green',alpha=0.5,label='signal',density=True)
plt.xlim((0,1))
plt.xlabel('parent mass ratio')
plt.ylabel('weight')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_pmr.png')
plt.clf()
# kt distance
tx=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[3])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=kt_bins,color='red',alpha=0.5,label='background',density=True)
tx=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[3])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=kt_bins,color='green',alpha=0.5,label='signal',density=True)
plt.xlim((0,1))
plt.xlabel('kt distance')
plt.ylabel('weight')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_kt.png')
plt.clf()
# helicity angle
tx=[]
tw=[]
for weight,vector in btopic:
    tx.append(vector[4])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=ha_bins,color='red',alpha=0.5,label='background',density=True)
tx=[]
tw=[]
for weight,vector in stopic:
    tx.append(vector[4])
    tw.append(weight)
plt.hist(tx,weights=tw,bins=ha_bins,color='green',alpha=0.5,label='signal',density=True)
plt.xlim((-1,1))
plt.xlabel('helicity angle')
plt.ylabel('weight')
plt.legend(loc='best')
plt.savefig('taggers/'+tagger_name+'/topicDistributions/'+tagger_name+'_ha.png')
plt.clf()
print('... 1D histograms of each observable have been plotted separately in the same folder, with signal and background histograms on the same plots'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
