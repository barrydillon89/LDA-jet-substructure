import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn import metrics
start_time = time.time()

topdir='LDA-jet-substructure'
if os.getcwd().split('/')[-1]==topdir:
    print('... parsing in progress')
if os.getcwd().split('/')[-1]!=topdir:
    print('... The script needs to be run from the top level directory, the default is '+topdir+'/.')
    print('... If you have renamed the directory you need to edit the "parser.py" script.')
    sys.exit(0)

cdir=os.getcwd()    

configFile=sys.argv[1]
file_in=sys.argv[2]
file_out=''.join([sys.argv[3],'.txt'])
dir_out=sys.argv[3]

# copying the config file to the directory for the parsed data
os.system('mkdir parsed-data/'+dir_out)
os.system('cp '+configFile+' parsed-data/'+dir_out+'/parserConfigFile.py')
sys.path.insert(0,cdir+'/parsed-data/'+dir_out)
from parserConfigFile import *

with open('parsed-data/'+dir_out+'/'+'command_used.txt','a+') as cmdfout:
    cmdfout.write(' '.join(sys.argv[0:]))

# opening the input file for reading
openIn=open(file_in,"r")

# opening the output file for appending the parsed data to
openOut=open('parsed-data/'+dir_out+'/'+file_out,"a+")

# opening an output file to store the event kinematics
file_out_kin=file_out[:-4]+'_kin.txt'
openOutKin=open('parsed-data/'+dir_out+'/'+file_out_kin,"a+")

# opening an output file to store the number of events before and after the cuts on substructure kinematics
# events which have features left after the cuts will be classified as background
file_out_lengths=file_out[:-4]+'_lengths.txt'
openOutLengths=open('parsed-data/'+dir_out+'/'+file_out_lengths,"a+")

# here we initialise the lengths for...
# events which pass all of the jet/event-level kinematic cuts, regardless of whether any features pass the substructure cuts
ev_len=0
# events which pass all of the jet/event-level kinematic cuts and have at least one feature passing the substructure cuts
ev_sub_len=0

# deriving the list of bins and mid-points for each observable
sj_bins=[i*sj_bin_size for i in range(round(sj_min_bin/sj_bin_size),round(sj_max_bin/sj_bin_size+1))]
md_bins=[i*md_bin_size for i in range(0,round(1/md_bin_size+1))]
kt_bins=[i*kt_bin_size for i in range(0,round(1/kt_bin_size+1))]
pmr_bins=[i*pmr_bin_size for i in range(0,round(1/pmr_bin_size+1))]
ha_bins=[i*ha_bin_size for i in range(-round(1/ha_bin_size),round(1/ha_bin_size+1))]
sj_bins_mid=[(i+0.5)*sj_bin_size for i in range(round(sj_min_bin/sj_bin_size),round(sj_max_bin/sj_bin_size))]
md_bins_mid=[(i+0.5)*md_bin_size for i in range(0,round(1/md_bin_size))]
pmr_bins_mid=[(i+0.5)*pmr_bin_size for i in range(0,round(1/pmr_bin_size))]
kt_bins_mid=[(i+0.5)*kt_bin_size for i in range(0,round(1/kt_bin_size))]
ha_bins_mid=[(i+0.5)*ha_bin_size for i in range(-round(1/ha_bin_size),round(1/ha_bin_size))]

# storing observables in vectors for plotting later
sj_vec=[]
md_vec=[]
pmr_vec=[]
kt_vec=[]
ha_vec=[]

# now we loop through the input file, analysing the raw data line by line
for line in openIn:
    # parsing the line by splitting on ' '
    line2=line.rstrip().split()
    # now going through the tags: 'doc_start', 'jet_start', 'jet_end', 'doc_end'.
    if line2[0]=='doc_start':
        # when starting a new event we re-initialise a bunch of arrays to...
        # store list of jet pTs for an event
        jetptL=[0]
        # store list of jet 4-momenta for an event
        jet4mL=[[0,0,0,0]]
        # store lists of features for an event
        eventL=[]
        # store jet pT
        jpt=0
        # store jet 4-momenta
        j4m=[0,0,0,0]
        # store features (or 'words') for each jet
        jw=[]
        # reset these
        sj_vec_ev_temp=[]
        md_vec_ev_temp=[]
        pmr_vec_ev_temp=[]
        kt_vec_ev_temp=[]
        ha_vec_ev_temp=[]
    if line2[0]=='jet_start':
        # these lines contain the kinematics of the jets
        # a new jet means new words
        jw=[]
        # reset these
        sj_vec_temp=[]
        md_vec_temp=[]
        pmr_vec_temp=[]
        kt_vec_temp=[]
        ha_vec_temp=[]
        # store jet pT
        jpt=0
        # store jet 4-momenta
        j4m=[0,0,0,0]
    if len(line2)==9:
        # these lines contain the kinematics of the jets        
        jet4mL.append(j4m)
        jetptL.append(jpt)
        eventL.append(jw)
        j4m=[eval(line2[2]),eval(line2[3]),eval(line2[4]),eval(line2[5])]
        jpt=np.sqrt(eval(line2[3])**2+eval(line2[4])**2)
    if len(line2)==5:
        # these are the lines that contain the observables at each splitting, so we bin and save
        sj=np.round(np.dot(sj_bins_mid,np.histogram(float(line2[0]),sj_bins)[0]))
        md=np.round(np.dot(md_bins_mid,np.histogram(float(line2[1]),md_bins)[0]),3)
        pmr=np.round(np.dot(pmr_bins_mid,np.histogram(float(line2[2]),pmr_bins)[0]),3)
        kt=np.round(np.dot(kt_bins_mid,np.histogram(float(line2[3]),kt_bins)[0]),3)
        ha=np.round(np.dot(ha_bins_mid,np.histogram(float(line2[4]),ha_bins)[0]),3)
        word='_'.join([str(sj),str(md),str(pmr),str(kt),str(ha)])
        # we only save those words that pass the cuts
        if sj_mass_cut[0]<=float(line2[0])<=sj_mass_cut[1] and kt_cut[0]<=float(line2[3])<=kt_cut[1]:
            jw.append(word)
            sj_vec_temp.append(sj)
            md_vec_temp.append(md)
            pmr_vec_temp.append(pmr)
            kt_vec_temp.append(kt)
            ha_vec_temp.append(ha)
    if line2[0]=='jet_end':
        sj_vec_ev_temp=sj_vec_ev_temp+sj_vec_temp
        md_vec_ev_temp=md_vec_ev_temp+md_vec_temp
        pmr_vec_ev_temp=pmr_vec_ev_temp+pmr_vec_temp
        kt_vec_ev_temp=kt_vec_ev_temp+kt_vec_temp
        ha_vec_ev_temp=ha_vec_ev_temp+ha_vec_temp
        # we only save those jets that pass the cuts
        if jptcut[0]<=jpt<=jptcut[1] and len(jw)>0:
            jet4mL.append(j4m)
            jetptL.append(jpt)
            eventL.append(jw)
    if line2[0]=='doc_end':
        # we only save those events that pass the cuts
        e4m=[sum(k) for k in [[i[j] for i in jet4mL] for j in range(len(jet4mL[0]))]]
        einvm=np.sqrt(e4m[0]**2-e4m[1]**2-e4m[2]**2-e4m[3]**2)
        if (len(jet4mL)>1) and (invmcut[0]<=einvm<=invmcut[1]):
            ev_len+=1
        if invmcut[0]<=einvm<=invmcut[1] and len(eventL)>0:
            openOut.write(' '.join([w for sublist in eventL for w in sublist])+'\n')
            openOutKin.write(str(einvm)+' '+str(jetptL)+' '+str(jet4mL)+'\n')
            ev_sub_len+=1
            sj_vec=sj_vec+sj_vec_ev_temp
            md_vec=md_vec+md_vec_ev_temp
            pmr_vec=pmr_vec+pmr_vec_ev_temp
            kt_vec=kt_vec+kt_vec_ev_temp
            ha_vec=ha_vec+ha_vec_ev_temp


# now we write out the num of events passing the different cuts and close open files
openOutLengths.write('num_events_passing_cuts'+' '+str(float(ev_len))+'\n'+'also_with_features'+' '+str(float(ev_sub_len)))
openOutLengths.close()
openOut.close()
openOutKin.close()
openIn.close()

print('... parsing done'+' (%s seconds)' % round(time.time() - start_time,4))
print('... the parsed data and a copy of the config file used is stored in parsed-data/'+dir_out)

# plotting the histograms for the pure samples
print('... generating histogram plots')

sj_bins=[i*sj_bin_size for i in range(round(sj_min_bin/sj_bin_size),round(sj_max_bin/sj_bin_size+1))]
md_bins=[i*md_bin_size for i in range(0,round(1/md_bin_size+1))]
kt_bins=[i*kt_bin_size for i in range(0,round(1/kt_bin_size+1))]
pmr_bins=[i*pmr_bin_size for i in range(0,round(1/pmr_bin_size+1))]
ha_bins=[i*ha_bin_size for i in range(-round(1/ha_bin_size),round(1/ha_bin_size+1))]

# plotting a 2D histogram of subjet mass vs mass drop
plt.hist2d(sj_vec,md_vec,cmap=plt.cm.Reds,bins=[sj_bins,md_bins],normed=True)
plt.xlim(sj_min_bin,sj_max_bin)
plt.ylim((0,1))
plt.xlabel('subjet mass (GeV)')
plt.ylabel('mass drop')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_sj_vs_md.png')
plt.clf()

print('... 2D histogram of subjet mass vs mass drop for the sample saved to '+'parsed-data/'+dir_out+'/'+dir_out+'_sj_vs_md.png'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# plotting the 1D histograms for the data
# subjet mass
plt.hist(sj_vec,bins=sj_bins,color='green',alpha=0.5,density=True)
plt.xlim(sj_min_bin,sj_max_bin)
plt.xlabel('subjet mass (GeV)')
plt.ylabel('weight')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_sj.png')
# mass drop
plt.hist(md_vec,bins=md_bins,color='green',alpha=0.5,density=True)
plt.xlim((0,1))
plt.xlabel('mass drop')
plt.ylabel('weight')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_md.png')
plt.clf()
# parent mass ratio
plt.hist(pmr_vec,bins=pmr_bins,color='green',alpha=0.5,density=True)
plt.xlim((0,1))
plt.xlabel('parent mass ratio')
plt.ylabel('weight')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_pmr.png')
plt.clf()
# kt distance
plt.hist(kt_vec,bins=kt_bins,color='green',alpha=0.5,density=True)
plt.xlim((0,1))
plt.xlabel('kt distance')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_kt.png')
plt.clf()
# helicity angle
plt.hist(ha_vec,bins=ha_bins,color='green',alpha=0.5,density=True)
plt.xlim((-1,1))
plt.xlabel('helicity angle')
plt.ylabel('weight')
plt.savefig('parsed-data/'+dir_out+'/'+dir_out+'_ha.png')
plt.clf()

print('... 1D histograms for the observables in the sample saved to '+'parsed-data/'+dir_out+'/'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)

# saving the plot data to file for use after parsing
with open('parsed-data/'+dir_out+'/'+dir_out+'_plot_data.txt','a+') as fout:
    fout.write(str([sj_vec,md_vec,pmr_vec,kt_vec,ha_vec]))

print('... data for the plots here has been saved to '+'parsed-data/'+dir_out+'/'+dir_out+'_plot_data.txt'+' (%s seconds)' % round(time.time() - start_time,4),flush=True)
