# number of background events
Nb=50000

# S/B fraction
sb=0.1

# number of signal events
Ns=round(sb*Nb)

# gensim parameters (keep the format as a string here)
gensim_parameters="num_topics=2,alpha=[0.9,0.1],passes=5,iterations=100,gamma_threshold=0.00000001,eval_every=10,update_every=1,chunksize=100,decay=0.5"
# the alpha parameters are the hyper-parameters that determine the topic concentrations in the docs and in the corpus
# the passes is how many times you train over the whole dataset
# for details on other parameters we refer the reader to the gensim webpage

# bin sizes for the observables, should match what was used in the parser
sj_bin_size=10
md_bin_size=0.05
pmr_bin_size=0.1
kt_bin_size=0.1
ha_bin_size=0.1

# max for subjet mass histogram axes
sj_min_bin=30
sj_max_bin=200
