# cuts on jet and event kinematics
# if the any of the observables for a jet or event do not fall within these limits then that it is dropped from the list
jptcut=[400.0,10000.0]
invmcut=[0.0,10000.0]

# cuts on jet substructure kinematics
# if the any of the observables at a splitting do not fall within these limits then that 'feature' is dropped from the list
# if an event passes the cuts above and doesn't have any features that pass the cuts below, it will be denoted as background
sj_mass_cut=[30.0,10000.0]
kt_cut=[0.00,10000.0]

# bin sizes for the observables
sj_bin_size=10
md_bin_size=0.05
pmr_bin_size=0.1
kt_bin_size=0.1
ha_bin_size=0.1

# max for subjet mass histogram axes
sj_min_bin=30
sj_max_bin=200

#########################################################
# The format of the raw data files should be as follows:
#########################################################
# doc_start 1
# jet_start
# top_tag w_tag p_0 p_1 p_2 p_3 tau_1 tau_2 tau_3
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# ...
# jet_end
# jet_start
# top_tag w_tag p_0 p_1 p_2 p_3 tau_1 tau_2 tau_3
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# ...
# jet_end
# jet_start
# top_tag w_tag p_0 p_1 p_2 p_3 tau_1 tau_2 tau_3
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# ...
# jet_end
# doc_end 1
# doc_start 2
# jet_start
# top_tag w_tag p_0 p_1 p_2 p_3 tau_1 tau_2 tau_3
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# ...
# jet_end
# jet_start
# top_tag w_tag p_0 p_1 p_2 p_3 tau_1 tau_2 tau_3
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# subjetmass massdrop parentratio kt theta
# ...
# jet_end
# doc_end 2
# doc_start 3
# ...
# ...
# ...
# ...
# ...
# doc_end N
##################################################################
# The documents are marked at their beginning and end by doc_start
# and doc_end, while the jets sare marked by jet_start and 
# jet_end.
# In the above example the first document has 3 reconstructed jets,
# and the second document has 2.
# In this example each document represents all jets from an event,
# so this is for an event tagger.
# You can build a top jet tagger by having just one jet per 
# document.
# The line immediately below jet_start contains jet observables.
# The top_tag and w_tag are the tags from the fastjet output, 
# these are just 0 for negatuve and 1 for positive and don't need 
# to be included.
# The n-subjettiness observables are also not neccesary, if
# you want to exclude any of these you will need to zero-pad.
# The only thing necessary in this line is the 4-momentum.
##################################################################

