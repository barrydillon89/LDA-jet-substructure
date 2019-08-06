The code in this repository implements unsupervised jet substructure classification based on Latent Dirichlet Allocation.  The code was developed for the work in the paper "Uncovering latent jet substructure" by B. M. Dillon, D. A. Faroughy, J. F. Kamenik.  The arxiv link is https://arxiv.org/abs/1904.04200.

The implementation here is simple in that:
 - there are only two topics: background and signal
 - only substructure observables at each splitting in the clustering history are used:
     subjet mass, mass drop, parent mass ratio, kt distance between parents, angle between child and heaviest parent
 - a document may contain one or more jets, and is treated as an unordered list of features
 - k-folding is not implemented automatically

There are 5 main subdirectories here:
 - raw-data/ - raw data for jet subtructure is stored here
 - parsed-data/ - here data parsed into the format we use is stored
 - configs/ - here config files for parsing data and running taggers are stored
 - scripts/ - scripts used by the taggers are stored here
 - taggers/ - here the data and scripts relevant for each tagger you build are stored

The scripts should be run from the top directory, which is by default assumed to be named "LDA-jet-substructure".  To change this you will need to make simple one-line updates to parser.py, run_classifier.py, and run_np_classifier.py.  All scripts in the scripts/ directory create other directories where they store data, when this is done a copy of the command used will always be stored in that directory in a file called 'command_used.txt'.

There are a few data files included in raw-dir/, each document in these files (see parser.py for a description of their structure) contains information on the clustering history of all jets in a single event.  The tt_events_test*.dat files were created using all hadronic final states in ttbar production, and the jj_events_test*.dat files were created from QCD background events.  For details on how the samples were produced see https://arxiv.org/abs/1904.04200.  These files are shorter extracts of larger event files which were too large too upload here, they contain a few hundred events each.

You can convert data files from raw-data/ into parsed data using the configuration specified in 'configs/parserConfig.py':

    >>  python3.7 scripts/parser.py configs/parserConfig.py raw-data/jj_events_test_a.dat jj_events_test_a
    ... parsing in progress
    ... parsing done (21.8639 seconds)
    ... the parsed data and a copy of the config file used is stored in parsed-data/jj_events_test_a
    ... generating histogram plots
    ... 2D histogram of subjet mass vs mass drop for the sample saved to parsed-data/jj_events_test_a/jj_events_test_a_sj_vs_md.png (22.0438 seconds)
    ... 1D histograms for the observables in the sample saved to parsed-data/jj_events_test_a/ (22.6028 seconds)
    ... data for the plots here has been saved to parsed-data/jj_events_test_a/jj_events_test_a_plot_data.txt (22.6089 seconds)
    
    >> python3.7 scripts/parser.py configs/parserConfig.py raw-data/tt_events_test_a.dat tt_events_test_a
    ... parsing in progress
    ... parsing done (22.5575 seconds)
    ... the parsed data and a copy of the config file used is stored in parsed-data/tt_events_test_a
    ... generating histogram plots
    ... 2D histogram of subjet mass vs mass drop for the sample saved to parsed-data/tt_events_test_a/tt_events_test_a_sj_vs_md.png (22.7205 seconds)
    ... 1D histograms for the observables in the sample saved to parsed-data/tt_events_test_a/ (23.2481 seconds)
    ... data for the plots here has been saved to parsed-data/tt_events_test_a/tt_events_test_a_plot_data.txt (23.2533 seconds)

In the above inputs the first is the config file, the second is the input file from raw-data/, and the third is a tag to name the parsed data with.  You can create custom config files and store them in configs/.  The required form of the raw data files is shown in the parserConfig.py file in configs/.  The outputs are all saved in a directory in parsed-data/ and includes observable histograms for the data sample.  You can perform the same commands for the test sets appended with 'test_b' to follow the examples below.

With the distributions for background and signal from the 'test_a' data you can construct the simple Neyman-Pearson classifier, and test it on the 'test_b' data:

    >>  python3.7 scripts/run_np_classifier.py jj_events_test_a tt_events_test_a jj_events_test_b tt_events_test_b tt_jj_event_test_np_classifier
    ... running the Neyman-Pearson classifier (0.0 seconds)
    ... a directory has been set up in taggers/tt_jj_event_test_np_classifier (0.0036 seconds)
    ... building the probability distributions (0.0036 seconds)
    ... probability distributions built and saved in taggers/tt_jj_event_test_np_classifier/tt_jj_event_test_np_classifier/
    ... calculating log-likelihood ratios done, results stored in taggers/tt_jj_event_test_np_classifier/ (0.0867 seconds)
    ... putting together the ROC curves using log-likelihood ratio
    ... ROC curves saved in taggers/tt_jj_event_test_np_classifier/ (0.6004 seconds)

The first two inputs are the background and signal data sources from parsed-data with which you want to construct the classifier.  The third and fourth arguments are the data sources in parsed-data on which you want to classify and construct the ROC curves for.  The last input is simply the name you want for the classifier in taggers/.

There is then one script which, using a tagger config file in configs/; constructs a mixed sample, extracts the LDA topics, constructs the ROC curves, and plots the topic visualisations:

    >>  python3.7 scripts/run_classifier.py configs/testClassifierConfig.py jj_events_test_a tt_events_test_a jj_events_test_b tt_events_test_b tt_jj_event_test_classifier
    ... running the classifier (0.0 seconds)
    ... a directory tree has been set up in taggers/tt_jj_event_test_classifier (0.0182 seconds)
    ... an unlabelled mixed sample with the specified S/B has been saved in taggers/tt_jj_event_test_classifier/docFiles/tmp_alleventstt_jj_event_test_classifier.txt (0.0602 seconds)
    ... logs will be output to taggers/tt_jj_event_test_classifier/modelLogs/tt_jj_event_test_classifier.log
    ... extracting topics
    ... topics extracted and stored in taggers/tt_jj_event_test_classifier/modelFiles/tt_jj_event_test_classifier (13.7706 seconds)
    ... inferring topic proportions on the documents for classification
    ... inferring topic proportions done, results stored in taggers/tt_jj_event_test_classifier/rocCurves/ (14.2148 seconds)
    ... putting together the ROC curves using topic proportions
    ... ROC curve saved in taggers/tt_jj_event_test_classifier/rocCurves/tt_jj_event_test_classifier_roc_tp1.png (14.3701 seconds)
    ... ROC curve saved in taggers/tt_jj_event_test_classifier/rocCurves/tt_jj_event_test_classifier_roc_tp2.png (14.7044 seconds)
    ... calculating log-likelihood ratios done, results stored in taggers/tt_jj_event_test_classifier/rocCurves/ (14.775 seconds)
    ... putting together the ROC curves using log-likelihood ratio
    ... ROC curve saved in taggers/tt_jj_event_test_classifier/rocCurves/tt_jj_event_test_classifier_roc_lr1.png (14.8892 seconds)
    ... ROC curve saved in taggers/tt_jj_event_test_classifier/rocCurves/tt_jj_event_test_classifier_roc_lr2.png (15.05 seconds)
    ... generating topic visualisations
    ... topic data saved out in taggers/tt_jj_event_test_classifier/topicDistributions/tt_jj_event_test_classifier_topicData.txt
    ... 2D histograms of subjet mass vs mass drop for topics saved to taggers/tt_jj_event_test_classifier/topicDistributions/ (15.2578 seconds)
    ... 1D histograms of each observable have been plotted separately in the same folder, with signal and background histograms on the same plots (15.9398 seconds)
    
The first input is the config you wish to use, the second and third are the background and signal sources from parsed-data from which you want to extract the topics, i.e. train on.  The fourth and fifth are the data sources from parsed-data for constructing the roc curves with.  Note that here it would be well justified to construct the ROC curves from the data you had trained on, since here we have an unsupervised training method which does not make use of the labels in the sample. The last input is the name you want for your classifier.  Each time you run this, the config file you used for the tagger will be copied to the directory for that tagger.  The relevant data is also saved out in case you want to load it into a notebook and re-work the plots.

