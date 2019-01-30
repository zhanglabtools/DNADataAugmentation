###############################################################################
We think these numerical results may be helpful to figure out how the CNN
tricks and models work for DNA sequences. We only reported AUC in the paper
due to space limitations. Here we present the results of area under the 
receiver operating characteristic curve (AUC), area under the precision-
recall curve (AUPR), accuracy and F1-score.
###############################################################################
01_evaluation_156_datasets.xlsx

This file reports the prediction results for our models, gkm-SVM and Zeng et al.
To make a fair comparison, the extending step is forbidden during testing. So,
the testing DNA sequences is 101bp long DNA sequences. 

For detail of our methods, please see section 3.4 of our paper. gkm-SVM is 
implemented using LS-GKM software with hyper-parameter l=11,k=7 and linear kernel. 
The results of Zeng et al. were downloaded from http://cnn.csail.mit.edu/motif_occupancy_pred/
on 2019/1/29. For each dataset, the results of /1layer_128motif/ is used. (This
may be slightly different from our re-implementation in the paper while the 
numerical results of AUC are almost the same.)

###############################################################################
02_final_prediction.rar

After uncompressing, each of the 156 files give the final prediction results of
out methods. Again, the extending step is forbidden during testing. These files 
may help you to make some comparison.

###############################################################################
03_deep_and_wide_model.xlsx

This file reports the prediction results for baseline, wide and deep CNN models
in the paper (similar to Figure 5). For each model, we also use different 
data augmentation tricks.

Remark: the extending step is allowed here during testing. So it is unfair to 
compare results in this table with others.
