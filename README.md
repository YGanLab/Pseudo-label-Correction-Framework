# Pseudo-label-Correction-Framework

This is the code implementation of the paper 'Push the Boundary of SAM: A Pseudo-label Correction Framework for Medical Segmentation'. \
code for framework: \
 step1_train_with_reweight.py: train the initial network with the multi-level reweighting strategy. \
 step2_update_label.py: update the noisy training labels. \
After iteration: \
 final_step_run_plain_unet.py: train an unet with the updated labels. \
Supporting code: \
 get_prediction_unet.ipynb: get the prediction results from the network. 
