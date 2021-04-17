%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Sample codes for machine learning analysis of classifying eyes-closed EEG and eyes-open EEG %%%%%
%%%% Written by Yiheng Tu, PhD                                                                   %%%%%
%%%% Date: 8/26/2018                                                                             %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clear all; close all;
 
load data_classification.mat
 
%% parameters
fs_test = 200; % sampling rate
N_Train = size(ec,2); % number of training trials
N_Test = size(test_samples,2); % number of test trials
 
 
%% PSD
nfft = 256; % Point of FFT
for n=1:N_Train
    [P_ec(:,n),f] = pwelch(detrend(ec(:,n)),[],[],nfft,fs_test); % calculate PSD for ec condition
    [P_eo(:,n),f] = pwelch(detrend(eo(:,n)),[],[],nfft,fs_test); % calculate PSD for eo condition
end
for n=1:N_Test
    [P_test(:,n),f] = pwelch(detrend(test_samples(:,n)),[],[],nfft,fs_test); % calculate PSD for test samples
end
 
%% feature extraction (alpha-band power in ec is significantly larger than in eo)
alpha_idx = find((f<=12)&(f>=8));  % frequency index of alpha band power
a_ec_train = mean(P_ec(alpha_idx,:)); % extract alpha band power from eo
a_eo_train = mean(P_eo(alpha_idx,:)); % extract alpha band power from ec
a_test = mean(P_test(alpha_idx,:)); % extract alpha band power from test data
 
%% 10-fold CV on training data
all_samples = [a_eo_train,a_ec_train]'; % all samples
all_labels = [ones(size(a_eo_train,2),1);zeros(size(a_ec_train,2),1)]; % labels of all samples: 1 for eo; 0 for ec
K = 10; % K-fold CV
indices = crossvalind('Kfold',all_labels,K); % generate indices for CV
for k = 1:K % K iterations
    cv_test_idx = find(indices == k); % indices for test samples in one trial of validation
    cv_train_idx = find(indices ~= k); % indices for training samples in one trial of validation
    cv_classout = classify(all_samples(cv_test_idx,:),all_samples(cv_train_idx,:),all_labels(cv_train_idx));
    cv_acc(k) = mean(cv_classout==all_labels(cv_test_idx)); % compute accuracy
    TP = sum((cv_classout==all_labels(cv_test_idx))&(cv_classout==1));
    TN = sum((cv_classout==all_labels(cv_test_idx))&(cv_classout==0));
    FP = sum((cv_classout~=all_labels(cv_test_idx))&(cv_classout==1));
    FN = sum((cv_classout~=all_labels(cv_test_idx))&(cv_classout==0));
    cv_sensitivity(k) = TP/(TP+FN); % compute specificity
    cv_specificity(k) = TN/(TN+FP); % compute sensitivity
end
cv_acc_avg = mean(cv_acc); % averaged accuracy
cv_sensitivity_avg = mean(cv_sensitivity);  % averaged sensitivity
cv_specificity_avg = mean(cv_specificity);  % averaged specificity
    
%% test on test data
% Concatenate training/test data and specify the labels
train_samples = [a_eo_train';a_ec_train']; %  training samples
train_labels = [ones(N_Train,1);zeros(N_Train,1)]; % labels of training samples: 1 for eo; 0 for ec
test_samples = [a_test']; % test samples
classout = classify(test_samples,train_samples,train_labels,'linear');
TP_test = sum((classout==test_labels)&(classout==1));
TN_test = sum((classout==test_labels)&(classout==0));
FP_test = sum((classout~=test_labels)&(classout==1));
FN_test = sum((classout~=test_labels)&(classout==0));
test_acc = sum(classout==test_labels)/N_Test; % compute accuracy
test_sensitivity = TP_test/(TP_test+FN_test); % compute specificity
test_specificity = TN_test/(TN_test+FP_test); % compute sensitivity