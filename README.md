# comp9417-project-21t2
Sleep stage classification via machine learning\
Arthur Sze (z5165205) and William Yang (z5162119)

## Libraries used
numpy\
pyplot\
pandas\
pickle\
mne\
scipy\
tdqm\
math\
sklearn\
(optional) pytorch\
(optional) torch\
seaborn\
tabulate

## How to run the code

### 1. Download the original dataset
#### Description
PhysioNet SleepEDF (~10.4GB): https://www.physionet.org/content/sleep-edfx/1.0.0/ \
If you do not wish to download the entire dataset, skip to step 4 where we have provided the data after feature extraction (subband_data.csv)

### 2. Run 1_EDF_to_pickle.py
#### Description
Converts the original .edf files that contains the raw EEG data and sleep stage annotations for each patient into a single pickle file, as labelled 30s epochs.
#### Parameters to change:
rawDir (get directory for original data) = "YourDirectoryHere/sleep-edf-database-expanded-1.0.0/sleep-cassette"\
dest_path (choose where to save pickled data) = "YourDirectoryHere/all_data.pkl"

### 3. Run 2_preprocess_feature_extraction.py
#### Description
Performs data cleanup and feature extraction from the raw EEG signal, saved into a csv file. The rows represents the sample/epoch, and columns represents the patient ID, features, and output sleep stage class.
#### Parameters to change:
raw_data_path (get pickled data from part 2) = "YourDirectoryHere/all_data.pkl"\
subband_data_pickle_path (choose where to save pickled data) = "YourDirectoryHere/subband_data.pkl"\
subband_data_csv_path (choose where to save csv file) = "YourDirectoryHere/subband_data.csv"

### 4. Run 3_Classification.py
#### Description
From the features extracted, apply the 3 splitting methodologies (intra-patient without balance, intra-patient with balance, inter-patient with balance) in nested cross-validation. The 3 classifiers used are LR, MLP, and DT. This will output the classifier models, best hyperparameters, and performance metrics for each splitting methodology.
#### Parameters to change:
data_csv (get subband_data.csv file) = 'YourDirectoryHere/subband_data.csv'\
The results folders are configured to save in the current directory, so there is no option to change the destination directory.

### 5. (optional) Run 4_train_perceptron.py
#### Description
Similar to step 3, we train an MLP with multiple layers and more hyperparameters as part of our preliminary testing.
#### Parameters to change:
data_csv (get subband_data.csv file) = 'YourDirectoryHere/subband_data.csv'

### 6. Run 5_plot_cv_results.py
#### Description
Plot the results in step 4.

### 7. (optional) Run 6_rnn_training.py
#### Description
Similar to step 3, we train an RNN as part of our preliminary testing.