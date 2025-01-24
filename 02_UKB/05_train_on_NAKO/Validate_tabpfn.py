#!/usr/bin/env python
# coding: utf-8

# In[26]:


# pip install scikit-learn numpy scipy tensorflow imblearn # if needed you can install these onse, depend to the environment 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import pointbiserialr, chi2_contingency
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import wandb
from time import sleep
import random
# Suppress TensorFlow warnings
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module='keras')
warnings.filterwarnings("ignore", category=UserWarning, module='absl')
tf.get_logger().setLevel('ERROR')
import subprocess
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
import os
import gc


# In[27]:


random_state = 42
tf.random.set_seed(random_state)
np.random.seed(random_state)
random.seed(random_state)


# In[28]:


log_file = "../99_logs/logging/training_log.csv"
subprocess.run(["rm", "-rf", "../99_logs/log_ml_loop.txt"], check=True)
subprocess.run(["rm", "-rf", log_file], check=True)


# In[29]:


###THIS IS FOR LOGGING TO CSV
columns = [
    "label_col", "mri_table", "number_of_pos", "number_of_neg","test_set_size","y_label_ratio", "sex_label_ratio",
    "Feature_extraction_applied", "Pretraining_applied", 
    "model_type", "Accuracy", "AUC", "Balanced_ACC", "Permutation_Balanced_ACC", "number_of_cross_validations", "cross_validation_count", "search_term"
]
log_df = pd.DataFrame(columns=columns)
log_file = "../99_logs/training_log.csv"
log_df.to_csv(log_file, index=False)

def write_to_csv(label_col,
                 mri_table,
                 number_of_pos,
                 number_of_neg,
                 test_set_size,
                 y_label_ratio,
                 sex_label_ratio,
                 Feature_extraction_applied,
                 Pretraining_applied,
                 model_type,
                 Accuracy,
                 AUC,
                 Balanced_ACC,
                 Permutation_Balanced_ACC,
                 number_of_cross_validations,
                 cross_validation_count,
                 file="../99_logs/training_log.csv"):
    log_df = pd.read_csv(file)
    new_row = pd.DataFrame([{
        "label_col": label_col,
        "mri_table": mri_table,
        "number_of_pos": number_of_pos,
        "number_of_neg": number_of_neg,
        "test_set_size": test_set_size,
        "y_label_ratio": y_label_ratio,
        "sex_label_ratio": sex_label_ratio,
        "Feature_extraction_applied": Feature_extraction_applied,
        "Pretraining_applied": Pretraining_applied,
        "model_type": model_type,
        "Accuracy": Accuracy,
        "AUC": AUC,
        "Balanced_ACC": Balanced_ACC,
        "Permutation_Balanced_ACC": Permutation_Balanced_ACC,
        "number_of_cross_validations": number_of_cross_validations,
        "cross_validation_count": cross_validation_count,
        "search_term": label_col + "_" + mri_table
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    log_df.to_csv(file, index=False)


# In[30]:


def smote_balancing(dataset, features, id_col, label_col, gender_col):
    # Create a combined label for stratification
    dataset['combined_label'] = dataset[label_col].astype(str) + '_' + dataset[gender_col].astype(str)
    
    # Apply SMOTE to balance the combined labels
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X, y = smote.fit_resample(dataset[features], dataset['combined_label'])
    
    # Identify synthetic samples
    is_synthetic = pd.Series(X.index).isin(dataset.index) == False

    # Assign IDs: keep original IDs for real data, assign new IDs for synthetic data
    synthetic_id_start = 999999
    synthetic_ids = list(range(synthetic_id_start, synthetic_id_start - is_synthetic.sum(), -1))
    new_ids = []

    for idx, is_syn in zip(X.index, is_synthetic):
        if is_syn:
            new_ids.append(synthetic_ids.pop(0))
        else:
            new_ids.append(dataset.at[idx, id_col])

    # Split the combined labels back into original columns
    new_labels = y.str.split('_', expand=True)
    y_balanced = new_labels[0].astype(float).astype(int)  # Original label
    gender_balanced = new_labels[1]                      # Original gender
    
    # Combine the balanced features with the separated labels and gender
    balanced_data = pd.concat([pd.DataFrame(X, columns=features),
                               pd.DataFrame({id_col: new_ids, label_col: y_balanced, gender_col: gender_balanced})], axis=1)
    
    return balanced_data


# In[31]:


# you can also change reading data to avoid merging them if you plan to use more labels for future research, or you can write me to change them
def prepare_dataset(brain_csv_path, labels_csv_path, id_col, label_col, additional_cols=None, smote_upsampling_for_sex=True, sex_col = "sex"):
    """
    Prepares the dataset for training by merging brain features and binary labels.

    Args:
    - brain_csv_path (str): Path to the CSV file containing brain info (features).
    - labels_csv_path (str): Path to the CSV file containing labels.
    - id_col (str): Column name representing unique IDs in both files.
    - label_col (str): Label to filter by (binary).
    - additional_cols (list, optional): Additional columns to include in the merged dataset.

    Returns:
    - pd.DataFrame: Merged and cleaned dataset.
    """
    additional_cols = additional_cols or [] 
    #filter out None addition cols
    additional_cols = [col for col in additional_cols if col is not None]
    brain_data = pd.read_csv(brain_csv_path)
    label_data = pd.read_csv(labels_csv_path)

    brain_data[id_col] = brain_data[id_col].astype(str)
    label_data[id_col] = label_data[id_col].astype(str)
    if "sub-" in label_data[id_col].iloc[0]:
        label_data[id_col] = label_data[id_col].str.replace("sub-", "")
    if "sub-" in brain_data[id_col].iloc[0]:
        brain_data[id_col] = brain_data[id_col].str.replace("sub-", "")

    # Filter labels to keep only rows where label is binary (0 or 1)
    #label_data = label_data[label_data[label_col].isin([0, 1])]

    if smote_upsampling_for_sex != None:
        # Merge datasets
        merged_data = pd.merge(
            brain_data,
            label_data[[id_col, label_col, sex_col] + additional_cols],
            on=id_col,
            how='inner'
        )
    else:
        # Merge datasets
        merged_data = pd.merge(
            brain_data,
            label_data[[id_col, label_col] + additional_cols],
            on=id_col,
            how='inner'
        )
    # Drop rows with missing values
    merged_data.dropna(inplace=True)
    
    return merged_data


# In[32]:


def write_to_file(input, output_file="../99_logs/log_ml_loop.txt", dict=False, print_inplace=True):
    with open(output_file, 'a') as f:
        f.write(input)
    if print_inplace:
        print(input)


# In[33]:


# i could not find a good baseline for dropping non correlated features, i got this idea from chatGPT and you are welcome to change or modify them based what you need 
def calculate_max_features(sample_size, min_features=5, scaling_factor=10):
    """
    Dynamically calculate the maximum number of features based on the sample size.

    Args:
        sample_size (int): The number of samples in the dataset.
        min_features (int): The minimum number of features to retain.
        scaling_factor (int): Scaling factor for feature-to-sample ratio (e.g., 10).

    Returns:
        int: Recommended maximum number of features.
    """
    # Rule-based calculations
    max_features_by_ratio = sample_size // scaling_factor
    max_features_by_log = scaling_factor * np.log(sample_size)
    
    # Choose the most restrictive limit, but ensure at least min_features
    return int(max(min_features, min(max_features_by_ratio, max_features_by_log)))

def validate_labels(data, label_col):
    """
    Validates the label column and ensures it exists in the dataset.

    Args:
        data (pd.DataFrame): The dataset.
        label_col (str): The label column to validate.

    Returns:
        pd.Series: The validated label column.

    Raises:
        KeyError: If the label column is not found in the dataset.
        ValueError: If the label column contains invalid data types or unexpected values.
    """
    if label_col not in data.columns:
        raise KeyError(f"The label column `{label_col}` is not found in the dataset.")
    
    y = data[label_col]  # Extract the label column
    if not isinstance(y, pd.Series):
        raise ValueError(f"The label column `{label_col}` must be a single column, but a different type was provided.")
    
    return y


def feature_extraction(data, features, label_col, excluded_labels=None, p_value_threshold=0.05, min_features=5, scaling_factor=10, use_feature_extraction=True):
    """
    Extracts features based on their correlation with the label.

    Args:
    - data (pd.DataFrame): Dataset containing features and labels.
    - features (list): List of feature columns to evaluate.
    - label_col (str): Name of the binary label column.
    - excluded_labels (list): List of additional columns to exclude during feature evaluation.
    - p_value_threshold (float): Threshold for selecting features based on p-value.
    - min_features (int): Minimum number of features to retain.
    - scaling_factor (int): Scaling factor for feature-to-sample ratio.
    - use_feature_extraction (bool): Whether to perform feature extraction.

    Returns:
    - pd.DataFrame: Dataset with filtered features and all excluded labels retained.
    - pd.DataFrame: DataFrame of correlation results for all features.
    """
    if not use_feature_extraction:
        print("Feature extraction is disabled. Returning the original dataset.")
        return data[features + [label_col]], None
    # Ensure excluded_labels is a list
    excluded_labels = excluded_labels or []
    excluded_labels = [col for col in excluded_labels if col is not None]
    if not isinstance(excluded_labels, list):
        raise ValueError("excluded_labels must be a list or None.")
    #if there is a None value in the list, drop it
    excluded_labels = [label for label in excluded_labels if label is not None]

    # Exclude labels only during p-value computation
    excluded_features = set(excluded_labels + [label_col])
    feature_subset = [f for f in features if f not in excluded_features]
    
    y = data[label_col].astype(int)  # Ensure binary label is numeric (0/1)
    X = data[feature_subset]  # Extract relevant feature columns
    write_to_file(f"Number of columns before feature extraction: {len(feature_subset)}")

    # Store correlation results
    correlation_results = {}
    for column in X.columns:
        if X[column].nunique() <= 1:
            continue  # Skip constant features
        
        if np.issubdtype(X[column].dtype, np.number):  # Numerical features
            try:
                _, p_value = pointbiserialr(X[column], y)
            except ValueError:  # Handle constant input issues
                continue
        else:  # Categorical features
            try:
                contingency_table = pd.crosstab(X[column], y)
                _, p_value, _, _ = chi2_contingency(contingency_table)
            except ValueError:
                continue

        correlation_results[column] = {'P-value': p_value}

    # Convert results to DataFrame
    correlation_df = pd.DataFrame(correlation_results).T

    # Dynamically calculate max_features based on sample size
    sample_size = len(data)
    max_features = calculate_max_features(sample_size, min_features=min_features, scaling_factor=scaling_factor)

    # Filter features based on p-value threshold
    filtered_features = correlation_df[correlation_df['P-value'] < p_value_threshold].index.tolist()

    if len(filtered_features) < max_features:
        # Not enough features, select the top `max_features` ranked by p-value
        filtered_features = correlation_df.nsmallest(max_features, 'P-value').index.tolist()

    # Print the 5 most significant features and their p-values
    most_significant = correlation_df.nsmallest(5, 'P-value')
    write_to_file(f"Top 5 most significant features and their p-values:\n{most_significant}\nNumber of columns after feature extraction: {len(filtered_features)}")
    
    # Retain excluded labels in the returned dataset
    return data[filtered_features + excluded_labels + [label_col]], correlation_df

# i faced milions of errors while was trying to uise the same function for binary and multiclass label as we needed for pretraining, i seprated them, so you can also choose other categorical features to make pretrain model
# if you plan to use binary features for pretraining you can also use binary function for preprocessing data before runing models 
# in addition, most of the calsses or labels are kind of balanced, i did not look at hem deeply but apparently, over and under sampling may not be so helpful, but it is your choice your Grace :) 
def preprocess_binary_data(data, features, label_col, test_size=0.2, normalize=True, sampling='over', random_state=42):
    """
    Preprocesses data for binary classification tasks.

    Args:
        data (pd.DataFrame): Input data.
        features (list): Features to use.
        label_col (str): Binary label column.
        test_size (float): Test set size.
        normalize (bool): Whether to normalize features.
        sampling (str): Sampling method ('over', 'under', or None).
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Train, validation, and test splits for features and labels.
    """
    # Validate the label column
    y = data[label_col]
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"The label column `{label_col}` must contain only binary values (0 and 1).")

    X = data[features]

    # Split the data
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=test_size, random_state=random_state, stratify=y_train_val
    )


    write_to_file(f"Real sample shape (before sampling):\nX_train: {X_train.shape}, y_train distribution:\n{y_train.value_counts()}")

    # Apply sampling if specified
    if sampling == 'over':
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("After oversampling (SMOTE):")
        print(f"X_train: {X_train.shape}, y_train distribution:\n{pd.Series(y_train).value_counts()}")
    elif sampling == 'under':
        undersampler = RandomUnderSampler(random_state=random_state)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)
        print("After undersampling:")
        print(f"X_train: {X_train.shape}, y_train distribution:\n{pd.Series(y_train).value_counts()}")
        write_to_file(f"After undersampling:\nX_train: {X_train.shape}, y_train distribution:\n{pd.Series(y_train).value_counts()}")
# i thinks we can also use more normalization mathods to compare the result, but would be the next steps 

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)
        print("After normalization:")
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    #labe ratio as float
    y_label_ratio = y_test.mean()
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_label_ratio": y_label_ratio,
        "sex_label_ratio": -99
    }


# Preprocess Multi-Class Data, I did not use sampling fot this part, but if you want i can add sampling as well 
# i did not used age classes( i mean i used the whole ages without using age range, that is why i had to drop stratify in train test split, so if you use age ranges you can add stratify in traintest split
# we should not have single class to use stratify, all classes should have more than one sample if yoy want use stratify spliting 
def preprocess_multiclass_data(data, features, label_col, test_size=0.2, normalize=True, random_state=42):
    y = validate_labels(data, label_col)
    X = data[features]
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, random_state=random_state)

    if normalize:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        print(y_train.shape)
        print(y_val.shape)
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test, "y_train": y_train, "y_val": y_val, "y_test": y_test}

 


# In[34]:


def create_balanced_sub_set_cross_validation(data, group_column='combined', test_size=0.2, random_state=None, set_type="TEST"):
    cross_validation_data = []
    group_sizes = data.groupby(group_column).size()
    smallest_group_size = group_sizes.min()
    n_splits = smallest_group_size // 6
    n_splits = min(n_splits, 5)
    if n_splits < 2:
        write_to_file(f"Skipping cross-validation for {set_type} set because there is not enough data.")
        print(f"Group sizes: {group_sizes}")
        remaining_data, test_set = create_balanced_sub_set(data, group_column=group_column, test_size=test_size, random_state=random_state, set_type=set_type)
        cross_validation_data.append((remaining_data, test_set, n_splits))
        return cross_validation_data
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_index, test_index in skf.split(data, data[group_column]):
        write_to_file(f"Example of test_index: {test_index[:5]} for split")
        #make test_index to df like data
        test_index_df = data.iloc[test_index]
        # Determine the smallest group size
        group_sizes = test_index_df.groupby(group_column).size()
        smallest_group_size = group_sizes.min()
        # Calculate the number of samples to take per group for the test set
        n_samples = int(smallest_group_size)
        if n_samples < 1:
            raise ValueError(f"The {set_type} set size is too small. Please reduce the test_size parameter.")
        if n_samples < 6:
            n_samples = 6
            write_to_file(f"Sampling {n_samples} samples per group for the {set_type} set because there is so less data.")
        print(f"Sampling {n_samples} samples per group for the {set_type} set.")
        print(f"{set_type} set size: {n_samples * len(group_sizes)} samples.")
    
        # Sample test set
        test_set = pd.DataFrame()
        remaining_data = data.copy()
    
        for group, group_data in test_index_df.groupby(group_column):
            # Sample for test set
            group_test = group_data.sample(n=n_samples, random_state=random_state)
            test_set = pd.concat([test_set, group_test])
            # Remove test samples from remaining data
            remaining_data = remaining_data.drop(group_test.index)
        
        cross_validation_data.append((remaining_data, test_set, n_splits))
    return cross_validation_data

def create_balanced_sub_set(data, group_column='combined', test_size=0.2, random_state=None, set_type="TEST"):
    # Determine the smallest group size
    group_sizes = data.groupby(group_column).size()
    smallest_group_size = group_sizes.min()
    # Calculate the number of samples to take per group for the test set
    n_samples = int(smallest_group_size * test_size)
    if n_samples < 1:
        raise ValueError(f"The {set_type} set size is too small. for n_samplies * 0.2 {n_samples} Please reduce the test_size parameter.")
    if n_samples < 6:
        n_samples = 6
        write_to_file(f"Sampling {n_samples} samples per group for the {set_type} set because there is so less data.")
    print(f"Sampling {n_samples} samples per group for the {set_type} set.")
    print(f"{set_type} set size: {n_samples * len(group_sizes)} samples.")
    
    # Sample test set
    test_set = pd.DataFrame()
    remaining_data = data.copy()
    
    for group, group_data in data.groupby(group_column):
        # Sample for test set
        group_test = group_data.sample(n=n_samples, random_state=random_state)
        test_set = pd.concat([test_set, group_test])
        
        # Remove test samples from remaining data
        remaining_data = remaining_data.drop(group_test.index)
    
    return test_set, remaining_data

def preprocess_balanced_binary_data(data, features, label_col, gender_col, test_size=0.2, normalize=True, random_state=42, smote_upsampling=False, standard_cross_validation=True, outer_folds=5):
    """
    Preprocesses data for binary classification tasks with equal representation of combined labels in the test set.

    Args:
        data (pd.DataFrame): Input data.
        features (list): Features to use.
        label_col (str): Binary label column.
        gender_col (str): Binary gender column (0: Male, 1: Female).
        test_size (float): Test set size.
        normalize (bool): Whether to normalize features.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: Train, validation, and test splits for features and labels.
    """
    # Create combined column for gender and label
    list_of_data_splits = []

    if standard_cross_validation:
        cross_validation_data = []
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
        for train_val_index, test_index in outer_cv.split(data, data[label_col]):
            train_val_set = data.iloc[train_val_index]
            test_set = data.iloc[test_index]
            number_of_cross_validations = outer_folds
            # Sample validation set so that in the y_val are at least 6 of the minority class
            cross_validation_data.append((train_val_set, test_set, number_of_cross_validations))
    for train_val_set, test_set, number_of_cross_validations in cross_validation_data:
        X_test = test_set[features]
        y_test = test_set[label_col]
        
        print("Test set distribution by combined label:")
        print(test_set[label_col].value_counts())

        # Sample validation set so that in the y_val are at least 6 of the minority class
        if standard_cross_validation:
                validation_set, train_set = train_test_split(
                train_val_set, test_size=test_size, stratify=train_val_set[label_col], random_state=random_state
                 )
        X_train = train_set[features]
        y_train = train_set[label_col]
        X_val = validation_set[features]
        y_val = validation_set[label_col]
        if smote_upsampling:
            # Apply SMOTE to the training set only
            smote = SMOTE(random_state=random_state)
            X_train_resampled, y_train = smote.fit_resample(X_train, y_train)
            
            write_to_file("\nTraining set distribution after SMOTE:\n")
            write_to_file(pd.Series(y_train).value_counts().to_string())
            
            # Aplly SMOTE to the validation set only
            X_val_resampled, y_val = smote.fit_resample(X_val, y_val)

            write_to_file("\nValidation set distribution after SMOTE:\n")
            write_to_file(pd.Series(y_val).value_counts().to_string())
        else:
            X_train_resampled, y_train = X_train, y_train
            X_val_resampled, y_val = X_val, y_val
        write_to_file("\nTraining set distribution after SMOTE:\n")
        write_to_file(y_train.value_counts().to_string())
        write_to_file("\n+++++FINISHED SMOTING AND PREPRCESSING+++++\n")
        # Normalize features if specified
        if normalize:
            scaler = StandardScaler()
            X_train_resampled = pd.DataFrame(scaler.fit_transform(X_train_resampled), columns=features)
            X_val_resampled = pd.DataFrame(scaler.transform(X_val_resampled), columns=features)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=features)
        
        #calculate how many samples where added to the training and test set
        added_samples = len(X_train_resampled) - len(X_train)
        added_samples_test = len(X_test) - len(test_set)
        write_to_file(f"Added {added_samples} samples to the training set and {added_samples_test} samples to the test set.")

        #get the ratio of posivte to negative and of sex in the test as one float
        ratio_pos_neg = y_test.sum() / len(y_test)
        sex_ratio = test_set[gender_col].sum() / len(test_set)

        list_of_data_splits.append(
            {
                "X_train": X_train_resampled,
                "X_val": X_val_resampled,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "y_label_ratio": ratio_pos_neg,
                "sex_label_ratio": sex_ratio,
                "number_of_cross_validations": number_of_cross_validations
            }
            )
    return list_of_data_splits


# In[35]:


# Early stopping,  you can increse number of epochs, and change 'patience' if you find the model is improving somehow
early_stopping = EarlyStopping(
    monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
    )
# Pretrain Models
def build_pretraining_model(input_shape, output_units, model_type="mlp"):
    """
    Builds a pretraining model for a categorical task.

    Args:
    - input_shape (tuple): Shape of the input data (features).
    - output_units (int): Number of output classes.
    - model_type (str): Type of model to build ('mlp' or 'cnn').

    Returns:
    - tf.keras.Model: Compiled model ready for training.
    """
    if model_type == "mlp":
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(output_units, activation='softmax')  # Multiclass output
        ])
    elif model_type == "cnn":
        model = models.Sequential([
            layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=(input_shape[0], 1)),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(output_units, activation='softmax')  # Multiclass output
        ])
    else:
        raise ValueError("Unsupported model type. Choose 'mlp' or 'cnn'.")

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model




# Pretraining Workflow, I dropped y and x val and used split during training because some tasks have inconsistency for input and output, does not affect the performance
# beause of some inconsistency of numpy and tensorflow as well as scipy, i have to change and convert the data several times for different models
# if you see some parts that you think is not neccasary when they are converting data and changing dim during runing, it is because of the python version of UKB_RAP, it may not be the same in different PC or Clouds

def pretrain_models(X_train, y_train, X_val, y_val, save_path_mlp, save_path_cnn, input_shape, num_classes, epochs=100, batch_size=32):
    """
    Pretrains MLP and CNN models for a categorical task.

    Args:
    - X_train, y_train: Training data and labels.
    - X_val, y_val: Validation data and labels.
    - save_path_mlp, save_path_cnn: Paths to save the pretrained models.
    - input_shape: Shape of the input data.
    - num_classes: Number of output classes.
    - epochs, batch_size: Training hyperparameters.

    Returns:
    - Tuple of trained MLP and CNN models.
    """
    # Dynamically determine the number of classes
    print(f"Number of classes (num_classes): {num_classes}")

    print("Pretraining MLP...")
    mlp_model = build_pretraining_model(input_shape, num_classes, model_type="mlp")
    mlp_model.fit(X_train, y_train, validation_split = 0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    mlp_model.save(save_path_mlp)
    print(f"MLP pretrained model saved to: {save_path_mlp}")

    print("Pretraining CNN...")
    # Adjust input shape for CNN
    X_train_cnn = X_train.to_numpy()[..., np.newaxis]  # Convert to NumPy array and add channel dimension
    X_val_cnn = X_val.to_numpy()[..., np.newaxis]  # Convert to NumPy array and add channel dimension

    cnn_model = build_pretraining_model((input_shape[0], 1), num_classes, model_type="cnn")
    cnn_model.fit(X_train_cnn, y_train, validation_split = 0.2, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    cnn_model.save(save_path_cnn)
    print(f"CNN pretrained model saved to: {save_path_cnn}")

    return mlp_model, cnn_model


def fine_tune_both_models(mlp_path, cnn_path, X_train, y_train, X_val, y_val, X_test, y_test, output_units, epochs=100, batch_size=32):
    """
    Fine-tunes both MLP and CNN pretrained models.

    Args:
    - mlp_path, cnn_path: Paths to the pretrained models.
    - X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and test data.
    - output_units: Number of output units for binary classification.
    - epochs, batch_size: Training hyperparameters.

    Returns:
    - dict: Results for MLP and CNN fine-tuning.
    """
    print("Fine-tuning MLP...")
    mlp_results = fine_tune_model(mlp_path, X_train, y_train, X_val, y_val, X_test, y_test, output_units, model_type="mlp", epochs=epochs, batch_size=batch_size)
    
    print("Fine-tuning CNN...")
    cnn_results = fine_tune_model(cnn_path, X_train, y_train, X_val, y_val, X_test, y_test, output_units, model_type="cnn", epochs=epochs, batch_size=batch_size)
    
    return {"MLP": mlp_results, "CNN": cnn_results}
# Fine-Tuning Pretrained Models
def fine_tune_model(pretrained_path, X_train, y_train, X_val, y_val, X_test, y_test, output_units, model_type="mlp", epochs=100, batch_size=32):
    """
    Fine-tunes a pretrained model for binary classification.

    Args:
    - pretrained_path: Path to the pretrained model.
    - X_train, y_train, X_val, y_val, X_test, y_test: Training, validation, and test data.
    - output_units: Number of output units (e.g., 1 for binary classification).
    - model_type: "mlp" or "cnn".
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.

    Returns:
    - dict: Evaluation metrics (ACC, AUC, Classification Report).
    """


    print(f"Loading pretrained model from {pretrained_path} for fine-tuning...")
    model = load_model(pretrained_path)
    # i did not change anything for input as you said we need to pretrain for reduced features, so if you need to train only on full data and try finetuning in reduced data, we can add an input layer to make it a little flexible
    
    # Modify the final layer for fine-tuning
    if hasattr(model, 'pop'):  # Some Keras models support pop()
        model.pop()
    else:  # Recreate the model without the last layer
        model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    model.add(Dense(output_units, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Adjust input shape for CNN
    if model_type == "cnn":
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
            X_val = X_val.values
            X_test = X_test.values
        X_train = X_train[..., np.newaxis]  # Add channel dimension
        X_val = X_val[..., np.newaxis]      # Add channel dimension
        X_test = X_test[..., np.newaxis]    # Add channel dimension

    # Debugging input shapes
    print(f"Fine-tuning {model_type.upper()} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Train the model
    print(f"Starting fine-tuning for {model_type.upper()}...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])
    print(f"Fine-tuning for {model_type.upper()} complete!")

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, np.round(y_pred))
    balanced_acc = balanced_accuracy_score(y_test, np.round(y_pred))
    report = classification_report(y_test, np.round(y_pred))

    #Evaluat against random permutation
    random_y_test = np.random.randint(0, 2, size=y_test.shape)
    random_balanced_acc = balanced_accuracy_score(random_y_test, np.round(y_pred))
    return {"ACC": acc, "AUC": auc, "Balanced ACC": balanced_acc, "Random Balanced ACC" : random_balanced_acc, "Classification Report": report}




# Debugging Preprocessing Steps
def debug_preprocessing(data, sampling, use_feature_extraction, label_col):
    print("\n--- Initial Data ---")
    print(data.head())

    if sampling:
        print("\n--- After Sampling ---")
        print(data.head())

    if use_feature_extraction:
        print("\n--- After Feature Extraction ---")
        print(data[label_col].head())







# In[36]:


# Build CNN model
def build_cnn_model(input_shape, output_units):
    model = models.Sequential([
        layers.Conv1D(256, kernel_size=3, activation='relu', input_shape=(input_shape[0], 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_units, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build MLP model
def build_mlp_model(input_shape, output_units):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_units, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate ML models
def train_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # For multiclass, we need to use predict_proba and different ROC AUC calculation
    y_pred_proba = model.predict_proba(X_test)
    
    # ROC AUC for multiclass using 'macro' average
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # For random comparison, use the number of classes instead of 2
    n_classes = len(np.unique(y_test))
    random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
    random_balanced_acc = balanced_accuracy_score(random_y_test, y_pred)
    
    return {
        "AUC": auc, 
        "ACC": acc, 
        "Balanced ACC": balanced_acc, 
        "Random Balanced ACC": random_balanced_acc, 
        "Classification Report": report
    }

# Train and evaluate DL models
# Train and evaluate DL models
def train_dl_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_type="mlp", epochs=100, batch_size=32):
    """
    Trains and evaluates a deep learning model.

    Args:
    - model: Keras model to train.
    - X_train, X_val, X_test: Training, validation, and test features (DataFrames or NumPy arrays).
    - y_train, y_val, y_test: Training, validation, and test labels.
    - model_type: "mlp" or "cnn".
    - epochs: Number of epochs for training.
    - batch_size: Batch size.

    Returns:
    - dict: Evaluation metrics (ACC, AUC, Balanced ACC, Classification Report).
    """


    # Convert DataFrames to NumPy arrays if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values

    # Add channel dimension for CNN input
    if model_type == "cnn":
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping])

    # Evaluate the model
    y_pred = model.predict(X_test).flatten()
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, np.round(y_pred))
    balanced_acc = balanced_accuracy_score(y_test, np.round(y_pred))
    report = classification_report(y_test, np.round(y_pred))

    random_y_test = np.random.randint(0, 2, size=y_test.shape)
    random_balanced_acc = balanced_accuracy_score(random_y_test, np.round(y_pred))
    return {"AUC": auc, "ACC": acc, "Balanced ACC": balanced_acc, "Random Balanced ACC" : random_balanced_acc, "Classification Report": report}


# In[37]:


def feature_extraction(data, excluded_labels=None, p_value_threshold=0.05, min_features=5, scaling_factor=10, use_feature_extraction=True):
    if not use_feature_extraction:
        print("Feature extraction is disabled. Returning the original dataset.")
        return data, None
    # Ensure excluded_labels is a list
    excluded_labels = excluded_labels or []
    excluded_labels = [col for col in excluded_labels if col is not None]
    if not isinstance(excluded_labels, list):
        raise ValueError("excluded_labels must be a list or None.")
    #if there is a None value in the list, drop it
    excluded_labels = [label for label in excluded_labels if label is not None]
    excluded_features = set(excluded_labels)
    

    y = data["y_train"].astype(int)  # Ensure binary label is numeric (0/1)
    X = data["X_train"]  # Extract relevant feature columns
    write_to_file(f"Number of columns before feature extraction: {len(X.columns)}")

    # Store correlation results
    correlation_results = {}
    for column in X.columns:
        if X[column].nunique() <= 1:
            continue  # Skip constant features
        
        if np.issubdtype(X[column].dtype, np.number):  # Numerical features
            try:
                _, p_value = pointbiserialr(X[column], y)
            except ValueError:  # Handle constant input issues
                continue
        else:  # Categorical features
            try:
                contingency_table = pd.crosstab(X[column], y)
                _, p_value, _, _ = chi2_contingency(contingency_table)
            except ValueError:
                continue

        correlation_results[column] = {'P-value': p_value}

    # Convert results to DataFrame
    correlation_df = pd.DataFrame(correlation_results).T

    # Dynamically calculate max_features based on sample size
    sample_size = len(X)
    max_features = calculate_max_features(sample_size, min_features=min_features, scaling_factor=scaling_factor)

    # Filter features based on p-value threshold
    filtered_features = correlation_df[correlation_df['P-value'] < p_value_threshold].index.tolist()

    if len(filtered_features) < max_features:
        # Not enough features, select the top `max_features` ranked by p-value
        filtered_features = correlation_df.nsmallest(max_features, 'P-value').index.tolist()

    # Print the 5 most significant features and their p-values
    most_significant = correlation_df.nsmallest(5, 'P-value')
    write_to_file(f"Top 5 most significant features and their p-values:\n{most_significant}\nNumber of columns after feature extraction: {len(filtered_features)}")
    data["X_train"] = data["X_train"][filtered_features]
    data["X_val"] = data["X_val"][filtered_features]
    data["X_test"] = data["X_test"][filtered_features]

    # Retain excluded labels in the returned dataset
    return data, correlation_df


# In[38]:


def train_models(perform_pretraining, use_feature_extraction, brain_csv_path, labels_csv_paths, label_col, age_col=None, id_col="ID", sex_col="sex"):
    try:
        number_of_positive= -99
        number_of_negative= -99
        number_of_cross_validation = -99
        mri_table_type = brain_csv_path.split('/')[-1]
        merged_data = prepare_dataset(
            brain_csv_path=brain_csv_path,
            labels_csv_path=labels_csv_paths,
            id_col=id_col,
            label_col=label_col,
            additional_cols=[age_col],
            smote_upsampling_for_sex=sex_col is not None,
            sex_col=sex_col
        )

        if len(merged_data) < 59:
            write_to_file(f"Dataset size for label {label_col} and {brain_csv_path.split('/')[-1]} is too small for training {len(merged_data)}")
        number_of_positive = len(merged_data[merged_data[label_col] == 1])
        number_of_negative = len(merged_data[merged_data[label_col] == 0])
        number_of_male = len(merged_data[merged_data[sex_col] == 1])
        number_of_female = len(merged_data[merged_data[sex_col] == 0])
        write_to_file(f"Number of positive, before sampling: {number_of_positive}")
        write_to_file(f"Number of negative, before sampling: {number_of_negative}")
        write_to_file(f"Number of male, before sampling {number_of_male}")
        write_to_file(f"Number of female, before sampling {number_of_female}")
        # Optional Feature Extraction True or False

        if sex_col is not None:
            processed_data_test_sampled = preprocess_balanced_binary_data(
                data=merged_data.copy(deep=True),
                features=[col for col in merged_data.columns if col not in [id_col, label_col, age_col, sex_col]],
                label_col=label_col,
                gender_col=sex_col,
            )
        
        for cross_val_count, processed_data in enumerate(processed_data_test_sampled):
            y_label_ratio = processed_data['y_label_ratio']
            sex_label_ratio = processed_data["sex_label_ratio"]
            number_of_cross_validation = processed_data["number_of_cross_validations"]
    
            if use_feature_extraction:
                print("Performing feature extraction...")
                processed_data, correlation_results = feature_extraction(
                    data = processed_data,
                    excluded_labels=None,
                    p_value_threshold=0.05
                )

            # Step 5: Training ML Models (Logistic Regression and Random Forest)
            # I just used RF and LR, i guess using more ML methods would not help us, as we only focus on deep side, we can also add more but i guess does not make sence, these two would be enough as baseline 
            ml_results = {}
            print('................................................................................................')
            print('ML is running... please wait :)')
            # Logistic Regression
            catgb_model = CatBoostClassifier(verbose=0, random_state=42, loss_function='MultiClass')
            ml_results['CatBoost'] = train_ml_model(
                model=catgb_model,
                X_train=processed_data['X_train'],
                y_train=processed_data['y_train'],
                X_test=processed_data['X_test'],
                y_test=processed_data['y_test']
            )

            """ lightgbm_model = LGBMClassifier(random_state=42, objective='multiclass', num_class=processed_data['y_train'].shape[1])
            ml_results['LightGBM'] = train_ml_model(
                model=lightgbm_model,
                X_train=processed_data['X_train'],
                y_train=processed_data['y_train'],
                X_test=processed_data['X_test'],
                y_test=processed_data['y_test']
            ) """

            xgb_model = XGBClassifier(random_state=42, objective='multi:softprob')
            ml_results['XGBoost'] = train_ml_model(
                model=xgb_model,
                X_train=processed_data['X_train'],
                y_train=processed_data['y_train'],
                X_test=processed_data['X_test'],
                y_test=processed_data['y_test']
            )
            

            tabpf_model = TabPFNClassifier()
            ml_results['TabPFN'] = train_ml_model(
                model=tabpf_model,
                X_train=processed_data['X_train'],
                y_train=processed_data['y_train'],
                X_test=processed_data['X_test'],
                y_test=processed_data['y_test']
            )

            #Random Forest
            rf_model = RandomForestClassifier(random_state=42)
            ml_results['Random Forest'] = train_ml_model(
                model=rf_model,
                X_train=processed_data['X_train'],
                y_train=processed_data['y_train'],
                X_test=processed_data['X_test'],
                y_test=processed_data['y_test']
            )
            # Display ML Results
            for model_name, result in ml_results.items():
                print(f"\n--- {model_name} Results ---")
                print(f"AUC: {result['AUC']:.4f}")
                print(f"Balanced ACC: {result['Balanced ACC']:.4f}")
                print(f"ACC: {result['ACC']:.4f}")
                print(f"Permutation Balanced ACC: {result['Random Balanced ACC']:.4f}")
                print("Classification Report:")
                print(result["Classification Report"])
                write_to_file(f"\n--- {model_name} Results ---\nAUC: {result['AUC']:.4f}\nBalanced ACC: {result['Balanced ACC']:.4f}\nACC: {result['ACC']:.4f}\nClassification Report:\n{result['Classification Report']}", print_inplace=False)
                write_to_csv(   label_col=label_col,
                                mri_table=mri_table_type,
                                number_of_pos=number_of_positive,
                                number_of_neg=number_of_negative,
                                test_set_size=len(processed_data['y_test']),
                                y_label_ratio= y_label_ratio,
                                sex_label_ratio=sex_label_ratio,
                                Feature_extraction_applied=use_feature_extraction,
                                Pretraining_applied=False,
                                model_type=model_name,
                                Accuracy=result['ACC'],
                                AUC=result['AUC'],
                                Balanced_ACC=result['Balanced ACC'],
                                Permutation_Balanced_ACC=result['Random Balanced ACC'],
                                number_of_cross_validations=number_of_cross_validation,
                                cross_validation_count=cross_val_count)
            del processed_data
            gc.collect()
        return True
    except Exception as e:
        write_to_file(f"Error: {e} for {label_col} and {brain_csv_path.split('/')[-1]}", print_inplace=True)
        write_to_csv(  label_col=label_col,
                        mri_table=mri_table_type,
                        number_of_pos=number_of_positive,
                        number_of_neg=number_of_negative,
                        test_set_size=-99,
                        y_label_ratio= -99,
                        sex_label_ratio=-99,
                        Feature_extraction_applied=use_feature_extraction,
                        Pretraining_applied=False,
                        model_type=f'Error {e}',
                        Accuracy=-99,
                        AUC=-99,
                        Balanced_ACC=-99,
                        Permutation_Balanced_ACC=-99,
                        number_of_cross_validations=number_of_cross_validation,
                        cross_validation_count=-99) 
        return False


# In[39]:


def choose_labels(path, label_col_types, specific_labels, cols_to_keep=['ID', 'sex'], path_to_save="/zi/home/esra.lenz/Documents/00_HITKIP/00_CLIP/00_NAKO/02_Validation_of_UKB/99_logs/"):
    # Load the dataset
    df = pd.read_csv(path)

    # Initialize a set to track unique columns to keep
    unique_columns = set(cols_to_keep)

    # Loop through each label column type
    for label_col_typ in label_col_types:
        # Filter columns that match the label column type
        matching_columns = [col for col in df.columns if label_col_typ in col]
        unique_columns.update(matching_columns)

    # Add specific labels to the unique columns
    unique_columns.update(specific_labels)

    #sort the columns
    unique_columns_list = sorted(list(unique_columns))
    # Ensure only unique columns are kept in the dataframe
    filtered_df = df[unique_columns_list]

    new_name = path.split('/')[-1].split('.')[0]
    new_path = f"{new_name}_filtered.csv"
    new_path = path_to_save + new_path
    # Save the filtered dataframe back to the same path
    filtered_df.to_csv(f"{new_path}", index=False)

    print(f"Filtered dataframe saved to {path} with columns: {unique_columns_list}")
    write_to_file(f"Filtered dataframe saved to {path} with columns: {unique_columns_list}")
    return new_path


# In[40]:


def load_mri_paths(mri_tables):
    mri_paths = []
    for file in os.listdir(mri_tables):
        if file.endswith(".csv"):
            mri_paths.append(os.path.join(mri_tables, file))
    return mri_paths

def get_all_labels(df, inidcator = "label_"):
    labels = []
    for col in df.columns:
        if inidcator in col:
            labels.append(col)
    return labels

def view_merge_details_fct(labels, all_label_df, mri_paths):
    for label in labels:
        filterd_df = all_label_df[["ID", label]]
        print(f"Value count for ", filterd_df[label].value_counts())
        for mri_path in mri_paths:
            mri_df = pd.read_csv(mri_path)
            mri = mri_path.split("/")[-1].replace(".csv", "")
            merged_df = filterd_df.merge(mri_df, on="ID", how="inner")
            print(f"Shape for {label} and {mri} is \n", merged_df.shape)
            print(f"Value count for {label} after merge \n", merged_df[label].value_counts())

def get_data_mri_and_label_path(
        all_labels_path = "/zi/home/esra.lenz/Documents/00_HITKIP/00_CLIP/00_NAKO/02_Validation_of_UKB/00_data/depression_3.csv", 
        mri_tables_path="/zi/home/esra.lenz/Documents/00_HITKIP/00_CLIP/00_NAKO/02_Validation_of_UKB/00_data/mri_table_for_loop/",
        view_merge_details=False):
    mri_paths = load_mri_paths(mri_tables_path)
    all_label_df = pd.read_csv(all_labels_path)
    labels= get_all_labels(df=all_label_df, inidcator = "label_")
    if view_merge_details:
        view_merge_details_fct(labels, all_label_df, mri_paths)
    return all_label_df, mri_paths, labels


# In[41]:


def iteration_loop(all_labels_path, mri_pathes, labels,time_tag):
    for label in labels:
        for mri_path in mri_pathes:
            print("Now Training")
            write_to_file(input=f"Label: {label}, MRI: {mri_path.split('/')[-1]}\n")
            write_to_file(input=f"search_term: {label}_{mri_path.split('/')[-1]}\n")
            write_to_file(input="\n##### With Feature Extraction #####\n") 
            succeeded = train_models(perform_pretraining=False, use_feature_extraction=True, brain_csv_path=mri_path, labels_csv_paths=all_labels_path, label_col=label)
            if succeeded == False:
                continue
            write_to_file(input="\n##### Without Feature Extraction #####\n")
            train_models(perform_pretraining=False, use_feature_extraction=False, brain_csv_path=mri_path, labels_csv_paths=all_labels_path, label_col=label)
            write_to_file(input="\n##############################################################################################################\n")
        sleep(5)


# In[42]:


all_labels_path = choose_labels(path="/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/age_label/all_ages.csv", label_col_types=[], specific_labels=["label_age_group"])


# In[43]:


all_label_df, mri_pathes, labels = get_data_mri_and_label_path(all_labels_path=all_labels_path,
    mri_tables_path="/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/deconfounded_but_age/")
time_tag = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(mri_pathes)
iteration_loop(all_labels_path, mri_pathes, labels,time_tag)
write_to_file(input="FINISHED TERMINATING INSTANCE")

