#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tabpfn import TabPFNClassifier
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
import gc
import torch
from tensorflow.keras import backend as K
import warnings
warnings.filterwarnings('ignore')


# In[4]:


def create_mlp_model(input_shape, num_classes):
    model = Sequential([
        Dense(1024, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[5]:


def clean_up_cuda(model):
    # Delete the Keras model
    K.clear_session()
    del model
    
    # Run garbage collection
    gc.collect()
    
    # Free CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print("CUDA memory cleared and model deleted.")


# In[6]:


def aggregate_cv_metrics(all_results):
    """
    Aggregate cross-validation metrics
    
    Parameters:
    all_results (list): List of results dictionaries from each fold
    
    Returns:
    dict: Aggregated metrics with means and standard deviations
    """
    # Initialize aggregation dictionary
    aggregated = {
        'accuracy': [],
        'balanced_accuracy': [],
        'random_balanced_accuracy': [],
        'roc_auc': []
    }
    
    # Collect metrics from each fold
    for result in all_results:
        aggregated['accuracy'].append(result['accuracy'])
        aggregated['balanced_accuracy'].append(result['balanced_accuracy'])
        aggregated['random_balanced_accuracy'].append(result['random_balanced_accuracy'])
        aggregated['roc_auc'].append(result['roc_auc'])
    # Compute mean and standard deviation
    summary = {
        'mean_accuracy': np.mean(aggregated['accuracy']),
        'std_accuracy': np.std(aggregated['accuracy']),
        'mean_balanced_accuracy': np.mean(aggregated['balanced_accuracy']),
        'std_balanced_accuracy': np.std(aggregated['balanced_accuracy']),
        'mean_random_balanced_accuracy': np.mean(aggregated['random_balanced_accuracy']),
        'std_random_balanced_accuracy': np.std(aggregated['random_balanced_accuracy']),
        'mean_roc_auc': np.mean(aggregated['roc_auc']),
        'std_roc_auc': np.std(aggregated['roc_auc'])
    }
    
    return summary


# In[7]:


df = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/deconfounded_but_age/aparc.thickness_aseg.volume_aparc.volume.csv")
label_df = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/age_label/all_ages_all_ids_healthy.csv")
n_splits = 5


label_df = label_df[['ID', 'label_age_group']]
merged_df = pd.merge(df, label_df, on='ID', how='inner')
merged_df.dropna(inplace=True)
df_sampled, _ = train_test_split(merged_df, train_size=10000, stratify=merged_df["label_age_group"], random_state=42)
df_sampled["label_age_group"].value_counts()


# In[8]:


df_control = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/01_Validation_data_set/00_data/final_folder/aparc.thickness_aparc.volume_aseg.volume.csv")
label_df_control = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/01_Validation_data_set/00_data/final_folder/aparc.thickness_aparc.volume_aseg.volume_label.csv")

label_df_control = label_df_control[['ID', 'label_age_group']]
df_control = df_control[df.columns]
merged_df_control = pd.merge(df_control, label_df_control, on='ID', how='inner')
merged_df_control.dropna(inplace=True)

X_control = merged_df_control.drop(["ID", "label_age_group"], axis=1)
y_control = merged_df_control["label_age_group"]

merged_df_control["label_age_group"].value_counts()


# In[9]:


y = df_sampled["label_age_group"]
X = df_sampled.drop(["ID", "label_age_group"], axis=1)


# In[10]:


skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_results = {
        'accuracy': [],
        'balanced_accuracy': [],
        'roc_auc': [],
        'classification_reports': []
    }

tab_pfn = TabPFNClassifier()


# In[11]:


def evaluate_model_performance_train(y_test, y_pred, y_pred_proba, y_val_bin=None):
    # Compute basic metrics
    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Random comparison
    n_classes = len(np.unique(y_test))
    random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
    random_balanced_acc = balanced_accuracy_score(random_y_test, y_pred)
    
    # ROC AUC (if probabilities provided)
    if y_val_bin is not None:
        y_test = y_val_bin
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    
    # Prepare results
    results = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'random_balanced_accuracy': random_balanced_acc,
        'classification_report': report
    }
    
    if auc is not None:
        results['roc_auc'] = auc
    
    return results, balanced_acc

# Example usage
def print_model_performance(results):
    """
    Print model performance metrics
    
    Parameters:
    results (dict): Performance metrics from evaluate_model_performance()
    """
    for metric, value in results.items():
        if metric == 'classification_report':
            print("\nClassification Report:")
            print(value)
        else:
            print(f"{metric.replace('_', ' ').title()}: {value}")


# In[12]:


def predict_and_evaluate(model, X_val, y_val, original_classes=None, multi_class=False):
    if multi_class:
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        #print(y_pred)
    else:
        y_pred_proba = model.predict_proba(X_val)
        y_pred = model.predict(X_val)
        #print(y_pred)
    
    # Get unique classes present in validation data
    present_classes = np.unique(y_val)
    
    # Get the indices of these classes in the original prediction probabilities
    class_indices = [np.where(original_classes == cls)[0][0] for cls in present_classes]
    
    # Select only the probability columns for present classes
    y_pred_proba_filtered = y_pred_proba[:, class_indices]
    
    # Binarize the true labels using only the present classes
    y_val_bin = label_binarize(y_val, classes=present_classes)

    results, balanced_acc = evaluate_model_performance_train(y_val, y_pred, y_pred_proba_filtered, y_val_bin)
    print_model_performance(results)
    return results, balanced_acc


# In[ ]:


tabpfn_results = []
tabpfn_results_eval = []
lgb_results = []
lgb_results_eval = []
random_results = []
mlp_results = []
mlp_results_eval = []
model_dict = {}
model_results = {}


best_balanced_accuracy_mlp = 0
best_balanced_accuracy_tabpfn = 0
best_balanced_accuracy_lgb = 0
for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[val_index]
        y_train, y_test = y.iloc[train_index], y.iloc[val_index]

        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        X_control = scaler.fit_transform(X_control)

        n_classes = len(np.unique(y_test))
        random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
        random_y_pred_proba = np.random.rand(len(y_test), n_classes)
        random_y_pred_proba /= random_y_pred_proba.sum(axis=1)[:, np.newaxis]
        results, balanced_accuracy  = evaluate_model_performance_train(y_test, random_y_test, random_y_pred_proba)
        print("RANDOM PERFORMANCE")
        print_model_performance(results)
        random_results.append(results)
                


        mlpclf = create_mlp_model(input_shape=X_train.shape[1], num_classes=len(y.unique()))
        mlpclf.fit(X_train, pd.get_dummies(y_train), epochs=10, batch_size=32, verbose=0)
        y_pred_proba = mlpclf.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        results, balanced_accuracy = evaluate_model_performance_train(y_test, y_pred, y_pred_proba)
        print("MLP PERFORMANCE")
        print_model_performance(results)
        mlp_results.append(results)
        #model_dict["mlp"] = mlpclf
        print("MLP PERFORMANCE FOR CONTROL")
        results, balanced_accuracy = predict_and_evaluate(mlpclf, X_control, y_control, original_classes = np.unique(y_train), multi_class=True)
        mlp_results_eval.append(results)
        if balanced_accuracy > best_balanced_accuracy_mlp:
                best_balanced_accuracy_mlp = balanced_accuracy
                model_dict["mlp"] = mlpclf
        clean_up_cuda(mlpclf)

        tabclf = TabPFNClassifier()
        tabclf.fit(X_train, y_train)
        y_pred_proba = tabclf.predict_proba(X_test)
        y_pred = tabclf.predict(X_test)
        results, balanced_accuracy = evaluate_model_performance_train(y_test, y_pred, y_pred_proba)
        print("tabpfn PERFORMANCE")
        print_model_performance(results)
        tabpfn_results.append(results)
        #model_dict["tabpfn"] = tabclf
        original_classes = tabclf.classes_
        print("tabpfn PERFORMANCE FOR CONTROL")
        results, balanced_accuracy = predict_and_evaluate(tabclf, X_control, y_control, original_classes=original_classes)
        tabpfn_results_eval.append(results)
        if balanced_accuracy > best_balanced_accuracy_tabpfn:
                best_balanced_accuracy_tabpfn = balanced_accuracy
                model_dict["tabpfn"] = tabclf
        clean_up_cuda(tabclf)
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
        params = {
        'objective': 'multiclass',
        'num_class': len(y.unique()),
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'seed': 42
        }
        lgbclf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=1000)
        y_pred_proba = lgbclf.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        results, balanced_accuracy = evaluate_model_performance_train(y_test, y_pred, y_pred_proba)
        print("LGBM PERFORMANCE")
        print_model_performance(results)
        lgb_results.append(results)
        print("LGBM PERFORMANCE FOR CONTROL")
        results, balanced_accuracy = predict_and_evaluate(lgbclf, X_control, y_control, original_classes=original_classes, multi_class=True)
        lgb_results_eval.append(results)
        if balanced_accuracy > best_balanced_accuracy_lgb:
                best_balanced_accuracy_lgb = balanced_accuracy
                model_dict["lgb"] = lgbclf
        clean_up_cuda(lgbclf)


# In[85]:


cv_summary_ = aggregate_cv_metrics(random_results)
print("\nRandom Classifier Performance:")
print_model_performance(cv_summary_)

cv_summary_ = aggregate_cv_metrics(tabpfn_results)
print("\nTabPFN Performance:")
print_model_performance(cv_summary_)

cv_summary_ = aggregate_cv_metrics(lgb_results)
print("\nLGBM Performance:")
print_model_performance(cv_summary_)

cv_summary_ = aggregate_cv_metrics(mlp_results)
print("\nMLP Performance:")
print_model_performance(cv_summary_)


# In[86]:


cv_summay_control = aggregate_cv_metrics(tabpfn_results_eval)
print("\nTabPFN Performance on control data:")
print_model_performance(cv_summay_control)

cv_summay_control = aggregate_cv_metrics(lgb_results_eval)
print("\nLGBM Performance on control data:")
print_model_performance(cv_summay_control)

cv_summay_control = aggregate_cv_metrics(mlp_results_eval)
print("\nMLP Performance on control data:")
print_model_performance(cv_summay_control)


# In[15]:


#load a model
import pickle
import os
save_dir = "../98_models/"
with open(os.path.join(save_dir, "tabpfn.pkl"), "rb") as f:
    model = pickle.load(f)
    original_classes = np.unique(y_control)
    results, balanced_accuracy = predict_and_evaluate(model, X_control, y_control, original_classes=original_classes)
    print_model_performance(results)


# In[ ]:




