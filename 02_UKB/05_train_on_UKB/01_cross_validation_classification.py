#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from tabpfn import TabPFNClassifier
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
import gc
import torch
from tensorflow.keras import backend as K
import  statsmodels.api as sm
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def create_mlp_model(input_shape, num_classes):
    model = Sequential([
        layers.Dense(1024, activation="relu", input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        layers.Conv1D(128, kernel_size=3, activation='relu', input_shape=(input_shape[0], 1)),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[3]:


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


# In[4]:


def feature_extraction_best_corr_with_target(X,X_val, X_control, y, threshold=0.6, df_columns=None, number_of_features=40):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        if df_columns is not None:
            X.columns = df_columns
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(x_val, np.ndarray):
        x_val = pd.DataFrame(x_val)
        if df_columns is not None:
            x_val.columns = df_columns
    if isinstance(x_control, np.ndarray):
        x_control = pd.DataFrame(x_control)
        if df_columns is not None:
            x_control.columns = df_columns
    correlation_matrix = X.corrwith(y).abs()
    to_keep = correlation_matrix.sort_values(ascending=False).head(number_of_features).index
    X = X[to_keep]
    X_val = X_val[to_keep]
    X_control = X_control[to_keep]
    X_ret = X.to_numpy().copy()
    X_val_ret = X_val.to_numpy().copy()
    X_control_ret = X_control.to_numpy().copy()
    return X_ret, X_val_ret, X_control_ret


def feature_extraction_with_Pearson(X, X_val, X_control, y, threshold=0.6, df_columns=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        if df_columns is not None:
            X.columns = df_columns
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
        if df_columns is not None:
            X_val.columns = df_columns
    if isinstance(X_control, np.ndarray):
        X_control = pd.DataFrame(X_control)
        if df_columns is not None:
            X_control.columns = df_columns
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    X = X.drop(columns=to_drop)
    X_val = X_val.drop(columns=to_drop)
    X_control = X_control.drop(columns=to_drop)
    X_ret = X.to_numpy().copy()
    X_val_ret = X_val.to_numpy().copy()
    X_control_ret = X_control.to_numpy().copy()
    return X_ret, X_val_ret, X_control_ret

def feature_extration_with_PCA(X, X_val, X_control, n_components):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_val_pca = pca.transform(X_val)
    X_control_pca = pca.transform(X_control)
    return X_pca, X_val_pca, X_control_pca

def feature_extration_with_BE(X, X_val, X_control, y, significance_level=0.05, df_columns=None):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
        if df_columns is not None:
            X.columns = df_columns
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
        if df_columns is not None:
            X_val.columns = df_columns
    if isinstance(X_control, np.ndarray):
        X_control = pd.DataFrame(X_control)
        if df_columns is not None:
            X_control.columns = df_columns
    # Add constant for intercept
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X = sm.add_constant(X)

    while True:
        # Fit the OLS model
        model = sm.OLS(y, X).fit()
        
        # Get the p-values for each feature
        p_values = model.pvalues
        
        # Find the feature with the highest p-value
        max_p_value = p_values.max()
        
        if max_p_value > significance_level:
            # Remove the feature with the highest p-value
            feature_to_remove = p_values.idxmax()
            print(f"Removing {feature_to_remove} with p-value {max_p_value:.4f}")
            X = X.drop(columns=[feature_to_remove])
            X_val = X_val.drop(columns=[feature_to_remove])
            X_control = X_control.drop(columns=[feature_to_remove])
        else:
            break
        print("Final Feature lengthe: ", len(X.columns))
    # Return the final selected feature set (excluding the intercept)
    X_ret = X.drop(columns=['const']).to_numpy().copy()
    X_val_ret = X_val.to_numpy().copy()
    X_control_ret = X_control.to_numpy().copy()
    return X_ret, X_val_ret, X_control_ret


# In[5]:


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
def aggregate_cv_metrics_and_print(all_results, model_name, tag="Validation"):
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
    
    print(f"\n {model_name} Classifier Performance {tag}:")
    print_model_performance(summary)
    return summary


# In[6]:


mri_table = "aseg.volume_aparc.volume_aparc.thickness.csv"
df = pd.read_csv(f"../00_data/deconfounded_but_age/{mri_table}")
label_df = pd.read_csv("../00_data/age_label/all_ages_all_ids_healthy.csv")
n_splits = 5
label_col= "label_age_group"

label_df = label_df[['ID', 'label_age_group']]
merged_df = pd.merge(df, label_df, on='ID', how='inner')
merged_df.dropna(inplace=True)
df_sampled, _ = train_test_split(merged_df, train_size=10000, stratify=merged_df["label_age_group"], random_state=42)
df_sampled["label_age_group"].value_counts()



# In[7]:


label_df_control = pd.read_csv("../00_data/validation_data/00_National_Cohort/all_ages_all_ids_subset_middle_age.csv")
df_control = pd.read_csv("../00_data/validation_data/00_National_Cohort/aparc.thickness_aseg.volume_aparc.volume_deconfounded_but_age.csv")

label_df_control = label_df_control[['ID', 'label_age_group']]
merged_df_control = pd.merge(df_control, label_df_control, on='ID', how='inner')
merged_df_control.dropna(inplace=True)

X_control = merged_df_control.drop(["ID", "label_age_group"], axis=1)
y_control = merged_df_control["label_age_group"]
merged_df_control["label_age_group"].value_counts()


# In[8]:


df_sampled.drop(['Left-Thalamus-Proper', 'Right-Thalamus-Proper', 'SupraTentorialVolNotVentVox'], axis=1, inplace=True)
column_control = df_sampled.drop(["ID", "label_age_group"], axis=1).columns
X_control = X_control[column_control]


# In[9]:


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




# In[10]:


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


percentage_of_the_data = [0.01]
data = []
percentage_dict = {}
best_mse_mlp = float('inf')
best_mse_lgb = float('inf')
best_mse_tab = float('inf')
deconfounding_strategies = ["BE", "Correlation_in_Feature","Correlation_with_target", "PCA" "Nothing"]
for percentage in percentage_of_the_data:
        percentage_dict[percentage] = {}
        for deconfounding_strategy in deconfounding_strategies:
                print(f"\n=== Deconfounding Strategy: {deconfounding_strategy} ===")
                if percentage == 1:
                        print(f"\n #### TRAINING WITH {percentage} OF THE DATA ####")
                        df_sampled_subset = df_sampled
                else:
                        print(f"\n #### TRAINING WITH {percentage} OF THE DATA ####")
                        df_sampled_subset, _ = train_test_split(
                        df_sampled,
                        train_size=percentage,  # Use train_size to get desired percentage
                        stratify=df_sampled["label_age_group"],
                        random_state=42
                        )

                y = df_sampled_subset["label_age_group"]
                X = df_sampled_subset.drop(["ID", "label_age_group"], axis=1)

                print(f"Training data shape: {X.shape}, length of y: {len(y)}")
                print(f"Training data class distribution: {y.value_counts()}")
                

                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_results = {
                        'accuracy': [],
                        'balanced_accuracy': [],
                        'roc_auc': [],
                        'classification_reports': []
                }

                tab_pfn = TabPFNClassifier()

                tabpfn_results = []
                tabpfn_results_eval = []
                lgb_results = []
                lgb_results_eval = []
                random_results = []
                mlp_results = []
                mlp_results_eval = []
                cnn_results = []
                cnn_results_eval = []
                model_dict = {}
                model_results = {}


                best_balanced_accuracy_mlp = 0
                best_balanced_accuracy_tabpfn = 0
                best_balanced_accuracy_lgb = 0
                for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
                        unique_classes = np.unique(y)
                        missing_classes = [cls for cls in unique_classes if cls not in y.iloc[val_index]]
                        for cls in missing_classes:
                                cls_indices = np.where(y == cls)[0]  # Get all indices of the missing class
                                # Check if removing a sample would leave train set empty for the class
                                train_cls_indices = np.intersect1d(cls_indices, train_index)

                                if len(train_cls_indices) <= 1:
                                        # If moving the last one, instead take a duplicate from the whole y array
                                        cls_idx_to_move = np.random.choice(cls_indices, 1)[0]
                                else:
                                        cls_idx_to_move = np.random.choice(train_cls_indices, 1)[0]
                                # Add to validation set
                                val_index = np.append(val_index, cls_idx_to_move)
                                # Remove only if it's not the last one in train
                                if len(train_cls_indices) > 1:
                                        train_index = np.setdiff1d(train_index, cls_idx_to_move)
                        print(f"\nFold {fold}")
                        X_train, X_test = X.iloc[train_index], X.iloc[val_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[val_index]

                        #check if columns between control and trai nare the same
                        try:
                                column_control = X_train.columns
                                X_control = X_control[column_control]
                        except Exception as e:
                                print("Columns are not the same")
                                print(e)
                        #scaler = MinMaxScaler()
                        df_columns = X.columns
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_control_scaled = scaler.fit_transform(X_control)
                        if deconfounding_strategy == "BE":
                                X_train, X_test, X_control = feature_extration_with_BE(X_train_scaled, X_test_scaled, X_control_scaled, y_train, df_columns=df_columns)
                        elif deconfounding_strategy == "PCA":
                                X_train, X_test, X_control= feature_extration_with_PCA(X_train_scaled, X_test_scaled, X_control_scaled,  n_components=50)
                        elif deconfounding_strategy == "Correlation_in_Feature":
                                X_train, X_test, X_control = feature_extraction_with_Pearson(X_train_scaled, X_test_scaled, X_control_scaled, y_train, threshold=0.6, df_columns=df_columns)
                        elif deconfounding_strategy == "Correlation_with_target":
                                X_train, X_test, X_control = feature_extraction_best_corr_with_target(X_train_scaled, X_test_scaled, X_control_scaled, y_train, threshold=0.6, df_columns=df_columns)
                        n_classes = len(np.unique(y_test))
                        random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
                        random_y_pred_proba = np.random.rand(len(y_test), n_classes)
                        random_y_pred_proba /= random_y_pred_proba.sum(axis=1)[:, np.newaxis]
                        results, balanced_accuracy  = evaluate_model_performance_train(y_test, random_y_test, random_y_pred_proba)
                        print("RANDOM PERFORMANCE")
                        print_model_performance(results)
                        random_results.append(results)

                        cnnclf = create_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=len(y.unique()))
                        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                        X_control_cnn = X_control.reshape(X_control.shape[0], X_control.shape[1], 1)
                        cnnclf.fit(X_train_cnn, pd.get_dummies(y_train), epochs=10, batch_size=32, verbose=0)
                        y_pred_proba = cnnclf.predict(X_test_cnn)
                        y_pred = np.argmax(y_pred_proba, axis=1)
                        results, balanced_accuracy = evaluate_model_performance_train(y_test, y_pred, y_pred_proba)
                        print("CNN PERFORMANCE")
                        print_model_performance(results)
                        cnn_results.append(results)
                        print("CNN PERFORMANCE FOR CONTROL")
                        results, balanced_accuracy = predict_and_evaluate(cnnclf, X_control_cnn, y_control, original_classes = np.unique(y_train), multi_class=True)
                        cnn_results_eval.append(results)
                        clean_up_cuda(cnnclf)

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
                        'seed': 42,
                        'verbose': -1
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

                random_summary = aggregate_cv_metrics_and_print(random_results, "Random")
                tabpfn_summary = aggregate_cv_metrics_and_print(tabpfn_results, "TabPFN")
                lgb_summary = aggregate_cv_metrics_and_print(lgb_results, "LGBM")
                mlp_summary = aggregate_cv_metrics_and_print(mlp_results, "MLP")
                cnn_summary = aggregate_cv_metrics_and_print(cnn_results, "CNN")

                tabpfn_eval_summary = aggregate_cv_metrics_and_print(tabpfn_results_eval, "TabPFN", "Control")
                lgb_eval_summary = aggregate_cv_metrics_and_print(lgb_results_eval, "LGBM", "Control")
                mlp_eval_summary = aggregate_cv_metrics_and_print(mlp_results_eval, "MLP", "Control")
                cnn_eval_summary = aggregate_cv_metrics_and_print(cnn_results_eval, "CNN", "Control")

                percentage_dict[percentage][deconfounding_strategy] = {
                "TabPFN": {
                        "results": tabpfn_summary,
                        "results_eval": tabpfn_eval_summary,
                        "cv_results": tabpfn_results,
                        "cv_results_eval": tabpfn_results_eval
                },
                "LGBM": {
                        "results": lgb_summary,
                        "results_eval": lgb_eval_summary,
                        "cv_results": lgb_results,
                        "cv_results_eval": lgb_results_eval
                },
                "Random": {
                        "results": random_summary,
                        "results_eval": random_summary,
                        "cv_results": random_results,
                        "cv_results_eval": random_results,

                },
                "MLP": {
                        "results": mlp_summary,
                        "results_eval": mlp_eval_summary,
                        "cv_results": mlp_results,
                        "cv_results_eval": mlp_results_eval
                },
                "CNN": {
                        "results": cnn_summary,
                        "results_eval": cnn_eval_summary,
                        "cv_results": cnn_results,
                        "cv_results_eval": cnn_results_eval
                }
        }
        


# In[23]:


for percentage, models in percentage_dict.items():
    print(f"\nResults for {percentage*100:.0f}% of the data:")
    for model, results in models.items():
        print(f"  {model} - Results: {results['results']}")
        if 'results_eval' in results:
            print(f"  {model} - Evaluation Results: {results['results_eval']}")


# In[ ]:


Feature_extraction_applied = False
Pretraining_applied = False
all_rows = []
for percentage, model in percentage_dict.items():
    for model_name, train_summary in models.items():
        for i, (cv_result, cv_result_eval) in enumerate(zip(train_summary["cv_results"], train_summary["cv_results_eval"])):
            row_train = {
                        "label_col": label_col,
                        "mri_table": mri_table,
                        "test_set_size": f"{(1 - percentage):.2%} (approx. of data left for test)",
                        "Feature_extraction_applied": Feature_extraction_applied,
                        "Pretraining_applied": Pretraining_applied,
                        "model_type": model_name,
                        "Accuracy": cv_result.get("accuracy", None),
                        "AUC": cv_result.get("roc_auc", None),  # or "auc" or whatever your aggregator uses
                        "Balanced_ACC": cv_result.get("balanced_accuracy", None),
                        "Permutation_Balanced_ACC": cv_result.get("random_balanced_accuracy", None), 
                        "number_of_cross_validations": n_splits,
                        "cross_validation_count": i,
                        "search_term": f"{percentage}_{model_name}_train",
                        "percentage_of_data": percentage,  # storing the used fraction
                        "eval_or_train": "train"
                    }
            row_eval = {
                        "label_col": label_col,
                        "mri_table": mri_table,
                        "test_set_size": f"{(1 - percentage):.2%} (approx. of data left for test)",
                        "Feature_extraction_applied": Feature_extraction_applied,
                        "Pretraining_applied": Pretraining_applied,
                        "model_type": model_name,
                        "Accuracy": cv_result_eval.get("accuracy", None),
                        "AUC": cv_result_eval.get("roc_auc", None),  # or "auc" or whatever your aggregator uses
                        "Balanced_ACC": cv_result_eval.get("balanced_accuracy", None),
                        "Permutation_Balanced_ACC": cv_result_eval.get("random_balanced_accuracy", None), 
                        "number_of_cross_validations": n_splits,
                        "cross_validation_count": i,
                        "search_term": f"{percentage}_{model_name}_eval",
                        "percentage_of_data": percentage,  # storing the used fraction
                        "eval_or_train": "eval"
                    }
        all_rows.append(row_eval)
        all_rows.append(row_train)
df_results = pd.DataFrame(all_rows)
df_results.to_csv("results_classification.csv", index=False)


# In[ ]:


print(percentage_dict)


# In[ ]:





# In[ ]:


""" #load a model
import pickle
import os
save_dir = "../98_models/"
with open(os.path.join(save_dir, "tabpfn.pkl"), "rb") as f:
    model = pickle.load(f)
    original_classes = np.unique(y_control)
    results, balanced_accuracy = predict_and_evaluate(model, X_control, y_control, original_classes=original_classes)
    print_model_performance(results) """

