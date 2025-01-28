#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier


# In[23]:


#load all dfs form a folder and merge them on ID
""" path = "/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/deconfounded_but_age/"
for i, file in enumerate(os.listdir(path)):
    if i == 0:
        df = pd.read_csv(path + file)
    else:
        df = pd.merge(df, pd.read_csv(path + file), on='ID') """
df = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/deconfounded_but_age/aseg.volume_aparc.volume_aparc.thickness.csv")
label_df = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/00_NAKO/00_data/age_label/all_ages.csv")

label_df = label_df[['ID', 'label_age_group']]


# In[24]:


for col in df.columns:
    print(col)


# In[25]:


#merge the dataframes on ID
merged_df = pd.merge(df, label_df, on='ID', how='inner')


# In[26]:


merged_df["label_age_group"].value_counts()
merged_df.dropna(inplace=True)

#get a sample exact from the data  10000 but having the same distribution of the labels
df_sampled, _ = train_test_split(merged_df, train_size=10000, stratify=merged_df["label_age_group"], random_state=42)

#drop specific labels for label_age_group
df_sampled = df_sampled[df_sampled.label_age_group != 10]


# In[27]:


len(df_sampled)


# In[28]:


df_sampled["label_age_group"].value_counts()


# In[29]:


y = df_sampled["label_age_group"]
X = df_sampled.drop(["ID", "label_age_group"], axis=1)


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# In[31]:


#get coutns from y_train and y_test
print(y_train.value_counts())
print(y_test.value_counts())


# In[32]:


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


# In[11]:


import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report


# Initialize the classifier
clf = TabPFNClassifier()

# Fit the model
clf.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = clf.predict_proba(X_test)

y_pred = clf.predict(X_test)




# In[12]:


# ROC AUC for multiclass using 'macro' average
# ROC AUC for multiclass using 'macro' average
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')

acc = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# For random comparison, use the number of classes instead of 2
n_classes = len(np.unique(y_test))
random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
random_balanced_acc = balanced_accuracy_score(random_y_test, y_pred)


# In[13]:


print(f"ROC AUC: {auc}")
print(f"Accuracy: {acc}")
print(f"Balanced Accuracy: {balanced_acc}")
print(f"Random Balanced Accuracy: {random_balanced_acc}")
print(report)


# In[14]:


#verify on onther dataset
df_control = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/01_Validation_data_set/00_data/deconfounded_but_age/aseg.volume_aparc.thickness_aparc.volume.csv")
label_df = pd.read_csv("/zi/home/esra.lenz/Documents/00_HITKIP/09_TABPFN/01_Validation_data_set/00_data/age_label/all_ages.csv")

label_df_control = label_df[['ID', 'label_age_group']]
df_control = df_control[df.columns]


# In[15]:


#get the label and the data
merged_df_control = pd.merge(df_control, label_df_control, on='ID', how='inner')
merged_df_control.dropna(inplace=True)

X_control = merged_df_control.drop(["ID", "label_age_group"], axis=1)
y_control = merged_df_control["label_age_group"]


# In[16]:


y_pred_control = clf.predict(X_control)
y_pred_proba_control = clf.predict_proba(X_control)


# In[17]:


#get the unqie from ypred
np.unique(y_pred_control)


# In[18]:


#get vlaue count for label age group
print(merged_df_control["label_age_group"].value_counts())
#get how often unique values are in y_pred_control
np.unique(y_pred_control, return_counts=True)


# In[19]:


from sklearn.preprocessing import label_binarize
# Get the classes present in the validation data
present_classes = np.unique(y_control)

# Get the indices of these classes in the original prediction probabilities
original_classes = clf.classes_
class_indices = [np.where(original_classes == cls)[0][0] for cls in present_classes]

# Select only the probability columns for present classes
y_pred_proba_filtered = y_pred_proba_control[:, class_indices]

# Binarize the true labels using only the present classes
y_train_bin = label_binarize(y_control, classes=present_classes)

# Calculate ROC AUC only for present classes
auc_control = roc_auc_score(y_train_bin, y_pred_proba_filtered, 
                           multi_class='ovr', average='macro')


# In[20]:


#ROC
#auc = roc_auc_score(y_control, y_pred_proba_control, multi_class='ovr', average='macro')
#auc_control = roc_auc_score(y_train, y_pred_proba_control, multi_class='ovr', average='macro')
acc = accuracy_score(y_control, y_pred_control)
balanced_acc = balanced_accuracy_score(y_control, y_pred_control)
report = classification_report(y_control, y_pred_control)

# For random comparison, use the number of classes instead of 2
n_classes = len(np.unique(y_control))
random_y_test = np.random.randint(0, n_classes, size=y_control.shape)
random_balanced_acc = balanced_accuracy_score(random_y_test, y_pred_control)

print(f" Control ROC AUC: {auc_control}")
print(f" Control Accuracy: {acc}")
print(f" Control Balanced Accuracy: {balanced_acc}")
print(f" Control Random Balanced Accuracy: {random_balanced_acc}")
print(report)


# In[23]:


import lightgbm as lgb


# Train LightGBM model
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

print("Training LightGBM model...")
print(f"lenght of X_train: {len(X_train)}")
clf = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=1000)

# Predictions
y_pred_proba = clf.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Evaluate performance
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
acc = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Compare to random prediction
n_classes = len(np.unique(y_test))
random_y_test = np.random.randint(0, n_classes, size=y_test.shape)
random_balanced_acc = balanced_accuracy_score(random_y_test, y_pred)

print(f"ROC AUC: {auc:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
print(f"Random Balanced Accuracy: {random_balanced_acc:.4f}")
print(report)



print("Training LightGBM model...for control")
print(f"lenght of X_train: {len(X_control)}")

# Predict on control dataset
y_pred_control_proba = clf.predict(X_control)
y_pred_control = np.argmax(y_pred_control_proba, axis=1)




# In[27]:


from sklearn.preprocessing import label_binarize
# Get the classes present in the validation data
present_classes = np.unique(y_control)

# Get the indices of these classes in the original prediction probabilities
class_indices = [np.where(original_classes == cls)[0][0] for cls in present_classes]

# Select only the probability columns for present classes
y_pred_proba_filtered = y_pred_proba_control[:, class_indices]

# Binarize the true labels using only the present classes
y_train_bin = label_binarize(y_control, classes=present_classes)

# Calculate ROC AUC only for present classes
auc_control = roc_auc_score(y_train_bin, y_pred_proba_filtered, 
                           multi_class='ovr', average='macro')


# In[28]:


# Evaluate control set performance
acc_control = accuracy_score(y_control, y_pred_control)
balanced_acc_control = balanced_accuracy_score(y_control, y_pred_control)
report_control = classification_report(y_control, y_pred_control)

# Compare to random prediction
random_y_control = np.random.randint(0, n_classes, size=y_control.shape)
random_balanced_acc_control = balanced_accuracy_score(random_y_control, y_pred_control)

print(f"Control ROC AUC: {auc_control:.4f}")
print(f"Control Accuracy: {acc_control:.4f}")
print(f"Control Balanced Accuracy: {balanced_acc_control:.4f}")
print(f"Control Random Balanced Accuracy: {random_balanced_acc_control:.4f}")
print(report_control)


# In[ ]:




