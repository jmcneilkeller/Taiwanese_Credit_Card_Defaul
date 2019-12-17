import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks


def resample_prep(X_train, y_train, target_var):
    """ Prepares training data for resampling.
    Train test split must be done first.
    Takes training data and name of the target variable column.
    Assumes pandas dataframe inputs.
    """
    # concatenate our training data back together
    resampling = X_train.copy()
    resampling[target_var] = y_train.values
    # separate minority and majority classes
    majority_class = resampling[resampling[target_var]==0]
    minority_class = resampling[resampling[target_var]==1]
    # Get a class count to understand the class imbalance.
    print('majority_class: '+ str(len(majority_class)))
    print('minority_class: '+ str(len(minority_class)))
    return majority_class, minority_class

def upsample(target_var, minority_class, majority_class, replace=False, ratio=1.0):
    """Upsamples minority class using scikit learn resample.
    The ratio argument is the percentage of the upsampled minority class in relation
    to the majority class. Default is 1.0.
    """
    minority_upsampled = resample(minority_class,
                          replace=replace, # sample with or without replacement
                          n_samples=round(len(majority_class)*ratio),
                          random_state=23) # reproducible results
    # combine majority and upsampled minority
    upsampled = pd.concat([majority_class, minority_upsampled])
    # check new class counts
    print(upsampled[target_var].value_counts())
    # return new upsampled X_train, y_train
    y_train_upsampled = upsampled[target_var]
    X_train_upsampled = upsampled.drop(target_var, axis=1)
    return X_train_upsampled, y_train_upsampled

def upsample_SMOTE(X_train, y_train, ratio=1.0):
    """Upsamples minority class using SMOTE.
    Ratio argument is the percentage of the upsampled minority class in relation
    to the majority class. Default is 1.0
    """
    sm = SMOTE(random_state=23, sampling_strategy=ratio)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print(len(X_train_sm), len(y_train_sm))
    return X_train_sm, y_train_sm

def downsample(target_var, minority_class, majority_class, replace=False):
    # downsample majority
    majority_downsampled = resample(majority_class,
                                    replace=replace, # sample with or without replacement
                                    n_samples=len(minority_class), # match minority class
                                    random_state=23) # reproducible results
    # combine majority and upsampled minority
    downsampled = pd.concat([majority_downsampled, minority_class])
    # check new class counts
    print(downsampled[target_var].value_counts())
    # return new downsampled X_train, y_train
    X_train_downsampled = downsampled.drop(target_var, axis=1)
    y_train_downsampled = downsampled[target_var]
    return X_train_downsampled, y_train_downsampled

def downsample_Tomek(X_train, y_train):
    tl = TomekLinks()
    X_train_tl, y_train_tl = tl.fit_resample(X_train, y_train)
    print(X_train_tl.count(), len(y_train_tl))
    return X_train_tl, y_train_tl
