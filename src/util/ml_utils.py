import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from xgboost import XGBRFRegressor, XGBRegressor

def k_fold_cross_val(model, X_all, y_all, k=10, up_sample_training=False,
                    up_sampler=lambda: ADASYN(random_state=42, n_neighbors=3), random_state=42, shuffle=True,
                    average='micro'):
    # average -> {None, micro, macro, weighted}
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin#:~:text=Suitability%20Macro%2Daverage%20method%20can,your%20dataset%20varies%20in%20size.
    if type(average) == list:
        if len(average) == 1:
            average = average[0]
        elif len(average) == 2:
            def recursive_call(avg_item):
                return k_fold_cross_val(model, X_all, y_all, k, up_sample_training, up_sampler, random_state, shuffle,
                                       [avg_item])

            return recursive_call(average[0]), recursive_call(average[1])
        else:
            raise NotImplementedError("This version of kfold_cross_val, only supports average param of of 2 max types")

    if shuffle:
        np.random.seed(random_state)
        idx = np.random.permutation(y_all.shape[0])
        x, y = X_all[idx], y_all[idx]
    else:
        x, y = X_all, y_all

    step = y.shape[0] / k
    folds_x = [x[int(i * step):int(0.01 + (i + 1) * step):, ] for i in range(k)]
    folds_y = [y[int(i * step):int(0.01 + (i + 1) * step):].reshape(-1, 1) for i in range(k)]

    accuracy = []
    recall = []
    precision = []
    for fold in np.arange(k):
        x_train = np.vstack([folds_x[i] for i in range(10) if i != fold])
        y_train = np.vstack([folds_y[i] for i in range(10) if i != fold]).reshape(-1)
        x_test = np.array(folds_x[fold])
        y_test = np.array(folds_y[fold]).reshape(-1)

        if up_sample_training:
            over_sampler = up_sampler()
            x_train, y_train = over_sampler.fit_resample(x_train, y_train)

        m = model()
        m.fit(x_train, y_train)
        pred = m.predict(x_test)
        if average is None:
            correct_pred = y_test[y_test == pred]
            fold_accuracy = [correct_pred[correct_pred == l].shape[0] / np.max((y_test[y_test == l].shape[0], 1)) for l
                             in np.unique(y_all)]
            # np.max(_,1) to deal with zero cases
        else:
            fold_accuracy = y_test[y_test == pred].shape[0] / y_test.shape[0]
        fold_precision, fold_recall, _, _ = metrics.precision_recall_fscore_support(y_test, pred, zero_division=0,
                                                                                    labels=np.unique(y),
                                                                                    average=average)
        # what to do with those that don't exist? 0, as if the model could not make any correct predictions.
        accuracy.append(fold_accuracy)
        recall.append(fold_recall)
        precision.append(fold_precision)

    return np.array(accuracy), np.array(precision), np.array(recall)
