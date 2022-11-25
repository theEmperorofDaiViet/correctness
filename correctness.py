import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred):
    return pd.crosstab(y_true, y_pred)

def accuracy(cm):
    
    return np.diag(cm).sum() / cm.sum().sum()

def precision(cm, average='binary', pos_label = 0):
    if average == None:
        pre = []
        for i in range(0, len(cm)):
            tp = cm.iloc[i,i]
            fp = cm.iloc[i, :].sum() - tp
            fn = cm.iloc[:, i].sum() - tp
            tn = cm.sum().sum() - (tp + fp + fn)
            pre.append(tp / (tp + fp))
        return np.asarray(pre)

    elif average == 'binary':
        tp = cm.iloc[pos_label, pos_label]
        fp = cm.iloc[pos_label, :].sum() - tp
        fn = cm.iloc[:, pos_label].sum() - tp
        tn = cm.sum().sum() - (tp + fp + fn)
        return tp / (tp + fp)

    elif average == 'macro':
        pre = precision(cm, average=None)
        return np.mean(pre)

    elif average == 'micro':
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(0, len(cm)):
            tp = cm.iloc[i,i]
            fp = cm.iloc[i, :].sum() - tp
            fn = cm.iloc[:, i].sum() - tp
            tn = cm.sum().sum() - (tp + fp + fn)
            TP += tp
            FP += fp
            FN += fn
            TN += tn
        return TP / (TP + FP)

    elif average == 'weighted':
        pre = precision(cm, average=None)
        sam = list(cm.sum())
        return np.average(pre, weights=sam)

def recall(cm, average='binary', pos_label = 0):
    if average == None:
        rec = []
        for i in range(0, len(cm)):
            tp = cm.iloc[i,i]
            fp = cm.iloc[i, :].sum() - tp
            fn = cm.iloc[:, i].sum() - tp
            tn = cm.sum().sum() - (tp + fp + fn)
            rec.append(tp / (tp + fn))
        return np.asarray(rec)

    elif average == 'binary':
        tp = cm.iloc[pos_label, pos_label]
        fp = cm.iloc[pos_label, :].sum() - tp
        fn = cm.iloc[:, pos_label].sum() - tp
        tn = cm.sum().sum() - (tp + fp + fn)
        return tp / (tp + fn)

    elif average == 'macro':
        rec = recall(cm, average=None)
        return np.mean(rec)

    elif average == 'micro':
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(0, len(cm)):
            tp = cm.iloc[i,i]
            fp = cm.iloc[i, :].sum() - tp
            fn = cm.iloc[:, i].sum() - tp
            tn = cm.sum().sum() - (tp + fp + fn)
            TP += tp
            FP += fp
            FN += fn
            TN += tn
        return TP / (TP + FN)

    elif average == 'weighted':
        rec = recall(cm, average=None)
        sam = list(cm.sum())
        return np.average(rec, weights=sam)

def f1_score(cm, average='binary', pos_label = 0):
    if average == None:
        f1 = []
        pre = precision(cm, average=None)
        rec = recall(cm, average=None)
        for i in range(0, len(cm)):
            f1.append(2 * pre[i] * rec[i] / (pre[i] + rec[i]))
        return np.asarray(f1)

    elif average == 'binary':
        pre = precision(cm, pos_label=pos_label)
        rec = recall(cm, pos_label=pos_label)
        return 2 * pre * rec / (pre + rec)

    elif average == 'macro':
        f1 = f1_score(cm, average=None)
        return np.mean(f1)

    elif average == 'micro':
        pre = precision(cm, average='micro')
        rec = precision(cm, average='micro')
        return 2 * pre * rec / (pre + rec)

    elif average == 'weighted':
        f1 = f1_score(cm, average=None)
        sam = list(cm.sum())
        return np.average(f1, weights=sam)

def support(cm, average = 'binary', pos_label=0):
    if average == 'binary':
        return cm.sum(axis=1)[pos_label]
    else:
        return cm.sum().sum()

def report(cm):
    print("CLASSIFICATION REPORT:")    
    report1 = pd.DataFrame([[precision(cm, pos_label=0),
                        recall(cm, pos_label=0),
                        f1_score(cm, pos_label=0),
                        support(cm, pos_label=0)]])
    for i in range(1, len(cm)):
        report1 = pd.concat([report1, pd.DataFrame([[precision(cm, pos_label=i),
                        recall(cm, pos_label=i),
                        f1_score(cm, pos_label=i),
                        support(cm, pos_label=i)]])], axis = 0, ignore_index=True)
    report1.columns = ['precision', 'recall', 'f1-score', 'support']
    
    report2 = pd.DataFrame([['macro',
                    precision(cm, average='macro'),
                    recall(cm, average='macro'),
                    f1_score(cm, average='macro'),
                    support(cm, average='macro')]])
    type = ['macro', 'micro', 'weighted']
    for i in type[1:]:
        report2 = pd.concat([report2, pd.DataFrame([[i,
                    precision(cm, average=i),
                    recall(cm, average=i),
                    f1_score(cm, average=i),
                    support(cm, average=i)]])], axis = 0, ignore_index=True)
    report2.columns = [' ', 'precision', 'recall', 'f1-score', 'support']
    report2.set_index(' ', inplace=True)
    
    report3 = pd.Series([accuracy(cm)])
    print(report1)
    print(report2)
    print(report3.set_axis(['accuracy'], axis=0))

################################################################################
# Test:
if __name__ == '__main__':
    y_target = [1, 1, 1, 0, 0, 2, 0, 3]
    y_predicted = [1, 0, 1, 0, 0, 2, 1, 3]

    test = pd.DataFrame([[7, 8, 9], [1, 2, 3], [3, 2, 1]])
    print(test)
    print()
    print(report(test))