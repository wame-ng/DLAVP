from __future__ import print_function
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

X_train = []
X_test = []
file1 = open('./data/train.txt','r')
file2 = open('./data/test.txt','r')
train_text = []
read_text = file1.readlines()
file1.close()
train_text.extend(read_text)
test_text = []
read_text = file2.readlines()
file2.close()
test_text.extend(read_text)

seq_length = 110
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC


def plot_roc_curve(labels, probality, legend_text, auc_tag=True):
    # fpr2, tpr2, thresholds = roc_curve(labels, pred_y)
    fpr, tpr, thresholds = roc_curve(labels, probality)  # probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    if auc_tag:
        rects1 = plt.plot(fpr, tpr, label=legend_text + ' (AUC=%6.3f) ' % roc_auc)
    else:
        rects1 = plt.plot(fpr, tpr, label=legend_text)


def lab_train():
    label = []
    for i in range(951):
        if i < 544:
            label.append(1)
        else:
            label.append(0)
    return label


def lab_test():
    label = []
    for i in range(105):
        if i < 60:
            label.append(1)
        else:
            label.append(0)
    return label


def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[seq_length - 1 - j] = protein_dict[line[len(line)-j-1]]
    return seq


for i in range(len(train_text)//2):
    line = train_text[i*2+1]
    line = line[0:len(line)-1]
    seq = seq_to_num(line, seq_length)
    X_train.append(seq)
X_train = np.array(X_train)


for i in range(len(test_text)//2):
    line = test_text[i*2+1]
    line = line[0:len(line)-1]
    seq = seq_to_num(line, seq_length)
    X_test.append(seq)
X_test = np.array(X_test)


def DLAVP(X_test):
    all_prob = {}
    all_prob[0] = []
    test = X_test
    test_label = lab_test()
    model = load_model('./DLAVP.h5')
    preds = model.predict([test])
    lstm_class = np.rint(preds)
    all_prob[0] = all_prob[0] + [val for val in preds]
    acc, precision, sensitivity, specificity, MCC = calculate_performace(len(test_label), lstm_class, test_label)
    roc = roc_auc_score(test_label, preds) * 100.0
    print(acc, precision, sensitivity, specificity, MCC, roc)
    plot_roc_curve(test_label, preds, 'DLAVP')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


DLAVP(X_test)





