from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split, \
    ShuffleSplit, LeaveOneOut, KFold, LeavePOut, GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import json
import argparse
import random
import itertools

parser = argparse.ArgumentParser(
    description='using svm to seperate')

parser.add_argument(
    '--feature_json',
    dest='feature_json',
    required=True,
    help='path to feature file')

parser.add_argument(
    '--label_json',
    dest='label_json',
    help='path to json file')

parser.add_argument(
    '--svm_model',
    dest='pickle',
    default='model/keypoint_recall.pkl',
    help='path to model file')

args = parser.parse_args()

from sklearn.externals import joblib
if __name__ == '__main__':
    clf = joblib.load(args.pickle)

    with open(args.feature_json, 'r') as input:
        feature = np.array([x['keypoints'] for x in json.load(input)])

    # feature_temp = list()
    # for a_feature in feature:
    #     a_feature = \
    #     np.append(a_feature,
    #               list(itertools.starmap(
    #                   lambda x, y: np.sign(x - y), # some non linear operation
    #                   (list(itertools.permutations(a_feature, 2))))))

    #     feature_temp.append(a_feature)

    # feature = np.array(feature_temp)

    #print(feature)
    rep = clf.predict_proba(feature)
    # for x in rep:
    #     print(x)
    #print(rep)

    resp_threshold_99 = [y > 0.99 for (x, y) in rep]

    rep = [x < y for (x, y) in rep]
    #print(rep)
    print("original positive ", sum(rep) / float(len(feature)))
    print("threshold 99 positive: ", sum(resp_threshold_99) / float(len(feature)))
    #print("total number of features: ", )

    # with open('label.json', 'r') as input:
    #     label = np.array(json.load(input))

    # print(sum([x == y for (x, y) in zip(rep, map(int, label))]))
    # print(len(label))

    # no = len([x for x in clf.predict(feature) if x == '0'])
    # print('no fight: ', no)
    # print('fight: ', len(feature) - no)
