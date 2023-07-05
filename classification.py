from multiprocessing import Pool

from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, StratifiedKFold, GroupShuffleSplit, KFold
from sklearn.utils import resample
####################################### harel #############################################
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
import time
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
import umap.umap_ as umap


from classification_preprocsesing import *
from postprocess import *


def list_flatten(lst):
    flat_list = []
    for sublist in lst:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def train_test_strat_split(x_data, labels, trainSize, rnd_state):
    x_train, x_test, y_train, y_test = train_test_split(x_data, labels,
                                                        test_size=1 - trainSize, train_size=trainSize,
                                                        stratify=labels, random_state=rnd_state)
    return x_train, y_train, x_test, y_test


def performance_vs_dataset_size(x_data, labels, path, antibio, n_splits, rand_state):
    # rand_state = 0
    auc_list = []
    x_data, labels, x_test, y_test = train_test_strat_split(x_data, labels, 1-0.037, rand_state)
    # size_arr = np.arange(0.1, 0.99, 0.07)
    data_size = labels.shape[0]
    # print(data_size)
    # size_arr = np.arange(0.1, 0.99, 35/data_size)
    size_arr = np.arange(0.9, 0.01, -35/data_size)
    # temp_size_arr = size_arr*data_size
    temp_arr = []
    temp_arr_prod = []
    j = 0
    for _ in size_arr:
        temp_arr.append(size_arr[j] / np.prod(temp_arr[0:j]))
        temp_arr_prod.append(np.prod(temp_arr))
        j = j + 1
    temp_arr = np.array(temp_arr)
    temp_arr_prod = np.array(temp_arr_prod)
    actual_size_arr_prod = temp_arr_prod[(temp_arr_prod*data_size) > 200]
    actual_size_arr = temp_arr[(temp_arr_prod*data_size) > 200]
    # size_arr = size_arr[(temp_arr*data_size) > 200]
    print(actual_size_arr)
    print(actual_size_arr_prod)
    x_train, y_train = x_data, labels
    auc_list.append(grid_search_and_test(x_train, y_train, x_test, y_test, path, antibio, n_splits, rand_state))
    # auc_list.append(test_k_fold(x_train, y_train, path, antibio, n_splits, rand_state))
    for i_size in actual_size_arr:
        print(i_size)
        x_train, y_train, _, _ = train_test_strat_split(x_train, y_train, i_size, rand_state)
        # auc_list.append(test_k_fold(x_train, y_train, path, antibio, n_splits, rand_state))
        auc_list.append(grid_search_and_test(x_train, y_train, x_test, y_test, path, antibio, n_splits, rand_state))
    actual_final = actual_size_arr_prod * data_size
    actual_final = np.append([data_size], actual_final)
    plt.figure()
    plt.plot(actual_final, auc_list, label=antibio)
    plt.xlabel('training set size')
    plt.ylabel('auc score')
    plt.title('auc vs. dataset size')
    plt.legend()
    # plt.show()
    size_arr = pd.DataFrame(actual_final, columns=['x'])
    auc_list = pd.DataFrame(auc_list, columns=['y'])
    res_df = pd.concat((auc_list, size_arr), axis=1)
    # size_arr = np.array(size_arr)
    # auc_list = np.array(auc_list)
    # res_df = pd.DataFrame(np.concatenate((size_arr, auc_list), axis=1), columns=['x', 'y'])
    res_df.to_csv(str(path + antibio + "_for_plot.csv"))
    plt.savefig(str(path + antibio + ".png"))
    return auc_list


def grid_search_and_test(x_train, y_train, x_test, y_test, path, antibio, n_splits, random_state):
    grid_clf = GridSearchCV(svm.SVC(random_state=0), {
        'C': [250], 'kernel': ['rbf'], 'gamma': [0.0000001, 0.000001, 0.000005, 0.00001, 0.00004, 0.0001, 0.0005, 0.001, 0.1, 1]},
                            # , 'gamma': [0.01,0.2]
                            cv=n_splits, return_train_score=True, scoring='roc_auc')
    # grid_clf = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini', oob_score=True),
    #                         {'max_depth': [2, 4, 6, 8, 10, 12, 14, 16]},
    #                         cv=10, return_train_score=True, scoring='roc_auc')
    # print(np.sum(y == 1))
    # print(np.sum(y == 0))
    grid_clf.fit(x_train, y_train)
    # gamma_param = grid_clf.best_params_['max_depth']
    gamma_param = grid_clf.best_params_['gamma']
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print(grid_df[['param_gamma', 'mean_train_score', 'mean_test_score']])
    clf = svm.SVC(C=250, kernel='rbf', gamma=gamma_param, probability=True)
    # clf = RandomForestClassifier(n_estimators=100, random_state=random_state,
    #                              criterion='gini', oob_score=True, max_depth=gamma_param)
    clf.fit(x_train, y_train)
    probs = np.array(clf.predict_proba(x_test)[:, 1])
    fpr, tpr, thresholds_keras = roc_curve(y_test, probs)
    # print(auc(fpr, tpr))
    y_test = np.array(list_flatten(y_test))
    res_arr = np.array([1 - probs, probs, y_test, np.arange(y_test.shape[0])])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + antibio + '.csv')
    return auc(fpr, tpr)


def grid_search_kfold_and_test(x_train, y_train, x_test, y_test, path, antibio, n_splits, random_state):
    # grid_clf = GridSearchCV(svm.SVC(random_state=0), {
    #     'C': [250], 'kernel': ['rbf'], 'gamma': [0.000001, 0.000005, 0.00001, 0.00004, 0.0001, 0.0005, 0.001, 0.1, 1]},
    #                         # , 'gamma': [0.01,0.2]
    #                         cv=n_splits, return_train_score=True, scoring='roc_auc')
    grid_clf = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini', oob_score=True),
                            {'max_depth': [2, 4, 6, 8, 10, 12, 14, 16]},
                            cv=10, return_train_score=True, scoring='roc_auc')
    # print(np.sum(y == 1))
    # print(np.sum(y == 0))
    grid_clf.fit(x_train, y_train)
    gamma_param = grid_clf.best_params_['max_depth']
    # gamma_param = grid_clf.best_params_['gamma']
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    # print(grid_df[['param_gamma', 'mean_train_score', 'mean_test_score']])
    print(grid_df[['param_max_depth', 'mean_train_score', 'mean_test_score']])
    test_k_fold(x_train, y_train, path, antibio, n_splits, random_state, gamma_param)
    # clf = svm.SVC(C=250, kernel='rbf', gamma=gamma_param, probability=True)
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state,
                                 criterion='gini', oob_score=True, max_depth=gamma_param)
    clf.fit(x_train, y_train)
    probs = np.array(clf.predict_proba(x_test)[:, 1])
    fpr, tpr, thresholds_keras = roc_curve(y_test, probs)
    # print(auc(fpr, tpr))
    y_test = np.array(list_flatten(y_test))
    res_arr = np.array([1 - probs, probs, y_test, np.arange(y_test.shape[0])])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    path_test = path+'test\\'
    try:
        os.mkdir(path_test)
    except OSError:
        print("Creation of the directory %s failed" % path_test)
    else:
        print("Successfully created the directory %s " % path_test)
    res_df.to_csv(path_test + antibio + '.csv', index=False)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)


def test_k_fold(x, y, path, antibio, n_splits, random_state, grid_param):
    # kf = KFold(n_splits=1287)
    # kf = KFold(n_splits=10, random_state=random_state, shuffle=True)
    # kf.get_n_splits(X)

    # grid_clf = GridSearchCV(svm.SVC(random_state=0), {
    #     'C': [250], 'kernel': ['rbf'], 'gamma': [0.000001, 0.000005, 0.00001, 0.00004, 0.0001, 0.0005, 0.001, 0.1, 1]},
    #                         # , 'gamma': [0.01,0.2]
    #                         cv=10, return_train_score=True, scoring='roc_auc')
    # grid_clf = GridSearchCV(RandomForestClassifier(n_estimators=100, random_state=random_state, criterion='gini', oob_score=True),
    #                         {'max_depth': [2, 4, 6, 8, 10, 12, 14, 16]},
    #                         cv=10, return_train_score=True, scoring='roc_auc')
    # print(np.sum(y == 1))
    # print(np.sum(y == 0))
    # grid_clf.fit(x, y)
    # gamma_param = grid_clf.best_params_['max_depth']
    # gamma_param = grid_clf.best_params_['gamma']
    # grid_df = pd.DataFrame(grid_clf.cv_results_)
    # print(grid_df[['param_gamma', 'mean_train_score', 'mean_test_score']])  # , 'param_gamma' 'param_degree',

    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(x, y)
    probs = []
    labels = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        pcs = 80
        x_train, scale_ = standard_scale(x_train)
        x_train, pca_ = pca_transformation(x_train, pcs)
        x_test = scale_.transform(x_test)
        x_test = pca_.transform(x_test)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=gamma_param, probability=True)
        clf_rf = RandomForestClassifier(n_estimators=500, random_state=random_state, max_depth=7,
                                        criterion='gini', oob_score=True)
        ann_clf = MLPClassifier(  # -----The architecture:------#
            activation="relu",
            # What is the activation function between neurons {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}?
            hidden_layer_sizes=(pcs, pcs, pcs,),  # What is the architecture? what happens if we add more layers?
            # -----The optimizer:------#
            solver="adam",  # Stochastic Gradient Descent, other optimizers are out of the scope of the course.
            learning_rate_init=0.01,  # What is the initial learning rate? in some optimizers the learning rate changes.
            alpha=2,
            learning_rate="invscaling",
            # How does the learning rate update itself? {‘constant’, ‘invscaling’, ‘adaptive’}
            power_t=0.5,
            # When we choose learning rate to be invscaling, it means that we multiply this number each epoch.

            early_stopping=False,
            # If True, then we set an internal validation data and stop training when there is no imporovement.
            tol=1e-4,  # A broad concept of converges, when we can say the algorithm converged?

            batch_size=32,  # The number of samples each batch.
            max_iter=300,
            # The total number of epochs.One epoch=one forward and one backard pass of all the training examples.
            warm_start=False,  # if we fit at the second time, do we start from the last fit?

            random_state=0  # seed
        )
        clf_ada = AdaBoostClassifier(random_state=0, n_estimators=20)
        clf_knn = KNeighborsClassifier(n_neighbors=35)
        clf_lda = LinearDiscriminantAnalysis()
        clf_svm = svm.SVC(C=1000, kernel='rbf', gamma=0.00001, probability=True)
        # # clf = svm.SVC(C=1, kernel='linear', probability=True)
        # clf_rf = RandomForestClassifier(random_state=random_state, max_depth=8,
        #                              criterion='gini', oob_score=True)
        clf_xgb = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
                                    gamma=0.1, learning_rate=0.1, max_depth=7, scale_pos_weight=1,
                                    reg_lambda=100, class_weight='balanced')
        clf = VotingClassifier(estimators=
                               [('ada', clf_ada), ('knn', clf_knn), ('lda', clf_lda), ('ann', ann_clf), ('rf', clf_rf), ('svm', clf_svm), ('xgb', clf_xgb)],
                               voting='soft', n_jobs=7, weights=[1, 1, 2, 1, 15, 1, 15])
        clf.fit(x_train, y_train)
        # print(clf.predict_proba(x_test)[0])
        probs.append(clf.predict_proba(x_test)[:, 1])
        labels.append(y_test[:])

        fpr, tpr, thresholds_keras = roc_curve(y_test[:], clf.predict_proba(x_test)[:, 1])
        print(auc(fpr, tpr))
    # loo = LeaveOneOut()
    # loo.get_n_splits(x)
    # for train_index, test_index in loo.split(x):#pool.map
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #     clf_svm = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True)
    #     # clf = svm.SVC(C=1, kernel='linear', probability=True)
    #     clf_rf = RandomForestClassifier(random_state=random_state, max_depth=8,
    #                                     criterion='gini', oob_score=True)
    #     clf_xgb = xgb.XGBClassifier(seed=0, objective='binary:logistic', colsample_bytree=0.5,
    #                                 gamma=0.1, learning_rate=0.1, max_depth=7, reg_lambda=0, scale_pos_weight=1)
    #     clf = VotingClassifier(estimators=[('rf', clf_rf), ('svm', clf_svm), ('xgb', clf_xgb)], voting='soft', n_jobs=3)
    #     # clf = RandomForestClassifier(random_state=0, min_samples_split=2,
    #     #                              criterion='gini', oob_score=True)
    #     clf.fit(x_train, y_train)
    #     # print(clf.predict_proba(x_test)[0])
    #     probs.append(clf.predict_proba(x_test)[0, 1])
    #     labels.append(y_test[0, 0])
    labels = np.array(list_flatten(labels))
    probs = np.array(list_flatten(probs))
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    # print(auc(fpr, tpr))
    res_arr = np.array([1 - probs, probs, labels, np.arange(labels.shape[0])])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + antibio + '.csv', index=False)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)


def get_minority_class(y):
    c1 = np.sum(y == 1)
    c0 = np.sum(y == 0)
    if c1 > c0:
        return 0, int(np.round(c1/c0, 0))
    else:
        return 1, int(np.round(c0/c1, 0))


def undersampling_kfold(x, y, path, antibio, n_splits, random_state, grid_param):
    start = time.time()
    n_estim = 500
    pcs = 70
    x_tmp = x
    x_tmp, _ = standard_scale(x_tmp)
    x_tmp, pca_ = pca_transformation(x_tmp, pcs)
    grid_clf = GridSearchCV(
        RandomForestClassifier(n_estimators=n_estim, random_state=0, criterion='gini', oob_score=True, class_weight='balanced'),
        {'max_depth': [7]},
        cv=10, return_train_score=True, scoring='roc_auc')
    grid_clf.fit(x_tmp, y)
    grid_df = pd.DataFrame(grid_clf.cv_results_)
    print("all data:")
    print(grid_df[['mean_train_score', 'mean_test_score']])

    minority_class, n_splits = get_minority_class(y)
    print("# splits: " + str(n_splits))
    x_min = x[y == minority_class]
    x_maj = x[y != minority_class]
    y_min = y[y == minority_class]
    y_maj = y[y != minority_class]
    kfold_data = []
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(x_maj)
    print("undersampling:")
    for train_index, test_index in kf.split(x_maj):  # pool.map
        x_test = x_maj[test_index]
        y_test = y_maj[test_index]
        x_fin = np.concatenate((x_test, x_min), axis=0)
        print(x_fin.shape)
        y_fin = np.concatenate((y_test, y_min), axis=0)
        print(y_fin.shape)
        # pcs = 80
        x_fin, _ = standard_scale(x_fin)
        x_fin, pca_ = pca_transformation(x_fin, pcs)
        grid_clf = GridSearchCV(
            RandomForestClassifier(n_estimators=n_estim, random_state=0, criterion='gini', oob_score=True),
            {'max_depth': [6]},
            cv=10, return_train_score=True, scoring='roc_auc')
        grid_clf.fit(x_fin, y_fin)
        grid_df = pd.DataFrame(grid_clf.cv_results_)
        print(grid_df[['mean_train_score', 'mean_test_score']])
    #
    #     x_train, x_test = x[train_index], x[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    #
    #     clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
    #                                  criterion='gini', oob_score=True)
    #     # clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
    #     #                             gamma=0.1, learning_rate=0.1, max_depth=9, scale_pos_weight=1, reg_lambda=100)
    #
    #     kfold_data.append([x_train, x_test, y_train, y_test, clf])
    #
    # p = Pool(processes=10)
    # res = p.map(cv_test, kfold_data)
    # labels = []
    # probs = []
    # for i_res in res:
    #     labels.extend(i_res[1])
    #     probs.extend(i_res[0])
    # # res = np.array(res)
    # # res[:, 0]
    # # print(res[:, 1])
    # p.close()
    # p.join()
    print('The cross-validation with undersampling took time (s): {}'.format(time.time() - start))
    # fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    # print(auc(fpr, tpr))


def pooled_strat_kfold(x, y, path, antibio, n_splits, random_state, grid_param):
    # cv_index = [(i, j) for i, j in sss.split(X, y)]
    # params = list(itertools.product(cv_index, classifiers))
    # kfold_indices = []
    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(x, y)

    for train_index, test_index in kf.split(x, y):  # pool.map
        # kfold_indices.append([train_index, test_index])
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
                                     criterion='gini', oob_score=True, class_weight='balanced')
        # clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
        #                             gamma=0.1, learning_rate=0.1, max_depth=9, scale_pos_weight=1, reg_lambda=100)

        kfold_data.append([x_train, x_test, y_train, y_test, clf])

    p = Pool(processes=10)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    labels = []
    probs = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
    # res = np.array(res)
    # res[:, 0]
    # print(res[:, 1])
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))


def one_group_split(df, i_anti, group_str, n_split_ratio, random_state):
    n_split_ratio = 7
    # group_str = 'index'
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    # calib_df = pd.DataFrame()
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx
    df_mean = df.groupby([group_str]).mean()
    kf = KFold(n_splits=n_split_ratio, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index)
    for train_index, test_index in kf.split(df_mean.index):
        df_train = df.loc[df[group_str].isin(train_index)]
        df_test = df.loc[df[group_str].isin(test_index)]
    print(test_index[:10])
    return df_train, df_test


def k_best_pca_estimation(df, i_anti):
    scores = []
    # k_best = [100, 200, 300]
    k_best = range(10, 170, 10)
    # k_best = [30, 50, 60, 70, 80, 90, 100, 110, 130, 150, 200]
    for ik in k_best:
        print('k_best_=' + str(ik))
        wl_start = "1801.263898"
        wl_end = "898.7034445"
        rawX = df.loc[:, wl_start:wl_end]
        rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
        rawY = df[i_anti].values
        rawX_der, _ = standard_scale(rawX_der)
        rawX_der, pca_ = pca_transformation(rawX_der, 469)
        rawX_der, kbest_ = k_best_transformation(rawX_der, rawY, ik)
        # grid_clf = GridSearchCV(QuadraticDiscriminantAnalysis(), {},
        #                         cv=10, return_train_score=True, scoring='roc_auc')
        grid_clf = GridSearchCV(LinearDiscriminantAnalysis(), {},
                                cv=10, return_train_score=True, scoring='roc_auc')
        # grid_clf = GridSearchCV(
        #     xgb.XGBClassifier(seed=0, class_weight='balanced', objective='binary:logistic', colsample_bytree=0.5), {
        #         'max_depth': [10], 'learning_rate': [0.1], 'n_estimators': [500],
        #         'gamma': [0], 'scale_pos_weight': [1], 'reg_lambda': [100]},
        #     cv=10, return_train_score=True, scoring='roc_auc', n_jobs=10)
        grid_clf.fit(rawX_der, rawY)
        grid_df = pd.DataFrame(grid_clf.cv_results_)
        scores.append(grid_clf.cv_results_['mean_test_score'][0])
    print(k_best[np.argmax(scores)])
    return k_best[np.argmax(scores)]


def k_best_estimation(df, i_anti):
    scores = []
    k_best = [100, 200, 300]
    for ik in k_best:
        print('k_best_=' + str(ik))
        wl_start = "1801.263898"
        wl_end = "898.7034445"
        rawX = df.loc[:, wl_start:wl_end]
        rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
        rawY = df[i_anti].values
        rawX_der, _ = standard_scale(rawX_der)
        rawX_der, kbest_ = k_best_transformation(rawX_der, rawY, ik)
        grid_clf = GridSearchCV(
            xgb.XGBClassifier(seed=0, class_weight='balanced', objective='binary:logistic', colsample_bytree=0.5), {
                'max_depth': [10], 'learning_rate': [0.1], 'n_estimators': [500],
                'gamma': [0], 'scale_pos_weight': [1], 'reg_lambda': [100]},
            cv=10, return_train_score=True, scoring='roc_auc', n_jobs=10)
        grid_clf.fit(rawX_der, rawY)
        grid_df = pd.DataFrame(grid_clf.cv_results_)
        scores.append(grid_clf.cv_results_['mean_test_score'][0])
    print(k_best[np.argmax(scores)])
    return k_best[np.argmax(scores)]


def n_pcs_estimation(df, i_anti, wl_start, wl_end, max_pcs):
    scores = []
    # n_pcs = [2,4,5,6,7,10]
    # n_pcs = [50, 60, 70, 80, 90, 100, 110, 150]
    # n_pcs = [60, 80, 100, 120, 140]
    step = round(max_pcs*2/10)
    n_pcs = range(20, max_pcs, step)
    for ipcs in n_pcs:
        print('i_pcs_=' + str(ipcs))
        wl_start = wl_start
        wl_end = wl_end
        # wl_start = "600"
        # wl_end = "1800"
        rawX = df.loc[:, wl_start:wl_end]
        rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
        rawX_der = rawX
        rawY = df[i_anti].values
        # rawX_der, _ = standard_scale(rawX_der)
        # rawX_der, _ = standard_scale(rawX)
        rawX_der, pca_ = pca_transformation(rawX_der, ipcs)
        # grid_clf = GridSearchCV(
        #     xgb.XGBClassifier(seed=0, class_weight='balanced', objective='binary:logistic', colsample_bytree=0.5), {
        #         'max_depth': [10], 'learning_rate': [0.1], 'n_estimators': [500],
        #         'gamma': [0], 'scale_pos_weight': [1], 'reg_lambda': [100]},
        #     cv=10, return_train_score=True, scoring='roc_auc', n_jobs=10)
        # grid_clf = GridSearchCV(LinearDiscriminantAnalysis(), {},
        #                         cv=10, return_train_score=True, scoring='roc_auc')
        # grid_clf = GridSearchCV(svm.SVC(random_state=0, class_weight='balanced'), {
        #     'C': [1], 'kernel': ['linear']}, cv=5, return_train_score=True, scoring='roc_auc')
        # grid_clf = GridSearchCV(RandomForestClassifier(class_weight='balanced',
        #                         max_depth=7, random_state=0, criterion='gini', oob_score=True),
        #                         {'n_estimators': [110]},
        #                         cv=5, return_train_score=True, scoring='roc_auc')
        clist = [100,1000,10000]
        glist = [1, 10, 100]
        # clist = [1000]
        # glist = [10]
        grid_clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced',
                                        random_state=0, probability=True),
                                {'C': clist, 'gamma': glist},
                                cv=5, return_train_score=True, scoring='roc_auc')

        grid_clf.fit(rawX_der, rawY)
        grid_df = pd.DataFrame(grid_clf.cv_results_)
        grid_df = grid_df[['param_C', 'param_gamma', 'mean_train_score', 'mean_test_score']]
        print(grid_df)
        scores.append(grid_df['mean_test_score'].max())
        # scores.append(grid_clf.cv_results_['mean_test_score'][0])
    print(scores)
    print(n_pcs[np.argmax(scores)])
    return n_pcs[np.argmax(scores)]


def pooled_strat_kbest_pca_nested_kfold_groups(df, path, i_anti, n_splits, random_state, grid_param, calib_flag):
    group_str = 'index'
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    # group_str = 'serial'
    # wl_start = "600"
    # wl_end = "1800"
    wls_str = df.loc[:, wl_start:wl_end].columns
    wls = [float(iwls) for iwls in wls_str]
    # print(df[group_str].unique().shape)
    mean_df = df.groupby([group_str]).mean()
    print(np.sum(mean_df[i_anti] == 0) / mean_df[i_anti].shape[0])
    if calib_flag == 1:
        df, calib_df = one_group_split(df, i_anti, group_str, n_splits, random_state)
    # print(df[group_str].unique().shape)
        mean_calib_df = calib_df.groupby(['index']).mean()
        print(np.sum(mean_calib_df[i_anti] == 0)/mean_calib_df[i_anti].shape[0])

    # rawX = df.loc[:, wl_start:wl_end]
    # rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
    # rawY = df[i_anti].values
    # tsne(rawX_der, rawY, 80)

    # rawY = df[i_anti].values
    # # print(df.columns[469:])
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx

    # print(df.columns)
    # # plt.plot(np.diff(df.groupby([group_str]).mean().index))
    # plt.plot(df.index[1:], np.diff(df[group_str]))
    # plt.show()
    df_mean = df.groupby([group_str]).mean()

    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index, df_mean[i_anti])
    pcs_dict = {}
    pcs_list = []
    for train_index, test_index in kf.split(df_mean.index, df_mean[i_anti]):  # pool.map

        df_train = df.loc[df[group_str].isin(train_index)]
        nested_df = df_train.groupby([group_str]).mean()
        # i_best = k_best_pca_estimation(nested_df, i_anti)
        i_best = 130
        pcs_list.append(i_best)
        # if calib_flag == 1:
        #     df_train, calib_df = one_group_split(df_train, i_anti, group_str, n_splits, random_state)
        y_train = df_train[i_anti].values
        x_train = df_train.loc[:, wl_start:wl_end]
        x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_train_ref1, scale_ = standard_scale(x_train)
        # print(x_train_ref1.shape)
        x_train_ref2, pca_ = pca_transformation(x_train_ref1, 469)
        # print(x_train_ref2.shape)
        x_train, kbest_ = k_best_transformation(x_train_ref2, y_train, i_best)
        # print(x_train.shape)
        # x_train, pca_ = pca_transformation(x_train_ref, i_best)
        train_group = df_train[group_str].values

        df_test = df.loc[df[group_str].isin(test_index)]
        x_test = df_test.loc[:, wl_start:wl_end]
        x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        # y_test = df_test[i_anti].values
        # x_test_ref, scale_ = standard_scale(x_test)
        # x_test_ref, pca_ = pca_transformation(x_test_ref, 469)
        # x_test, kbest_ = k_best_transformation(x_test_ref, y_test, i_best)
        # print(x_test.shape)
        x_test_ref1 = scale_.transform(x_test)
        x_test_ref2 = pca_.transform(x_test_ref1)
        # print(x_test_ref2.shape)
        x_test = kbest_.transform(x_test_ref2)
        # print(x_test.shape)

        y_test = df_test[i_anti].values
        test_group = df_test[group_str].values

        if calib_flag == 1:
            y_calib = calib_df[i_anti].values
            x_calib = calib_df.loc[:, wl_start:wl_end]
            x_calib = sgf(x_calib, window_length=13, polyorder=3, deriv=2, mode="nearest")
            x_calib = scale_.transform(x_calib)
            x_calib = pca_.transform(x_calib)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True, class_weight='balanced')
        # clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
        #                              criterion='gini', oob_score=True, class_weight='balanced')
        clf = LinearDiscriminantAnalysis()
        # clf = QuadraticDiscriminantAnalysis()

        # clf = PLSRegression(n_components=16)
        # clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
        #                         gamma=0, learning_rate=0.1, max_depth=10, scale_pos_weight=1,
        #                         reg_lambda=100, class_weight='balanced')
        # clf = LinearDiscriminantAnalysis()
        # clf = LogisticRegression(C=1, random_state=0, class_weight='balanced', penalty='l2')
        # clf = svm.SVC(kernel='linear', C=1, random_state=0, class_weight='balanced', probability=True)

        if calib_flag == 1:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf,
                               calib_flag, x_calib, y_calib])
        else:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf, calib_flag])

    p = Pool(processes=5)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    res_df_avg, fpr_avg, tpr_avg, best_point_avg = calc_group_results(res)
    _save_optimal_cut_(fpr_avg, tpr_avg, best_point_avg, path + '/' + i_anti)
    res_df_avg.to_csv(path + i_anti + '_res.csv')

    labels = []
    probs = []
    group = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
        group.extend(i_res[2])
    labels, probs, group = np.array(labels), np.array(probs), np.array(group)
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    plot_calibration_curve(labels, probs, path+i_anti)
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))

    res_arr = np.array([1 - probs, probs, labels, group])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + i_anti + '.csv', index=False)
    pcs_dict[i_anti] = pcs_list
    pcs_df = pd.DataFrame.from_dict(pcs_dict)
    pcs_df.to_csv(path + i_anti + 'pcs.csv', index=False)


def pooled_strat_kbest_pca_nested_kfold_groups(df, path, i_anti, n_splits, random_state, grid_param, calib_flag):
    group_str = 'index'
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    # group_str = 'serial'
    # wl_start = "600"
    # wl_end = "1800"
    wls_str = df.loc[:, wl_start:wl_end].columns
    wls = [float(iwls) for iwls in wls_str]
    # print(df[group_str].unique().shape)
    mean_df = df.groupby([group_str]).mean()
    print(np.sum(mean_df[i_anti] == 0) / mean_df[i_anti].shape[0])
    if calib_flag == 1:
        df, calib_df = one_group_split(df, i_anti, group_str, n_splits, random_state)
    # print(df[group_str].unique().shape)
        mean_calib_df = calib_df.groupby(['index']).mean()
        print(np.sum(mean_calib_df[i_anti] == 0)/mean_calib_df[i_anti].shape[0])

    # rawX = df.loc[:, wl_start:wl_end]
    # rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
    # rawY = df[i_anti].values
    # tsne(rawX_der, rawY, 80)

    # rawY = df[i_anti].values
    # # print(df.columns[469:])
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx

    # print(df.columns)
    # # plt.plot(np.diff(df.groupby([group_str]).mean().index))
    # plt.plot(df.index[1:], np.diff(df[group_str]))
    # plt.show()
    df_mean = df.groupby([group_str]).mean()

    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index, df_mean[i_anti])
    pcs_dict = {}
    pcs_list = []
    for train_index, test_index in kf.split(df_mean.index, df_mean[i_anti]):  # pool.map

        df_train = df.loc[df[group_str].isin(train_index)]
        nested_df = df_train.groupby([group_str]).mean()
        i_best = k_best_pca_estimation(nested_df, i_anti)
        # i_best = 130
        pcs_list.append(i_best)
        # if calib_flag == 1:
        #     df_train, calib_df = one_group_split(df_train, i_anti, group_str, n_splits, random_state)
        y_train = df_train[i_anti].values
        x_train = df_train.loc[:, wl_start:wl_end]
        x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_train_ref1, scale_ = standard_scale(x_train)
        # print(x_train_ref1.shape)
        x_train_ref2, pca_ = pca_transformation(x_train_ref1, 469)
        # print(x_train_ref2.shape)
        x_train, kbest_ = k_best_transformation(x_train_ref2, y_train, i_best)
        # print(x_train.shape)
        # x_train, pca_ = pca_transformation(x_train_ref, i_best)
        train_group = df_train[group_str].values

        df_test = df.loc[df[group_str].isin(test_index)]
        x_test = df_test.loc[:, wl_start:wl_end]
        x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        # y_test = df_test[i_anti].values
        # x_test_ref, scale_ = standard_scale(x_test)
        # x_test_ref, pca_ = pca_transformation(x_test_ref, 469)
        # x_test, kbest_ = k_best_transformation(x_test_ref, y_test, i_best)
        # print(x_test.shape)
        x_test_ref1 = scale_.transform(x_test)
        x_test_ref2 = pca_.transform(x_test_ref1)
        # print(x_test_ref2.shape)
        x_test = kbest_.transform(x_test_ref2)
        # print(x_test.shape)

        y_test = df_test[i_anti].values
        test_group = df_test[group_str].values

        if calib_flag == 1:
            y_calib = calib_df[i_anti].values
            x_calib = calib_df.loc[:, wl_start:wl_end]
            x_calib = sgf(x_calib, window_length=13, polyorder=3, deriv=2, mode="nearest")
            x_calib = scale_.transform(x_calib)
            x_calib = pca_.transform(x_calib)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True, class_weight='balanced')
        # clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
        #                              criterion='gini', oob_score=True, class_weight='balanced')
        clf = LinearDiscriminantAnalysis()
        # clf = QuadraticDiscriminantAnalysis()

        # clf = PLSRegression(n_components=16)
        # clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
        #                         gamma=0, learning_rate=0.1, max_depth=10, scale_pos_weight=1,
        #                         reg_lambda=100, class_weight='balanced')
        # clf = LinearDiscriminantAnalysis()
        # clf = LogisticRegression(C=1, random_state=0, class_weight='balanced', penalty='l2')
        # clf = svm.SVC(kernel='linear', C=1, random_state=0, class_weight='balanced', probability=True)

        if calib_flag == 1:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf,
                               calib_flag, x_calib, y_calib])
        else:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf, calib_flag])

    p = Pool(processes=5)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    res_df_avg, fpr_avg, tpr_avg, best_point_avg = calc_group_results(res)
    _save_optimal_cut_(fpr_avg, tpr_avg, best_point_avg, path + '/' + i_anti)
    res_df_avg.to_csv(path + i_anti + '_res.csv')

    labels = []
    probs = []
    group = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
        group.extend(i_res[2])
    labels, probs, group = np.array(labels), np.array(probs), np.array(group)
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    plot_calibration_curve(labels, probs, path+i_anti)
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))

    res_arr = np.array([1 - probs, probs, labels, group])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + i_anti + '.csv', index=False)
    pcs_dict[i_anti] = pcs_list
    pcs_df = pd.DataFrame.from_dict(pcs_dict)
    pcs_df.to_csv(path + i_anti + 'pcs.csv', index=False)


def pooled_strat_pca_nested_kfold_groups(df, path, i_anti, n_splits, random_state, grid_param, calib_flag):
    group_str = 'index'
    # group_str = 'serial'
    # wl_end = "898.70345"
    # wl_start = "1801.2639"
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    # group_str = 'serial'
    # wl_start = "600"
    # wl_end = "1800"
    wls_str = df.loc[:, wl_start:wl_end].columns
    wls = [float(iwls) for iwls in wls_str]
    # print(df[group_str].unique().shape)
    mean_df = df.groupby([group_str]).mean()
    print(np.sum(mean_df[i_anti] == 0) / mean_df[i_anti].shape[0])
    if calib_flag == 1:
        df, calib_df = one_group_split(df, i_anti, group_str, n_splits, random_state)
    # print(df[group_str].unique().shape)
        mean_calib_df = calib_df.groupby(['index']).mean()
        print(np.sum(mean_calib_df[i_anti] == 0)/mean_calib_df[i_anti].shape[0])

    # rawX = df.loc[:, wl_start:wl_end]
    # rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
    # rawY = df[i_anti].values
    # tsne(rawX_der, rawY, 80)

    # rawY = df[i_anti].values
    # # print(df.columns[469:])
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx
    df_mean = df.groupby([group_str]).mean()

    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index, df_mean[i_anti])
    pcs_dict = {}
    pcs_list = []
    for train_index, test_index in kf.split(df_mean.index, df_mean[i_anti]):  # pool.map

        df_train = df.loc[df[group_str].isin(train_index)]
        nested_df = df_train.groupby([group_str]).mean()
        # i_best = n_pcs_estimation(nested_df, i_anti)
        i_best = 100
        pcs_list.append(i_best)
        # if calib_flag == 1:
        #     df_train, calib_df = one_group_split(df_train, i_anti, group_str, n_splits, random_state)
        y_train = df_train[i_anti].values
        x_train = df_train.loc[:, wl_start:wl_end]
        x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_train_ref, scale_ = standard_scale(x_train)
        x_train, pca_ = pca_transformation(x_train_ref, i_best)
        train_group = df_train[group_str].values

        df_test = df.loc[df[group_str].isin(test_index)]
        x_test = df_test.loc[:, wl_start:wl_end]
        x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_test_ref = scale_.transform(x_test)
        x_test = pca_.transform(x_test_ref)

        y_test = df_test[i_anti].values
        test_group = df_test[group_str].values

        if calib_flag == 1:
            y_calib = calib_df[i_anti].values
            x_calib = calib_df.loc[:, wl_start:wl_end]
            x_calib = sgf(x_calib, window_length=13, polyorder=3, deriv=2, mode="nearest")
            x_calib = scale_.transform(x_calib)
            x_calib = pca_.transform(x_calib)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True, class_weight='balanced')
        clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=10, max_features='sqrt',
                                     criterion='gini', oob_score=True, class_weight='balanced') #balanced_subsample
        # clf = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=8, max_features='sqrt',
        #                              criterion='gini', oob_score=True, class_weight='balanced') #balanced_subsample
        # clf = PLSRegression(n_components=16)
        # clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
        #                         gamma=0, learning_rate=0.1, max_depth=10, scale_pos_weight=1,
        #                         reg_lambda=100, class_weight='balanced')
        # clf = LinearDiscriminantAnalysis()
        # clf = LogisticRegression(C=1, random_state=0, class_weight='balanced', penalty='l2')
        # clf = svm.SVC(kernel='linear', C=1, random_state=0, class_weight='balanced', probability=True)

        if calib_flag == 1:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf,
                               calib_flag, x_calib, y_calib])
        else:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf, calib_flag])

    p = Pool(processes=5)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    res_df_avg, fpr_avg, tpr_avg, best_point_avg = calc_group_results(res)
    _save_optimal_cut_(fpr_avg, tpr_avg, best_point_avg, path + '/' + i_anti)
    res_df_avg.to_csv(path + i_anti + '_res.csv')

    labels = []
    probs = []
    group = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
        group.extend(i_res[2])
    labels, probs, group = np.array(labels), np.array(probs), np.array(group)
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    plot_calibration_curve(labels, probs, path+i_anti)
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))

    res_arr = np.array([1 - probs, probs, labels, group])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + i_anti + '.csv', index=False)
    pcs_dict[i_anti] = pcs_list
    pcs_df = pd.DataFrame.from_dict(pcs_dict)
    pcs_df.to_csv(path + i_anti + 'pcs.csv', index=False)


def _save_optimal_cut_(fpr, tpr, best_point, path):
    """
    print the optimal point
    *not for users*
    """
    plt.figure(dpi=600)
    plt.plot(fpr, tpr)
    # plt.scatter(best_point[0], best_point[1], c="red")
    plt.xlabel("1-specificity")
    plt.ylabel("sensitivity")
    plt.plot([0, 1], [0, 1], "--r")
    plt.axes().set_aspect(aspect='equal')
    plt.savefig(path+'.tiff')
    plt.close()


def calc_group_results(res):
    cnt = 0
    best_points = []
    tpr_all_folds = []
    fpr_ = np.arange(0, 1.01, 0.01)
    for i_res in res:
        labels = i_res[1]
        probs = i_res[0]
        group = i_res[2]
        labels, probs, group = np.array(labels), np.array(probs), np.array(group)
        fpr, tpr, thresholds_keras = roc_curve(labels, probs)
        print(auc(fpr, tpr))
        res_arr = np.array([1 - probs, probs, labels, group])
        res_df_raw = pd.DataFrame(res_arr.T,
                              columns=['0', '1', 'labels', 'group'])

        df = build_llr_table(res_df_raw)
        labels, probs = df['labels'], df['1']
        labels, probs = switch_R_S(labels, probs)
        report_all, threshold, best_point = optimal_cut_point_on_roc__(labels, probs, delta_max=0.2, tpr_low_bound=0.5)
        # print(threshold)
        best_points.append(best_point)
        fpr, tpr, thresholds = roc_curve(labels, probs)
        # _save_optimal_cut_(fpr, tpr, best_point, path + '/' + i_anti)
        #     print(calc_SE_SP_ACC(labels, probs, threshold))
        labels, probs = switch_R_S(labels, probs)
        # report_all['R'] = np.sum(labels == 1)
        # report_all['S'] = np.sum(labels == 0)
        # report_all['#samples'] = labels.shape[0]
        # report_all['antibotics'] = i_anti
        if cnt == 0:
            res_df = pd.DataFrame.from_dict(report_all)
        else:
            tmp = pd.DataFrame.from_dict(report_all)
            res_df = res_df.append(tmp, ignore_index=True)
        tmp_interp = np.interp(fpr_, fpr, tpr)
        tmp_interp[0] = 0
        tpr_all_folds.append(tmp_interp)
        cnt = cnt + 1

    res_df = res_df.mean(axis=0)
    tpr_avg = np.mean(tpr_all_folds, axis=0)
    best_points_avg = np.mean(best_points, axis=0)
    new_best_points_avg = [best_points_avg[0], np.interp(best_points_avg[0], fpr_, tpr_avg)]

    return res_df, fpr_, tpr_avg, new_best_points_avg


def pooled_strat_kbest_nested_kfold_groups(df, path, i_anti, n_splits, random_state, grid_param, calib_flag):
    group_str = 'index'
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    wls_str = df.loc[:, wl_start:wl_end].columns
    wls = [float(iwls) for iwls in wls_str]
    # print(df[group_str].unique().shape)
    mean_df = df.groupby(['index']).mean()
    print(np.sum(mean_df[i_anti] == 0) / mean_df[i_anti].shape[0])
    if calib_flag == 1:
        df, calib_df = one_group_split(df, i_anti, group_str, n_splits, random_state)
    # print(df[group_str].unique().shape)
        mean_calib_df = calib_df.groupby(['index']).mean()
        print(np.sum(mean_calib_df[i_anti] == 0)/mean_calib_df[i_anti].shape[0])

    # rawX = df.loc[:, wl_start:wl_end]
    # rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
    # rawY = df[i_anti].values
    # tsne(rawX_der, rawY, 80)

    # rawY = df[i_anti].values
    # # print(df.columns[469:])
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx
    df_mean = df.groupby([group_str]).mean()

    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index, df_mean[i_anti])
    k_bast_list = []
    for train_index, test_index in kf.split(df_mean.index, df_mean[i_anti]):  # pool.map

        df_train = df.loc[df[group_str].isin(train_index)]
        nested_df = df_train.groupby([group_str]).mean()
        ik_best = k_best_estimation(nested_df, i_anti)
        k_bast_list.append(ik_best)
        # if calib_flag == 1:
        #     df_train, calib_df = one_group_split(df_train, i_anti, group_str, n_splits, random_state)
        y_train = df_train[i_anti].values
        x_train = df_train.loc[:, wl_start:wl_end]
        x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_train_ref, scale_ = standard_scale(x_train)
        x_train, selectkbest_ = k_best_transformation(x_train_ref, y_train, ik_best)
        train_group = df_train[group_str].values

        df_test = df.loc[df[group_str].isin(test_index)]
        x_test = df_test.loc[:, wl_start:wl_end]
        x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_test_ref = scale_.transform(x_test)
        x_test = selectkbest_.transform(x_test_ref)

        y_test = df_test[i_anti].values
        test_group = df_test[group_str].values

        if calib_flag == 1:
            y_calib = calib_df[i_anti].values
            x_calib = calib_df.loc[:, wl_start:wl_end]
            x_calib = sgf(x_calib, window_length=13, polyorder=3, deriv=2, mode="nearest")
            x_calib = scale_.transform(x_calib)
            x_calib = selectkbest_.transform(x_calib)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True, class_weight='balanced')
        # clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
        #                              criterion='gini', oob_score=True, class_weight='balanced')
        # clf = PLSRegression(n_components=16)
        clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
                                gamma=0, learning_rate=0.1, max_depth=10, scale_pos_weight=1,
                                reg_lambda=100, class_weight='balanced')
        if calib_flag == 1:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf,
                               calib_flag, x_calib, y_calib])
        else:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf, calib_flag])

    p = Pool(processes=10)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    labels = []
    probs = []
    group = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
        group.extend(i_res[2])
    labels, probs, group = np.array(labels), np.array(probs), np.array(group)
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    plot_calibration_curve(labels, probs, path+i_anti)
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))

    res_arr = np.array([1 - probs, probs, labels, group])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + i_anti + '.csv', index=False)


def pooled_strat_kfold_groups(df, path, i_anti, n_splits, random_state, grid_param, calib_flag):
    group_str = 'index'
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    wls_str = df.loc[:, wl_start:wl_end].columns
    wls = [float(iwls) for iwls in wls_str]
    # print(df[group_str].unique().shape)
    mean_df = df.groupby(['index']).mean()
    print(np.sum(mean_df[i_anti] == 0) / mean_df[i_anti].shape[0])
    if calib_flag == 1:
        df, calib_df = one_group_split(df, i_anti, group_str, n_splits, random_state)
    # print(df[group_str].unique().shape)
        mean_calib_df = calib_df.groupby(['index']).mean()
        print(np.sum(mean_calib_df[i_anti] == 0)/mean_calib_df[i_anti].shape[0])

    # rawX = df.loc[:, wl_start:wl_end]
    # rawX_der = sgf(rawX, window_length=13, polyorder=3, deriv=2, mode="nearest")
    # rawY = df[i_anti].values
    # tsne(rawX_der, rawY, 80)

    # rawY = df[i_anti].values
    # # print(df.columns[469:])
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx
    df_mean = df.groupby([group_str]).mean()

    kfold_data = []
    kf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    kf.get_n_splits(df_mean.index, df_mean[i_anti])
    for train_index, test_index in kf.split(df_mean.index, df_mean[i_anti]):  # pool.map

        df_train = df.loc[df[group_str].isin(train_index)]
        # if calib_flag == 1:
        #     df_train, calib_df = one_group_split(df_train, i_anti, group_str, n_splits, random_state)
        x_train = df_train.loc[:, wl_start:wl_end]
        x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        pcs = 90
        x_train_ref, scale_ = standard_scale(x_train)
        x_train, pca_ = pca_transformation(x_train_ref, pcs)

        # inv_x_train = pca_.inverse_transform(x_train)
        # mse_score = ((x_train_ref - inv_x_train)**2).sum(axis=1)
        # # plot_y(np.sort(mse_score))
        # # outlier_indices = np.argsort(mse_score)[-1500:]
        # outlier_indices = np.argwhere(mse_score > 1)
        # outlier_indices = np.array(list_flatten(outlier_indices))
        # df_train = df_train.loc[~df_train.index.isin(outlier_indices)]
        # x_train = df_train.loc[:, wl_start:wl_end]
        # x_train = sgf(x_train, window_length=13, polyorder=3, deriv=2, mode="nearest")
        # pcs = 90
        # x_train_ref, scale_ = standard_scale(x_train)
        # x_train, pca_ = pca_transformation(x_train_ref, pcs)
        # # outlier_indices = largest_indices(mse_score, 100)
        # # transformed_data_df = pd.DataFrame(x_train, columns=wls_str)
        # # df_train_avg = pd.concat([df_train[i_anti], df_train[group_str], transformed_data_df], axis=1).groupby([group_str]).mean()
        train_group = df_train[group_str].values
        y_train = df_train[i_anti].values

        df_test = df.loc[df[group_str].isin(test_index)]
        x_test = df_test.loc[:, wl_start:wl_end]
        x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        x_test_ref = scale_.transform(x_test)
        x_test = pca_.transform(x_test_ref)

        # inv_x_test = pca_.inverse_transform(x_test)
        # mse_score = ((x_test_ref - inv_x_test) ** 2).sum(axis=1)
        # # plot_y(np.sort(mse_score))
        # outlier_indices = np.argwhere(mse_score > 1)
        # outlier_indices = np.array(list_flatten(outlier_indices))
        # df_test = df_test.loc[~df_test.index.isin(outlier_indices)]
        # x_test = df_test.loc[:, wl_start:wl_end]
        # x_test = sgf(x_test, window_length=13, polyorder=3, deriv=2, mode="nearest")
        # x_test_ref = scale_.transform(x_test)
        # x_test = pca_.transform(x_test_ref)
        y_test = df_test[i_anti].values
        test_group = df_test[group_str].values

        if calib_flag == 1:
            y_calib = calib_df[i_anti].values
            x_calib = calib_df.loc[:, wl_start:wl_end]
            x_calib = sgf(x_calib, window_length=13, polyorder=3, deriv=2, mode="nearest")
            x_calib = scale_.transform(x_calib)
            x_calib = pca_.transform(x_calib)
        # clf = svm.SVC(C=250, kernel='rbf', gamma=0.0001, probability=True, class_weight='balanced')
        # clf = RandomForestClassifier(n_estimators=500, random_state=0, max_depth=grid_param,
        #                              criterion='gini', oob_score=True, class_weight='balanced')
        # clf = PLSRegression(n_components=16)
        clf = xgb.XGBClassifier(n_estimators=500, seed=0, objective='binary:logistic', colsample_bytree=0.5,
                                gamma=0, learning_rate=0.1, max_depth=10, scale_pos_weight=1,
                                reg_lambda=100, class_weight='balanced')
        if calib_flag == 1:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf,
                               calib_flag, x_calib, y_calib])
        else:
            kfold_data.append([x_train, x_test, y_train, y_test, train_group, test_group, clf, calib_flag])

    p = Pool(processes=10)
    start = time.time()
    res = p.map(cv_test, kfold_data)
    labels = []
    probs = []
    group = []
    for i_res in res:
        labels.extend(i_res[1])
        probs.extend(i_res[0])
        group.extend(i_res[2])
    labels, probs, group = np.array(labels), np.array(probs), np.array(group)
    p.close()
    p.join()
    print('The cross-validation with stratified sampling on 5 cores took time (s): {}'.format(time.time() - start))
    plot_calibration_curve(labels, probs, path+i_anti)
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    print(auc(fpr, tpr))

    res_arr = np.array([1 - probs, probs, labels, group])
    res_df = pd.DataFrame(res_arr.T,
                          columns=['0', '1', 'labels', 'group'])
    res_df.to_csv(path + i_anti + '.csv', index=False)


def cv_test(params):
    # global X
    # global y
    # train_index = params[0][0]
    # test_index = params[0][1]
    # train_index = params[0]
    # test_index = params[1]

    x_tr, x_tes, y_tr, y_tes, train_group, test_group = params[0], params[1], params[2], params[3], params[4], params[5]
    clf = params[6]
    calibr_flag = params[7]
    # clf = params[1]
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    # name = clf.__class__.__name__
    if calibr_flag == 1:
        x_cal, y_cal = params[8], params[9]
        clf.fit(x_tr, y_tr)
        y_pred_probas = clf.predict_proba(x_cal)
        rl = LogisticRegression(random_state=0)
        tmp = y_pred_probas[:, 1]
        tmp = tmp.reshape(tmp.shape[0], 1)
        rl.fit(tmp, y_cal)

        y_pred_probas = clf.predict_proba(x_tes)
        tmp = y_pred_probas[:, 1]
        # pred_hist(tmp)
        tmp = tmp.reshape(tmp.shape[0], 1)
        y_pred_probas = rl.predict_proba(tmp)
        # pred_hist(y_pred_probas[:, 1])
    else:
        clf.fit(x_tr, y_tr)
        y_pred_probas = clf.predict_proba(x_tes)



    # acc = accuracy_score(y_test, y_pred)
    # loss = log_loss(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_tes, y_pred_probas[:, 1])
    auc_score = auc(fpr, tpr)
    print(auc_score)
    return [y_pred_probas[:, 1], y_tes, test_group]


def cv_test_avg(params):
    x_tr, x_tes, y_tr, y_tes,  = params[0], params[1], params[2], params[3]
    clf = params[4]
    clf.fit(x_tr, y_tr)

    # y_pred_probas = clf.predict_proba(x_tes)
    # fpr, tpr, thresholds = roc_curve(y_tes, y_pred_probas[:, 1])
    # auc_score = auc(fpr, tpr)
    # print(auc_score)
    # return [y_pred_probas[:, 1], y_tes]

    # tr_pred = clf.predict(x_tr)
    # tr_acc_score = accuracy_score(y_tr, tr_pred)
    # y_pred_probas = clf.predict(x_tes)
    # acc_score = accuracy_score(y_tes, y_pred_probas)
    # return [y_pred_probas, y_tes, acc_score, tr_acc_score]

    y_pred_probas_tr = clf.predict_proba(x_tr)
    fpr, tpr, thresholds = roc_curve(y_tr, y_pred_probas_tr[:, 1])
    tr_auc_score = auc(fpr, tpr)
    y_pred_probas = clf.predict_proba(x_tes)
    fpr, tpr, thresholds = roc_curve(y_tes, y_pred_probas[:, 1])
    auc_score = auc(fpr, tpr)
    # print(auc_score)

    y_tr, probs = switch_R_S(y_tr, y_pred_probas_tr[:, 1])
    report_all, threshold, best_point = \
        optimal_cut_point_on_roc__(y_tr, probs, delta_max=0.2, tpr_low_bound=0.5)
    print(report_all)
    y_tr, probs = switch_R_S(y_tr, probs)
    y_tes, probs = switch_R_S(y_tes, y_pred_probas[:, 1])
    fold_results = calc_results(y_tes, probs, threshold)
    y_tes, probs = switch_R_S(y_tes, probs)
    return [y_pred_probas, y_tes, auc_score, tr_auc_score, fold_results]

    # y_pred_probas = clf.predict_proba(x_tes)
    # return [y_pred_probas, y_tes]


def plot_y(y):
    plt.figure()
    plt.plot(y)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.show()
    # plt.close()
    return


def plot_calibration_curve(y_test, y_pred, path):
    br_score = brier_score_loss(y_test, y_pred)
    # plt.rcParams.update({'font.size': 10})
    frac_of_positives, pred_prob = calibration_curve(y_test, y_pred, n_bins=10)
    plt.figure()
    plt.plot(pred_prob, frac_of_positives)
    # sns.lineplot(x=pred_prob, y=frac_of_positives)
    plt.grid(linestyle='-', linewidth=0.2)
    plt.title("Probability vs Fraction of Positives, brier score = " + str(np.round(br_score, 2)))
    plt.xlabel("Probability of positive")
    plt.ylabel("Fraction of positives")
    plt.savefig(path + 'calib_curve.png')
    # plt.show()
    # plt.close()
    # xlabel = plt.xlabel("Probability of positive")
    # ylabel = plt.ylabel("Fraction of positives")
    # ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # xticks = plt.xticks(ticks)
    # yticks = plt.yticks(ticks)
    return


def dbscan_outliers(x, epsilon, min_samp):
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samp).fit(x)
    return dbscan


def optics_outliers(x, min_samp):
    optics = OPTICS(min_samples=min_samp).fit(x)
    return optics


def standard_scale(x):
    standard_scaler = StandardScaler()
    standard_scaler.fit(x)
    scaled_X = standard_scaler.transform(x)
    return scaled_X, standard_scaler


def robust_scale(x):
    robust_scaler = RobustScaler()
    robust_scaler.fit(x)
    scaled_X = robust_scaler.transform(x)
    return scaled_X, robust_scaler


def minmax_scale(X):
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X)
    scaled_X = min_max_scaler.transform(X)
    return scaled_X, min_max_scaler


def umap_transformation(x, n_comp, n_neib, min_dist):
    reducer = umap.UMAP(random_state=0, n_components=n_comp, min_dist=min_dist, n_neighbors=n_neib)#n_neighbors=5,20,100,300
    x = reducer.fit_transform(x)
    return x, reducer


def pca_transformation(x, n_comp):
    pca = decomposition.PCA(n_components=n_comp)
    # pca = decomposition.PCA(n_components=n_comp, svd_solver='full')
    pca.fit(x)
    X_pca = pca.transform(x)
    return X_pca, pca


def pca_transformation_99perc(x, n_comp):
    # pca = decomposition.PCA(n_components=n_comp)
    pca = decomposition.PCA(n_components=n_comp, svd_solver='full')
    pca.fit(x)
    X_pca = pca.transform(x)
    return X_pca, pca


def k_best_transformation(x, y, n_comp):
    # kbest = SelectKBest(score_func=mutual_info_classif, k=n_comp)
    kbest = SelectKBest(score_func=f_classif, k=n_comp)
    kbest.fit(x, y)
    x_kbest = kbest.transform(x)
    return x_kbest, kbest


def tsne(x, y, pcs):
    x, _ = standard_scale(x)
    x, pca_ = pca_transformation(x, pcs)
    #     %matplotlib notebook
    #     %matplotlib inline
    x_sne = TSNE(n_components=2, random_state=0,
                 method='barnes_hut', angle=0.5).fit_transform(x)  # perplexity=50.0
    fig = plt.figure()
    #     ax = Axes3D(fig)
    #     pls2 = PLSRegression(n_components=2)
    #     pls2.fit(srawX, rawY)
    #     x_pls = pls2.transform(srawX)
    for name, label in [('R', 1), ('S', 0)]:
        x_ = x_sne[y == label]
        plt.scatter(x_[:, 0], x_[:, 1], label=name)
    #         ax.scatter(X_[:,0], X_[:,1], X_[:,2], label=name)
    plt.legend()
    plt.show()


def groups_to_index(groups):
    groups_indx = []
    group_num = 0
    for i_group in list(range(len(groups) - 1)):
        if (groups[i_group + 1] - groups[i_group]) == 0:
            groups_indx.append(group_num)
        else:
            groups_indx.append(group_num)
            group_num = group_num + 1
    groups_indx.append(group_num)
    return groups_indx


def pred_hist(y_prob):
    ax = plt.subplot(1, 1, 1)
    ax.hist(y_prob, color='blue', edgecolor='black')  # , bins=[-0.5, 0.5, 1.5, 2.5])
    ax.set_title('Hist Of Predicted probabilities')
    ax.set_ylabel('Freq')
    ax.set_xlabel('Pred Prob')
    plt.tight_layout()
    plt.show()
    plt.close()


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def pooling_avg(x, m_size):
    newx = []
    for i_x in x:
        tmp = np.mean(np.resize(i_x, (int(i_x.shape[0]/m_size), m_size)), axis=1)
        newx.append(tmp)
    return np.array(newx)


def get_closest_wls(wl, wls):
    return wls[np.argmin(np.abs(wl-wls))]


def remove_dups_from_groups(df, group_str):
    wl_start = "1801.263898"
    wl_end = "898.7034445"
    groups = df[group_str].values
    group_indx = np.array(groups_to_index(groups))
    df[group_str] = group_indx
    df_mean = df.groupby([group_str]).mean()
    new_groups = []
    for i_group in df_mean[group_str].values:
        x = df[df[group_str] == i_group].loc[:, wl_start:wl_end].values
