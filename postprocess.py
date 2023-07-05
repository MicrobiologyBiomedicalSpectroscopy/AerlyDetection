import math
from sklearn import metrics
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from preprocsesing import lists_solver
from sklearn.metrics import roc_curve, auc, accuracy_score
warnings.simplefilter("ignore")


def build_best_acc_df(labels, probs):
    fpt, tpr, thresholds = metrics.roc_curve(labels, probs)
    accuracy_ls = []
    for thres in thresholds:
        y_pred = np.where(probs > thres, 1, 0)
        accuracy_ls.append(metrics.accuracy_score(labels, y_pred, normalize=True))

    accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls), pd.Series(fpt), pd.Series(tpr)],
                            axis=1)
    accuracy_ls.columns = ['thresholds', 'accuracy', '1-SP', 'SE']
    accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
    # print(accuracy_ls.head(50))
    # print(np.array(accuracy_ls.iloc[0]))
    # accuracy_ls.head()
    return accuracy_ls


def rejection_proc(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    margins = np.concatenate((probs[probs > 0.5], 1 - probs[probs < 0.5]))
    margins = np.sort(margins)

    auc_list = []
    fpr, tpr, thresholds_keras = roc_curve(labels, probs)
    auc_list.append(auc(fpr, tpr))
    print(auc_list)
    for i_margin in np.arange(margins.shape[0] - 200):
        i_labels, i_probs = excepted_samples(labels, probs, margins[i_margin])
        #         print(i_labels.shape, i_probs.shape)
        fpr, tpr, thresholds_keras = roc_curve(i_labels, i_probs)
        auc_list.append(auc(fpr, tpr))
    plt.figure()
    plt.plot(auc_list)
    plt.title('Rejections - ciprofloxacin')
    plt.xlabel('n drops')
    plt.ylabel('auc score')
    plt.show()


def excepted_samples(labels, probs, margin):
    indices = np.concatenate(((probs > margin).nonzero()[0], (probs < 0.5 - (margin - 0.5)).nonzero()[0]))
    probs = probs[indices]
    labels = labels[indices]
    return labels, probs


def reject_specific_range(labels, probs, margin):
    labels = np.array(labels)
    probs = np.array(probs)
    smaples_all = labels.shape[0]
    all_1 = (labels == 1).sum()
    all_0 = (labels == 0).sum()
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_all_samples = auc(fpr, tpr)

    indices = np.concatenate(((probs > margin[1]).nonzero()[0], (probs < margin[0]).nonzero()[0]))
    probs = probs[indices]
    labels = labels[indices]
    smaples_rej = labels.shape[0]
    rej_1 = (labels == 1).sum()
    rej_0 = (labels == 0).sum()
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_rejected = auc(fpr, tpr)
    col_list = ['#_smaples_all', 'all_0', 'all_1', 'auc_all_samples', '#_smaples_all', 'rej_0', 'rej_1', 'auc_rejected']
    res_list_tmp = [smaples_all, all_0, all_1, auc_all_samples, smaples_rej, rej_0, rej_1, auc_rejected]
    res_list = np.array(res_list_tmp)
    #     res_list = []
    #     for i in res_list_tmp:
    #         res_list.append([i])
    #     print(col_list)
    #     print(res_list)
    print(pd.DataFrame([res_list], columns=col_list))
    print('rejection:')
    print(smaples_rej / smaples_all)

    return labels, probs


def confidance_probs(labels, probs, percent, bins, name):
    #     bins_array = np.append(np.arange(0,1,1/bins),1)
    all_1 = (labels == 1).sum()
    all_0 = (labels == 0).sum()
    bins_array = np.concatenate((np.arange(0, 0.4, 0.1), np.arange(0.4, 0.6, 0.05), np.arange(0.6, 1.01, 0.1)))
    bins_array[0] = -0.001
    bins_array[-1] = 1.001
    bins = bins_array.shape[0]
    probs_df = pd.DataFrame(np.array([labels, probs]).T, columns=['labels', 'probs'])
    true_conf_list = []
    false_conf_list = []
    x_probs = []
    n_spectra = []
    for i_bin in np.arange(bins - 1):
        tmp_df = probs_df[probs_df['probs'] > bins_array[i_bin]]
        tmp_df = tmp_df[tmp_df['probs'] <= bins_array[i_bin + 1]]

        # true_rate_tmp = (((tmp_df['labels'] == 1).sum() / tmp_df['labels'].shape[0]) / (probs_df['labels'] == 1).sum())
        # false_rate_tmp = (((tmp_df['labels'] == 0).sum() / tmp_df['labels'].shape[0]) / (probs_df['labels'] == 0).sum())
        # # print(true_rate_tmp)
        # # print(false_rate_tmp)
        # true_rate = true_rate_tmp/(true_rate_tmp + false_rate_tmp)
        # false_rate = false_rate_tmp/(true_rate_tmp + false_rate_tmp)
        # true_conf_list.append(true_rate)
        # false_conf_list.append(false_rate)

        # p_prior = tmp_df['labels'].shape[0]/probs_df['labels'].shape[0]
        # prior_1 = ((probs_df['labels'] == 1).sum()) / probs_df['labels'].shape[0]
        # prior_0 = ((probs_df['labels'] == 0).sum()) / probs_df['labels'].shape[0]
        # true_conf_list.append((prior_1/p_prior)*(tmp_df['labels'] == 1).sum() / tmp_df['labels'].shape[0])
        # false_conf_list.append((prior_0/p_prior)*(tmp_df['labels'] == 0).sum() / tmp_df['labels'].shape[0])

        true_conf_list.append((tmp_df['labels'] == 1).sum() / tmp_df['labels'].shape[0])
        false_conf_list.append((tmp_df['labels'] == 0).sum() / tmp_df['labels'].shape[0])
        x_probs.append((bins_array[i_bin + 1] + bins_array[i_bin]) / 2)
        n_spectra.append(tmp_df['labels'].shape[0])
    #     print(x_probs)
    #     print(true_conf_list)
    #     print(len(x_probs),len(true_conf_list))
    #     plt.figure(figsize = (5, 5))
    #     plt.plot(x_probs,true_conf_list)
    #     plt.plot(x_probs,false_conf_list)
    #     plt.show()

    #     m1, b1 = np. polyfit(x_probs, true_conf_list, 1)   ##########
    #     m0, b0 = np. polyfit(x_probs, false_conf_list, 1)   ##########
    #     x_probs_interp = np.append(np.arange(0,1,1/1000),1)   ##########
    #     true_conf = m1*x_probs_interp+b1   ##########
    #     false_conf = m0*x_probs_interp+b0   ##########

    #     print(true_conf, false_conf)
    plt.figure(figsize=(10, 5))
    wdth = 0.025
    shft = 0.015
    plt.bar(np.array(x_probs) + shft, true_conf_list, width=wdth, label='% true rate - 1, R')
    plt.bar(np.array(x_probs) - shft, false_conf_list, width=wdth, label='% true rate - 0, S')
    #     plt.plot(true_conf)
    #     plt.plot(false_conf)
    #     print(n_spectra)
    cnt = 0
    #     print((np.array(n_spectra)).sum())
    for itext in n_spectra:
        plt.text(x_probs[cnt] - shft, 0.2, str(itext))
        cnt = cnt + 1
    plt.legend()

    plt.title(
        name.split('/')[-1][0].capitalize() + name.split('/')[-1][1:] + ', #R=' + str(all_1) + ', #S=' + str(all_0))
    plt.xlabel('Classifier output (probabilities)')
    #     plt.show()
    plt.savefig(name + '_confidence_hist.png')

    #     true_conf = x_probs_interp[np.argmin(abs(true_conf-(1-percent)))]##########
    #     false_conf = x_probs_interp[np.argmin(abs(false_conf-(1-percent)))]##########
    #     return true_conf, false_conf ##########
    return


def sigmoid_confidence_interval(labels, probs, name):
    fpt, tpr, thresholds = metrics.roc_curve(labels, probs)
    best_acc_df = build_best_acc_df(labels, probs)
    #     print(best_acc_df.head())
    best_thres_prob = np.array(best_acc_df.iloc[0])[0]
    z_thresh = sigmoid_inverse(best_thres_prob)
    z_arr = sigmoid_inverse(probs)
    calibrated_probs = sigmoid(z_arr - z_thresh)
    confidance_probs((labels), (calibrated_probs), 0.05, 10, name)


def sigmoid(z):
    p = 1 / (1 + np.exp(-z))
    return p


def sigmoid_inverse(p):
    z = np.log2(p) - np.log2(1 - p)
    return z


def build_best_acc_df(labels, probs):
    fpt, tpr, thresholds = metrics.roc_curve(labels, probs)
    accuracy_ls = []
    for thres in thresholds:
        y_pred = np.where(probs > thres, 1, 0)
        accuracy_ls.append(accuracy_score(labels, y_pred, normalize=True))

    accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls), pd.Series(fpt), pd.Series(tpr)],
                            axis=1)
    accuracy_ls.columns = ['thresholds', 'accuracy', '1-SP', 'SE']
    accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
    #     print(accuracy_ls.head(50))
    #     print(np.array(accuracy_ls.iloc[0]))
    #     accuracy_ls.head()
    return accuracy_ls


def switch_R_S(labels, probs):
    labels = np.array(labels)
    probs = np.array(probs)
    labels = (labels == 0)
    probs = 1-probs
    return labels, probs


def _save_optimal_cut(fpr, tpr, best_point, path):
    """
    print the optimal point
    *not for users*
    """
    plt.figure(dpi=600)
    plt.plot(fpr, tpr)
    plt.scatter(best_point[0], best_point[1], c="red")
    plt.xlabel("1-specificity")
    plt.ylabel("sensitivity")
    plt.plot([0, 1], [0, 1], "--r")
    plt.axes().set_aspect(aspect='equal')
    plt.savefig(path+'.tiff')
    plt.close()


def _plot_optimal_cut(fpr, tpr, best_point):
    """
    print the optimal point
    *not for users*
    """
    plt.plot(fpr, tpr)
    plt.scatter(best_point[0], best_point[1], c="red")
    plt.xlabel("1-specificity")
    plt.ylabel("sensitivity")
    plt.plot([0, 1], [0, 1], "--r")
    plt.axes().set_aspect(aspect='equal')
    plt.show()
    plt.close()


def optimal_cut_point_on_roc__(labels, probs, delta_max=0.8, tpr_low_bound=0.5):
    """
    print the optimal cut on you're roc curve
    :param delta_max: the maximum delta between tpr and fpr (type: flute between 0 to 1)
    :param plot_point_on_ROC: is you like to show the roc curve now (type:bool)
    :return: report on you're optimal working point (type: dictionary)
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    n_n = labels[labels == 0].shape[0]
    n_p = labels[labels == 1].shape[0]
    # sen = fpr[fpr > 0.55]
    # spe = 1 - tpr[fpr > 0.55]
    sen = tpr[tpr > tpr_low_bound]
    spe = 1 - fpr[tpr > tpr_low_bound]
    thresholds = thresholds[tpr > tpr_low_bound]

    delt = abs(sen - spe)
    ix_1 = np.argwhere(delt <= delta_max)

    acc = (n_p / (n_p + n_n)) * sen[ix_1] + (n_n / (n_p + n_n)) * spe[ix_1]
    acc_max_index = ix_1[np.argmax(acc)][0]
    best_point = (1 - spe[acc_max_index], sen[acc_max_index])
#         auc = np.around(np.trapz(tpr, fpr), 2)

    recall_1 = sen[acc_max_index]
    recall_2 = spe[acc_max_index]
    precision_1 = (n_p * sen[acc_max_index]) / (n_p * sen[acc_max_index] + n_n * (1 - spe[acc_max_index]))
    precision_2 = (n_n * spe[acc_max_index]) / (n_n * spe[acc_max_index] + n_p * (1 - sen[acc_max_index]))

    report = {"auc": [np.around(auc_score, 2)], "acc": [np.around(acc.max(), 2)], "SE": [np.around(recall_1, 2)],
              "SP": [np.around(recall_2, 2)], "PPV": [np.around(precision_1, 2)],
              "NPV": [np.around(precision_2, 2)]
              }
#         report = {"auc": np.around(auc_score, 2), "acc": np.around(acc.max(), 2), "SE": np.around(recall_1, 2),
#                   "SP": np.around(recall_2, 2), "PPV (prcsion)": np.around(precision_1, 2),
#                   "NPV": np.around(precision_2, 2)
#                   }

#     _plot_optimal_cut(fpr, tpr, best_point)
    # if plot_point_on_ROC:
    #     p = Process(target=ClassificationPreprocessing, args=(fpr, tpr, best_point,))
    #     p.start()

    return report, thresholds[acc_max_index], best_point


def build_llr_table(df):
    df_llr = pd.DataFrame()
    df['log_prob_0'] = -np.log2(df['0'])
    df['log_prob_1'] = -np.log2(df['1'])
    df_test_prod = df.groupby(['group']).sum()
    df_test_mean = df.groupby(['group']).mean()
    df_test_cnt = df.groupby(['group']).count()
    df_test_mean['LLR'] = (df_test_prod['log_prob_0'] - df_test_prod['log_prob_1']) / df_test_cnt['labels']
    # df_test_mean['LLR'] = 1 / (1 + np.exp(-df_test_mean['LLR']))
    # print(df_test_mean.head(20))
    df_llr['1'] = 1 / (1 + np.exp(-df_test_mean['LLR']))
    df_llr['0'] = 1 - df_test_mean['1']
    df_llr['group'] = df_test_mean.index
    df_llr['labels'] = df_test_mean['labels']
    return df_llr


def calc_SE_SP_ACC(labels, probs, threshold, path):
    labels = np.array(labels)
    preds = probs > threshold
    preds = preds.astype(int)
    # print(labels.shape)
    # print(preds.shape)
    sen = np.round(np.sum(preds[labels == 1] == labels[labels == 1])/np.sum(labels == 1), 2)
    spe = np.round(np.sum(preds[labels == 0] == labels[labels == 0])/np.sum(labels == 0), 2)
    acc = np.round(np.sum(preds == labels)/labels.shape[0], 2)
    ppv = np.round(np.sum(preds[preds == 1] == labels[preds == 1])/np.sum(preds == 1), 2)
    npv = np.round(np.sum(preds[preds == 0] == labels[preds == 0])/np.sum(preds == 0), 2)
    res = np.array([[acc], [sen], [spe], [ppv], [npv]])
    res_df = pd.DataFrame(res.T, columns=['Acc', 'SE', 'SP', 'PPV', 'NPV'])
    res_df.to_csv(path + '/hard_res.csv')
    return acc, sen, spe, ppv, npv


def calc_results(labels, probs, threshold):
    labels = np.array(labels)
    probs = np.array(probs)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    labels = np.array(labels)
    preds = probs > threshold
    preds = preds.astype(int)
    # print(labels.shape)
    # print(preds.shape)
    sen = np.round(np.sum(preds[labels == 1] == labels[labels == 1])/np.sum(labels == 1), 2)
    spe = np.round(np.sum(preds[labels == 0] == labels[labels == 0])/np.sum(labels == 0), 2)
    acc = np.round(np.sum(preds == labels)/labels.shape[0], 2)
    ppv = np.round(np.sum(preds[preds == 1] == labels[preds == 1])/np.sum(preds == 1), 2)
    npv = np.round(np.sum(preds[preds == 0] == labels[preds == 0])/np.sum(preds == 0), 2)
    # res = np.array([[acc], [sen], [spe], [ppv], [npv]])
    # res_df = pd.DataFrame(res.T, columns=['Acc', 'SE', 'SP', 'PPV', 'NPV'])
    return auc_score, acc, sen, spe, ppv, npv


def compare_errors(y1, pred1, y2, pred2):
    hit_indices1 = (y1 == pred1).nonzero()[0]
    y1_hit = y1[hit_indices1]
    miss_indices1 = (y1 != pred1).nonzero()[0]
    y1_miss = y1[miss_indices1]

    hit_indices2 = (y2 == pred2).nonzero()[0]
    y2_hit = y2[hit_indices2]
    miss_indices2 = (y2 != pred2).nonzero()[0]
    y2_miss = y2[miss_indices2]

    intersection = np.intersect1d(miss_indices1, miss_indices2)
    print(intersection.shape)
    print(miss_indices1.shape)
    print(miss_indices2.shape)


#     plt.figure()
#     plt.scatter(1hit_indices[y1_hit==0], y1_hit[y1_hit==0])
#     plt.scatter(miss_indices1[y1_miss==0], y1_miss[y1_miss==0])
#     x = np.arange(y1.shape[0])
#     plt.plot(x, y2==pred2)
#     plt.plot(x, y1==pred1)
#     plt.show()
#     plt.close()

# def analysis_strat_k_fold(orig_x, orig_y, labels, probs, n_splits, random_stt, groups):
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_stt)
#     kf.get_n_splits(orig_x, orig_y)
#     auc_list = []
#     acc_list = []
#     icumsum = 0
#     for ix_train, ix_test in kf.split(orig_x, orig_y):
#         test_size = len(ix_test)
#         #         print(groups[icumsum:icumsum+test_size])
#         #         print(ix_test)
#         preds = probs[icumsum:icumsum + test_size] >= 0.5
#         acc_list.append(np.mean(preds == labels[icumsum:icumsum + test_size]))
#         fpr, tpr, thresholds_keras = roc_curve(labels[icumsum:icumsum + test_size], probs[icumsum:icumsum + test_size])
#         auc_list.append(auc(fpr, tpr))
#         #         error_analysis(labels[ix_test], probs[ix_test])
#         plot_roc(fpr, tpr, auc(fpr, tpr), 'cnn')
#         icumsum = icumsum + test_size
#     return auc_list, acc_list


def error_analysis(labels, probs, groups, threshold):
    pred = probs > threshold
    pred = pred.astype(int)
    miss_indices = (labels != pred).nonzero()[0]
    hit_indices = (labels == pred).nonzero()[0]
    y_hit = labels[hit_indices]
    y_hit = y_hit.astype(int)
    y_miss = labels[miss_indices]
    y_miss = y_miss.astype(int)
    miss_groups = groups[miss_indices]
    miss_probs = probs[miss_indices]
    hit_groups = groups[hit_indices]
    hit_probs = probs[hit_indices]


    plt.figure()
    for i_label in [0, 1]:
        print(miss_groups[y_miss == i_label][:3])
        print(miss_probs[y_miss == i_label][:3])
        print(hit_groups[y_hit == i_label][:3])
        print(hit_probs[y_hit == i_label][:3])
        plt.scatter(miss_groups[y_miss == i_label], miss_probs[y_miss == i_label],
                    label='class ' + str(i_label) + ',#=' + str(np.sum(y_miss == i_label)))
        # plt.scatter(groups[labels == i_label], probs[labels == i_label])
    plt.legend()
    plt.show()
    plt.close()


def plot_misclassified(x, y, pred, n):
    miss_indices = (y != pred).nonzero()[0]
    hit_indices = (y == pred).nonzero()[0]

    x_hit = x[hit_indices]
    y_hit = y[hit_indices]
    x_miss = x[miss_indices]
    y_miss = y[miss_indices]
    n_hits_0 = np.sum(y_hit == 0)
    n_hits_1 = np.sum(y_hit == 1)
    n_miss_0 = np.sum(y_miss == 0)
    n_miss_1 = np.sum(y_miss == 1)

    miss_indices_y1 = np.intersect1d(miss_indices, (y == 1).nonzero()[0])
    miss_indices_y0 = np.intersect1d(miss_indices, (y == 0).nonzero()[0])
    x_0_mean = np.mean(np.array(x_hit[y_hit == 0]), axis=0)
    x_1_mean = np.mean(np.array(x_hit[y_hit == 1]), axis=0)
    x_0_miss_mean = np.mean(np.array(x_miss[y_miss == 0]), axis=0)
    x_1_miss_mean = np.mean(np.array(x_miss[y_miss == 1]), axis=0)
    #     fig, ax = plt.subplots(nrows=1, ncols=n, sharex=True, sharey=True)
    #     ax = ax.flatten()
    #     for i_plot in range(n):
    #         xi = np.random.choice(miss_indices,1)
    #         ax[i_plot].plot(x[xi[0]], label='i='+str(xi)+', y='+str(y[xi]))
    #         ax[i_plot].plot(x_0_mean, label='0-mean')
    #         ax[i_plot].plot(x_1_mean, label='1-mean')
    #     plt.legend()
    #     plt.show()
    #     plt.close()

    plt.figure(figsize=(20, 10))
    #     for i_plot in range(n):
    #         xi = np.random.choice(miss_indices_y0,1)
    #         print(y[xi[0]],' ', pred[xi[0]])
    #         plt.plot(x[xi[0]], label='i='+str(xi)+', y='+str(y[xi]))
    #         xi = np.random.choice(miss_indices_y1,1)
    #         print(y[xi[0]],' ', pred[xi[0]])
    #         plt.plot(x[xi[0]], label='i='+str(xi)+', y='+str(y[xi]))

    #         xi = np.random.choice(hit_indices,1)
    #         print(y[xi[0]],' ', pred[xi[0]])
    #         plt.plot(x[xi[0]], label='i='+str(xi)+', y='+str(y[xi]))
    start = 150
    end = 200
    plt.plot(x_0_mean[start:end], label='0-mean-hit, #=' + str(n_hits_0))
    plt.plot(x_1_mean[start:end], label='1-mean-hit, #=' + str(n_hits_1))
    plt.plot(x_0_miss_mean[start:end], label='0-mean-miss, #=' + str(n_miss_0))
    plt.plot(x_1_miss_mean[start:end], label='1-mean-miss, #=' + str(n_miss_1))

    plt.legend()
    plt.show()
    plt.close()