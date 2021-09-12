import statistics
import os.path

import numpy as np
from scipy.stats import ttest_ind

from common.common import load_pickle

PROJECT_FOLDER = '/data/PycharmProjects/SpiderMonkeysNew'
OUTPUT_FOLDER = PROJECT_FOLDER + '/Results'


def trial_average(summary_list, name, return_list=False):
    value_list = list()
    for s in summary_list:
        if name in s.keys():
            value_list.append(s[name])
    if len(value_list) > 1:
        m_v = statistics.mean(value_list)
        std_v = statistics.stdev(value_list)
        max_v = max(value_list)
    elif len(value_list) == 0:
        m_v = 0.0
        std_v = 0.0
        max_v = 0.0
    else:
        m_v = value_list[0]
        std_v = 0.0
        max_v = value_list[0]

    if return_list:
        return (m_v, std_v, max_v), value_list
    else:
        return (m_v, std_v, max_v), value_list


def t_test_2_methods(method_names):
    method_1, method_2 = method_names
    print()
    print(method_1, method_2)

    trial_summaries_1 = list()

    for t in range(10):
        if not os.path.exists(OUTPUT_FOLDER + "/" + method_1 + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        # print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + method_1 + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        # print(results_summary["whinny_single"])
        trial_summaries_1.append(results_summary["whinny_single"])
        # print(results_summary["whinny_single"])

    trial_summaries_2 = list()

    for t in range(10):
        if not os.path.exists(OUTPUT_FOLDER + "/" + method_2 + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        # print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + method_2 + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        # print(results_summary["whinny_single"])
        trial_summaries_2.append(results_summary["whinny_single"])
        # print(results_summary["whinny_single"])

    for measure_name in ["best_devel_pos_au_pr",
                         "test_macro_au_roc",
                         "test_pos_au_pr",
                         "test_neg_au_pr",
                         "test_macro_f1",
                         "test_macro_recall",
                         "test_macro_precision",
                         "test_pos_f1",
                         "test_pos_recall",
                         "test_pos_precision"
                         ]:
        _, array_1 = trial_average(trial_summaries_1, measure_name)
        _, array_2 = trial_average(trial_summaries_2, measure_name)

        array_1 = np.array(array_1, dtype=np.float32)
        array_2 = np.array(array_2, dtype=np.float32)

        t_stat, p_value = ttest_ind(array_1,
                                    array_2,
                                    equal_var=False)
        print(measure_name, p_value)


t_test_2_methods(["CNN14_PANN-avg",
                  "VGG16-avg"])
