import statistics
import os.path

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
        return (m_v, std_v, max_v)


results_dict = dict()


for name in [
    "ResNet38_PANN-avg",
    "SEResNet38_PANN-avg",
    "SEResNet38_PANN-avg-max",
    "Hong-avg",
    "CNN14_PANN-avg",
    "VGG16-avg",
    "2DCRNN-avg",
    "1DCRNN-avg",
]:

    print(name)
    trial_summaries = list()

    for t in range(10):
        if not os.path.exists(OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"):
            continue
        # print("Trial: ", t)
        filepath = OUTPUT_FOLDER + "/" + name + "/results_summary_trial" + repr(t) + ".pkl"
        try:
            results_summary = load_pickle(filepath)
        except FileNotFoundError:
            continue
        # print(results_summary["whinny_single"])
        results_dict[name] = results_summary["whinny_single"]
        trial_summaries.append(results_summary["whinny_single"])
        # print(results_summary["whinny_single"])
        # try:
        #     print("Best devel POS AU PR:", results_summary["whinny_single"]["best_devel_pos_au_pr"])
        # except KeyError:
        #     pass

        # if True:
        #     print("Test  Macro AU PR:    ", results_summary["whinny_single"]["test_macro_au_pr"])
        #     print("Test  Macro AU ROC:   ", results_summary["whinny_single"]["test_macro_au_roc"])
        #     print("Test  POS AU PR:      ", results_summary["whinny_single"]["test_pos_au_pr"])
        #     print("Test  NEG AU PR:      ", results_summary["whinny_single"]["test_neg_au_pr"])
        #     print("Test  Macro F1:       ", results_summary["whinny_single"]["test_macro_f1"])
        #     print("Test  Macro Recall:   ", results_summary["whinny_single"]["test_macro_recall"])
        #     print("Test  Macro Precision:", results_summary["whinny_single"]["test_macro_precision"])
        #     print("Test  POS F1:         ", results_summary["whinny_single"]["test_pos_f1"])
        #     print("Test  POS Recall:     ", results_summary["whinny_single"]["test_pos_recall"])
        #     print("Test  POS Precision:  ", results_summary["whinny_single"]["test_pos_precision"])

    print("Trial averages.")
    print("Best devel POS AU PR:", trial_average(trial_summaries, "best_devel_pos_au_pr"))

    if True:
        print("Test  Macro AU PR:    ", trial_average(trial_summaries, "test_macro_au_pr"))
        print("Test  Macro AU ROC:   ", trial_average(trial_summaries, "test_macro_au_roc"))
        print("Test  POS AU PR:      ", trial_average(trial_summaries, "test_pos_au_pr"))
        print("Test  NEG AU PR:      ", trial_average(trial_summaries, "test_neg_au_pr"))
        print("Test  Macro F1:       ", trial_average(trial_summaries, "test_macro_f1"))
        print("Test  Macro Recall:   ", trial_average(trial_summaries, "test_macro_recall"))
        print("Test  Macro Precision:", trial_average(trial_summaries, "test_macro_precision"))
        print("Test  POS F1:         ", trial_average(trial_summaries, "test_pos_f1"))
        print("Test  POS Recall:     ", trial_average(trial_summaries, "test_pos_recall"))
        print("Test  POS Precision:  ", trial_average(trial_summaries, "test_pos_precision"))
