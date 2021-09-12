import numpy as np
import sklearn

from common.evaluation.monitor import PerformanceMonitorVirtual, CustomSaverVirtual


class CustomSaver(CustomSaverVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 target_to_measures,
                 keras_model_test):
        super().__init__(output_folder,
                         method_string,
                         target_to_measures,
                         keras_model_test)


class PerformanceMonitor(PerformanceMonitorVirtual):
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 target_to_measures,
                 are_test_labels_available,
                 model_configuration):
        super().__init__(output_folder,
                 method_string,
                 custom_saver,
                 target_to_measures,
                 are_test_labels_available,
                 model_configuration)

    def get_measures(self,
                     items,
                     partition):
        global_pooling = self.model_configuration["global_pooling"]
        pred_logits = items["whinny_single"]["pred"]
        true_indicator = items["whinny_single"]["true"]

        # pred_prob = np.sigmoid(pred_logits)

        pred_logits = np.nan_to_num(pred_logits)

        if global_pooling == "Prediction":
            # pred_prob = sigmoid(pred_logits)
            pred_logits[pred_logits < 1e-7] = 1e-7
            pred_logits[pred_logits > 1. - 1e-7] = 1. - 1e-7

            pred_logits = np.log(pred_logits / (1. - pred_logits))
        else:
            pass

        pred_prob = stable_softmax(pred_logits)

        # pred_prob_continuous = stable_softmax(items["whinny_continuous"]["pred"].reshape((-1, 2)))

        # print(np.hstack([pred_prob, true_indicator]))
        # np.save("pred_true.npy", np.hstack([pred_prob, true_indicator]))
        # print(pred_prob_continuous)
        # print(items["whinny_continuous"]["true"])

        # print(true_indicator.sum(axis=0), true_indicator.shape[0])
        # print(items["whinny_continuous"]["true"].reshape((-1, 2)).sum(axis=0), items["whinny_continuous"]["true"].reshape((-1, 2)).shape[0])

        true_labels = np.argmax(true_indicator, axis=-1)
        pred_labels = np.argmax(pred_logits, axis=-1)

        measures = dict()
        measures["whinny_single"] = dict()

        # Accuracy.
        accuracy = sklearn.metrics.accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
        measures["whinny_single"]["accuracy"] = accuracy

        # AU-ROC.
        au_roc_macro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="macro")
        au_roc_micro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="micro")
        measures["whinny_single"]["macro_au_roc"] = au_roc_macro
        measures["whinny_single"]["micro_au_roc"] = au_roc_micro

        # AU-PR
        au_pr_classes = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average=None)
        au_prc_macro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="macro")
        au_prc_micro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="micro")
        measures["whinny_single"]["macro_au_pr"] = au_prc_macro
        measures["whinny_single"]["micro_au_pr"] = au_prc_micro
        measures["whinny_single"]["pos_au_pr"] = au_pr_classes[1]
        measures["whinny_single"]["neg_au_pr"] = au_pr_classes[0]

        # Precision, Recall, F1
        precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                           pred_labels,
                                                                                                           average=None)
        precision_macro, recall_macro, f1_macro, _ = sklearn.metrics.precision_recall_fscore_support(true_labels,
                                                                                                     pred_labels,
                                                                                                     average="macro")
        measures["whinny_single"]["macro_f1"] = precision_classes[1]
        measures["whinny_single"]["macro_f1"] = np.mean(precision_classes)
        measures["whinny_single"]["macro_f1"] = precision_macro
        measures["whinny_single"]["pos_precision"] = precision_classes[1]
        measures["whinny_single"]["macro_precision"] = np.mean(precision_classes)
        measures["whinny_single"]["micro_precision"] = precision_macro
        measures["whinny_single"]["pos_recall"] = recall_classes[1]
        measures["whinny_single"]["macro_recall"] = np.mean(recall_classes)
        measures["whinny_single"]["micro_recall"] = recall_macro
        measures["whinny_single"]["pos_f1"] = f1_classes[1]
        measures["whinny_single"]["macro_f1"] = np.mean(f1_classes)
        measures["whinny_single"]["micro_f1"] = f1_macro

        self.measures[partition] = measures

    def report_measures(self,
                        partition):
        measures = self.measures[partition]

        print("Macro AU PR:    ", measures["whinny_single"]["macro_au_pr"])
        print("Macro AU ROC:   ", measures["whinny_single"]["macro_au_roc"])
        # print("Macro F1:       ", measures["whinny_single"]["macro_f1"])
        # print("Macro Recall:   ", measures["whinny_single"]["macro_recall"])
        # print("Macro Precision:", measures["whinny_single"]["macro_precision"])
        print("POS AU PR:      ", measures["whinny_single"]["pos_au_pr"])
        print("NEG AU PR:      ", measures["whinny_single"]["neg_au_pr"])
        # print("POS F1:         ", measures["whinny_single"]["pos_f1"])
        # print("POS Recall:     ", measures["whinny_single"]["pos_recall"])
        # print("POS Precision:  ", measures["whinny_single"]["pos_precision"])

    def report_best_performance_measures(self):
        for target in self.target_to_measures.keys():
            for measure in self.target_to_measures[target]:
                print("Best devel POS AU PR:", self.best_performance_dict[target][measure])

                if self.are_test_labels_available:
                    print("Test  Macro AU PR:    ",
                          self.test_measures_dict[target][measure]["whinny_single"]["macro_au_pr"])
                    print("Test  Macro AU ROC:   ",
                          self.test_measures_dict[target][measure]["whinny_single"]["macro_au_roc"])
                    print("Test  POS AU PR:      ",
                          self.test_measures_dict[target][measure]["whinny_single"]["pos_au_pr"])
                    print("Test  NEG AU PR:      ",
                          self.test_measures_dict[target][measure]["whinny_single"]["neg_au_pr"])
                    print("Test  Macro F1:       ",
                          self.test_measures_dict[target][measure]["whinny_single"]["macro_f1"])
                    print("Test  Macro Recall:   ",
                          self.test_measures_dict[target][measure]["whinny_single"]["macro_recall"])
                    print("Test  Macro Precision:",
                          self.test_measures_dict[target][measure]["whinny_single"]["macro_precision"])
                    print("Test  POS F1:         ",
                          self.test_measures_dict[target][measure]["whinny_single"]["pos_f1"])
                    print("Test  POS Recall:     ",
                          self.test_measures_dict[target][measure]["whinny_single"]["pos_recall"])
                    print("Test  POS Precision:  ",
                          self.test_measures_dict[target][measure]["whinny_single"]["pos_precision"])


# def initialise_best_performance_dict():
#     best_performance_dict = dict()
#     best_performance_dict["whinny_single"] = dict()
#     best_performance_dict["whinny_single"]["pos_au_pr"] = - 1.0
#     return best_performance_dict


# def monitor_improvement(best_performance_dict,
#                         devel_measures,
#                         saver_paths,
#                         saver_dict,
#                         sess):
#     noticed_improvement = False
#
#     y_names = list(best_performance_dict.keys())
#
#     for y_name in y_names:
#         if best_performance_dict[y_name]["pos_au_pr"] < devel_measures[y_name]["pos_au_pr"]:
#             for measure_name in ["pos_au_pr", ]:
#                 best_performance_dict[y_name][measure_name] = devel_measures[y_name][measure_name]
#             # saver_dict[y_name].save(sess, saver_paths[y_name])
#             saver_dict[y_name].save(saver_paths[y_name] + "_model")
#             noticed_improvement = True
#
#     return noticed_improvement


def stable_softmax(X):
    exps = np.exp(X - np.max(X, 1).reshape((X.shape[0], 1)))
    return exps / np.sum(exps, 1).reshape((X.shape[0], 1))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


# def get_measures(items,
#                  model_configuration):
#     global_pooling = model_configuration["global_pooling"]
#     pred_logits = items["whinny_single"]["pred"]
#     true_indicator = items["whinny_single"]["true"]
#
#     # pred_prob = np.sigmoid(pred_logits)
#
#     pred_logits = np.nan_to_num(pred_logits)
#
#     if global_pooling == "Prediction":
#         # pred_prob = sigmoid(pred_logits)
#         pred_logits[pred_logits < 1e-7] = 1e-7
#         pred_logits[pred_logits > 1. - 1e-7] = 1. - 1e-7
#
#         pred_logits = np.log(pred_logits / (1. - pred_logits))
#     else:
#         pass
#
#     pred_prob = stable_softmax(pred_logits)
#
#     # pred_prob_continuous = stable_softmax(items["whinny_continuous"]["pred"].reshape((-1, 2)))
#
#     # print(np.hstack([pred_prob, true_indicator]))
#     # np.save("pred_true.npy", np.hstack([pred_prob, true_indicator]))
#     # print(pred_prob_continuous)
#     # print(items["whinny_continuous"]["true"])
#
#     # print(true_indicator.sum(axis=0), true_indicator.shape[0])
#     # print(items["whinny_continuous"]["true"].reshape((-1, 2)).sum(axis=0), items["whinny_continuous"]["true"].reshape((-1, 2)).shape[0])
#
#     true_labels = np.argmax(true_indicator, axis=-1)
#     pred_labels = np.argmax(pred_logits, axis=-1)
#
#     measures = dict()
#     measures["whinny_single"] = dict()
#
#     # Accuracy.
#     accuracy = sklearn.metrics.accuracy_score(true_labels, pred_labels, normalize=True, sample_weight=None)
#     measures["whinny_single"]["accuracy"] = accuracy
#
#     # AU-ROC.
#     au_roc_macro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="macro")
#     au_roc_micro = sklearn.metrics.roc_auc_score(true_indicator, pred_prob, average="micro")
#     measures["whinny_single"]["macro_au_roc"] = au_roc_macro
#     measures["whinny_single"]["micro_au_roc"] = au_roc_micro
#
#     # AU-PR
#     au_pr_classes = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average=None)
#     au_prc_macro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="macro")
#     au_prc_micro = sklearn.metrics.average_precision_score(true_indicator, pred_prob, average="micro")
#     measures["whinny_single"]["macro_au_pr"] = au_prc_macro
#     measures["whinny_single"]["micro_au_pr"] = au_prc_micro
#     measures["whinny_single"]["pos_au_pr"] = au_pr_classes[1]
#     measures["whinny_single"]["neg_au_pr"] = au_pr_classes[0]
#
#     # Precision, Recall, F1
#     precision_classes, recall_classes, f1_classes, _ = sklearn.metrics.precision_recall_fscore_support(true_labels, pred_labels, average=None)
#     precision_macro, recall_macro, f1_macro, _ = sklearn.metrics.precision_recall_fscore_support(true_labels, pred_labels, average="macro")
#     measures["whinny_single"]["macro_f1"] = precision_classes[1]
#     measures["whinny_single"]["macro_f1"] = np.mean(precision_classes)
#     measures["whinny_single"]["macro_f1"] = precision_macro
#     measures["whinny_single"]["pos_precision"] = precision_classes[1]
#     measures["whinny_single"]["macro_precision"] = np.mean(precision_classes)
#     measures["whinny_single"]["micro_precision"] = precision_macro
#     measures["whinny_single"]["pos_recall"] = recall_classes[1]
#     measures["whinny_single"]["macro_recall"] = np.mean(recall_classes)
#     measures["whinny_single"]["micro_recall"] = recall_macro
#     measures["whinny_single"]["pos_f1"] = f1_classes[1]
#     measures["whinny_single"]["macro_f1"] = np.mean(f1_classes)
#     measures["whinny_single"]["micro_f1"] = f1_macro
#
#     return measures


# def report_measures(measures):
#     # print("Macro AU PR:    ", measures["whinny_single"]["macro_au_pr"])
#     # print("Macro AU ROC:   ", measures["whinny_single"]["macro_au_roc"])
#     # print("Macro F1:       ", measures["whinny_single"]["macro_f1"])
#     # print("Macro Recall:   ", measures["whinny_single"]["macro_recall"])
#     # print("Macro Precision:", measures["whinny_single"]["macro_precision"])
#
#     print("Macro AU PR:    ", measures["whinny_single"]["macro_au_pr"])
#     print("Macro AU ROC:   ", measures["whinny_single"]["macro_au_roc"])
#     # print("Macro F1:       ", measures["whinny_single"]["macro_f1"])
#     # print("Macro Recall:   ", measures["whinny_single"]["macro_recall"])
#     # print("Macro Precision:", measures["whinny_single"]["macro_precision"])
#     print("POS AU PR:      ", measures["whinny_single"]["pos_au_pr"])
#     print("NEG AU PR:      ", measures["whinny_single"]["neg_au_pr"])
#     # print("POS F1:         ", measures["whinny_single"]["pos_f1"])
#     # print("POS Recall:     ", measures["whinny_single"]["pos_recall"])
#     # print("POS Precision:  ", measures["whinny_single"]["pos_precision"])


# def report_best_performance_measures(best_performance_dict,
#                                      are_test_labels_available,
#                                      test_measures_dict):
#     print("Best devel POS AU PR:", best_performance_dict["whinny_single"]["pos_au_pr"])
#
#     if are_test_labels_available:
#         print("Test  Macro AU PR:    ", test_measures_dict["whinny_single"]["whinny_single"]["macro_au_pr"])
#         print("Test  Macro AU ROC:   ", test_measures_dict["whinny_single"]["whinny_single"]["macro_au_roc"])
#         print("Test  POS AU PR:      ", test_measures_dict["whinny_single"]["whinny_single"]["pos_au_pr"])
#         print("Test  NEG AU PR:      ", test_measures_dict["whinny_single"]["whinny_single"]["neg_au_pr"])
#         print("Test  Macro F1:       ", test_measures_dict["whinny_single"]["whinny_single"]["macro_f1"])
#         print("Test  Macro Recall:   ", test_measures_dict["whinny_single"]["whinny_single"]["macro_recall"])
#         print("Test  Macro Precision:", test_measures_dict["whinny_single"]["whinny_single"]["macro_precision"])
#         print("Test  POS F1:         ", test_measures_dict["whinny_single"]["whinny_single"]["pos_f1"])
#         print("Test  POS Recall:     ", test_measures_dict["whinny_single"]["whinny_single"]["pos_recall"])
#         print("Test  POS Precision:  ", test_measures_dict["whinny_single"]["whinny_single"]["pos_precision"])


# def get_results_summary(output_folder,
#                         method_string,
#                         best_performance_dict,
#                         test_measures_dict,
#                         test_items_dict,
#                         are_test_labels_available):
#     results = dict()
#     results["method_string"] = method_string
#
#     for y_name in test_measures_dict.keys():
#         results[y_name] = dict()
#         results[y_name]["test_pred"] = test_items_dict[y_name][y_name]["pred"]
#         np.save(output_folder + "/" + method_string + "/" + y_name + "_" + "test_pred.npy", test_items_dict[y_name][y_name]["pred"])
#
#     if are_test_labels_available:
#         for y_name in test_measures_dict.keys():
#             if y_name in test_measures_dict[y_name].keys():
#                 for measure_name in test_measures_dict[y_name][y_name].keys():
#                     results[y_name]["test_" + measure_name] = test_measures_dict[y_name][y_name][measure_name]
#                     results[y_name]["test_true"] = test_items_dict[y_name][y_name]["true"]
#                     np.save(output_folder + "/" + method_string + "/" + y_name + "_" + "test_true.npy", test_items_dict[y_name][y_name]["true"])
#
#     for y_name in best_performance_dict.keys():
#         for measure_name in best_performance_dict[y_name].keys():
#             results[y_name]["best_devel_" + measure_name] = best_performance_dict[y_name][measure_name]
#
#     return results
