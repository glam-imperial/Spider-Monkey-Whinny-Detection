import tensorflow as tf
import numpy as np

from common.common import safe_make_dir

BEST_VALUE_INITIALISER = dict()
BEST_VALUE_INITIALISER["pos_au_pr"] = -1.0


# TODO: lambda comparison function.
# TODO: Store other stuff, apart from the prediction.
# TODO: Separate target/measure to learn from target/measure to monitor.


class CustomSaverVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 target_to_measures,
                 keras_model_test):
        self.output_folder = output_folder
        self.method_string = method_string
        self.target_to_measures = target_to_measures
        self.keras_model_test = keras_model_test
        self.saver_paths = dict()
        self.saver_dict = dict()

        self.method_output_prefix = output_folder + "/" + method_string

        safe_make_dir(self.method_output_prefix)

        for target in target_to_measures.keys():
            self.saver_paths[target] = dict()
            self.saver_dict[target] = dict()
            safe_make_dir(self.method_output_prefix + "/" + target)
            for measure in target_to_measures[target]:
                self.saver_paths[target][measure] = self.method_output_prefix + "/" + target + "/" + measure
                self.saver_dict[target][measure] = self.keras_model_test
                # safe_make_dir(self.method_output_prefix + "/" + target + "/" + measure)

    def save_model(self,
                   target,
                   measure):
        self.saver_dict[target][measure].save(self.saver_paths[target][measure] + "_model")

    def load_model(self,
                   target,
                   measure):
        self.saver_dict[target][measure] = tf.keras.models.load_model(self.saver_paths[target][measure] + "_model")


class PerformanceMonitorVirtual:
    def __init__(self,
                 output_folder,
                 method_string,
                 custom_saver,
                 target_to_measures,
                 are_test_labels_available,
                 model_configuration):
        self.output_folder = output_folder
        self.method_string = method_string
        self.custom_saver = custom_saver
        self.target_to_measures = target_to_measures
        self.are_test_labels_available = are_test_labels_available
        self.model_configuration = model_configuration

        self.method_output_prefix = output_folder + "/" + method_string

        # Contains measure summary for last run.
        self.measures = dict()

        # I may want to monitor multiple performance measures per multiple tasks/targets separately.

        # Contains test items and summary, dependent on target and measure
        self.test_measures_dict = dict()
        self.test_items_dict = dict()

        self.best_performance_dict = dict()
        for target in self.target_to_measures.keys():
            self.best_performance_dict[target] = dict()
            self.test_measures_dict[target] = dict()
            self.test_items_dict[target] = dict()
            for measure in self.target_to_measures[target]:
                self.best_performance_dict[target][measure] = BEST_VALUE_INITIALISER[measure]

    def get_measures(self,
                     items,
                     partition):
        raise NotImplementedError

    def report_measures(self,
                        partition):
        raise NotImplementedError

    def monitor_improvement(self):
        noticed_improvement = False

        for target in self.target_to_measures.keys():
            for measure in self.target_to_measures[target]:
                # TODO: lambda comparison function.
                if self.best_performance_dict[target][measure] < self.measures["devel"][target][measure]:
                    self.best_performance_dict[target][measure] = self.measures["devel"][target][measure]
                    noticed_improvement = True
                    self.custom_saver.save_model(target=target,
                                                 measure=measure)
        return noticed_improvement

    def get_test_measures(self,
                          test_items,
                          target,
                          measure):
        if self.are_test_labels_available:
            self.get_measures(items=test_items,
                              partition="test")
            self.test_measures_dict[target][measure] = self.measures["test"]
        self.test_items_dict[target][measure] = test_items

    def report_best_performance_measures(self):
        raise NotImplementedError

    def get_results_summary(self):
        results = dict()
        results["method_string"] = self.method_string

        for target in self.target_to_measures.keys():
            results[target] = dict()
            for measure in self.target_to_measures[target]:
                results[target][measure] = dict()

                results[target][measure]["test_pred"] = self.test_items_dict[target][measure][target]["pred"]
                np.save(self.output_folder + "/" + self.method_string + "/" + target + "/" + measure + "_" + "test_pred.npy",
                        self.test_items_dict[target][measure][target]["pred"])

                # TODO: Store other stuff, apart from the prediction.

        if self.are_test_labels_available:
            for target in self.target_to_measures:
                for measure in self.target_to_measures[target]:
                    if target in self.test_measures_dict[target][measure].keys():
                        for measure_name in self.test_measures_dict[target][measure][target].keys():
                            results[target][measure]["test_" + measure_name] = self.test_measures_dict[target][measure][target][measure_name]
                            results[target][measure]["test_true"] = self.test_items_dict[target][measure][target]["true"]
                            np.save(self.output_folder + "/" + self.method_string + "/" + target + "/" + measure + "_" + "test_true.npy",
                                    self.test_items_dict[target][measure][target]["true"])

        for target in self.best_performance_dict.keys():
            for measure in self.best_performance_dict[target].keys():
                results[target][measure]["best_devel_" + measure] = self.best_performance_dict[target][measure]

        return results
