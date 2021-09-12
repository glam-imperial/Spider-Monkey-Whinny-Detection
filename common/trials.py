import os.path

from common.experiment import experiment_run
from common.common import store_pickle


def run_experiments(config_dict_list):
    all_experiments_results_summary_list = list()
    for config_dict in config_dict_list:
        single_experiment_results_summary = run_trials(config_dict=config_dict)
        all_experiments_results_summary_list.append(single_experiment_results_summary)

    return all_experiments_results_summary_list


def run_trials(config_dict):
    all_trials_results_summary_list = list()
    print(config_dict["method_string"])
    for t in range(config_dict["number_of_trials"]):
        config_dict_effective = {k: v for k, v in config_dict.items()}
        config_dict_effective["current_trial"] = t
        single_trial_results_summary = run_single_trial(config_dict=config_dict_effective)
        single_trial_results_summary["configuration_dict"] = config_dict_effective

        t_eff = t
        while os.path.exists(config_dict["results_summary_path"] + "_trial" + repr(t_eff) + ".pkl"):
            t_eff += 1

        store_pickle(config_dict["results_summary_path"] + "_trial" + repr(t_eff) + ".pkl",
                     single_trial_results_summary)

        all_trials_results_summary_list.append(single_trial_results_summary)

    # TODO: Best across trials (measures & model) -- option to keep all, or just the best?

    return all_trials_results_summary_list


def run_single_trial(config_dict):
    single_trial_results_summary = experiment_run(config_dict=config_dict)
    return single_trial_results_summary
