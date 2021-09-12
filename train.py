from common.trials import run_experiments
from configuration import get_config_dict_from_yaml


def make_config_dict_list():
    config_dict_list = list()

    for name in [
                 "SEResNet28-attention-4",
                 # "ResNet28-avg",
                 # "SEResNet28-avg"
                 ]:
        config_dict = get_config_dict_from_yaml(name)
        config_dict_list.append(config_dict)

    return config_dict_list


if __name__ == '__main__':
    run_experiments(make_config_dict_list())
