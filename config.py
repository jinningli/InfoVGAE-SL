import time

import torch


def update_arg_with_config_name(args, config_name, phrase):
    """
    :param args: argparse dict
    :param config_name: name of config
    :param phrase: train / evaluate
    :return: argparse dict
    """

    if config_name == "InfoVGAE-SL_election_3D":
        setattr(args, "data_path", "dataset/election/data.csv")
        setattr(args, "stopword_path", "dataset/election/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.01)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "kthreshold", 5)
        setattr(args, "uthreshold", 3)
        setattr(args, "tthreshold", 0)
        setattr(args, "beta", 5e-7)
        setattr(args, "epochs", 73)

    elif config_name == "InfoVGAE-SL_eurovision_3D":
        setattr(args, "data_path", "dataset/eurovision/data.csv")
        setattr(args, "stopword_path", "dataset/eurovision/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "decode_mode", "partial")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.03)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "kthreshold", 5)
        setattr(args, "uthreshold", 3)
        setattr(args, "tthreshold", 0)
        setattr(args, "beta", 1e-6)
        setattr(args, "pos_weight_lambda", 1.4)
        setattr(args, "epochs", 94)

    elif config_name == "InfoVGAE-SL_bill_3D":
        setattr(args, "data_path", "")
        setattr(args, "stopword_path", "")
        setattr(args, "model", "VGAE")
        setattr(args, "decode_mode", "partial")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.01)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "kthreshold", 5)
        setattr(args, "uthreshold", 3)
        setattr(args, "tthreshold", 0)
        setattr(args, "beta", 5e-7)
        setattr(args, "pos_weight_lambda", 1.0)
        setattr(args, "epochs", 132)

    elif config_name == "InfoVGAE-SL_war_3D":
        setattr(args, "data_path", "dataset/war/data.csv")
        setattr(args, "stopword_path", "dataset/war/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 2)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "kthreshold", 1)
        setattr(args, "uthreshold", 10)
        setattr(args, "tthreshold", 20)
        setattr(args, "beta", 5e-7)
        setattr(args, "pos_weight_lambda", 2.0)
        setattr(args, "epochs", 19)

    else:
        raise NotImplementedError("Unknown config name")

    setattr(args, "output_path", "./output/{}_{}_{}".format(config_name, args.dataset, time.strftime("%Y%m%d%H%M%S", time.localtime())))
    return args
