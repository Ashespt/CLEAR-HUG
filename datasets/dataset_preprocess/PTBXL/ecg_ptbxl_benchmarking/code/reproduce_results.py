from experiments.scp_experiment import SCP_Experiment
from configs.fastai_configs import *
from configs.wavelet_configs import *


def main(args):

    datafolder = args.datafolder
    datafolder_icbeb = "../data/ICBEB/"
    outputfolder = "datasets/ecg_datasets/PTBXL/"

    models = [
        conf_fastai_xresnet1d101,
        conf_fastai_resnet1d_wang,
        conf_fastai_lstm,
        conf_fastai_lstm_bidir,
        conf_fastai_fcn_wang,
        conf_fastai_inception1d,
        conf_wavelet_standard_nn,
    ]

    experiments = [
        ("diagnostic", "diagnostic"),
        ("subdiagnostic", "subdiagnostic"),
        ("superdiagnostic", "superdiagnostic"),
        ("form", "form"),
        ("rhythm", "rhythm"),
        ("all", "all"),
    ]

    for name, task in experiments:
        print(f"========task: {task}========")
        e = SCP_Experiment(name, task, datafolder, outputfolder, models)
        e.prepare()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("preprocess", add_help=False)
    parser.add_argument("--datafolder", default='datasets/PTB-XL/', type=str)
    args = parser.parse_args()
    main(args)
