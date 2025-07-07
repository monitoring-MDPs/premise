from premise.interval.conformal_prediction.train_stoch_seq_nsc import *
from premise.interval.conformal_prediction.train_seq_se import *
from premise.interval.conformal_prediction.train_seq_nsc import *
from premise.interval.conformal_prediction.CP_Classification import *
from premise.interval.conformal_prediction.CP_Regression import *
from premise.interval.conformal_prediction.SeqDataset import *
import torch
import pickle 
from torch.autograd import Variable
import premise.interval.conformal_prediction.utility_functions as utils
import numpy as np
import argparse
from premise.interval.conformal_prediction.InvertedPendulum import *
from premise.interval.conformal_prediction.MC_model import *
import time
import torch.nn.functional #ANTONINA
from premise.interval.loading import build_suo, build_suo_args_parser




def infer_on_new_batch(new_noisy, state_estimator, error_estimator, rej_classifier, max_min, cp_comb_class): 
    
    new_noisy_scaled = -1+2*(new_noisy - max_min['dataset.MIN[1]'])/(max_min['dataset.MAX[1]']-max_min['dataset.MIN[1]'])
    Y1 = np.transpose(new_noisy_scaled, (0,2,1))
    Y1t = Variable(FloatTensor(Y1))

    print(Y1t.shape)


    state_estimator.eval()    
    state_estim = state_estimator(Y1t)
    error_estimator.eval()
    label_hypothesis = error_estimator(state_estim)
    
    label_prob = torch.nn.functional.softmax(label_hypothesis, dim=1)
    error_prob = label_prob[:, 1]

    pool_conf_cred = cp_comb_class.compute_confidence_credibility(np.transpose(new_noisy_scaled,(0,2,1)))
    keep_mask = utils.apply_svc_query_strategy(rej_classifier, pool_conf_cred)
    
    return error_prob, keep_mask

def conformal_testing_main(args: argparse.Namespace, se_path, error_path, rej_path, stats_path, cp_classification_path):
    
    #se_path = args.se_path
    #error_path = args.error_path
    #rej_path = args.rej_path
    #stats_path = args.stats_path
    #cp_classification_path = args.cp_classification_path

    suo, initial_amount, horizon = build_suo(args)
    horizon = args.horizon

    models_dict = {"IP": InvertedPendulum(), "MC": mc_model(horizon)}
    model = models_dict[args.model_name]
    model_name = args.model_name 

    model = mc_model(horizon)

    evaluation_samples = []
    for x in range(args.testing_samples):
            path = suo.generate_random_traces([], args.length)[0]
            evaluation_samples.append(tuple(path))


    traces = suo.generate_random_traces([], args.length, args.testing_samples)

    noisy_measurements = model.get_noisy_measurments(traces, horizon)
    labels = model.gen_labels(traces, horizon)

    state_estimator = torch.load(se_path, weights_only=False)
    error_estimator = torch.load(error_path, weights_only=False)
    cp_comb_class = torch.load(cp_classification_path, weights_only=False)

    with open(rej_path, 'rb') as f:
        rej_classifier = pickle.load(f)
    
    rej_classifier = rej_classifier['rej_rule']

    with open(stats_path, 'rb') as f:
        max_min = pickle.load(f)


    error_prob, keep_mask = infer_on_new_batch(noisy_measurements, state_estimator, error_estimator, rej_classifier, max_min, cp_comb_class)
    
    print(labels)
    print(error_prob)
    print(keep_mask)

    return labels, error_prob, keep_mask


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument("--model_name", type=str, default="MC", help="Name of the model (first letters code).")
    #group.add_argument("--do_refinement", type=bool, default=True, help="Flag: refine of the rejection rule.")
    #group.add_argument("--nb_active_iterations", type=int, default=1, help="Number of active learning iterations.")
    #group.add_argument("--nb_epochs", type=int, default=200, help="Number of epochs.")
    #group.add_argument("--nb_epochs_active", type=int, default=400, help="Number of epochs in active learning.")
    #group.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    #group.add_argument("--lr", type=float, default=0.00001, help="Adam: learning rate")
    #group.add_argument("--lr_tuning", type=float, default=0.000001, help="Adam: learning rate for fine tuning")
    #group.add_argument("--net_type", type=str, default="Conv", help="Type of the net: Conv or FF.")
    #group.add_argument("--nb_filters", type=int, default=128, help="Number of filters per conv layer.")
    #group.add_argument("--epsilon", type=float, default=0.05, help="CP significance level.")
    #group.add_argument("--split_rate", type=float, default=10/14, help="adam: learning rate")
    #group.add_argument("--pool_size_ref", type=int, default=25000, help="Size of the pool for the refinement step.")
    #group.add_argument("--pool_size", type=int, default=50000, help="Size of the pool for one active learning step.")
    #group.add_argument("--reinit_weights", type=bool, default=False, help="Flag: do reinitialize the weights in active learning steps.")
    #group.add_argument("--do_finetuning", type=bool, default=False, help="Flag: do fine-tuning of the two step process.")
    #group.add_argument("--nb_epochs_tuning", type=int, default=100, help="Number of epochs of fine-tuning.")
    #group.add_argument("--nb_epochs_active_tuning", type=int, default=200, help="Number of epochs of fine-tuning in active learning.")

    group.add_argument("-s", "--testing_samples", type=int, help="Total number of samples used in learning")
    group.add_argument("-l", "--length", type=int, help="Path lenth, with horizon")
    group.add_argument("-ho", "--horizon", type=int, help="Horizon length")
    group.add_argument("--no-target", action="store_true", help="Do not use the target monitor" )


def conformal_testing_argsparser():
    parser = argparse.ArgumentParser(description="Learn an IMC")
    build_suo_args_parser(parser)
    build_learning_parser(parser)

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (can be used multiple times)",
    )

    parser.add_argument(
        "-ri",
        "--run-id",
        type=int,
        default=0,
        help="Run ID to use for the experiment. Used to distinguish between different runs in the same model path.",
    )

    parser.add_argument('--se_path', 
                        type = str, 
                        help = 'Path to state estimator',
    )

    parser.add_argument('--error_path', 
                        type = str, 
                        help ='Path to error estimator',
    )

    parser.add_argument('--rej_path', 
                        type = str, 
                        help ='Path to rejection classifier',
    )

    parser.add_argument('--stats_path',
                        type = str, 
                        help = 'Path to dataset statistics',
    )

    parser.add_argument('--cp_classification_path',
                        type = str, 
                        help = 'Path to CP classifier'
    )

    return parser


if __name__ == "__main__":
    parser = conformal_testing_argsparser()
    args = parser.parse_args()
    conformal_testing_main(args, args.se_path, args.error_path, args.rej_path, args.stats_path, args.cp_classification_path)

#python -m premise.interval.conformal_prediction.conformal_prediction_testing --mc airportA-7-10-10 --testing_samples 500 --length 40 -ho 20 --no-target --se_path /workspaces/premise/premise/interval/conformal_prediction/test_results/conformal_state_estimator.pt --error_path /workspaces/premise/premise/interval/conformal_prediction/test_results/conformal_error_estimator.pt --rej_path /workspaces/premise/premise/interval/conformal_prediction/test_results/conformal_rej.pickle --cp_classification_path /workspaces/premise/premise/interval/conformal_prediction/test_results/cp_classification.pt --stats_path /workspaces/premise/premise/interval/conformal_prediction/test_results/conformal_stats.pickle
