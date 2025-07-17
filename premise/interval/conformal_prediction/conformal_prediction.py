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
import torch.nn.functional
from premise.interval.loading import build_suo, build_suo_args_parser
from premise.interval.utils import setup_logging, logger
import dill


def learn_conformal_prediction_model(
    args, model, suo, dataset, initial_amount, horizon, amount
):
    # horizon = args.horizon
    model_name = args.model_name

    se = Train_SeqSE(model_name, dataset, net_type=args.net_type)
    start_time = time.time()
    se.train(args.nb_epochs, args.batch_size, lr=args.lr)
    logger.info(f"SE TRAINING TIME: {time.time()-start_time}")

    nsc = Train_SeqNSC(
        model_name, dataset, net_type=args.net_type, nb_filters=args.nb_filters
    )
    start_time = time.time()
    nsc.train(args.nb_epochs, args.batch_size, args.lr)
    logger.info(f"NSC TRAINING TIME: {time.time()-start_time}")

    nsc_info = (nsc.idx, args.nb_epochs)
    se_info = (se.idx, args.nb_epochs)

    comb_ponsc = Train_StochSeqNSC(
        model_name,
        dataset,
        net_type=args.net_type,
        fine_tuning_flag=args.do_finetuning,
        seq_nsc_idx=nsc_info,
        seq_se_idx=se_info,
    )
    start_time = time.time()
    comb_ponsc.train(args.nb_epochs_tuning, args.batch_size, args.lr_tuning)
    logger.info(f"FINE TUNING TRAINING TIME: {time.time()-start_time}")

    comb_ponsc.generate_test_results()

    nsc_fnc = (
        lambda inp: comb_ponsc.seq_nsc(Variable(FloatTensor(inp)))
        .cpu()
        .detach()
        .numpy()
    )  # after fine-tuning
    se_fnc = (
        lambda inp: comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy()
    )  # after fine-tuning

    ponsc_fnc = (
        lambda meas: comb_ponsc.seq_nsc(comb_ponsc.seq_se(Variable(FloatTensor(meas))))
        .cpu()
        .detach()
        .numpy()
    )  # after fine-tuning

    # compute test validity and efficiency
    meas_test = np.transpose(dataset.Y_test_scaled, (0, 2, 1))
    meas_cal = np.transpose(dataset.Y_cal_scaled, (0, 2, 1))
    state_test = np.transpose(dataset.X_test_scaled, (0, 2, 1))
    state_cal = np.transpose(dataset.X_cal_scaled, (0, 2, 1))
    output_cal = dataset.L_cal
    output_test = dataset.L_test

    # MEMO: the calibration set MUST come from the same distribution of the train set
    cp_class = ICP_Classification(
        Xc=state_cal, Yc=output_cal, trained_model=nsc_fnc, mondrian_flag=False
    )

    cp_comb_class = ICP_Classification(
        Xc=meas_cal, Yc=output_cal, trained_model=ponsc_fnc, mondrian_flag=False
    )

    cp_regr = ICP_Regression(Xc=meas_cal, Yc=state_cal, trained_model=se_fnc)

    logger.info("----- Computing CP Regression validity and (box) efficiency...")
    se_box_coverage = cp_regr.get_box_coverage(args.epsilon, meas_test, state_test)
    se_box_efficiency = cp_regr.get_efficiency(box_flag=True)
    logger.info(
        f"Box-Coverage for significance = {1 - args.epsilon}: {se_box_coverage}; Box Efficiency = {se_box_efficiency}",
    )

    logger.info("----- Computing CP Regression validity and NON-BOX efficiency...")
    se_coverage = cp_regr.get_coverage(args.epsilon, meas_test, state_test)
    se_efficiency = cp_regr.get_efficiency(box_flag=False)
    logger.info(
        f"Regr (NON-BOX) Coverage for significance = {1 - args.epsilon}: {se_coverage}; Efficiency = {se_efficiency}",
    )

    logger.info("----- Computing test CP classification validity...")
    logger.info("Coverage on the test set states:")
    nsc_coverage = cp_class.compute_coverage(
        eps=args.epsilon, inputs=state_test, outputs=output_test
    )
    nsc_efficiency = cp_class.compute_efficiency()
    logger.info(
        f"Test empirical coverage: {nsc_coverage}; Efficiency = {nsc_efficiency}"
    )

    logger.info("Coverage on the test states estimated by the SE:")
    estim_state_test = se_fnc(meas_test)
    ponsc_coverage = cp_class.compute_coverage(
        eps=args.epsilon, inputs=estim_state_test, outputs=output_test
    )
    logger.info(
        f"Test empirical coverage on ESTIM STATES: {ponsc_coverage} (Expected = {1 - args.epsilon})"
    )

    logger.info("----- Computing test CP COMB classification validity...")
    logger.info("Coverage on the test set measurments:")
    ponsc_coverage = cp_comb_class.compute_coverage(
        eps=args.epsilon, inputs=meas_test, outputs=output_test
    )
    ponsc_efficiency = cp_comb_class.compute_efficiency()
    logger.info(
        f"Test empirical coverage: {ponsc_coverage}; Efficiency = {ponsc_efficiency}"
    )

    logger.info("----- Labeling correct/incorrect predictions...")
    cal_errors = utils.label_correct_incorrect_pred(
        np.argmax(cp_comb_class.cal_pred_lkh, axis=1), output_cal
    )
    test_pred_lkh = ponsc_fnc(meas_test)
    test_errors = utils.label_correct_incorrect_pred(
        np.argmax(test_pred_lkh, axis=1), output_test
    )

    if np.sum(cal_errors) == len(cal_errors):
        cal_errors[0] = -1
    if np.sum(cal_errors) == -1 * len(cal_errors):
        cal_errors[0] = 1
    logger.info("----- Computing calibration confidence and credibility...")
    cal_conf_cred = cp_comb_class.compute_cross_confidence_credibility()

    kernel_type = "rbf"
    logger.info("----- Training the query strategy on calibration data...")
    query_fnc = utils.train_svc_query_strategy(kernel_type, cal_conf_cred, cal_errors)

    test_conf_cred = cp_comb_class.compute_confidence_credibility(meas_test)
    test_pred_errors = utils.apply_svc_query_strategy(query_fnc, test_conf_cred)

    rej_rate = utils.compute_rejection_rate(test_pred_errors)
    logger.info(f"----- Rejection rate = {rej_rate}")

    nb_detected, nb_errors, detection_rate = utils.compute_error_detection_rate(
        test_pred_errors, test_errors
    )
    logger.info(
        f"----- Error detection rate = {detection_rate} ({nb_detected}/{nb_errors})"
    )

    fp_indexes, fn_indexes = utils.label_fp_fn(
        np.argmax(test_pred_lkh, axis=1), output_test
    )
    fp_detection_rate, fn_detection_rate, res = utils.compute_fp_fn_detection_rate(
        test_pred_errors, fp_indexes, fn_indexes
    )

    nb_detected_fp, nb_fp, nb_detected_fn, nb_fn = res

    logger.info(
        f"FP Detection rate: {fp_detection_rate}; FN Detection rate: {fn_detection_rate}"
    )

    if args.do_refinement:
        logger.info("----- REFINEMENT of the Rejection Rule...")
        ref_samples = round((amount - 50) * 5 / 41)
        logger.info(f"Requires: {ref_samples} samples")

        # unc_meas_ref, unc_states_ref, unc_outputs_ref = utils.Comb_PONSC_active_sample_query(pool_size = opt.pool_size_ref, model_class = model, conf_pred = cp_comb_class, trained_svc = query_fnc, se_fnc= se_fnc, dataset=dataset)
        # unc_meas_ref, unc_states_ref, unc_outputs_ref = utils.Comb_PONSC_active_sample_query(ref_samples, horizon, model_class = model, conf_pred = cp_comb_class, trained_svc = query_fnc, se_fnc = se_fnc, dataset=dataset)
        unc_meas_ref, unc_states_ref, unc_outputs_ref = (
            utils.Comb_PONSC_active_sample_query(
                suo,
                ref_samples,
                (initial_amount + horizon),
                horizon,
                model_class=model,
                conf_pred=cp_comb_class,
                trained_svc=query_fnc,
                se_fnc=se_fnc,
                dataset=dataset,
            )
        )

        if len(unc_meas_ref) < ref_samples:
            logger.info("Not enough samples for refinement step")
        else:
            unc_meas_ref = unc_meas_ref[:ref_samples]
            unc_states_ref = unc_states_ref[:ref_samples]
            unc_outputs_ref = unc_outputs_ref[:ref_samples]

        n_ref_points = len(unc_meas_ref)
        logger.info(f"Nb of points to add: {n_ref_points}")

        logger.info("Shapes")
        logger.info(f"Calibration data shape: {dataset.Y_cal_scaled.shape}")
        logger.info(f"Uncertain measurements shape: {unc_meas_ref.shape}")

        meas_cal_ref = np.vstack((dataset.Y_cal_scaled, unc_meas_ref))
        state_cal_ref = np.vstack((dataset.X_cal_scaled, unc_states_ref))
        output_cal_ref = np.hstack((dataset.L_cal, unc_outputs_ref))

        meas_cal_ref = np.transpose(meas_cal_ref, (0, 2, 1))
        state_cal_ref = np.transpose(state_cal_ref, (0, 2, 1))

        ref_cp_comb_class = ICP_Classification(
            Xc=meas_cal_ref,
            Yc=output_cal_ref,
            trained_model=ponsc_fnc,
            mondrian_flag=False,
        )

        ref_cal_conf_cred = ref_cp_comb_class.compute_cross_confidence_credibility()

        ref_cal_errors = utils.label_correct_incorrect_pred(
            np.argmax(ponsc_fnc(meas_cal_ref), axis=1), output_cal_ref
        )

        logger.info(
            "----- Training a REFINED query strategy on enlarged calibration data..."
        )
        ref_query_fnc = utils.train_svc_query_strategy(
            kernel_type, ref_cal_conf_cred, ref_cal_errors
        )
        results_dict = {"rej_rule": query_fnc, "ref_rej_rule": ref_query_fnc}

        query_fnc = ref_query_fnc
    else:
        results_dict = {"rej_rule": query_fnc}

    # filename = model_name+"/Conv_StochSeqNSC_results/ID_"+comb_ponsc.idx+"/rejection_rule.pickle"
    # with open(filename, 'wb') as handle:
    # 	pickle.dump(results_dict, handle)
    # handle.close()

    curr_cp_comb_class = cp_comb_class
    curr_query_fnc = query_fnc
    curr_dataset = dataset
    curr_se_fnc = se_fnc
    for k in range(args.nb_active_iterations):
        logger.info(f"--- xxx ACTIVE ITERATION NB. {k}")

        logger.info("----- Active selection of additional (uncertain) points...")
        start_active = time.time()
        # unc_meas, unc_states, unc_outputs = utils.Comb_PONSC_active_sample_query(pool_size = opt.pool_size, model_class = model, conf_pred = curr_cp_comb_class, trained_svc = curr_query_fnc, se_fnc= curr_se_fnc, dataset=curr_dataset)
        # unc_meas, unc_states, unc_outputs = utils.Comb_PONSC_active_sample_query(active_samples, horizon, model_class = model, conf_pred = curr_cp_comb_class, trained_svc = curr_query_fnc, se_fnc = curr_se_fnc, dataset=curr_dataset)
        active_samples = round((amount - 50) * 10 / 41)
        unc_meas, unc_states, unc_outputs = utils.Comb_PONSC_active_sample_query(
            suo,
            active_samples,
            (initial_amount + horizon),
            horizon,
            model_class=model,
            conf_pred=cp_comb_class,
            trained_svc=query_fnc,
            se_fnc=se_fnc,
            dataset=dataset,
        )

        if len(unc_meas) < round(active_samples):
            logger.info("Not enough samples for active learning step")
        else:
            unc_meas = unc_meas[:active_samples]
            unc_states = unc_states[:active_samples]
            unc_outputs = unc_outputs[:active_samples]

        logger.info(
            f"XXX time to active query points for pool: {time.time() - start_active}"
        )

        n_active_points = len(unc_outputs)
        logger.info(f"Nb of points to add: {n_active_points}")

        # n_retrain = int(np.round(opt.nb_active_points*opt.split_rate))
        n_retrain = int(np.round(n_active_points * args.split_rate))  # ANTONINA

        meas_retrain = np.vstack((curr_dataset.Y_train_scaled, unc_meas[:n_retrain]))
        state_retrain = np.vstack((curr_dataset.X_train_scaled, unc_states[:n_retrain]))
        output_retrain = np.hstack((curr_dataset.L_train, unc_outputs[:n_retrain]))

        meas_recal = np.vstack((curr_dataset.Y_cal_scaled, unc_meas[n_retrain:]))
        state_recal = np.vstack((curr_dataset.X_cal_scaled, unc_states[n_retrain:]))
        output_recal = np.hstack((curr_dataset.L_cal, unc_outputs[n_retrain:]))

        active_dataset = dataset
        active_dataset.n_training_points = len(output_retrain)
        active_dataset.n_cal_points = len(output_recal)
        active_dataset.Y_train_scaled = meas_retrain
        active_dataset.Y_cal_scaled = meas_recal
        active_dataset.X_train_scaled = state_retrain
        active_dataset.X_cal_scaled = state_recal
        active_dataset.L_train = output_retrain
        active_dataset.L_cal = output_recal

        # tuned_info = (comb_ponsc.idx, n_epochs_tuning)
        # RIFACCIO SOLO IL FINE TUNING
        logger.info("----- ACTIVE RETRAINING...")

        if False:  # Retrain everything from scratch
            active_se = Train_SeqSE(model_name, active_dataset, net_type=opt.net_type)
            active_se.train(opt.nb_epochs, opt.batch_size, lr=opt.lr)

            active_nsc = Train_SeqNSC(
                model_name,
                active_dataset,
                net_type=opt.net_type,
                nb_filters=opt.nb_filters,
            )
            active_nsc.train(opt.nb_epochs, opt.batch_size, opt.lr)

            active_nsc_info = (active_nsc.idx, opt.nb_epochs)
            active_se_info = (active_se.idx, opt.nb_epochs)
        else:  # redo only the finetuning
            active_nsc_info = (nsc.idx, args.nb_epochs)
            active_se_info = (se.idx, args.nb_epochs)

        active_comb_ponsc = Train_StochSeqNSC(
            model_name,
            active_dataset,
            net_type=args.net_type,
            fine_tuning_flag=args.do_finetuning,
            seq_nsc_idx=active_nsc_info,
            seq_se_idx=active_se_info,
        )
        active_comb_ponsc.train(
            args.nb_epochs_active_tuning, args.batch_size, args.lr_tuning
        )
        active_comb_ponsc.generate_test_results()

        # active_nsc_fnc = lambda inp: active_comb_ponsc.seq_nsc(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning
        # active_se_fnc = lambda inp: active_comb_ponsc.seq_se(Variable(FloatTensor(inp))).cpu().detach().numpy() # after fine-tuning

        active_ponsc_fnc = (
            lambda meas: active_comb_ponsc.seq_nsc(
                active_comb_ponsc.seq_se(Variable(FloatTensor(meas)))
            )
            .cpu()
            .detach()
            .numpy()
        )  # after fine-tuning

        # compute test validity and efficiency
        meas_recal = np.transpose(active_dataset.Y_cal_scaled, (0, 2, 1))
        state_recal = np.transpose(active_dataset.X_cal_scaled, (0, 2, 1))

        # MEMO: the calibration set MUST come from the same distribution of the train set
        # active_cp_class = ICP_Classification(Xc = state_recal, Yc = output_recal, trained_model = active_nsc_fnc, mondrian_flag = False)

        active_cp_comb_class = ICP_Classification(
            Xc=meas_recal,
            Yc=output_recal,
            trained_model=active_ponsc_fnc,
            mondrian_flag=False,
        )

        # active_cp_regr = ICP_Regression(Xc = meas_recal, Yc = state_recal, trained_model = active_se_fnc)

        # print("----- ACTIVE Computing CP Regression validity and (box) efficiency...")
        # active_se_coverage = active_cp_regr.get_box_coverage(args.epsilon, meas_test, state_test)
        # active_se_efficiency = active_cp_regr.get_efficiency(box_flag = True)
        # print("Box-Coverage for significance = ", 1-args.epsilon, ": ", active_se_coverage, "; Box Efficiency = ", active_se_efficiency)

        # print("----- ACTIVE Computing test CP classification validity...")
        # print("- Coverage on the test set states:")
        # active_nsc_coverage = active_cp_class.compute_coverage(eps=args.epsilon, inputs=state_test, outputs=output_test)
        # active_nsc_efficiency = active_cp_class.compute_efficiency()
        # print("Test empirical coverage: ", active_nsc_coverage, " Efficiency: ", active_nsc_efficiency)

        # print("- Coverage on the test states estimated by the SE:")
        # active_estim_state_test = active_se_fnc(meas_test)
        # active_ponsc_coverage = active_cp_class.compute_coverage(eps=args.epsilon, inputs=active_estim_state_test, outputs=output_test)
        # print("Test empirical coverage on ESTIM STATES: ", active_ponsc_coverage, " (Expected = ", 1-args.epsilon, ")")

        # print("- Coverage on the test set measurments (CP COMB):")
        # active_ponsc_coverage = active_cp_comb_class.compute_coverage(eps=args.epsilon, inputs=meas_test, outputs=output_test)
        # active_ponsc_efficiency = active_cp_comb_class.compute_efficiency()
        # print("Test empirical coverage: ", active_ponsc_coverage, " Efficiency: ", active_ponsc_efficiency)

        # print("----- ACTIVE Labeling correct/incorrect predictions...")
        active_cal_errors = utils.label_correct_incorrect_pred(
            np.argmax(active_cp_comb_class.cal_pred_lkh, axis=1), output_recal
        )
        active_test_pred_lkh = active_ponsc_fnc(meas_test)
        active_test_errors = utils.label_correct_incorrect_pred(
            np.argmax(active_test_pred_lkh, axis=1), output_test
        )

        # print("----- ACTIVE Computing calibration confidence and credibility...")
        active_cal_conf_cred = (
            active_cp_comb_class.compute_cross_confidence_credibility()
        )

        # print("----- ACTIVE Training the query strategy on calibration data...")
        kernel_type = "rbf"
        active_query_fnc = utils.train_svc_query_strategy(
            kernel_type, active_cal_conf_cred, active_cal_errors
        )

        active_test_conf_cred = active_cp_comb_class.compute_confidence_credibility(
            meas_test
        )
        active_test_pred_errors = utils.apply_svc_query_strategy(
            active_query_fnc, active_test_conf_cred
        )

        # active_rej_rate = utils.compute_rejection_rate(active_test_pred_errors)
        # print("----- ACTIVE Rejection rate = ", active_rej_rate)

        # active_nb_detected, active_nb_errors, active_detection_rate = utils.compute_error_detection_rate(active_test_pred_errors, active_test_errors)
        # print("----- ACTIVE Error detection rate = ", active_detection_rate, "({}/{})".format(active_nb_detected, active_nb_errors))

        # active_fp_indexes, active_fn_indexes = utils.label_fp_fn(np.argmax(active_test_pred_lkh, axis=1), output_test)
        # active_fp_detection_rate, active_fn_detection_rate, active_res = utils.compute_fp_fn_detection_rate(active_test_pred_errors, active_fp_indexes, active_fn_indexes)

        # print("ACTIVE FP Detection rate: ", active_fp_detection_rate, "FN Detection rate: ", active_fn_detection_rate)

    return (
        active_comb_ponsc,
        active_query_fnc,
        active_cp_comb_class,
        n_ref_points,
        n_active_points,
    )


def conformal_prediction_main(args: argparse.Namespace):
    setup_logging()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"

    logger.info(f"Starting conformal prediction training... ({args})")
    suo, initial_amount, horizon = build_suo(args)
    # horizon = args.horizon

    models_dict = {"IP": InvertedPendulum(), "MC": mc_model(horizon)}
    model = models_dict[args.model_name]
    model_name = args.model_name

    model = mc_model(horizon)

    trainset = []
    for x in range(round((args.amount - 50) * 20 / 41)):
        path = suo.generate_random_traces([], (initial_amount + horizon))[0]
        # trainset.append(tuple(path))
        trainset.append(path)

    calibrset = []
    for x in range(round((args.amount - 50) * 6 / 41)):
        path = tuple(suo.generate_random_traces([], (initial_amount + horizon))[0])
        calibrset.append(path)

    testset = []
    for x in range(100):
        path = tuple(suo.generate_random_traces([], (initial_amount + horizon))[0])
        testset.append(path)

    validset = []
    for x in range(50):
        path = tuple(suo.generate_random_traces([], (initial_amount + horizon))[0])
        validset.append(path)

    logger.info(f"Trainset size: {len(trainset)}")
    logger.info(f"Calibrset size: {len(calibrset)}")
    logger.info(f"Testset size: {len(testset)}")
    logger.info(f"Validset size: {len(validset)}")

    for i in range(8, 9):
        logger.info(
            "___________________________________________________________________"
        )
        logger.info(f"Learning interation : {i}")
        logger.info(
            "___________________________________________________________________"
        )
        end_index = int((i / 8) * len(trainset))
        current_trainset = trainset[:end_index]

        end_index = int((i / 8) * len(calibrset))
        current_calibrset = calibrset[:end_index]

        amount = int((i / 8) * args.amount)

        logger.info(f"AMOUNT: {amount}")
        logger.info(f"Current trainset size: {len(current_trainset)}")
        logger.info(f"Current calibrset size: {len(current_calibrset)}")

        logger.info(
            f"Without active and refinement: {len(current_trainset) + len(current_calibrset) + 50}"
        )

        sets = {
            "trainset": current_trainset,
            "calibrset": current_calibrset,
            "testset": testset,
            "validset": validset,
        }

        dicts = {}

        for name, s in sets.items():
            paths = model.gen_trajectories(s, horizon)
            noisy_measurments = model.get_noisy_measurments(s, horizon)
            labels = model.gen_labels(s, horizon)
            dicts[f"{name}_fn"] = {
                "x": paths,
                "y": noisy_measurments,
                "cat_labels": labels,
            }

        dataset = SeqDataset(
            dicts["trainset_fn"], dicts["testset_fn"], dicts["validset_fn"]
        )
        dataset.load_data()
        dataset.add_calibration_path(dicts["calibrset_fn"])
        dataset.load_calibration_data()

        (
            active_comb_ponsc,
            query_fnc,
            active_cp_comb_class,
            n_ref_points,
            n_active_points,
        ) = learn_conformal_prediction_model(
            args, model, suo, dataset, initial_amount, horizon, amount
        )

        results_dict = {"rej_rule": query_fnc}

        # SAVING REJECTION CLASSIFIER

        rej_filename = os.path.join(
            args.dump_stats,
            f"{args.mc}_comp_conformal_pred_rejection_classifier_{args.run_id}_{amount}.pickle",
        )
        with open(rej_filename, "wb") as handle:
            pickle.dump(results_dict, handle)

        nn_filename = os.path.join(
            args.dump_model,
            f"{args.mc}_comp_conformal_pred_state_estimator_{args.run_id}_{amount}.pt",
        )
        torch.save(active_comb_ponsc.seq_se, nn_filename)

        nn_filename = os.path.join(
            args.dump_model,
            f"{args.mc}_comp_conformal_pred_label_estimator_{args.run_id}_{amount}.pt",
        )
        torch.save(active_comb_ponsc.seq_nsc, nn_filename)

        nn_filename = os.path.join(
            args.dump_model,
            f"{args.mc}_comp_conformal_pred_cp_classification_{args.run_id}_{amount}.pt",
        )

        torch.save(active_cp_comb_class, nn_filename, pickle_module=dill)

        sample_count = (
            n_ref_points
            + n_active_points
            + len(trainset)
            + len(calibrset)
            + len(validset)
        )

        dataset_stats = {
            "dataset.MIN[1]": dataset.MIN[1],
            "dataset.MAX[1]": dataset.MAX[1],
            "learned_on": sample_count,
            "length_with_horizon": (initial_amount + horizon),
            "horizon": horizon,
        }

        stats_filename = os.path.join(
            args.dump_stats,
            f"{args.mc}_comp_conformal_pred_conformal_stats_{args.run_id}_{amount}.pickle",
        )

        with open(stats_filename, "wb") as handle:
            pickle.dump(dataset_stats, handle)


def build_learning_parser(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Learning Parameters")

    group.add_argument(
        "--model-name",
        type=str,
        default="MC",
        help="Name of the model (first letters code).",
    )
    group.add_argument(
        "--do-refinement",
        type=bool,
        default=True,
        help="Flag: refine of the rejection rule.",
    )
    group.add_argument(
        "--nb-active-iterations",
        type=int,
        default=1,
        help="Number of active learning iterations.",
    )
    group.add_argument("--nb-epochs", type=int, default=200, help="Number of epochs.")
    group.add_argument(
        "--nb-epochs-active",
        type=int,
        default=400,
        help="Number of epochs in active learning.",
    )
    group.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    group.add_argument("--lr", type=float, default=0.00001, help="Adam: learning rate")
    group.add_argument(
        "--lr-tuning",
        type=float,
        default=0.000001,
        help="Adam: learning rate for fine tuning",
    )
    group.add_argument(
        "--net-type", type=str, default="Conv", help="Type of the net: Conv or FF."
    )
    group.add_argument(
        "--nb-filters", type=int, default=128, help="Number of filters per conv layer."
    )
    group.add_argument(
        "--epsilon", type=float, default=0.05, help="CP significance level."
    )
    group.add_argument(
        "--split-rate", type=float, default=10 / 14, help="adam: learning rate"
    )
    group.add_argument(
        "--pool-size-ref",
        type=int,
        default=25000,
        help="Size of the pool for the refinement step.",
    )
    group.add_argument(
        "--pool-size",
        type=int,
        default=50000,
        help="Size of the pool for one active learning step.",
    )
    group.add_argument(
        "--reinit-weights",
        type=bool,
        default=False,
        help="Flag: do reinitialize the weights in active learning steps.",
    )
    group.add_argument(
        "--do-finetuning",
        type=bool,
        default=False,
        help="Flag: do fine-tuning of the two step process.",
    )
    group.add_argument(
        "--nb-epochs-tuning",
        type=int,
        default=100,
        help="Number of epochs of fine-tuning.",
    )
    group.add_argument(
        "--nb-epochs-active-tuning",
        type=int,
        default=200,
        help="Number of epochs of fine-tuning in active learning.",
    )

    group.add_argument(
        "-a", "--amount", type=int, help="Total number of samples used in learning"
    )
    # group.add_argument("-l", "--sample_length", type=int, help="Path lenth, with horizon")
    # group.add_argument("-ho", "--horizon", type=int, help="Horizon length")
    group.add_argument(
        "--no-target", action="store_true", help="Do not use the target monitor"
    )

    group.add_argument(
        "-m",
        "--dump-model",
        type=str,
        help="Path to dump the model to",
    )


def conformal_prediction_argsparser():
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
        "--dump-stats",
        type=str,
        help="Path to dump the model to",
    )

    parser.add_argument(
        "-ri",
        "--run-id",
        type=int,
        default=0,
        help="Run ID to use for the experiment. Used to distinguish between different runs in the same model path.",
    )

    return parser


if __name__ == "__main__":
    parser = conformal_prediction_argsparser()
    args = parser.parse_args()
    conformal_prediction_main(args)


# python -m premise.interval.conformal_prediction.conformal_prediction --mc airportA-7-10-10 -a 55 --no-target --dump-stats /workspaces/premise/premise/analysis/test_sets --dump-model /workspaces/premise/premise/analysis/test_sets
