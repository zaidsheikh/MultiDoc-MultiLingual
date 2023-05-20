
import os
import json
import time
import bert_score
from t5_score import T5Scorer
import numpy as np


def run_bertscore(mt: list, ref: list):
    """ Runs BERTScores and returns precision, recall and F1 BERTScores ."""
    _, _, f1 = bert_score.score(
        cands=mt,
        refs=ref,
        batch_size=32,
        verbose=True,
        model_type="bert-base-multilingual-cased",
        device='cuda:0',
    )

    return f1.numpy()


def run_t5score(scorer, mt: list, ref: list):
    hypo_ref = np.array(scorer.score(mt, ref,
                                     batch_size=8))
    ref_hypo = np.array(scorer.score(ref, mt,
                                     batch_size=8))
    return ref_hypo, hypo_ref


def main():
    t5_ckpt = "./model/T5Score/"
    # BASELINE = ["single_oracle", "multi_oracle", "lead_oracle", "lead_random"]
    BASELINE = ["TR_oracle", "TR_lead"]
    LANGUAGE = ["cantonese"]
    for baseline in BASELINE:
        print(
            "*****************************{}****************************".format(baseline))
        for language in LANGUAGE:
            print("---------------------{}-------------------".format(language))
            # TODO: do not hardcode these
            ground_truth_file = "./Multi-Doc-Sum/Mtl_data_aug_filtered/split/filtered/{}_test.jsonl".format(
                language)
            if baseline == "single_mt5":
                if language == "EN":
                    generation_file = "./baseline_results/oracle/single_languae/lr_5e-4_ada_all_epoch_20_bs_8_acc_2/{}/test/test_generations.txt".format(
                        language)
                else:
                    generation_file = "./baseline_results/oracle/single_languae/lr_5e-4_ada_all_epoch_20_bs_8_acc_4/{}/test/test_generations.txt".format(
                        language)
            elif baseline == "multi_mt5":
                generation_file = "./baseline_results/oracle/multi_languae/lr_5e-5_linear_schedual_ada_all_epoch__bs_8_acc_16_max_steps_20000/checkpoint-best/test/{}/test_generations.txt".format(
                    language)
            else:
                generation_file = "./baseline_results/clean_dataset/{}_{}.jsonl".format(
                    language, baseline)
            if "mt5" in baseline:
                generations = []
                for line in open(generation_file, 'r'):
                    generations.append(line)
            else:
                generations = []
                for line in open(generation_file, 'r'):
                    data = json.loads(line)
                    generations.append(data)

            ground_truth = []
            for line in open(ground_truth_file, 'r'):
                data = json.loads(line)
                ground_truth.append(data["summary"])

            start = time.time()
            print(f'Begin calculating BERTScore.')
            scores = run_bertscore(generations, ground_truth)
            print(
                f'Finished calculating BERTScore, time passed {time.time() - start}s.')
            print("BERTScore:")
            print(scores.mean())

            # t5_scorer = T5Scorer(device='cuda:0', checkpoint=t5_ckpt)
            t5_scorer = T5Scorer(device='cpu', checkpoint=t5_ckpt)
            start = time.time()
            print(f'Begin calculating T5Score.')
            scores_precision, scores_recall = run_t5score(
                t5_scorer, generations, ground_truth)
            scores = (scores_precision + scores_recall)/2
            print(
                f'Finished calculating T5Score, time passed {time.time() - start}s.')
            print("T5Score:")
            print(scores.mean())


if __name__ == '__main__':
    main()
