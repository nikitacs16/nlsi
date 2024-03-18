import string
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import pearsonr


class Evaluation:
    """
    Module to work independently of the task
    """

    def __init__(
        self,
        ground_truth: List[List[str]],
        predictions: List[List[str]],
        datum_ids: Optional[List[str]] = None,
        profiles: Optional[List[List[str]]] = None,
        preprocess: bool = False,
        remove_hallucinations: bool = False,
        example_labels: Optional[List[str]] = None,
    ):
        """
        predictions: List of list of strings, where each list of strings represents the predicted set of strings for a single instance
        ground_truth: List of list of strings, where each list of strings represents the ground truth set of strings for a single instance
        preprocess: Whether to preprocess the strings before evaluation. Default: True
        """
        self.predictions = predictions
        self.ground_truth = ground_truth
        self.preprocess = preprocess
        self.profiles = profiles
        self.datum_ids = datum_ids
        self.example_labels = example_labels
        if self.preprocess:
            print("self.predictions[0] = ", self.predictions[0])
            self.predictions = [
                [Evaluation.process_string(s) for s in ss] for ss in self.predictions
            ]
            self.ground_truth = [
                [Evaluation.process_string(s) for s in ss] for ss in self.ground_truth
            ]
            if self.profiles is not None:
                self.profiles = [
                    [Evaluation.process_string(s) for s in ss] for ss in self.profiles
                ]
        if remove_hallucinations:
            assert self.profiles is not None
            self.predictions = [
                Evaluation.get_predictions_without_hallucinations(ss, pp)
                for ss, pp in zip(self.predictions, self.profiles)
            ]
        # add some simple asserts
        assert len(self.predictions) == len(self.ground_truth)
        if self.profiles is not None:
            assert len(self.predictions) == len(self.profiles)
        if datum_ids is not None:
            assert len(self.predictions) == len(datum_ids)

    def exact_match(self) -> float:
        score = [
            set(self.ground_truth[i]) == set(self.predictions[i])
            for i in range(len(self.ground_truth))
        ]
        return sum(score) * 1.0 / len(score)

    def sample_f1(self) -> float:
        predictions, ground_truth = self.predictions, self.ground_truth
        f1_score, _ = Evaluation.get_f1_score(predictions, ground_truth)
        return f1_score

    def get_metrics(
        self, prefix: str = "", is_selection_exp: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
        """
        Compute all the metrics
        Returns a tuple of two dictionaries:
        1. The first dictionary contains the aggregated scores for each metric
        2. The second dictionary contains the scores for each instance for each metric
        """
        predictions, ground_truth = self.predictions, self.ground_truth
        em_score, em_scores = Evaluation.get_set_exact_match(predictions, ground_truth)
        f1_score, f1_scores = Evaluation.get_f1_score(predictions, ground_truth)
        prec_score, prec_scores = Evaluation.get_prec_score(predictions, ground_truth)
        rec_score, rec_scores = Evaluation.get_rec_score(predictions, ground_truth)
        aggr = {
            f"{prefix}em_score": em_score,
            f"{prefix}f1_score": f1_score,
            f"{prefix}prec_score": prec_score,
            f"{prefix}rec_score": rec_score,
        }
        instance_wise = {
            f"{prefix}em_scores": em_scores,
            f"{prefix}f1_scores": f1_scores,
            f"{prefix}prec_scores": prec_scores,
            f"{prefix}rec_scores": rec_scores,
        }
        if is_selection_exp:
            assert self.profiles is not None
            hallucination = []
            for i in range(len(predictions)):
                hallucination_label: List[bool] = self.get_hallucination(
                    predictions[i], self.profiles[i]
                )
                # compute the fractio  of hallucination_label that are true
                if len(hallucination_label) == 0:
                    hallucination.append(0.0)
                else:
                    hallucination.append(
                        sum(hallucination_label) / len(hallucination_label)
                    )
            aggr[f"{prefix}hallucination"] = sum(hallucination) / len(hallucination)
            instance_wise[f"{prefix}hallucination"] = hallucination
        # compute avg number of strings in prediction and ground_truth
        avg_num_pred = sum([len(p) for p in predictions]) / len(predictions)
        avg_num_gt = sum([len(g) for g in ground_truth]) / len(ground_truth)
        aggr[f"{prefix}avg_num_pred"] = avg_num_pred
        aggr[f"{prefix}avg_num_gt"] = avg_num_gt
        # add count of total predictions
        aggr[f"{prefix}total_cnt"] = len(predictions)
        # add list of dialog ids to instance_wise
        if self.datum_ids is not None:
            instance_wise[f"{prefix}datum_ids"] = self.datum_ids
        # instance_wise: add length of profile, prediction and ground_truth
        if is_selection_exp and self.profiles is not None:
            instance_wise[f"{prefix}profile_len"] = [len(p) for p in self.profiles]
        instance_wise[f"{prefix}pred_len"] = [len(p) for p in predictions]
        instance_wise[f"{prefix}gt_len"] = [len(g) for g in ground_truth]
        return aggr, instance_wise

    @classmethod
    def process_string(cls, s: str):
        """
        Process the string to remove punctuations and convert to lower case
        """
        return s.lower().translate(str.maketrans("", "", string.punctuation))

    @classmethod
    def get_set_exact_match(
        cls, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Tuple[float, List[float]]:
        """
        Compute exact match score for each instance: if the set of strings in the two lists represent the same set of strings
        """
        em_scores = []
        for i in range(len(ground_truth)):
            if set(predictions[i]) == set(ground_truth[i]):
                em_scores.append(1.0)
            else:
                em_scores.append(0.0)
        return sum(em_scores) / len(em_scores), em_scores

    @classmethod
    def get_f1_score(
        cls, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Tuple[float, List[float]]:
        """
        Compute f1 score for each instance, and then report macro f1 score
        Also keep track of the f1 score for each instance
        """
        f1_scores = []
        for i in range(len(ground_truth)):
            f1_scores.append(cls._get_f1_score(predictions[i], ground_truth[i]))
        return sum(f1_scores) / len(f1_scores), f1_scores

    @classmethod
    def _get_f1_score(cls, prediction: List[str], ground_truth: List[str]) -> float:
        """
        Compute f1 score for a single instance
        """
        tp = len(set(prediction).intersection(set(ground_truth)))
        if len(ground_truth) == 0 and len(prediction) == 0:
            return 1.0
        if tp == 0:
            return 0.0
        precision = tp / len(prediction)
        recall = tp / len(ground_truth)
        return 2 * precision * recall / (precision + recall)

    @classmethod
    def get_prec_score(
        cls, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Tuple[float, List[float]]:
        prec_scores = []
        for i in range(len(ground_truth)):
            prec_scores.append(cls._get_prec_score(predictions[i], ground_truth[i]))
        return sum(prec_scores) / len(prec_scores), prec_scores

    @classmethod
    def _get_prec_score(cls, prediction: List[str], ground_truth: List[str]) -> float:
        """
        Compute prec score for a single instance
        """
        tp = len(set(prediction).intersection(set(ground_truth)))
        if tp == 0:
            return 0.0
        return tp / len(prediction)

    @classmethod
    def get_rec_score(
        cls, predictions: List[List[str]], ground_truth: List[List[str]]
    ) -> Tuple[float, List[float]]:
        rec_scores = []
        for i in range(len(ground_truth)):
            rec_scores.append(cls._get_rec_score(predictions[i], ground_truth[i]))
        return sum(rec_scores) / len(rec_scores), rec_scores

    @classmethod
    def _get_rec_score(cls, prediction: List[str], ground_truth: List[str]) -> float:
        """
        Compute recall score for a single instance
        """
        tp = len(set(prediction).intersection(set(ground_truth)))
        if tp == 0:
            return 0.0
        return tp / len(ground_truth)

    @classmethod
    def label_recall_on_groundtruth(
        cls, prediction: List[str], ground_truth: List[str]
    ) -> List[bool]:
        """
        a method to 1) label each prediction string in prediction[List[str]] as present or not using ground_truth[List[str]];
        """
        recall: List[bool] = []
        for s in ground_truth:
            if s in prediction:
                recall.append(True)
            else:
                recall.append(False)
        return recall

    @classmethod
    def label_precision_on_prediction(
        cls, prediction: List[str], ground_truth: List[str]
    ) -> List[bool]:
        """
        a method to label each prediction string in prediction[List[str]] as present or not using ground_truth[List[str]];
        """
        precision: List[bool] = []
        for s in prediction:
            if s in ground_truth:
                precision.append(True)
            else:
                precision.append(False)
        return precision

    @classmethod
    def get_hallucination(cls, prediction: List[str], profile: List[str]):
        """
        a method to label each prediction string in prediction[List[str]] as halllucination or not using profile[List[str]];
        """
        hallucination: List[bool] = []
        for s in prediction:
            if s in profile:
                hallucination.append(False)
            else:
                hallucination.append(True)
        return hallucination

    @classmethod
    def get_predictions_without_hallucinations(
        cls, prediction: List[str], profile: List[str]
    ):
        """
        a method to remove hallucination from prediction[List[str]] using profile[List[str]];
        """
        prediction = [s for s in prediction if s in profile]
        return prediction

    @staticmethod
    def get_correlation_score(
        scores_a: List[Union[int, float]],
        scores_b: List[Union[int, float]],
        a_name: str = "a",
        b_name: str = "b",
        plot_vals: bool = False,
        plot_save_path: Optional[str] = None,
    ) -> float:
        # compute correlation scores
        # compute correrlation scores after normalizing values to 0,1 range
        # plot a 2d graph
        # return the correlation score and the plot

        # compute correlation scores
        corr, _ = pearsonr(scores_a, scores_b)

        if plot_vals:
            # clear any previous plots
            plt.clf()

            # plot a 2d graph; use colors to show density since points can be overlapping; use a light bakcground
            # plt.hist2d(scores_a, scores_b, bins=(50, 50), cmap=plt.cm.jet)
            h = plt.hist2d(scores_a, scores_b, bins=25, norm=LogNorm())
            plt.colorbar(h[3])

            # use correlation scores as title
            plt.title(f"Correlation: {corr:.3f}")

            # use a_name and b_name as labels
            plt.xlabel(a_name)
            plt.ylabel(b_name)
            if plot_save_path:
                plt.savefig(plot_save_path)
            else:
                plt.show()

        return corr

    def get_per_example_type_exact_match(self):
        """
        Compute evaluation metrics for each example type
        """
        # get the set of example types
        example_types = set(self.example_labels)
        # get the indices of each example type
        example_type_indices = {
            example_type: [
                i
                for i in range(len(self.example_labels))
                if self.example_labels[i] == example_type
            ]
            for example_type in example_types
        }
        example_type_exact_match = {}
        for example_type, example_type_indices in example_type_indices.items():
            predictions = [
                [j.lower() for j in self.predictions[i]] for i in example_type_indices
            ]
            ground_truth = [
                [j.lower() for j in self.ground_truth[i]] for i in example_type_indices
            ]
            score, score_list = self.get_set_exact_match(predictions, ground_truth)
            example_type_exact_match[example_type] = score

        return example_type_exact_match
