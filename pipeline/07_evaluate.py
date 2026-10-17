import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy import stats
from typing import TypedDict
from uk_dkcot.config import (
    ACCURACY_COLUMN,
    CLASSIFICATION_CONFUSION_MATRICES_PNG_FILE_NAME,
    CLASSIFICATION_METRICS_PNG_FILE_NAME,
    CLASSIFICATION_RESULTS_CSV_FILE_NAME,
    DKCOT_KNOWLEDGE_LEVELS,
    DKCOT_MODEL_NAME,
    DKCOT_PREDICTIONS_CSV_FILE_NAME,
    FINBERT_DISPLAY_NAME,
    FINBERT_MODEL_NAME,
    FINBERT_PREDICTIONS_CSV_FILE_NAME,
    FIGURE_DEEP_BLUE,
    FIGURE_NAVY,
    FIGURE_PRIMARY_TEAL,
    FIGURE_SECONDARY_TEAL,
    GOLD_LABELS_CSV_FILE_NAME,
    HEADLINE_ID_COLUMN,
    KNOWLEDGE_LEVEL_COLUMN,
    LABEL_GOLD_COLUMN,
    LABEL_PRED_COLUMN,
    MACRO_F1_COLUMN,
    MODEL_COLUMN,
    MODEL_ID_COLUMN,
    NEGATIVE_LABEL,
    NEUTRAL_LABEL,
    NO_KNOWLEDGE_LEVEL,
    PAIRED_SIGNIFICANCE_TESTS_CSV_FILE_NAME,
    POSITIVE_LABEL,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)


SENTIMENT_LABELS = [NEGATIVE_LABEL, NEUTRAL_LABEL, POSITIVE_LABEL]
BOOTSTRAP_ITERATIONS = 2_000
RANDOM_SEED = 2025

MODEL_VARIANTS = [
    (FINBERT_DISPLAY_NAME, FINBERT_MODEL_NAME, NO_KNOWLEDGE_LEVEL),
    *[
        (f"DK-CoT ({knowledge_level})", DKCOT_MODEL_NAME, knowledge_level)
        for knowledge_level in DKCOT_KNOWLEDGE_LEVELS
    ],
]

# Create class to use as type for the McNemar results
class McNemarTestResult(TypedDict):
    model_a: str
    model_b: str
    model_a_correct_model_b_wrong: int
    model_b_correct_model_a_wrong: int
    p_value: float
    significant_at_0_05: bool

class ClassificationMetrics(TypedDict):
    accuracy: float # Must match the variable ACCURACY_COLUMN from the config.py
    macro_f1: float # Must match the variable MACRO_F1_COLUMN from the config.py

class BootstrapConfidenceIntervals(TypedDict):
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    macro_f1_ci_lower: float
    macro_f1_ci_upper: float


def calculate_accuracy_and_f1_score(gold_labels, predictions) -> ClassificationMetrics:
    """Return accuracy and Macro-F1 for one prediction set."""

    return ClassificationMetrics(
        accuracy=accuracy_score(gold_labels, predictions),
        macro_f1=f1_score(
            gold_labels,
            predictions,
            labels=SENTIMENT_LABELS,
            average="macro",
            zero_division=0,
        ),
    )


def calculate_bootstrap_confidence_intervals(gold_labels, model_predictions, seed) -> BootstrapConfidenceIntervals:
    """Estimate 95% confidence intervals by resampling headlines"""

    # Convert labels into NumPy arrays
    gold_labels_np_array = np.asarray(gold_labels)
    model_predictions_np_array = np.asarray(model_predictions)
    # Create a reproducible random-number generator using the supplied seed
    random_generator = np.random.default_rng(seed)

    # Instantiate lists to store the metrics calculated from each bootstrap sample
    bootstrap_accuracy_values = []
    bootstrap_macro_f1_values = []

    # Repeat the resampling process 2,000 times
    for _ in range(BOOTSTRAP_ITERATIONS):
        # Randomly select 300 row positions from indices 0 to 299. Selection is with replacement, so a headline may appear more than once while another may not appear in that particular bootstrap sample
        bootstrap_sample_indices = random_generator.integers(
            0,
            len(gold_labels_np_array),
            len(gold_labels_np_array),
        )
        # Apply the same indices to both arrays so every prediction remains paired with its correct manually assigned gold label
        bootstrap_metrics = calculate_accuracy_and_f1_score(
            gold_labels_np_array[bootstrap_sample_indices],
            model_predictions_np_array[bootstrap_sample_indices],
        )
        # Store the metrics calculated from this bootstrap sample
        bootstrap_accuracy_values.append(bootstrap_metrics[ACCURACY_COLUMN])
        bootstrap_macro_f1_values.append(bootstrap_metrics[MACRO_F1_COLUMN])

    # Use the 2.5th and 97.5th percentiles as the lower and upper limits of a 95% percentile bootstrap confidence interval. This leaves 2.5% of the bootstrap results outside each end of the interval
    # NumPy quantile: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    # Bootstrap method: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    # This is the percentile bootstrap method
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
    return BootstrapConfidenceIntervals(
        accuracy_ci_lower=np.quantile(bootstrap_accuracy_values, 0.025),
        accuracy_ci_upper=np.quantile(bootstrap_accuracy_values, 0.975),
        macro_f1_ci_lower=np.quantile(bootstrap_macro_f1_values, 0.025),
        macro_f1_ci_upper=np.quantile(bootstrap_macro_f1_values, 0.975),
    )


def get_mcnemar_test_values(first_correct, second_correct) -> tuple[int, int, float]:
    """
    Return the exact two-sided McNemar p-value for paired predictions.

    McNemar's test is used to compare the two classifiers evaluated on the same headlines when, one model disagrees about which headline classification is correct. It asks the questions of does one model win (classify correctly based on the manually labelled data) noticeably more often?

    Accuracy scores show which model classified more headlines correctly, but they do not indicate whether the difference could reasonably result from variation in the evaluation sample. Because every model classified the same 300 headlines, their outcomes were paired. The exact McNemar test was therefore used to determine whether one model was correct significantly more often than another on the headlines where their correctness differed. This reduced the risk of treating a small numerical accuracy difference as evidence of a genuine performance improvement.

    Notes for me:
    - https://sci2s.ugr.es/keel/pdf/algorithm/articulo/dietterich1998.pdf (original doi: https://doi.org/10.1162/089976698300017197)

    - https://www.cambridge.org/core/journals/psychometrika/article/abs/note-on-the-sampling-error-of-the-difference-between-correlated-proportions-or-percentages/698C2461BE63F5848763502D54E534FD
        - https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/mcnemar.htm
    - https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html
    """

    # Follow the McNemar logic defined here (from statsmodels):
    # https://www.statsmodels.org/stable/_modules/statsmodels/stats/contingency_tables.html

    # Count headlines classified correctly only by the first model
    n1 = int(np.sum(first_correct & ~second_correct))
    # Count headlines classified correctly only by the second model
    n2 = int(np.sum(~first_correct & second_correct))

    # McNemar's test uses only cases where the two models disagree
    int_sm = n1 + n2

    # If the models never disagree, there is no evidence of a difference
    if int_sm == 0:
        return n1, n2, 1.0

    # Under the null hypothesis, either model is equally likely to be correct for each disagreement therefore, use the smaller count for the lower binomial value
    statistic = min(n1, n2)

    # Account for a result favouring either model
    # Under the null hypothesis, either model has a 50% chance of being correct on each disagreement. The binomial (`stats.biom`) `.cdf` calculates the probability of the smaller count. Then, multiply by 2 to test either model being better. Cap the result at 1 because a p-value cannot exceed 1.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
    p_value = stats.binom.cdf(statistic, int_sm, 0.5) * 2
    p_value = np.minimum(p_value, 1)

    # Return the following:
    # - n1: The number of headlines classified correctly only by the first model
    # - n2: The number of headlines classified correctly only by the second model
    # - p_value: The `p_value` which is evidence of whether the difference between those two counts is statistically significant
    return n1, n2, p_value


def load_and_get_gold_labels_and_model_predictions_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load gold labels and all four model prediction variants."""

    gold_labels_df = pd.read_csv(PROCESSED_DATA_DIR / GOLD_LABELS_CSV_FILE_NAME)
    finbert_predictions_df = pd.read_csv(PROCESSED_DATA_DIR / FINBERT_PREDICTIONS_CSV_FILE_NAME)
    dkcot_predictions_df = pd.read_csv(PROCESSED_DATA_DIR / DKCOT_PREDICTIONS_CSV_FILE_NAME)

    predictions_df = pd.concat([finbert_predictions_df, dkcot_predictions_df], ignore_index=True)

    return gold_labels_df, predictions_df


def evaluate_models(gold_labels_df, predictions_df):
    """Calculate classification metrics and retain paired predictions."""

    # Use to store one summary row per model
    result_rows = []
    # Use to store each prediction of the model's along with it's gold label. This will be returned to then be used for the McNemar comparisons
    gold_and_predicted_labels_by_model = {}

    # For each model variant, get the model name, id and knowledge level
    for model_index, (name, model_id, knowledge_level) in enumerate(MODEL_VARIANTS):
        # Filtering
        model_predictions = predictions_df.loc[
            (predictions_df[MODEL_ID_COLUMN] == model_id)
            & (predictions_df[KNOWLEDGE_LEVEL_COLUMN] == knowledge_level)
        ].copy()

        # Join predictions to the 300 manually verified labels using the headline_id
        model_variant_predictions_df = gold_labels_df.merge(
            model_predictions,
            on=HEADLINE_ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

        accuracy_and_f1_score_metrics = calculate_accuracy_and_f1_score(
            model_variant_predictions_df[LABEL_GOLD_COLUMN],
            model_variant_predictions_df[LABEL_PRED_COLUMN],
        )
        bootStrap_confidence_intervals = calculate_bootstrap_confidence_intervals(
            model_variant_predictions_df[LABEL_GOLD_COLUMN],
            model_variant_predictions_df[LABEL_PRED_COLUMN],
            RANDOM_SEED + model_index,
        )

        result_rows.append(
            {
                MODEL_COLUMN: name,
                MODEL_ID_COLUMN: model_id,
                KNOWLEDGE_LEVEL_COLUMN: knowledge_level,
                "n_headlines": len(model_variant_predictions_df),
                **accuracy_and_f1_score_metrics,
                **bootStrap_confidence_intervals,
            }
        )
        gold_and_predicted_labels_by_model[name] = model_variant_predictions_df[
            [HEADLINE_ID_COLUMN, LABEL_GOLD_COLUMN, LABEL_PRED_COLUMN]
        ].copy()

    return pd.DataFrame(result_rows), gold_and_predicted_labels_by_model


def run_mcnemar_tests_per_model_variant(gold_and_predicted_labels_by_model) -> pd.DataFrame:
    """Run McNemar tests for every pair of model variants."""

    # Use to store the results
    mcnemar_test_result_list: list[McNemarTestResult] = []

    # Use `itertools.combinations` to create every unique pair without repeating or reversing pairs. The param `gold_and_predicted_labels_by_model` is a dictionary and below, the iteration is per its models name keys. The four models will produce six comparisons. The number 2 is to have two models per loop, this is because the `get_mcnemar_test_values` expects two params and is what is needed to calculate the McNemar's test.
    # https://docs.python.org/3/library/itertools.html
    for first_model_name, second_model_name in itertools.combinations(
        gold_and_predicted_labels_by_model,
        2,
    ):
        # The below results in the following (a total of six comparisons):
        # FinBERT with DK-CoT (none)
        # FinBERT with DK-CoT (sector)
        # FinBERT with DK-CoT (firm)
        # DK-CoT (none) with DK-CoT (sector)
        # DK-CoT (none) with DK-CoT (firm)
        # DK-CoT (sector) with DK-CoT (firm)
        print("---------------------------------Models being compared together:---------------------------------")
        print(f"Comparing {first_model_name} with {second_model_name}")

        # Obtain the gold and predicted labels for each model
        first_model_metrics_df = gold_and_predicted_labels_by_model[first_model_name]
        second_df_model_metrics = gold_and_predicted_labels_by_model[second_model_name]
        # Merge into a pandas df using the headline id column
        paired_model_gold_and_predicted_labels_df = first_model_metrics_df.merge(
            second_df_model_metrics[[HEADLINE_ID_COLUMN, LABEL_PRED_COLUMN]],
            on=HEADLINE_ID_COLUMN,
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )

        first_model_correct_by_headline = (
            paired_model_gold_and_predicted_labels_df[f"{LABEL_PRED_COLUMN}_first"] == paired_model_gold_and_predicted_labels_df[LABEL_GOLD_COLUMN]
        ).to_numpy()
        second_model_correct_by_headline = (
            paired_model_gold_and_predicted_labels_df[f"{LABEL_PRED_COLUMN}_second"] == paired_model_gold_and_predicted_labels_df[LABEL_GOLD_COLUMN]
        ).to_numpy()
        first_model_only_correct_count, second_model_only_correct_count, p_value = get_mcnemar_test_values(
            first_model_correct_by_headline,
            second_model_correct_by_headline,
        )

        mcnemar_result = McNemarTestResult(
            model_a=first_model_name,
            model_b=second_model_name,
            model_a_correct_model_b_wrong=first_model_only_correct_count,
            model_b_correct_model_a_wrong=second_model_only_correct_count,
            p_value=float(p_value),
            significant_at_0_05=p_value < 0.05,
        )

        mcnemar_test_result_list.append(mcnemar_result)

    return pd.DataFrame(mcnemar_test_result_list)


def create_accuracy_and_macro_f1_figure(results_df, output_path):
    """Create a figure for the comparison of accuracy and Macro-F1."""

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    positions = np.arange(len(results_df))
    width = 0.36
    colours = {
        ACCURACY_COLUMN: FIGURE_PRIMARY_TEAL,
        MACRO_F1_COLUMN: FIGURE_DEEP_BLUE,
    }

    figure, axis = plt.subplots(figsize=(9, 5.3))

    for offset, metric, label in [
        (-width / 2, ACCURACY_COLUMN, "Accuracy"),
        (width / 2, MACRO_F1_COLUMN, "Macro-F1"),
    ]:
        values = results_df[metric].to_numpy()
        lower = results_df[f"{metric}_ci_lower"].to_numpy()
        upper = results_df[f"{metric}_ci_upper"].to_numpy()
        error = np.vstack([values - lower, upper - values])
        bars = axis.bar(
            positions + offset,
            values,
            width,
            label=label,
            color=colours[metric],
            yerr=error,
            capsize=4,
        )
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    axis.set_title("Sentiment classification performance")
    axis.set_ylabel("Score")
    axis.set_xticks(positions)
    axis.set_xticklabels(results_df[MODEL_COLUMN])
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_confusion_matrices_for_each_model_variant(paired_predictions, output_path):
    """Save one confusion matrix for each evaluated model variant."""

    figure, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    colour_gradient = LinearSegmentedColormap.from_list(
        "defined_colours",
        ["white", FIGURE_SECONDARY_TEAL, FIGURE_NAVY],
    )
    matrices = {
        name: confusion_matrix(
            data[LABEL_GOLD_COLUMN],
            data[LABEL_PRED_COLUMN],
            labels=SENTIMENT_LABELS,
        )
        for name, data in paired_predictions.items()
    }
    maximum_count = max(matrix.max() for matrix in matrices.values())

    for axis, name in zip(axes.flat, paired_predictions):
        matrix = matrices[name]
        image = axis.imshow(matrix, cmap=colour_gradient, vmin=0, vmax=maximum_count)
        for row in range(len(SENTIMENT_LABELS)):
            for column in range(len(SENTIMENT_LABELS)):
                colour = "white" if matrix[row, column] > matrix.max() / 2 else "black"
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=colour,
                )
        axis.set_title(name)
        axis.set_xticks(range(len(SENTIMENT_LABELS)), SENTIMENT_LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(SENTIMENT_LABELS)), SENTIMENT_LABELS)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Gold label")

    figure.suptitle("Confusion matrices on the 300-headline sample", y=1.01)
    figure.subplots_adjust(wspace=0.35, hspace=0.42, right=0.82)
    colour_axis = figure.add_axes([0.86, 0.2, 0.025, 0.6])
    figure.colorbar(
        image,
        cax=colour_axis,
        label="Headline count",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    """Evaluate FinBERT and all three DK-CoT knowledge treatments."""

    tables_directory = PROJECT_ROOT / "results" / "tables"
    figures_directory = PROJECT_ROOT / "results" / "figures"
    tables_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    gold_df, predictions_df = load_and_get_gold_labels_and_model_predictions_data()
    results_df, paired_predictions = evaluate_models(gold_df, predictions_df)
    mc_nemar_tests_df = run_mcnemar_tests_per_model_variant(paired_predictions)

    classification_results_path = tables_directory / CLASSIFICATION_RESULTS_CSV_FILE_NAME
    significance_path = tables_directory / PAIRED_SIGNIFICANCE_TESTS_CSV_FILE_NAME
    results_df.to_csv(classification_results_path, index=False)
    mc_nemar_tests_df.to_csv(significance_path, index=False)

    create_accuracy_and_macro_f1_figure(
        results_df,
        figures_directory / CLASSIFICATION_METRICS_PNG_FILE_NAME,
    )
    create_confusion_matrices_for_each_model_variant(
        paired_predictions,
        figures_directory / CLASSIFICATION_CONFUSION_MATRICES_PNG_FILE_NAME,
    )


if __name__ == "__main__":
    main()
