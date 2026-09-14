import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from uk_dkcot.config import (
    MODEL_VARIANTS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
)


LABELS = ["negative", "neutral", "positive"]
BOOTSTRAP_ITERATIONS = 2_000
GOLD_COLUMNS = ["headline_id", "label_gold", "annotator_pass", "qa_flag"]


def calculate_metrics(gold_labels, predictions):
    """Return accuracy and Macro-F1 for one prediction set."""

    return {
        "accuracy": accuracy_score(gold_labels, predictions),
        "macro_f1": f1_score(
            gold_labels,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def bootstrap_confidence_intervals(gold_labels, predictions, seed):
    """Estimate 95% confidence intervals by resampling headlines."""

    gold = np.asarray(gold_labels)
    predicted = np.asarray(predictions)
    random_generator = np.random.default_rng(seed)
    accuracy_values = []
    macro_f1_values = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        indices = random_generator.integers(0, len(gold), len(gold))
        metrics = calculate_metrics(gold[indices], predicted[indices])
        accuracy_values.append(metrics["accuracy"])
        macro_f1_values.append(metrics["macro_f1"])

    return {
        "accuracy_ci_lower": np.quantile(accuracy_values, 0.025),
        "accuracy_ci_upper": np.quantile(accuracy_values, 0.975),
        "macro_f1_ci_lower": np.quantile(macro_f1_values, 0.025),
        "macro_f1_ci_upper": np.quantile(macro_f1_values, 0.975),
    }


def exact_mcnemar_test(first_correct, second_correct):
    """Return the exact two-sided McNemar p-value for paired predictions."""

    first_only = int(np.sum(first_correct & ~second_correct))
    second_only = int(np.sum(~first_correct & second_correct))
    discordant = first_only + second_only

    if discordant == 0:
        return first_only, second_only, 1.0

    smaller_count = min(first_only, second_only)
    lower_tail = sum(
        math.comb(discordant, count)
        for count in range(smaller_count + 1)
    ) / (2**discordant)
    return first_only, second_only, min(1.0, 2 * lower_tail)


def holm_adjust(p_values):
    """Return Holm-adjusted p-values for one family of tests."""

    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.empty_like(p_values)
    running_maximum = 0.0

    for rank, index in enumerate(np.argsort(p_values)):
        running_maximum = max(
            running_maximum,
            (len(p_values) - rank) * p_values[index],
        )
        adjusted[index] = min(1.0, running_maximum)

    return adjusted


def load_evaluation_data():
    """Load gold labels and all four model prediction variants."""

    gold_df = pd.read_csv(PROCESSED_DATA_DIR / "gold_labels.csv")
    finbert_df = pd.read_csv(PROCESSED_DATA_DIR / "finbert_predictions.csv")
    dkcot_df = pd.read_csv(PROCESSED_DATA_DIR / "dkcot_predictions.csv")

    assert gold_df.columns.tolist() == GOLD_COLUMNS
    assert gold_df["headline_id"].is_unique
    assert set(gold_df["label_gold"]) == set(LABELS)

    first_pass_only = int(gold_df["annotator_pass"].ne(2).sum())
    if first_pass_only:
        print(
            f"Warning: {first_pass_only} gold labels have not been through "
            "the second annotation pass."
        )

    predictions_df = pd.concat([finbert_df, dkcot_df], ignore_index=True)
    expected_keys = {
        (model_id, knowledge_level)
        for _, model_id, knowledge_level in MODEL_VARIANTS
    }
    actual_keys = set(
        predictions_df[["model_id", "knowledge_level"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert actual_keys == expected_keys

    return gold_df, predictions_df


def evaluate_models(gold_df, predictions_df):
    """Calculate classification metrics and retain paired predictions."""

    result_rows = []
    paired_predictions = {}

    for model_index, (name, model_id, knowledge_level) in enumerate(MODEL_VARIANTS):
        model_predictions = predictions_df.loc[
            (predictions_df["model_id"] == model_id)
            & (predictions_df["knowledge_level"] == knowledge_level)
        ].copy()

        evaluation_df = gold_df.merge(
            model_predictions,
            on="headline_id",
            how="left",
            validate="one_to_one",
        )
        assert len(evaluation_df) == len(gold_df)
        assert not evaluation_df["label_pred"].isna().any()

        metrics = calculate_metrics(
            evaluation_df["label_gold"],
            evaluation_df["label_pred"],
        )
        intervals = bootstrap_confidence_intervals(
            evaluation_df["label_gold"],
            evaluation_df["label_pred"],
            RANDOM_SEED + model_index,
        )

        parse_ok = model_predictions["parse_ok"].astype(str).str.lower().eq("true")
        evaluation_parse_ok = (
            evaluation_df["parse_ok"].astype(str).str.lower().eq("true")
        )

        result_rows.append(
            {
                "model": name,
                "model_id": model_id,
                "knowledge_level": knowledge_level,
                "n_headlines": len(evaluation_df),
                **metrics,
                **intervals,
                "parse_failures_sample": int((~evaluation_parse_ok).sum()),
                "parse_failures_full": int((~parse_ok).sum()),
            }
        )
        paired_predictions[name] = evaluation_df[
            ["headline_id", "label_gold", "label_pred"]
        ].copy()

    return pd.DataFrame(result_rows), paired_predictions


def compare_paired_predictions(paired_predictions):
    """Run exact McNemar tests for every pair of model variants."""

    rows = []

    for first_name, second_name in itertools.combinations(
        paired_predictions,
        2,
    ):
        first_df = paired_predictions[first_name]
        second_df = paired_predictions[second_name]
        paired_df = first_df.merge(
            second_df[["headline_id", "label_pred"]],
            on="headline_id",
            suffixes=("_first", "_second"),
            validate="one_to_one",
        )

        first_correct = (
            paired_df["label_pred_first"] == paired_df["label_gold"]
        ).to_numpy()
        second_correct = (
            paired_df["label_pred_second"] == paired_df["label_gold"]
        ).to_numpy()
        first_only, second_only, p_value = exact_mcnemar_test(
            first_correct,
            second_correct,
        )

        rows.append(
            {
                "model_a": first_name,
                "model_b": second_name,
                "a_correct_b_wrong": first_only,
                "b_correct_a_wrong": second_only,
                "p_value": p_value,
            }
        )

    # The pairwise tests form one family, so significance uses Holm-adjusted p-values.
    significance_df = pd.DataFrame(rows)
    significance_df["p_value_holm"] = holm_adjust(significance_df["p_value"])
    significance_df["significant_at_0_05"] = significance_df["p_value_holm"] < 0.05
    return significance_df


def save_classification_figure(results_df, output_path):
    """Save a report-ready comparison of accuracy and Macro-F1."""

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    positions = np.arange(len(results_df))
    width = 0.36
    colours = {"accuracy": "#0B6E75", "macro_f1": "#D17A22"}

    figure, axis = plt.subplots(figsize=(9, 5.3))

    for offset, metric, label in [
        (-width / 2, "accuracy", "Accuracy"),
        (width / 2, "macro_f1", "Macro-F1"),
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
    axis.set_xticklabels(results_df["model"])
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrices(paired_predictions, output_path):
    """Save one confusion matrix for each evaluated model variant."""

    figure, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    matrices = {
        name: confusion_matrix(
            data["label_gold"],
            data["label_pred"],
            labels=LABELS,
        )
        for name, data in paired_predictions.items()
    }
    maximum_count = max(matrix.max() for matrix in matrices.values())

    for axis, (name, data) in zip(axes.flat, paired_predictions.items()):
        matrix = matrices[name]
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=maximum_count)
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
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
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=25, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Gold label")

    headline_count = len(next(iter(paired_predictions.values())))
    figure.suptitle(
        f"Confusion matrices on the {headline_count}-headline sample",
        y=1.01,
    )
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

    gold_df, predictions_df = load_evaluation_data()
    results_df, paired_predictions = evaluate_models(gold_df, predictions_df)
    significance_df = compare_paired_predictions(paired_predictions)

    results_path = tables_directory / "classification_results.csv"
    significance_path = tables_directory / "paired_significance_tests.csv"
    results_df.to_csv(results_path, index=False)
    significance_df.to_csv(significance_path, index=False)

    save_classification_figure(
        results_df,
        figures_directory / "classification_metrics.png",
    )
    save_confusion_matrices(
        paired_predictions,
        figures_directory / "classification_confusion_matrices.png",
    )

    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nPaired exact McNemar tests:")
    print(significance_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved tables to {tables_directory}")
    print(f"Saved figures to {figures_directory}")


if __name__ == "__main__":
    main()
