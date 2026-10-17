import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uk_dkcot.config import (
    BACKTEST_CUMULATIVE_RETURNS_PNG_FILE_NAME,
    BACKTEST_METRICS_PNG_FILE_NAME,
    BACKTEST_RESULTS_CSV_FILE_NAME,
    BACKTEST_SUMMARY_CSV_FILE_NAME,
    BACKTEST_SUMMARY_TABLE_PNG_FILE_NAME,
    DAILY_SIGNALS_CSV_FILE_NAME,
    DKCOT_KNOWLEDGE_LEVELS,
    DKCOT_MODEL_NAME,
    DKCOT_PREDICTIONS_CSV_FILE_NAME,
    FINBERT_DISPLAY_NAME,
    FINBERT_MODEL_NAME,
    FINBERT_PREDICTIONS_CSV_FILE_NAME,
    HEADLINE_ID_COLUMN,
    HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME,
    KNOWLEDGE_LEVEL_COLUMN,
    LABEL_PRED_COLUMN,
    MODEL_COLUMN,
    MODEL_ID_COLUMN,
    NEGATIVE_LABEL,
    NEUTRAL_LABEL,
    NO_KNOWLEDGE_LEVEL,
    PD_DATE_FORMAT,
    POSITIVE_LABEL,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    YFINANCE_PRICES_CSV_FILE_NAME,
)


SENTIMENT_SCORES = {NEGATIVE_LABEL: -1, NEUTRAL_LABEL: 0, POSITIVE_LABEL: 1}
TRANSACTION_COST_BPS = 10
BOOTSTRAP_ITERATIONS = 2_000
BLOCK_LENGTH = 5
RANDOM_SEED = 2025

LONG_ONLY = "long-only"
LONG_SHORT = "long-short"
STRATEGIES = (LONG_ONLY, LONG_SHORT)

MODEL_VARIANTS = [
    (FINBERT_DISPLAY_NAME, FINBERT_MODEL_NAME, NO_KNOWLEDGE_LEVEL),
    *[
        (f"DK-CoT ({knowledge_level})", DKCOT_MODEL_NAME, knowledge_level)
        for knowledge_level in DKCOT_KNOWLEDGE_LEVELS
    ],
]


def calculate_sharpe(daily_returns):
    """Calculate annualised Sharpe ratio with a zero risk-free rate."""

    returns = np.asarray(daily_returns, dtype=float)
    standard_deviation = returns.std(ddof=1)
    if standard_deviation == 0 or np.isnan(standard_deviation):
        return 0.0
    return np.sqrt(252) * returns.mean() / standard_deviation


def block_bootstrap_intervals(daily_results, seed):
    """Estimate Sharpe and hit-rate intervals using five-day blocks."""

    random_generator = np.random.default_rng(seed)
    day_count = len(daily_results)
    block_count = int(np.ceil(day_count / BLOCK_LENGTH))
    maximum_start = day_count - BLOCK_LENGTH + 1
    sharpe_values = []
    hit_rate_values = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        starts = random_generator.integers(0, maximum_start, block_count)
        indices = np.concatenate(
            [np.arange(start, start + BLOCK_LENGTH) for start in starts]
        )[:day_count]
        sample = daily_results.iloc[indices]
        sharpe_values.append(calculate_sharpe(sample["net_return"]))

        active_trades = sample["active_trades"].sum()
        hit_rate_values.append(
            sample["correct_trades"].sum() / active_trades
            if active_trades
            else 0.0
        )

    return {
        "Sharpe_ci_lower": np.quantile(sharpe_values, 0.025),
        "Sharpe_ci_upper": np.quantile(sharpe_values, 0.975),
        "hit_rate_ci_lower": np.quantile(hit_rate_values, 0.025),
        "hit_rate_ci_upper": np.quantile(hit_rate_values, 0.975),
    }


def load_predictions():
    """Load FinBERT and DK-CoT predictions under one shared schema."""

    finbert_df = pd.read_csv(PROCESSED_DATA_DIR / FINBERT_PREDICTIONS_CSV_FILE_NAME)
    dkcot_df = pd.read_csv(PROCESSED_DATA_DIR / DKCOT_PREDICTIONS_CSV_FILE_NAME)
    return pd.concat([finbert_df, dkcot_df], ignore_index=True)


def create_daily_signal_panels(headlines_df, prices_df, predictions_df):
    """Aggregate headline sentiment and build one daily panel per model."""

    headline_keys = headlines_df[[HEADLINE_ID_COLUMN, "ticker", "trade_date"]]
    scored_predictions = predictions_df.merge(
        headline_keys,
        on=HEADLINE_ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    scored_predictions["sentiment_score"] = scored_predictions[LABEL_PRED_COLUMN].map(
        SENTIMENT_SCORES
    )
    assert not scored_predictions["sentiment_score"].isna().any()

    daily_scores = (
        scored_predictions.groupby(
            ["trade_date", "ticker", MODEL_ID_COLUMN, KNOWLEDGE_LEVEL_COLUMN],
            as_index=False,
        )
        .agg(
            sentiment_score=("sentiment_score", "mean"),
        )
    )

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["next_open"] = prices.groupby("ticker")["open"].shift(-1)
    prices["forward_return"] = prices["next_open"] / prices["open"] - 1
    prices["date"] = prices["date"].dt.strftime(PD_DATE_FORMAT)

    panels = {}
    output_rows = []

    for name, model_id, knowledge_level in MODEL_VARIANTS:
        model_scores = daily_scores.loc[
            (daily_scores[MODEL_ID_COLUMN] == model_id)
            & (daily_scores[KNOWLEDGE_LEVEL_COLUMN] == knowledge_level),
            ["trade_date", "ticker", "sentiment_score"],
        ]
        panel = prices.merge(
            model_scores,
            left_on=["date", "ticker"],
            right_on=["trade_date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        panel["sentiment_score"] = panel["sentiment_score"].fillna(0.0)
        panel["position"] = np.sign(panel["sentiment_score"]).astype(int)
        panel[MODEL_ID_COLUMN] = model_id
        panel[KNOWLEDGE_LEVEL_COLUMN] = knowledge_level
        panels[name] = panel

        output_rows.append(
            panel[
                [
                    "date",
                    "ticker",
                    MODEL_ID_COLUMN,
                    KNOWLEDGE_LEVEL_COLUMN,
                    "sentiment_score",
                    "position",
                ]
            ]
        )

    return panels, pd.concat(output_rows, ignore_index=True)


def run_strategy(panel, strategy):
    """Apply one strategy to a model's daily company-level signals."""

    tested = panel.copy().sort_values(["ticker", "date"])
    if strategy == LONG_ONLY:
        tested["strategy_position"] = tested["position"].clip(lower=0)
    elif strategy == LONG_SHORT:
        tested["strategy_position"] = tested["position"]
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # A position cannot be tested when the following opening price is unavailable.
    tested.loc[tested["forward_return"].isna(), "strategy_position"] = 0
    previous_position = tested.groupby("ticker")["strategy_position"].shift(1)
    previous_position = previous_position.fillna(0)
    tested["turnover"] = (tested["strategy_position"] - previous_position).abs()

    tested["gross_return"] = (
        tested["strategy_position"] * tested["forward_return"].fillna(0)
    )
    tested["transaction_cost"] = (
        tested["turnover"] * TRANSACTION_COST_BPS / 10_000
    )
    tested["net_return"] = tested["gross_return"] - tested["transaction_cost"]

    active = (
        tested["strategy_position"].ne(0)
        & tested["forward_return"].notna()
    )
    tested["active_trade"] = active.astype(int)
    tested["correct_trade"] = (
        active
        & (tested["strategy_position"] * tested["forward_return"] > 0)
    ).astype(int)

    # Each company has a fixed 10% portfolio weight. Unused allocations remain cash.
    company_count = tested["ticker"].nunique()
    daily = (
        tested.groupby("date", as_index=False)
        .agg(
            gross_return=("gross_return", "sum"),
            transaction_cost=("transaction_cost", "sum"),
            net_return=("net_return", "sum"),
            active_trades=("active_trade", "sum"),
            correct_trades=("correct_trade", "sum"),
        )
        .sort_values("date")
    )
    # Daily company returns are summed and divided by ten, giving each company an effective 10% portfolio weight.
    # Where no position is taken for a company, its contribution is zero and the unused allocation is treated as remaining in cash.
    daily[["gross_return", "transaction_cost", "net_return"]] /= company_count
    daily["cumulative_return"] = (1 + daily["net_return"]).cumprod() - 1
    return daily


def evaluate_backtests(panels):
    """Evaluate both trading rules for every sentiment model."""

    result_rows = []
    daily_curves = {}
    seed = RANDOM_SEED

    for name, model_id, knowledge_level in MODEL_VARIANTS:
        for strategy in STRATEGIES:
            daily = run_strategy(panels[name], strategy)
            active_trades = int(daily["active_trades"].sum())
            correct_trades = int(daily["correct_trades"].sum())
            intervals = block_bootstrap_intervals(daily, seed)
            seed += 1

            result_rows.append(
                {
                    MODEL_COLUMN: name,
                    MODEL_ID_COLUMN: model_id,
                    KNOWLEDGE_LEVEL_COLUMN: knowledge_level,
                    "strategy": strategy,
                    "Sharpe_ratio": calculate_sharpe(daily["net_return"]),
                    **intervals,
                    "hit_rate": (
                        correct_trades / active_trades if active_trades else 0.0
                    ),
                    "cumulative_return": daily["cumulative_return"].iloc[-1],
                    "active_trades": active_trades,
                    "trading_days": len(daily),
                    "transaction_cost_bps": TRANSACTION_COST_BPS,
                }
            )
            daily_curves[(name, strategy)] = daily

    return pd.DataFrame(result_rows), daily_curves


def create_report_table(results_df):
    """Create the concise backtest table used in the dissertation."""

    report_table = results_df[
        [
            MODEL_COLUMN,
            "strategy",
            "Sharpe_ratio",
            "hit_rate",
            "cumulative_return",
            "active_trades",
        ]
    ].copy()

    report_table["strategy"] = report_table["strategy"].str.capitalize()
    report_table["Sharpe_ratio"] = report_table["Sharpe_ratio"].round(3)
    report_table["hit_rate"] = (report_table["hit_rate"] * 100).round(2)
    report_table["cumulative_return"] = (
        report_table["cumulative_return"] * 100
    ).round(2)

    return report_table.rename(
        columns={
            MODEL_COLUMN: "Condition",
            "strategy": "Strategy",
            "Sharpe_ratio": "Sharpe ratio",
            "hit_rate": "Hit rate (%)",
            "cumulative_return": "Cumulative net return (%)",
            "active_trades": "Active company-days",
        }
    )


def save_backtest_summary_figure(report_table, output_path):
    """Save the concise backtest results as a report-ready table figure."""

    display_table = report_table.copy()
    display_table["Sharpe ratio"] = display_table["Sharpe ratio"].map(
        lambda value: f"{value:.3f}"
    )
    display_table["Hit rate (%)"] = display_table["Hit rate (%)"].map(
        lambda value: f"{value:.2f}%"
    )
    display_table["Cumulative net return (%)"] = display_table[
        "Cumulative net return (%)"
    ].map(lambda value: f"{value:.2f}%")
    display_table["Active company-days"] = display_table[
        "Active company-days"
    ].map(lambda value: f"{value:,.0f}")

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9})
    figure, axis = plt.subplots(figsize=(12, 4.2))
    axis.axis("off")
    table = axis.table(
        cellText=display_table.to_numpy(),
        colLabels=display_table.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.19, 0.13, 0.13, 0.13, 0.23, 0.18],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D6DEDA")
        if row == 0:
            cell.set_facecolor("#0B6E75")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F1F5F3")
        if row > 0 and column in (0, 1):
            cell.set_text_props(ha="left")

    axis.set_title("Backtest performance summary", fontsize=13, pad=12)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_cumulative_return_figure(daily_curves, output_path):
    """Save cumulative net-return curves for both trading strategies."""

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    colours = ["#0B6E75", "#D17A22", "#4C78A8", "#8C564B"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for axis, strategy in zip(axes, STRATEGIES):
        for colour, (name, _, _) in zip(colours, MODEL_VARIANTS):
            daily = daily_curves[(name, strategy)]
            axis.plot(
                pd.to_datetime(daily["date"]),
                daily["cumulative_return"] * 100,
                label=name,
                color=colour,
                linewidth=1.7,
            )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_title(strategy.title())
        axis.set_ylabel("Cumulative return (%)")
        axis.grid(alpha=0.22)

    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Trading date")
    figure.suptitle("Cumulative portfolio returns after transaction costs")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_backtest_metrics_figure(results_df, output_path):
    """Save Sharpe-ratio and hit-rate comparisons with confidence intervals."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colours = {LONG_ONLY: "#0B6E75", LONG_SHORT: "#D17A22"}
    positions = np.arange(len(MODEL_VARIANTS))
    width = 0.36

    for offset, strategy in [(-width / 2, LONG_ONLY), (width / 2, LONG_SHORT)]:
        subset = results_df.loc[results_df["strategy"] == strategy].set_index(MODEL_COLUMN)
        subset = subset.loc[[item[0] for item in MODEL_VARIANTS]]

        for axis, metric, lower_name, upper_name in [
            (axes[0], "Sharpe_ratio", "Sharpe_ci_lower", "Sharpe_ci_upper"),
            (axes[1], "hit_rate", "hit_rate_ci_lower", "hit_rate_ci_upper"),
        ]:
            values = subset[metric].to_numpy()
            error = np.vstack(
                [
                    values - subset[lower_name].to_numpy(),
                    subset[upper_name].to_numpy() - values,
                ]
            )
            axis.bar(
                positions + offset,
                values,
                width,
                yerr=error,
                capsize=3,
                label=strategy.title(),
                color=colours[strategy],
            )

    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_title("Annualised Sharpe ratio")
    axes[1].set_title("Directional hit rate")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [item[0] for item in MODEL_VARIANTS],
            rotation=18,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.22)

    axes[0].legend(frameon=False)
    figure.suptitle("Trading performance with 95% block-bootstrap intervals")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    """Create daily signals and backtest all model variants consistently."""

    tables_directory = PROJECT_ROOT / "results" / "tables"
    figures_directory = PROJECT_ROOT / "results" / "figures"
    tables_directory.mkdir(parents=True, exist_ok=True)
    figures_directory.mkdir(parents=True, exist_ok=True)

    headlines_df = pd.read_csv(
        PROCESSED_DATA_DIR / HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME
    )
    prices_df = pd.read_csv(RAW_DATA_DIR / YFINANCE_PRICES_CSV_FILE_NAME)
    predictions_df = load_predictions()

    assert headlines_df[HEADLINE_ID_COLUMN].is_unique
    assert prices_df["ticker"].nunique() == 10

    panels, daily_signals_df = create_daily_signal_panels(
        headlines_df,
        prices_df,
        predictions_df,
    )
    results_df, daily_curves = evaluate_backtests(panels)
    report_table = create_report_table(results_df)

    signals_path = PROCESSED_DATA_DIR / DAILY_SIGNALS_CSV_FILE_NAME
    results_path = PROCESSED_DATA_DIR / BACKTEST_RESULTS_CSV_FILE_NAME
    report_table_path = tables_directory / BACKTEST_SUMMARY_CSV_FILE_NAME
    daily_signals_df.to_csv(signals_path, index=False)
    results_df.to_csv(results_path, index=False)
    report_table.to_csv(report_table_path, index=False)

    save_cumulative_return_figure(
        daily_curves,
        figures_directory / BACKTEST_CUMULATIVE_RETURNS_PNG_FILE_NAME,
    )
    save_backtest_metrics_figure(
        results_df,
        figures_directory / BACKTEST_METRICS_PNG_FILE_NAME,
    )
    save_backtest_summary_figure(
        report_table,
        figures_directory / BACKTEST_SUMMARY_TABLE_PNG_FILE_NAME,
    )

    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved daily signals to {signals_path}")
    print(f"Saved backtest results to {results_path}")
    print(f"Saved report table to {report_table_path}")
    print(f"Saved figures to {figures_directory}")


if __name__ == "__main__":
    main()
