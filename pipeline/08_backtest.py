import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uk_dkcot.config import (
    MODEL_VARIANTS,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
)


SENTIMENT_SCORES = {"negative": -1, "neutral": 0, "positive": 1}
STRATEGIES = ["long-only", "long-short"]
TRANSACTION_COST_BPS = 10
COST_SENSITIVITY_BPS = [0, 5, 10, 20]
BOOTSTRAP_ITERATIONS = 2_000
BLOCK_LENGTH = 5
COMPANY_COUNT = 10
BENCHMARK_NAME = "Buy and hold (equal weight)"


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

        active_positions = sample["active_positions"].sum()
        hit_rate_values.append(
            sample["correct_positions"].sum() / active_positions
            if active_positions
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

    finbert_df = pd.read_csv(PROCESSED_DATA_DIR / "finbert_predictions.csv")
    dkcot_df = pd.read_csv(PROCESSED_DATA_DIR / "dkcot_predictions.csv")
    return pd.concat([finbert_df, dkcot_df], ignore_index=True)


def add_forward_returns(prices_df):
    """Add each day's open-to-next-open return for every ticker."""

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["next_open"] = prices.groupby("ticker")["open"].shift(-1)
    prices["forward_return"] = prices["next_open"] / prices["open"] - 1
    prices["date"] = prices["date"].dt.strftime("%Y-%m-%d")
    return prices


def create_benchmark_panel(prices):
    """Hold every company long for the whole period as a passive comparison."""

    panel = prices.copy()
    panel["position"] = 1
    return panel


def create_daily_signal_panels(headlines_df, prices, predictions_df):
    """Aggregate headline sentiment and build one daily panel per model."""

    headline_keys = headlines_df[["headline_id", "ticker", "trade_date"]]
    scored_predictions = predictions_df.merge(
        headline_keys,
        on="headline_id",
        how="inner",
        validate="many_to_one",
    )
    scored_predictions["sentiment_score"] = scored_predictions["label_pred"].map(
        SENTIMENT_SCORES
    )
    assert not scored_predictions["sentiment_score"].isna().any()

    daily_scores = (
        scored_predictions.groupby(
            ["trade_date", "ticker", "model_id", "knowledge_level"],
            as_index=False,
        )
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            headline_count=("headline_id", "size"),
        )
    )

    panels = {}
    output_rows = []

    for name, model_id, knowledge_level in MODEL_VARIANTS:
        model_scores = daily_scores.loc[
            (daily_scores["model_id"] == model_id)
            & (daily_scores["knowledge_level"] == knowledge_level),
            ["trade_date", "ticker", "sentiment_score", "headline_count"],
        ]
        panel = prices.merge(
            model_scores,
            left_on=["date", "ticker"],
            right_on=["trade_date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        panel["sentiment_score"] = panel["sentiment_score"].fillna(0.0)
        panel["headline_count"] = panel["headline_count"].fillna(0).astype(int)
        panel["position"] = np.sign(panel["sentiment_score"]).astype(int)
        panel["model"] = name
        panel["model_id"] = model_id
        panel["knowledge_level"] = knowledge_level
        panels[name] = panel

        output_rows.append(
            panel[
                [
                    "date",
                    "ticker",
                    "model_id",
                    "knowledge_level",
                    "sentiment_score",
                    "position",
                ]
            ]
        )

    return panels, pd.concat(output_rows, ignore_index=True)


def run_strategy(panel, strategy, cost_bps=TRANSACTION_COST_BPS):
    """Apply one strategy to a model's daily company-level signals."""

    tested = panel.copy().sort_values(["ticker", "date"])
    if strategy == "long-only":
        tested["strategy_position"] = tested["position"].clip(lower=0)
    elif strategy == "long-short":
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
    tested["transaction_cost"] = tested["turnover"] * cost_bps / 10_000
    tested["net_return"] = tested["gross_return"] - tested["transaction_cost"]

    # One row per company-day with an open position; hit rate is before costs.
    active = (
        tested["strategy_position"].ne(0)
        & tested["forward_return"].notna()
    )
    tested["active_position"] = active.astype(int)
    tested["correct_position"] = (
        active
        & (tested["strategy_position"] * tested["forward_return"] > 0)
    ).astype(int)

    # Each company has a fixed 10% portfolio weight; unused allocations remain cash.
    daily = (
        tested.groupby("date", as_index=False)
        .agg(
            gross_return=("gross_return", "sum"),
            transaction_cost=("transaction_cost", "sum"),
            net_return=("net_return", "sum"),
            active_positions=("active_position", "sum"),
            correct_positions=("correct_position", "sum"),
        )
        .sort_values("date")
    )
    daily[["gross_return", "transaction_cost", "net_return"]] /= COMPANY_COUNT
    daily["cumulative_return"] = (1 + daily["net_return"]).cumprod() - 1
    return tested, daily


def summarise_backtest(tested, daily, seed):
    """Return the performance metrics of one backtest."""

    position_days = int(tested["active_position"].sum())
    correct_positions = int(tested["correct_position"].sum())
    gross_cumulative_return = (1 + daily["gross_return"]).prod() - 1

    return {
        "Sharpe_ratio": calculate_sharpe(daily["net_return"]),
        **block_bootstrap_intervals(daily, seed),
        "hit_rate": correct_positions / position_days if position_days else 0.0,
        "cumulative_return": daily["cumulative_return"].iloc[-1],
        "gross_Sharpe_ratio": calculate_sharpe(daily["gross_return"]),
        "gross_cumulative_return": gross_cumulative_return,
        "position_days": position_days,
        "trading_days": len(daily),
        "transaction_cost_bps": TRANSACTION_COST_BPS,
    }


def evaluate_backtests(panels, benchmark_panel):
    """Evaluate both trading rules for every model, then the benchmark."""

    result_rows = []
    daily_curves = {}
    seed = RANDOM_SEED

    for name, model_id, knowledge_level in MODEL_VARIANTS:
        for strategy in STRATEGIES:
            tested, daily = run_strategy(panels[name], strategy)
            result_rows.append(
                {
                    "model": name,
                    "model_id": model_id,
                    "knowledge_level": knowledge_level,
                    "strategy": strategy,
                    **summarise_backtest(tested, daily, seed),
                }
            )
            daily_curves[(name, strategy)] = daily
            seed += 1

    # Evaluated last so the model rows keep the same bootstrap seeds.
    tested, daily = run_strategy(benchmark_panel, "long-only")
    result_rows.append(
        {
            "model": BENCHMARK_NAME,
            "model_id": "buy_and_hold",
            "knowledge_level": "none",
            "strategy": "long-only",
            **summarise_backtest(tested, daily, seed),
        }
    )
    daily_curves[(BENCHMARK_NAME, "long-only")] = daily

    return pd.DataFrame(result_rows), daily_curves


def evaluate_cost_sensitivity(panels):
    """Recalculate each backtest's Sharpe ratio and return at several costs."""

    rows = []

    for name, model_id, knowledge_level in MODEL_VARIANTS:
        for strategy in STRATEGIES:
            for cost_bps in COST_SENSITIVITY_BPS:
                _, daily = run_strategy(panels[name], strategy, cost_bps)
                rows.append(
                    {
                        "model": name,
                        "model_id": model_id,
                        "knowledge_level": knowledge_level,
                        "strategy": strategy,
                        "transaction_cost_bps": cost_bps,
                        "Sharpe_ratio": calculate_sharpe(daily["net_return"]),
                        "cumulative_return": daily["cumulative_return"].iloc[-1],
                    }
                )

    return pd.DataFrame(rows)


def save_cumulative_return_figure(daily_curves, output_path):
    """Save cumulative net-return curves for both trading strategies."""

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
    colours = ["#0B6E75", "#D17A22", "#4C78A8", "#8C564B"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    benchmark = daily_curves[(BENCHMARK_NAME, "long-only")]

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
        axis.plot(
            pd.to_datetime(benchmark["date"]),
            benchmark["cumulative_return"] * 100,
            label=BENCHMARK_NAME,
            color="#777777",
            linestyle="--",
            linewidth=1.2,
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
    colours = {"long-only": "#0B6E75", "long-short": "#D17A22"}
    positions = np.arange(len(MODEL_VARIANTS))
    width = 0.36

    for offset, strategy in zip([-width / 2, width / 2], STRATEGIES):
        subset = results_df.loc[results_df["strategy"] == strategy].set_index("model")
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
        PROCESSED_DATA_DIR / "headlines_aligned_prices.csv"
    )
    prices_df = pd.read_csv(RAW_DATA_DIR / "yfinance_prices.csv")
    predictions_df = load_predictions()

    assert headlines_df["headline_id"].is_unique
    assert prices_df["ticker"].nunique() == COMPANY_COUNT

    prices = add_forward_returns(prices_df)
    panels, daily_signals_df = create_daily_signal_panels(
        headlines_df,
        prices,
        predictions_df,
    )
    results_df, daily_curves = evaluate_backtests(
        panels,
        create_benchmark_panel(prices),
    )
    sensitivity_df = evaluate_cost_sensitivity(panels)

    signals_path = PROCESSED_DATA_DIR / "daily_signals.csv"
    results_path = PROCESSED_DATA_DIR / "backtest_results.csv"
    sensitivity_path = tables_directory / "backtest_cost_sensitivity.csv"
    daily_signals_df.to_csv(signals_path, index=False)
    results_df.to_csv(results_path, index=False)
    sensitivity_df.to_csv(sensitivity_path, index=False)

    save_cumulative_return_figure(
        daily_curves,
        figures_directory / "backtest_cumulative_returns.png",
    )
    save_backtest_metrics_figure(
        results_df,
        figures_directory / "backtest_metrics.png",
    )

    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved daily signals to {signals_path}")
    print(f"Saved backtest results to {results_path}")
    print(f"Saved cost sensitivity to {sensitivity_path}")
    print(f"Saved figures to {figures_directory}")


if __name__ == "__main__":
    main()
