import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import TypedDict

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
    FIGURE_DARK_TEAL,
    FIGURE_DEEP_BLUE,
    FIGURE_NAVY,
    FIGURE_PRIMARY_TEAL,
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
    get_figure_font_settings,
)

SENTIMENT_SCORES = {NEGATIVE_LABEL: -1, NEUTRAL_LABEL: 0, POSITIVE_LABEL: 1}
# Trading transaction costs
# I need to reference this for the report: https://arxiv.org/pdf/2302.10175
TRANSACTION_COST_BASIS_POINTS = 10
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

# Local DataFrame column names used throughout the backtest.
DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
TRADE_DATE_COLUMN = "trade_date"
OPEN_COLUMN = "open"
NEXT_OPEN_COLUMN = "next_open"
SENTIMENT_SCORE_COLUMN = "sentiment_score"
POSITION_COLUMN = "position"
STRATEGY_COLUMN = "strategy"
STRATEGY_POSITION_COLUMN = "strategy_position"
TURNOVER_COLUMN = "turnover"
FORWARD_RETURN_COLUMN = "forward_return"
GROSS_RETURN_COLUMN = "gross_return"
TRANSACTION_COST_COLUMN = "transaction_cost"
NET_RETURN_COLUMN = "net_return"
ACTIVE_TRADE_COLUMN = "active_trade"
CORRECT_TRADE_COLUMN = "correct_trade"
ACTIVE_TRADES_COLUMN = "active_trades"
CORRECT_TRADES_COLUMN = "correct_trades"
CUMULATIVE_GROSS_RETURN_COLUMN = "cumulative_gross_return"
CUMULATIVE_RETURN_COLUMN = "cumulative_return"
TRANSACTION_COST_DRAG_COLUMN = "transaction_cost_drag"
SHARPE_RATIO_COLUMN = "Sharpe_ratio"
SHARPE_CI_LOWER_COLUMN = "Sharpe_ci_lower"
SHARPE_CI_UPPER_COLUMN = "Sharpe_ci_upper"
HIT_RATE_COLUMN = "hit_rate"
HIT_RATE_CI_LOWER_COLUMN = "hit_rate_ci_lower"
HIT_RATE_CI_UPPER_COLUMN = "hit_rate_ci_upper"
TRADING_DAYS_COLUMN = "trading_days"
TRANSACTION_COST_BPS_COLUMN = "transaction_cost_bps"

# Local display-column names used in the dissertation backtest table and figure.
CONDITION_REPORT_COLUMN = "Condition"
STRATEGY_REPORT_COLUMN = "Strategy"
SHARPE_RATIO_REPORT_COLUMN = "Sharpe ratio"
HIT_RATE_REPORT_COLUMN = "Hit rate (%)"
CUMULATIVE_GROSS_RETURN_REPORT_COLUMN = "Cumulative gross return (%)"
TRANSACTION_COST_DRAG_REPORT_COLUMN = (
    "Transaction-cost drag (percentage points)"
)
CUMULATIVE_RETURN_REPORT_COLUMN = "Cumulative net return (%)"
ACTIVE_TRADES_REPORT_COLUMN = "Active company-days"


class BacktestConfidenceIntervals(TypedDict):
    Sharpe_ci_lower: float
    Sharpe_ci_upper: float
    hit_rate_ci_lower: float
    hit_rate_ci_upper: float


class BacktestResult(TypedDict):
    model: str
    model_id: str
    knowledge_level: str
    strategy: str
    Sharpe_ratio: float
    Sharpe_ci_lower: float
    Sharpe_ci_upper: float
    hit_rate_ci_lower: float
    hit_rate_ci_upper: float
    hit_rate: float
    cumulative_gross_return: float
    cumulative_return: float
    active_trades: int
    trading_days: int
    transaction_cost_bps: int


def convert_basis_points_to_decimal(basis_points: float) -> float:
    """Convert basis points into a decimal return."""

    return basis_points / 10_000


def calculate_annualised_sharpe_ratio(daily_returns) -> float:
    """Calculate annualised Sharpe ratio with a zero risk-free rate."""

    # Convert the daily returns into a NumPy array and calculate their sample standard deviation.
    daily_returns_np_array = np.asarray(daily_returns, dtype=float)
    daily_returns_standard_deviation = daily_returns_np_array.std(ddof=1)

    # A constant return series has no measurable volatility, so its Sharpe ratio is returned as zero.
    if daily_returns_standard_deviation == 0 or np.isnan(
        daily_returns_standard_deviation
    ):
        return 0.0

    # Sharpe (1966) introduced the reward-to-variability measure now known as the Sharpe ratio.
    # https://doi.org/10.1086/294846
    # Multiplication by the square root of 252 annualises a daily ratio under the usual independent-return assumption. Lo (2002) explains why serial correlation can affect this scaling.
    # https://doi.org/10.2469/faj.v58.n4.2453
    # The number 252 was used because it is a standard approximation of the trading days in a year
    # https://doi.org/10.1007/s10489-021-02249-x
    return (
        np.sqrt(252) * daily_returns_np_array.mean() / daily_returns_standard_deviation
    )


def calculate_backtest_confidence_intervals(
    daily_backtest_results_df,
    seed,
) -> BacktestConfidenceIntervals:
    """Estimate Sharpe and hit-rate intervals using five-day blocks."""

    # Use the supplied seed so that rerunning the bootstrap produces the same samples.
    random_generator = np.random.default_rng(seed)
    # Record how many trading days each bootstrap sample must contain.
    trading_day_count = len(daily_backtest_results_df)
    # Calculate how many five-day blocks are needed to recreate that number of days
    bootstrap_block_count = int(np.ceil(trading_day_count / BLOCK_LENGTH))
    # Restrict block starts so that every selected block remains inside the dataset
    maximum_block_start_index = trading_day_count - BLOCK_LENGTH + 1
    # Store the Sharpe ratio and hit rate calculated from each bootstrap sample.
    bootstrap_sharpe_ratio_values = []
    bootstrap_hit_rate_values = []

    # Sample consecutive five-day blocks with replacement. This retains the ordering within each selected trading week rather than treating every daily return as independent.
    # This follows the moving-block bootstrap introduced for dependent observations by Kunsch (1989).
    # https://doi.org/10.1214/aos/1176347265
    for _ in range(BOOTSTRAP_ITERATIONS):
        bootstrap_block_start_indices = random_generator.integers(
            0,
            maximum_block_start_index,
            bootstrap_block_count,
        )
        bootstrap_sample_indices = np.concatenate(
            [
                np.arange(start_index, start_index + BLOCK_LENGTH) # Create the row positions for a block of five consecutive trading days, beginning at the randomly selected starting row
                for start_index in bootstrap_block_start_indices
            ]
        )[:trading_day_count] # Trim array so the bootstrap sample contains the same number of days as the original dataset
        bootstrap_sample_df = daily_backtest_results_df.iloc[bootstrap_sample_indices]
        bootstrap_sharpe_ratio_values.append(
            calculate_annualised_sharpe_ratio(bootstrap_sample_df[NET_RETURN_COLUMN])
        )

        bootstrap_active_trade_count = bootstrap_sample_df[ACTIVE_TRADES_COLUMN].sum()
        bootstrap_hit_rate_values.append(
            bootstrap_sample_df[CORRECT_TRADES_COLUMN].sum()
            / bootstrap_active_trade_count
            if bootstrap_active_trade_count
            else 0.0
        )

    # Use the 2.5th and 97.5th percentiles as the limits of each 95% percentile-bootstrap confidence interval
    # NumPy quantile documentation: https://numpy.org/doc/stable/reference/generated/numpy.quantile.html
    return BacktestConfidenceIntervals(
        Sharpe_ci_lower=np.quantile(bootstrap_sharpe_ratio_values, 0.025),
        Sharpe_ci_upper=np.quantile(bootstrap_sharpe_ratio_values, 0.975),
        hit_rate_ci_lower=np.quantile(bootstrap_hit_rate_values, 0.025),
        hit_rate_ci_upper=np.quantile(bootstrap_hit_rate_values, 0.975),
    )


def load_FinBERT_and_DK_CoT_predictions() -> pd.DataFrame:
    """Load and combine FinBERT and DK-CoT predictions into one dataframe."""

    finbert_predictions_df = pd.read_csv(
        PROCESSED_DATA_DIR / FINBERT_PREDICTIONS_CSV_FILE_NAME
    )
    dkcot_predictions_df = pd.read_csv(
        PROCESSED_DATA_DIR / DKCOT_PREDICTIONS_CSV_FILE_NAME
    )

    return pd.concat(
        [finbert_predictions_df, dkcot_predictions_df],
        ignore_index=True,
    )


def create_daily_sentiment_signal_panels(
    headlines_aligned_prices_df,
    yfinance_prices_df,
    model_predictions_df,
):
    """Average sentiment by company and trading day, then create one daily dataset for each model."""

    # Retain the identifiers needed to connect every prediction to its company and aligned trading date
    headline_trade_dates_df = headlines_aligned_prices_df[
        [HEADLINE_ID_COLUMN, TICKER_COLUMN, TRADE_DATE_COLUMN]
    ]
    sentiment_scored_predictions_df = model_predictions_df.merge(
        headline_trade_dates_df,
        on=HEADLINE_ID_COLUMN,
        how="inner",
        validate="many_to_one",
    )
    # Convert each sentiment label into a score: negative=-1, neutral=0 and positive=1
    sentiment_scored_predictions_df[SENTIMENT_SCORE_COLUMN] = (
        sentiment_scored_predictions_df[LABEL_PRED_COLUMN].map(SENTIMENT_SCORES)
    )

    # Average multiple headlines for the same company and trading date into one daily score per model variant.
    daily_sentiment_scores_df = sentiment_scored_predictions_df.groupby(
        [
            TRADE_DATE_COLUMN,
            TICKER_COLUMN,
            MODEL_ID_COLUMN,
            KNOWLEDGE_LEVEL_COLUMN,
        ],
        as_index=False, # Keep TRADE_DATE_COLUMN, TICKER_COLUMN, MODEL_ID_COLUMN and KNOWLEDGE_LEVEL_COLUMN as dataframe columns rather than index levels in the resulting dataframe
        # https://stackoverflow.com/questions/51866908/difference-between-as-index-false-and-reset-index-in-pandas-groupby
    ).agg(
        sentiment_score=(SENTIMENT_SCORE_COLUMN, "mean"), # Create an output column named SENTIMENT_SCORE_COLUMN. The value is the "mean" of SENTIMENT_SCORE_COLUMN within each group. For example, scores of [1, 1, -1] produce a mean score of 0.33. Later on, `np.sign()`` will convert that into a positive position considering a zero mean becomes neutral, a positive mean becomes positive and a negative number mean becomes a negative trading position.
    )

    # Work on a copy so the original price DataFrame remains unchanged
    prices_with_forward_returns_df = yfinance_prices_df.copy()
    # Convert the date strings to datetime values so they can be sorted later on
    prices_with_forward_returns_df[DATE_COLUMN] = pd.to_datetime(
        prices_with_forward_returns_df[DATE_COLUMN]
    )
    # Sort each company price by date so that the `.shift(-1)` can get the next trading day's opening price
    prices_with_forward_returns_df = prices_with_forward_returns_df.sort_values(
        [TICKER_COLUMN, DATE_COLUMN]
    )

    # Within each company, place the following trading day's opening price on the current row.
    # Pandas GroupBy.shift: https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.shift.html
    prices_with_forward_returns_df[NEXT_OPEN_COLUMN] = (
        prices_with_forward_returns_df.groupby(TICKER_COLUMN)[OPEN_COLUMN].shift(-1)
    )

    # Calculate the return between two opening prices (current and next). For example, a move from £100 to £102 will be as follows:
    # Opening price for day 10 = £100
    # Openning price for day 12 = £102
    # Price difference = £2
    # (102/100) - 1 = 0.02
    # The above value is then later on converted into a percentage of 2% for display purposes. The 2% value is the company's share price between two opening prices that are consecutive.
    # The example will be used by the backtest to measure the trading result:
    # - Long position: 1 × 0.02 = 0.02, an 2% gross return.
    # - Short position: -1 × 0.02 = -0.02, a 2% gross loss.
    # As for the reason why there is a `1` being subtracted. This was found when an issue faced when the price ratio equals one (as in it has not changed). So the `-1` is to cater for that and provide a value of `0` when there is no price change.
    # This was tricky initially but the following source was used to make this work (see "Example 1.6 (Simple return calculation)" from the source specifically) (Zivot, 2022):
    # https://bookdown.org/compfinezbook/introFinRbook/Return-Calculations.html
    # Zivot, E. (2022) Introduction to computational finance and financial econometrics with R. Available at: https://bookdown.org/compfinezbook/introFinRbook/Return-Calculations.html (Accessed: 20 August 2026).
    prices_with_forward_returns_df[FORWARD_RETURN_COLUMN] = (
        prices_with_forward_returns_df[NEXT_OPEN_COLUMN]
        / prices_with_forward_returns_df[OPEN_COLUMN]
        - 1
    )
    # Convert the datetime values into the same date-string format used by the sentiment trade dates so that the two datasets can be merged
    prices_with_forward_returns_df[DATE_COLUMN] = prices_with_forward_returns_df[
        DATE_COLUMN
    ].dt.strftime(PD_DATE_FORMAT)

    daily_signal_panels_by_model = {}
    daily_signal_dataframes = []

    # Loop to repeat the process for FinBERT and each DK-CoT knowledge condition.
    for model_name, model_id, knowledge_level in MODEL_VARIANTS:
        # Select the daily sentiment scores belonging to the current model condition
        model_daily_sentiment_scores_df = daily_sentiment_scores_df.loc[
            (daily_sentiment_scores_df[MODEL_ID_COLUMN] == model_id)
            & (daily_sentiment_scores_df[KNOWLEDGE_LEVEL_COLUMN] == knowledge_level),
            [TRADE_DATE_COLUMN, TICKER_COLUMN, SENTIMENT_SCORE_COLUMN],
        ]
        model_daily_signal_panel_df = prices_with_forward_returns_df.merge(
            model_daily_sentiment_scores_df,
            left_on=[DATE_COLUMN, TICKER_COLUMN],
            right_on=[TRADE_DATE_COLUMN, TICKER_COLUMN],
            how="left",
            validate="one_to_one",
        )

        # A missing score means no headline signal exists for that company-date, so the position remains zero
        model_daily_signal_panel_df[SENTIMENT_SCORE_COLUMN] = (
            model_daily_signal_panel_df[SENTIMENT_SCORE_COLUMN].fillna(0.0)
        )
        # Convert into a trading position from the mean value using `np.sign`.
        # https://numpy.org/doc/stable/reference/generated/numpy.sign.html
        model_daily_signal_panel_df[POSITION_COLUMN] = np.sign(
            model_daily_signal_panel_df[SENTIMENT_SCORE_COLUMN]
        ).astype(int)
        # Add the model identifier to each row
        model_daily_signal_panel_df[MODEL_ID_COLUMN] = model_id
        # Add the model knowledge condition to each row
        model_daily_signal_panel_df[KNOWLEDGE_LEVEL_COLUMN] = knowledge_level
        # Store the complete company day dataset under its display name. The values here will be later passed to the backtest for each model condition
        daily_signal_panels_by_model[model_name] = model_daily_signal_panel_df

        # Include the columns needed for the daily signals output
        daily_signal_dataframes.append(
            model_daily_signal_panel_df[
                [
                    DATE_COLUMN,
                    TICKER_COLUMN,
                    MODEL_ID_COLUMN,
                    KNOWLEDGE_LEVEL_COLUMN,
                    SENTIMENT_SCORE_COLUMN,
                    POSITION_COLUMN,
                ]
            ]
        )
        # Print the merged data before missing scores are replaced with zero
        print_single_model_daily_signal(model_daily_signal_panel_df, model_name)

    return daily_signal_panels_by_model, pd.concat(
        daily_signal_dataframes,
        ignore_index=True,
    )

def print_single_model_daily_signal(model_daily_signal_panel_df: pd.DataFrame, model_name: str) -> None:
    """Print a summary and sample of company day signals for one model condition."""

    matched_signal_mask = model_daily_signal_panel_df[SENTIMENT_SCORE_COLUMN].notna()

    print(
        f"\n---------------------------------Daily company-day signal preview:---------------------------------"
    )
    print(f"\nDaily signal panel for {model_name}")
    print(f"Total company day rows: {len(model_daily_signal_panel_df):,}")
    print(f"Rows with sentiment signals: {matched_signal_mask.sum():,}")

    print(
        model_daily_signal_panel_df.loc[
            matched_signal_mask,
            [
                DATE_COLUMN,
                TICKER_COLUMN,
                OPEN_COLUMN,
                NEXT_OPEN_COLUMN,
                FORWARD_RETURN_COLUMN,
                SENTIMENT_SCORE_COLUMN,
            ],
        ]
        .head(10)
        .to_string(index=False)
    )

def run_backtest_strategy(daily_signal_panel_df, strategy) -> pd.DataFrame:
    """Apply one strategy to a model's daily company-level signals."""

    # Create a new dataframe with sorted rows by date for the position turnover calculations. A copy is done as to not mutate the dataframe coming from the parameter
    company_backtest_results_df = daily_signal_panel_df.copy().sort_values(
        [TICKER_COLUMN, DATE_COLUMN]
    )

    # Long-only uses positive signals and otherwise remains in cash. Long-short also takes a short position for negative signals.
    # Replace the values below zero with zero using `clip(lower=0)` for LONG_ONLY because negative positions are prohibited. This means that a negative number signal (like, -1) will be retained as cash of 0.
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.clip.html

    # In a nutshell:
    # - long-only: negative signals become cash
    # - long-short: negative signals become short positions.
    # https://www.investor.gov/introduction-investing/investing-basics/how-stock-markets-work/stock-purchases-and-sales-long-and

    # The implementation here is also backed up by Suenot (no date):
    # https://github.com/suenot/061-llm-alpha-mining
    # Specifically the `04: Backtesting LLM Alphas` section in the README file
    # suenot (no date) 061-llm-alpha-mining [source code]. GitHub. Available at: https://github.com/suenot/061-llm-alpha-mining (Accessed: 20 August 2026).
    if strategy == LONG_ONLY:
        company_backtest_results_df[STRATEGY_POSITION_COLUMN] = (
            company_backtest_results_df[POSITION_COLUMN].clip(lower=0)
        )
    # Long-short permits both positive and negative positions, so retain
    elif strategy == LONG_SHORT:
        company_backtest_results_df[STRATEGY_POSITION_COLUMN] = (
            company_backtest_results_df[POSITION_COLUMN]
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # A position cannot be tested when the following opening price is unavailable.
    company_backtest_results_df.loc[
        company_backtest_results_df[FORWARD_RETURN_COLUMN].isna(),
        STRATEGY_POSITION_COLUMN,
    ] = 0

    # Get the previous trading position separately for each company
    previous_strategy_position = company_backtest_results_df.groupby(TICKER_COLUMN)[
        STRATEGY_POSITION_COLUMN
    ].shift(1) # Use `shift(1)` to move each company’s positions down by one row so the current position can be compared with its position on the previous available trading date
    # The first position for each company has no previous value, so assume that the strategy started in cash with a position of zero
    previous_strategy_position = previous_strategy_position.fillna(0)
    # Calculate how much the position changed. The absolute value makes both entries and exits positive. For example, 0 to 1 gives turnover of 1, 1 to 0 gives 1, and -1 to 1 gives 2
    company_backtest_results_df[TURNOVER_COLUMN] = (
        company_backtest_results_df[STRATEGY_POSITION_COLUMN]
        - previous_strategy_position
    ).abs()

    # Multiplying the position by the following open-to-open return makes a short position profitable when the price falls.
    company_backtest_results_df[GROSS_RETURN_COLUMN] = company_backtest_results_df[
        STRATEGY_POSITION_COLUMN
    ] * company_backtest_results_df[FORWARD_RETURN_COLUMN].fillna(0)
    # Divide basis points by 10,000 to show the transaction cost as a decimal return, then deduct it from the gross return.
    company_backtest_results_df[TRANSACTION_COST_COLUMN] = (
        company_backtest_results_df[TURNOVER_COLUMN]
        * convert_basis_points_to_decimal(TRANSACTION_COST_BASIS_POINTS)
    )
    company_backtest_results_df[NET_RETURN_COLUMN] = (
        company_backtest_results_df[GROSS_RETURN_COLUMN]
        - company_backtest_results_df[TRANSACTION_COST_COLUMN]
    )

    # An active company day has a non-zero position and an available following-day return.
    active_trade_mask = (
        company_backtest_results_df[STRATEGY_POSITION_COLUMN].ne(0)
        & company_backtest_results_df[FORWARD_RETURN_COLUMN].notna()
    )
    company_backtest_results_df[ACTIVE_TRADE_COLUMN] = active_trade_mask.astype(int)
    company_backtest_results_df[CORRECT_TRADE_COLUMN] = (
        active_trade_mask
        & (
            company_backtest_results_df[STRATEGY_POSITION_COLUMN]
            * company_backtest_results_df[FORWARD_RETURN_COLUMN]
            > 0
        )
    ).astype(int)

    # Each company has a fixed 10% portfolio weight. Unused allocations remain cash.
    company_count = company_backtest_results_df[TICKER_COLUMN].nunique()
    daily_portfolio_results_df = (
        company_backtest_results_df.groupby(DATE_COLUMN, as_index=False)
        .agg(
            gross_return=(GROSS_RETURN_COLUMN, "sum"),
            transaction_cost=(TRANSACTION_COST_COLUMN, "sum"),
            net_return=(NET_RETURN_COLUMN, "sum"),
            active_trades=(ACTIVE_TRADE_COLUMN, "sum"),
            correct_trades=(CORRECT_TRADE_COLUMN, "sum"),
        )
        .sort_values(DATE_COLUMN)
    )
    # Daily company returns are summed and divided by ten, giving each company an effective 10% portfolio weight.
    # Where no position is taken for a company, its contribution is zero and the unused allocation is treated as remaining in cash.
    daily_portfolio_results_df[
        [GROSS_RETURN_COLUMN, TRANSACTION_COST_COLUMN, NET_RETURN_COLUMN]
    ] /= company_count

    # Compound the daily gross and net returns to produce both cumulative return paths
    daily_portfolio_results_df[CUMULATIVE_GROSS_RETURN_COLUMN] = (
        1 + daily_portfolio_results_df[GROSS_RETURN_COLUMN]
    ).cumprod() - 1
    daily_portfolio_results_df[CUMULATIVE_RETURN_COLUMN] = (
        1 + daily_portfolio_results_df[NET_RETURN_COLUMN]
    ).cumprod() - 1

    return daily_portfolio_results_df


def evaluate_model_backtests(daily_signal_panels_by_model):
    """Evaluate both trading rules for every sentiment model."""

    backtest_result_list: list[BacktestResult] = []
    daily_backtest_results_by_model_and_strategy = {}
    bootstrap_seed = RANDOM_SEED

    # Run both trading strategies independently for each of the four model variants.
    for model_name, model_id, knowledge_level in MODEL_VARIANTS:
        for strategy in STRATEGIES:
            print(
                "---------------------------------Backtest being evaluated:---------------------------------"
            )
            print(f"Evaluating {model_name} with the {strategy} strategy")

            daily_backtest_results_df = run_backtest_strategy(
                daily_signal_panels_by_model[model_name],
                strategy,
            )
            total_active_company_days = int(
                daily_backtest_results_df[ACTIVE_TRADES_COLUMN].sum()
            )
            correct_active_company_days = int(
                daily_backtest_results_df[CORRECT_TRADES_COLUMN].sum()
            )
            backtest_confidence_intervals = calculate_backtest_confidence_intervals(
                daily_backtest_results_df,
                bootstrap_seed,
            )
            bootstrap_seed += 1

            # Hit rate is the proportion of active company-days for which the position direction matched the following return direction.
            hit_rate = (
                correct_active_company_days / total_active_company_days if total_active_company_days else 0.0
            )

            print(
                "---------------------------------Hit rate calculation result:---------------------------------"
            )
            print(
                f"Hit rate calculation: {correct_active_company_days:,} correct active company-days "
                f"/ {total_active_company_days:,} total active company-days "
                f"= {hit_rate:,}"
            )

            backtest_result = BacktestResult(
                model=model_name,
                model_id=model_id,
                knowledge_level=knowledge_level,
                strategy=strategy,
                Sharpe_ratio=calculate_annualised_sharpe_ratio(
                    daily_backtest_results_df[NET_RETURN_COLUMN]
                ),
                Sharpe_ci_lower=backtest_confidence_intervals[SHARPE_CI_LOWER_COLUMN],
                Sharpe_ci_upper=backtest_confidence_intervals[SHARPE_CI_UPPER_COLUMN],
                hit_rate_ci_lower=backtest_confidence_intervals[
                    HIT_RATE_CI_LOWER_COLUMN
                ],
                hit_rate_ci_upper=backtest_confidence_intervals[
                    HIT_RATE_CI_UPPER_COLUMN
                ],
                hit_rate=hit_rate,
                cumulative_gross_return=daily_backtest_results_df[
                    CUMULATIVE_GROSS_RETURN_COLUMN
                ].iloc[-1],
                cumulative_return=daily_backtest_results_df[
                    CUMULATIVE_RETURN_COLUMN
                ].iloc[-1],
                active_trades=total_active_company_days,
                trading_days=len(daily_backtest_results_df),
                transaction_cost_bps=TRANSACTION_COST_BASIS_POINTS,
            )
            # Use `.3f` to display to 3 decimal places 
            print(
                f"Annualised Sharpe ratio: "
                f"{backtest_result[SHARPE_RATIO_COLUMN]:.3f}"
            )
            # Use `.2%` is used to convert values to percentages and 2 decimal places
            print(f"Hit rate: {backtest_result[HIT_RATE_COLUMN]:.2%}")
            print(
                f"Cumulative net return: "
                f"{backtest_result[CUMULATIVE_RETURN_COLUMN]:.2%}"
            )
            print(f"Active company-days: " f"{backtest_result[ACTIVE_TRADES_COLUMN]:,}")
            backtest_result_list.append(backtest_result)
            daily_backtest_results_by_model_and_strategy[(model_name, strategy)] = (
                daily_backtest_results_df
            )

    return (
        pd.DataFrame(backtest_result_list),
        daily_backtest_results_by_model_and_strategy,
    )


def create_backtest_table_figure(backtest_results_df) -> pd.DataFrame:
    """Create a backtest table figure."""

    backtest_report_df = backtest_results_df[
        [
            MODEL_COLUMN,
            STRATEGY_COLUMN,
            SHARPE_RATIO_COLUMN,
            HIT_RATE_COLUMN,
            CUMULATIVE_GROSS_RETURN_COLUMN,
            CUMULATIVE_RETURN_COLUMN,
            ACTIVE_TRADES_COLUMN,
        ]
    ].copy()

    # Format the numeric results for the concise dissertation summary table.
    backtest_report_df[STRATEGY_COLUMN] = backtest_report_df[
        STRATEGY_COLUMN
    ].str.capitalize()
    backtest_report_df[SHARPE_RATIO_COLUMN] = backtest_report_df[
        SHARPE_RATIO_COLUMN
    ].round(3)
    backtest_report_df[HIT_RATE_COLUMN] = (
        backtest_report_df[HIT_RATE_COLUMN] * 100
    ).round(2)
    backtest_report_df.insert(
        backtest_report_df.columns.get_loc(CUMULATIVE_RETURN_COLUMN),
        TRANSACTION_COST_DRAG_COLUMN,
        backtest_report_df[CUMULATIVE_GROSS_RETURN_COLUMN]
        - backtest_report_df[CUMULATIVE_RETURN_COLUMN],
    )
    backtest_report_df[CUMULATIVE_GROSS_RETURN_COLUMN] = (
        backtest_report_df[CUMULATIVE_GROSS_RETURN_COLUMN] * 100
    ).round(2)
    backtest_report_df[TRANSACTION_COST_DRAG_COLUMN] = (
        backtest_report_df[TRANSACTION_COST_DRAG_COLUMN] * 100
    ).round(2)
    backtest_report_df[CUMULATIVE_RETURN_COLUMN] = (
        backtest_report_df[CUMULATIVE_RETURN_COLUMN] * 100
    ).round(2)

    return backtest_report_df.rename(
        columns={
            MODEL_COLUMN: CONDITION_REPORT_COLUMN,
            STRATEGY_COLUMN: STRATEGY_REPORT_COLUMN,
            SHARPE_RATIO_COLUMN: SHARPE_RATIO_REPORT_COLUMN,
            HIT_RATE_COLUMN: HIT_RATE_REPORT_COLUMN,
            CUMULATIVE_GROSS_RETURN_COLUMN: (
                CUMULATIVE_GROSS_RETURN_REPORT_COLUMN
            ),
            TRANSACTION_COST_DRAG_COLUMN: TRANSACTION_COST_DRAG_REPORT_COLUMN,
            CUMULATIVE_RETURN_COLUMN: CUMULATIVE_RETURN_REPORT_COLUMN,
            ACTIVE_TRADES_COLUMN: ACTIVE_TRADES_REPORT_COLUMN,
        }
    )


def create_backtest_summary_figure(backtest_report_df, output_path):
    """Create the backtest results as a figure."""

    # Convert the report values to their final display strings without modifying the saved numeric table.
    formatted_backtest_report_df = backtest_report_df.copy()
    formatted_backtest_report_df[SHARPE_RATIO_REPORT_COLUMN] = (
        formatted_backtest_report_df[SHARPE_RATIO_REPORT_COLUMN].map(
            lambda value: f"{value:.3f}"
        )
    )
    formatted_backtest_report_df[HIT_RATE_REPORT_COLUMN] = formatted_backtest_report_df[
        HIT_RATE_REPORT_COLUMN
    ].map(lambda value: f"{value:.2f}%")
    formatted_backtest_report_df[CUMULATIVE_GROSS_RETURN_REPORT_COLUMN] = (
        formatted_backtest_report_df[CUMULATIVE_GROSS_RETURN_REPORT_COLUMN].map(
            lambda value: f"{value:.2f}%"
        )
    )
    formatted_backtest_report_df[TRANSACTION_COST_DRAG_REPORT_COLUMN] = (
        formatted_backtest_report_df[TRANSACTION_COST_DRAG_REPORT_COLUMN].map(
            lambda value: f"{value:.2f}"
        )
    )
    formatted_backtest_report_df[CUMULATIVE_RETURN_REPORT_COLUMN] = (
        formatted_backtest_report_df[CUMULATIVE_RETURN_REPORT_COLUMN].map(
            lambda value: f"{value:.2f}%"
        )
    )
    formatted_backtest_report_df[ACTIVE_TRADES_REPORT_COLUMN] = (
        formatted_backtest_report_df[ACTIVE_TRADES_REPORT_COLUMN].map(
            lambda value: f"{value:,.0f}"
        )
    )

    figure_column_labels = [
        CONDITION_REPORT_COLUMN,
        STRATEGY_REPORT_COLUMN,
        "Sharpe\nratio",
        "Hit rate\n(%)",
        "Cumulative gross\nreturn (%)",
        "Transaction-cost drag\n(percentage points)",
        "Cumulative net\nreturn (%)",
        "Active\ncompany-days",
    ]

    plt.rcParams.update(get_figure_font_settings(9))
    figure, axis = plt.subplots(figsize=(15, 3.2))
    axis.axis("off")
    figure_table = axis.table(
        cellText=formatted_backtest_report_df.to_numpy(),
        colLabels=figure_column_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.15, 0.10, 0.095, 0.095, 0.14, 0.16, 0.14, 0.12],
        loc="center",
    )
    figure_table.auto_set_font_size(False)
    figure_table.set_fontsize(8)
    figure_table.scale(1, 1.45)

    for (row_index, column_index), cell in figure_table.get_celld().items():
        cell.set_edgecolor("#D6DEDA")
        if row_index == 0:
            cell.set_facecolor(FIGURE_NAVY)
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F1F5F3")
        if row_index > 0 and column_index in (0, 1):
            cell.set_text_props(ha="left")

    axis.set_title("Backtest performance summary", fontsize=13, pad=6)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(figure)


def create_cumulative_return_figure(
    daily_backtest_results_by_model_and_strategy,
    output_path,
) -> None:
    """Create the cumulative net-return curves for both trading strategies."""

    plt.rcParams.update(get_figure_font_settings(10))
    colours = [
        FIGURE_NAVY,
        FIGURE_DEEP_BLUE,
        FIGURE_DARK_TEAL,
        FIGURE_PRIMARY_TEAL,
    ]
    line_styles = ["-", "--", "-.", ":"]
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for axis, strategy in zip(axes, STRATEGIES):
        for colour, line_style, (model_name, _, _) in zip(
            colours,
            line_styles,
            MODEL_VARIANTS,
        ):
            daily_backtest_results_df = daily_backtest_results_by_model_and_strategy[
                (model_name, strategy)
            ]
            axis.plot(
                pd.to_datetime(daily_backtest_results_df[DATE_COLUMN]),
                daily_backtest_results_df[CUMULATIVE_RETURN_COLUMN] * 100,
                label=model_name,
                color=colour,
                linestyle=line_style,
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


def create_backtest_metrics_figure(backtest_results_df, output_path):
    """Create the Sharpe-ratio and hit-rate comparisons figure with confidence intervals."""

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colours = {LONG_ONLY: FIGURE_PRIMARY_TEAL, LONG_SHORT: FIGURE_DEEP_BLUE}
    positions = np.arange(len(MODEL_VARIANTS))
    width = 0.36

    for offset, strategy in [(-width / 2, LONG_ONLY), (width / 2, LONG_SHORT)]:
        strategy_results_df = backtest_results_df.loc[
            backtest_results_df[STRATEGY_COLUMN] == strategy
        ].set_index(MODEL_COLUMN)
        strategy_results_df = strategy_results_df.loc[
            [model_variant[0] for model_variant in MODEL_VARIANTS]
        ]

        # Plot each metric with asymmetric error bars extending to its bootstrap confidence limits.
        for axis, metric, lower_name, upper_name in [
            (
                axes[0],
                SHARPE_RATIO_COLUMN,
                SHARPE_CI_LOWER_COLUMN,
                SHARPE_CI_UPPER_COLUMN,
            ),
            (
                axes[1],
                HIT_RATE_COLUMN,
                HIT_RATE_CI_LOWER_COLUMN,
                HIT_RATE_CI_UPPER_COLUMN,
            ),
        ]:
            metric_values = strategy_results_df[metric].to_numpy()
            confidence_interval_error = np.vstack(
                [
                    metric_values - strategy_results_df[lower_name].to_numpy(),
                    strategy_results_df[upper_name].to_numpy() - metric_values,
                ]
            )
            axis.bar(
                positions + offset,
                metric_values,
                width,
                yerr=confidence_interval_error,
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
            [model_variant[0] for model_variant in MODEL_VARIANTS],
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

    headlines_aligned_prices_df = pd.read_csv(
        PROCESSED_DATA_DIR / HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME
    )
    yfinance_prices_df = pd.read_csv(RAW_DATA_DIR / YFINANCE_PRICES_CSV_FILE_NAME)
    model_predictions_df = load_FinBERT_and_DK_CoT_predictions()

    daily_signal_panels_by_model, daily_signals_df = (
        create_daily_sentiment_signal_panels(
            headlines_aligned_prices_df,
            yfinance_prices_df,
            model_predictions_df,
        )
    )
    (
        backtest_results_df,
        daily_backtest_results_by_model_and_strategy,
    ) = evaluate_model_backtests(daily_signal_panels_by_model)
    backtest_report_df = create_backtest_table_figure(backtest_results_df)

    daily_signals_file_path = PROCESSED_DATA_DIR / DAILY_SIGNALS_CSV_FILE_NAME
    backtest_results_file_path = PROCESSED_DATA_DIR / BACKTEST_RESULTS_CSV_FILE_NAME
    backtest_summary_file_path = tables_directory / BACKTEST_SUMMARY_CSV_FILE_NAME
    daily_signals_df.to_csv(daily_signals_file_path, index=False)
    backtest_results_df.to_csv(backtest_results_file_path, index=False)
    backtest_report_df.to_csv(backtest_summary_file_path, index=False)

    create_cumulative_return_figure(
        daily_backtest_results_by_model_and_strategy,
        figures_directory / BACKTEST_CUMULATIVE_RETURNS_PNG_FILE_NAME,
    )
    create_backtest_metrics_figure(
        backtest_results_df,
        figures_directory / BACKTEST_METRICS_PNG_FILE_NAME,
    )
    create_backtest_summary_figure(
        backtest_report_df,
        figures_directory / BACKTEST_SUMMARY_TABLE_PNG_FILE_NAME,
    )


if __name__ == "__main__":
    main()
