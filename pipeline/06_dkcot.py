import re

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from uk_dkcot.config import PROCESSED_DATA_DIR, load_companies


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
KNOWLEDGE_LEVELS = ("none", "sector", "firm")
BATCH_SIZE = 16
MAX_NEW_TOKENS = 128
OUTPUT_COLUMNS = [
    "headline_id",
    "model_id",
    "knowledge_level",
    "label_pred",
    "raw_output",
    "parse_ok",
]


def build_knowledge(company, knowledge_level):
    """Return the controlled knowledge supplied for one treatment."""

    if knowledge_level == "none":
        return "No additional company or sector knowledge is supplied."

    if knowledge_level == "sector":
        return f"Sector: {company.sector}."

    if knowledge_level == "firm":
        return (
            f"Sector: {company.sector}. "
            f"Main products or services: {company.products}. "
            f"Key business risks: {company.risks}."
        )

    raise ValueError(f"Unknown knowledge level: {knowledge_level}")


def build_prompt(headline, company, knowledge_level):
    """Build the UK headline adaptation of Chen et al.'s DK-CoT prompt."""

    knowledge = build_knowledge(company, knowledge_level)

    return f"""You are an expert in UK financial-news sentiment classification.

Classify the likely sentiment of the headline for investors in the named company:
- positive: likely favourable for the company or its investors
- negative: likely unfavourable for the company or its investors
- neutral: no clear favourable or unfavourable implication

Use the same reasoning order for every headline:
1. Entity: identify the company discussed.
2. Event: identify what happened.
3. Impact: explain the likely implication for the company or its investors.
4. Sentiment: select positive, neutral, or negative.

Examples:
Headline: "Example plc reports profit above market expectations."
Analysis: {{"Entity":"Example plc","Event":"Profit exceeded expectations","Impact":"The result is likely favourable for investors","Sentiment":"positive"}}

Headline: "The regulator rejects Example plc's key product application."
Analysis: {{"Entity":"Example plc","Event":"A key application was rejected","Impact":"The decision may reduce expected future revenue","Sentiment":"negative"}}

Headline: "Example plc announces the date of its annual general meeting."
Analysis: {{"Entity":"Example plc","Event":"The annual meeting date was announced","Impact":"The administrative announcement has no clear directional effect","Sentiment":"neutral"}}

Target company: {company.company_name}
Knowledge treatment: {knowledge_level}
Supplied knowledge: {knowledge}
Headline: "{headline}"

Return only one JSON object with the fields Entity, Event, Impact, and Sentiment.
"""


def parse_sentiment(raw_output):
    """Extract a valid sentiment label from the model's JSON response."""

    matches = re.findall(
        r"[\"']?sentiment[\"']?\s*[:：]\s*[\"']?\s*(positive|neutral|negative)\b",
        raw_output,
        flags=re.IGNORECASE,
    )

    if matches:
        return matches[-1].lower(), True

    # Chen et al.'s code also defaults an unparseable response to neutral.
    return "neutral", False


def load_model():
    """Load the local instruction model on the NVIDIA GPU."""

    if not torch.cuda.is_available():
        raise RuntimeError("DK-CoT requires a CUDA-enabled PyTorch installation.")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    # Greedy decoding makes repeated experiment runs deterministic.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return tokenizer, model


def generate_responses(prompts, tokenizer, model):
    """Generate one concise DK-CoT JSON response for each prompt."""

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    model_inputs = tokenizer(
        chat_prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    input_length = model_inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(
            output_tokens[input_length:],
            skip_special_tokens=True,
        ).strip()
        for output_tokens in generated_tokens
    ]


def load_completed_keys(output_path):
    """Read completed predictions so an interrupted long run can resume."""

    if not output_path.exists():
        return set()

    existing_df = pd.read_csv(output_path)
    assert list(existing_df.columns) == OUTPUT_COLUMNS
    assert set(existing_df["model_id"]) == {MODEL_NAME}

    return set(
        zip(existing_df["headline_id"], existing_df["knowledge_level"])
    )


def save_checkpoint(predictions, output_path):
    """Append a small prediction batch without discarding earlier work."""

    if not predictions:
        return

    pd.DataFrame(predictions, columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a",
        header=not output_path.exists(),
        index=False,
    )
    predictions.clear()


def main():
    """Generate DK-CoT predictions under all three knowledge treatments."""

    input_path = PROCESSED_DATA_DIR / "headlines_aligned_prices.csv"
    output_path = PROCESSED_DATA_DIR / "dkcot_predictions.csv"

    headlines_df = pd.read_csv(input_path)
    required_columns = ["headline_id", "ticker", "headline_text"]
    assert not headlines_df[required_columns].isna().any().any()
    assert headlines_df["headline_id"].is_unique

    companies = {company.ticker: company for company in load_companies()}
    assert set(headlines_df["ticker"]).issubset(companies)

    completed_keys = load_completed_keys(output_path)
    total_predictions = len(headlines_df) * len(KNOWLEDGE_LEVELS)

    print(f"Loading model: {MODEL_NAME}")
    tokenizer, model = load_model()
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    predictions = []
    completed_count = len(completed_keys)

    all_rows = list(headlines_df.itertuples(index=False))

    for knowledge_level in KNOWLEDGE_LEVELS:
        pending_rows = [
            row
            for row in all_rows
            if (row.headline_id, knowledge_level) not in completed_keys
        ]

        for start_index in range(0, len(pending_rows), BATCH_SIZE):
            batch_rows = pending_rows[start_index:start_index + BATCH_SIZE]
            prompts = [
                build_prompt(
                    row.headline_text,
                    companies[row.ticker],
                    knowledge_level,
                )
                for row in batch_rows
            ]
            raw_outputs = generate_responses(prompts, tokenizer, model)

            for row, raw_output in zip(batch_rows, raw_outputs):
                label_pred, parse_ok = parse_sentiment(raw_output)
                predictions.append(
                    {
                        "headline_id": row.headline_id,
                        "model_id": MODEL_NAME,
                        "knowledge_level": knowledge_level,
                        "label_pred": label_pred,
                        "raw_output": raw_output,
                        "parse_ok": parse_ok,
                    }
                )

            completed_count += len(batch_rows)
            save_checkpoint(predictions, output_path)
            print(
                f"Processed {completed_count:,} / "
                f"{total_predictions:,} predictions"
            )

    save_checkpoint(predictions, output_path)

    results_df = pd.read_csv(output_path)
    valid_labels = {"positive", "neutral", "negative"}
    assert len(results_df) == total_predictions
    assert not results_df.duplicated(["headline_id", "knowledge_level"]).any()
    assert set(results_df["knowledge_level"]) == set(KNOWLEDGE_LEVELS)
    assert set(results_df["label_pred"]).issubset(valid_labels)

    print("Prediction counts by knowledge level:")
    print(
        results_df.groupby(["knowledge_level", "label_pred"])
        .size()
        .to_string()
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
