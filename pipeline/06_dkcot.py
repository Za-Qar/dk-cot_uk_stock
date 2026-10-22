import re
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from uk_dkcot.config import (
    Company,
    DKCOT_KNOWLEDGE_LEVELS,
    FIRM_KNOWLEDGE_LEVEL,
    NO_KNOWLEDGE_LEVEL,
    DKCOT_MODEL_NAME,
    DKCOT_PREDICTIONS_CSV_FILE_NAME,
    HEADLINE_ID_COLUMN,
    HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME,
    KNOWLEDGE_LEVEL_COLUMN,
    LABEL_PRED_COLUMN,
    MODEL_ID_COLUMN,
    NEUTRAL_LABEL,
    SECTOR_KNOWLEDGE_LEVEL,
    PROCESSED_DATA_DIR,
    load_companies,
)

# Qwen2.5-3B-Instruct is an LLM that maintains a KV (key-value) cache while producing tokens. Processing more prompts increases GPU memory use. Therefore, a batch size of 16 was selected (as opposed to 32 like FinBERT) considering it requires more GPU memory (from the NVIDIA GeForce RTX 3060 used for this research)
# https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# https://huggingface.co/docs/transformers/kv_cache
BATCH_SIZE = 16
# The number 128 is the maximum response length for QWEN through the Hugging Face Transformer library. The output is a JSON object containing 4 fields (used by Chen et al. (2025): https://github.com/XMUT-Service-Computing-Group/GLM-Sentiment/blob/main/GLM.py), so 128 tokens provides decent space for the response. However, this number is a ceiling and QWEN can stop earlier. The number is to prevent very long outputs. This number is also for the hugging face transformer used.
# https://huggingface.co/docs/transformers/main_classes/text_generation
MAX_NEW_TOKENS = 128
OUTPUT_COLUMNS = [
    HEADLINE_ID_COLUMN,
    MODEL_ID_COLUMN,
    KNOWLEDGE_LEVEL_COLUMN,
    LABEL_PRED_COLUMN,
    "raw_output",
    "parse_ok",
]


def build_knowledge(company, knowledge_level) -> str:
    """Return the controlled knowledge supplied for one treatment."""

    if knowledge_level == NO_KNOWLEDGE_LEVEL:
        return "No additional company or sector knowledge is supplied."

    if knowledge_level == SECTOR_KNOWLEDGE_LEVEL:
        return f"Sector: {company.sector}."

    if knowledge_level == FIRM_KNOWLEDGE_LEVEL:
        return (
            f"Sector: {company.sector}. "
            f"Main products or services: {company.products}. "
            f"Key business risks: {company.risks}."
        )


def build_prompt(headline, company, knowledge_level) -> str:
    """Build the UK headline adaptation of Chen et al.'s DK-CoT prompt."""

    knowledge = build_knowledge(company, knowledge_level)

    # This prompt was translated from the original Mandarin used by Chen et al. (2025) (https://github.com/XMUT-Service-Computing-Group/GLM-Sentiment/blob/main/GLM.py) to English with some adaptations discussed in the dissertation report. The translation was done by Chat GPT and cross-checked with Google translate (https://translate.google.com/)
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


def parse_sentiment(raw_output) -> tuple[str, bool]:
    """Extract a valid sentiment label from the model's JSON response."""

    # Regex to search through Qwen's response and return them as a type of list of string
    matches: list[str] = re.findall(
        r"[\"']?sentiment[\"']?\s*[:：]\s*[\"']?\s*(positive|neutral|negative)\b",
        raw_output,
        flags=re.IGNORECASE,
    )

    # Returns the final sentiment from Qwen's text response. An example of the response is:
    # {"Entity":"GSK","Event":"Profit increased","Impact":"Favourable","Sentiment":"positive"}
    # So in the case of the example above, a tuple is returned with ("positive", True)
    if matches:
        return matches[-1].lower(), True

    # Chen et al.'s (2025) code also defaults an unparseable response to neutral.
    return NEUTRAL_LABEL, False


def load_model():
    """Load the local instruction model on the GPU or CPU."""

    # Use CUDA if possible, otherwise default to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Qwen's tokenizer, which converts prompt text into token IDs and later converts the generated token IDs back into readable text.
    # https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
    tokeniser = AutoTokenizer.from_pretrained(
        DKCOT_MODEL_NAME,
    )
    # Many prompts are processed together in one batch, but may contain different numbers of tokens. This adds padding tokens to the beginning of shorter prompts so every prompt has the same length and still end with its real text. This is required for reliable batched generation
    # https://huggingface.co/docs/transformers/llm_tutorial#padding-side
    tokeniser.padding_side = "left"

    # Load the pretrained Qwen language model using half precision to reduce memory use. `low_cpu_mem_usage=True` reduces temporary memory use while the model is loading.
    model = (
        AutoModelForCausalLM.from_pretrained(
            DKCOT_MODEL_NAME,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )

    # This disables random sampling so Qwen selects the highest-probability token at each step. The reasoning behind this is explained in the dissertation report.
    # https://huggingface.co/docs/transformers/generation_strategies
    model.generation_config.do_sample = False

    return tokeniser, model


def generate_responses(prompts, tokenizer, model):
    """Generate one concise DK-CoT JSON response for each prompt."""

    INPUT_IDS_KEY = "input_ids"

    # This simulates a user speaking to QWEN. This is the expected chat structure for QWEN:
    # https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
    # https://huggingface.co/docs/transformers/chat_templating
    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        # Repeat formatting for every prompt from the `prompts` param passed into this def
        for prompt in prompts
    ]
    # Converts all formatted prompts into token numbers understood by Qwen
    model_inputs = tokenizer(
        chat_prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Disable gradient tracking because the model is generating responses rather than being trained. This reduces inference memory and processing overhead.
    # https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
    with torch.inference_mode():
        generated_tokens = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # This is used to enable greedy decoding: https://huggingface.co/docs/transformers/generation_strategies and https://huggingface.co/docs/transformers/v5.20.2/en/main_classes/text_generation
        )

    print(
        "---------------------------------Formatted prompt:---------------------------------"
    )
    print(chat_prompts[0])

    print(
        "---------------------------------Token tensor shape:---------------------------------"
    )
    print(model_inputs[INPUT_IDS_KEY].shape)

    # Obtain the padded input length. Qwen's returned sequences contain both the original prompt tokens and the newly generated response tokens.
    input_length = model_inputs[INPUT_IDS_KEY].shape[1]
    # Convert every generated response from token IDs back into readable text.
    return [
        tokenizer.decode(
            output_tokens[
                input_length:
            ],  # Remove the original prompt tokens and retain only Qwen's response
            skip_special_tokens=True,  # Remove Qwen's internal formatting tokens from the decoded response
        ).strip()  # Remove any whitespace from around the response
        for output_tokens in generated_tokens
    ]


def load_completed_keys(output_path):
    """
    Read completed predictions so an interrupted long run can resume. This  can be considered a non-functional requirement to save time and continue off from the last point of process if something were to go wrong with the processing. To load checkpoints, saving them (in the `save_checkpoint` def) is required.

    It follows the concept of the "Checkpoint/Restart Processing" outlined here: https://www.ibm.com/docs/en/zos/3.1.0?topic=checkpointrestart-processing
    """

    # If no prediction CSV exists, return an empty set because nothing has been completed
    if not output_path.exists():
        return set()

    # Load the existing predictions
    existing_df = pd.read_csv(output_path)

    # `zip` pairs values from the two columns:
    return set(
        zip(existing_df[HEADLINE_ID_COLUMN], existing_df[KNOWLEDGE_LEVEL_COLUMN])
    )


def save_checkpoint(predictions, output_path) -> None:
    """
    Append a small prediction batch without losing the CSV progess

    This follows the concept (design pattern) of the "Checkpoint/Restart Processing" outlined here: https://www.ibm.com/docs/en/zos/3.1.0?topic=checkpointrestart-processing
    """

    if not predictions:
        return

    # Convert the list of prediction dictionaries into a dataframe with the expected column order
    pd.DataFrame(predictions, columns=OUTPUT_COLUMNS).to_csv(
        output_path,
        mode="a",  # "a" is to append to the existing CSV instead of overwriting
        header=not output_path.exists(),
        index=False,
    )
    # Clear the list considering that the batch has been saved
    predictions.clear()


def main():
    """Generate DK-CoT predictions under all three knowledge treatments."""

    input_path = PROCESSED_DATA_DIR / HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME
    output_path = PROCESSED_DATA_DIR / DKCOT_PREDICTIONS_CSV_FILE_NAME

    headlines_aligned_prices_df = pd.read_csv(input_path)
    required_columns = [HEADLINE_ID_COLUMN, "ticker", "headline_text"]

    # Load the companies and assign them to a dictionary
    companies: [str, Company] = {
        company.ticker: company for company in load_companies()
    }

    # Check the checkpoint
    completed_keys = load_completed_keys(output_path)
    # Calculate the amount of DK-CoT predictions that should exist when processing is completed (this is used for the checkpoint logic)
    total_predictions = len(headlines_aligned_prices_df) * len(DKCOT_KNOWLEDGE_LEVELS)

    print(
        "---------------------------------Loading model:---------------------------------"
    )
    print(DKCOT_MODEL_NAME)

    tokenizer, model = load_model()

    # Instantiate an empty list to populate the predictions later on (note: this list is cleared in the `save_checkpoint` def after a successful batch save)
    predictions = []
    # Counts how many predictions were already saved before this current run started
    completed_predictions_count = len(completed_keys)

    headline_rows = list(headlines_aligned_prices_df.itertuples(index=False))

    for knowledge_level in DKCOT_KNOWLEDGE_LEVELS:
        # Select headlines that have not been processed for the current knowledge level
        pending_predictions_headline_rows = [
            row
            for row in headline_rows
            if (row.headline_id, knowledge_level) not in completed_keys
        ]

        # Process the pending headlines in batches of 16
        for start_index in range(0, len(pending_predictions_headline_rows), BATCH_SIZE):
            # Select the current group of up to 16 headline rows
            batch_rows = pending_predictions_headline_rows[
                start_index : start_index + BATCH_SIZE
            ]
            # Build one DK-CoT prompt for every headline in the current batch and assign to a list
            prompts = [
                build_prompt(
                    row.headline_text,
                    companies[row.ticker],
                    knowledge_level,
                )
                for row in batch_rows
            ]
            # Send the prompts to Qwen and obtain one text response per headline.
            qwen_generated_raw_outputs = generate_responses(prompts, tokenizer, model)

            # Pair each headline row with its corresponding Qwen response
            for row, raw_output in zip(batch_rows, qwen_generated_raw_outputs):
                # Extract the sentiment label and whether parsing succeeded
                label_pred, parse_ok = parse_sentiment(raw_output)

                # Append the completed prediction in memory
                predictions.append(
                    {
                        HEADLINE_ID_COLUMN: row.headline_id,
                        MODEL_ID_COLUMN: DKCOT_MODEL_NAME,
                        KNOWLEDGE_LEVEL_COLUMN: knowledge_level,
                        LABEL_PRED_COLUMN: label_pred,
                        "raw_output": raw_output,
                        "parse_ok": parse_ok,
                    }
                )

            # Update the number of completed predictions
            completed_predictions_count += len(batch_rows)
            # Append this batch to the CSV and clear the predictions list
            save_checkpoint(predictions, output_path)
            print(
                "---------------------------------Processed predictions count::---------------------------------"
            )
            print(
                f"Processed {completed_predictions_count:,} / "
                f"{total_predictions:,} predictions"
            )

    save_checkpoint(predictions, output_path)

    predictions_results_df = pd.read_csv(output_path)

    print(
        "---------------------------------Prediction counts by knowledge level:---------------------------------"
    )
    print(
        predictions_results_df.groupby([KNOWLEDGE_LEVEL_COLUMN, LABEL_PRED_COLUMN])
        .size()
        .to_string()
    )
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
