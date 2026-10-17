import json

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uk_dkcot.config import (
    FINBERT_MODEL_NAME,
    FINBERT_PREDICTIONS_CSV_FILE_NAME,
    HEADLINE_ID_COLUMN,
    HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME,
    KNOWLEDGE_LEVEL_COLUMN,
    LABEL_PRED_COLUMN,
    MODEL_ID_COLUMN,
    NO_KNOWLEDGE_LEVEL,
    PROCESSED_DATA_DIR,
    VALID_SENTIMENT_LABELS,
)

# The model's name comes from https://huggingface.co/ProsusAI/finbert
# A batch size of 32 was selected as a practical setting for inference. NVidia suggests the following (direct quote):
# "Choose the batch size and the number of inputs and outputs to be divisible by 4 (TF32) / 8(FP16) / 16 (INT8) to run efficiently on Tensor Cores. For best efficiency on A100, choose these parameters to be divisible by 32 (TF32) / 64 (FP16) / 128 (INT8")."
# Considering an NVidia GTX 3060 used to run this experiment, the number 32 was chosen.
# Source: https://docs.nvidia.com/deeplearning/performance/pdf/Optimizing-Linear-Fully-Connected-Layers-User-Guide.pdf
BATCH_SIZE = 32


def predict_headlines(headlines, tokeniser, model, device) -> pd.DataFrame :
    """Run FinBERT in batches and return one prediction per headline."""

    predictions = []

    # for-loop to split the headlines into batches of 32 and calculates the batch end to ensure that the processing does not exceed the dataframe length
    for start_index in range(0, len(headlines), BATCH_SIZE):
        end_index = min(start_index + BATCH_SIZE, len(headlines))
        batch = headlines.iloc[start_index:end_index]

        # The `truncation=True` is used to cut anything exceeding the 512 tokens which is the maximum number of input finBERT can accept per input
        # https://huggingface.co/ProsusAI/finbert/blob/main/config.json 
        # https://huggingface.co/docs/transformers/main_classes/tokenizer
        # # Headlines contain different numbers of tokens, but a batch must form one fixed-size tensor. Padding adds special padding tokens to shorter headlines. Therefore, every headline matches the longest sequence in the current batch. Hugging Face states: "Batched inputs are often different lengths, so they can’t be converted to fixed-size tensors.". That is why padding is used.
        # Source: https://huggingface.co/docs/transformers/main/pad_truncation
        model_inputs = tokeniser(
            batch["headline_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            # Pass the tokenised headlines into FinBERT and obtain its raw class scores.
            # logits are finBERT's raw scores: https://huggingface.co/docs/transformers/model_doc/bert
            logits = model(**model_inputs).logits
            print("---------------------------------FinBERT's raw scores (logits):---------------------------------")
            print(logits)
            # Use `softmax` to convert FinBERT's logits to a number between 0 and 1 and also "sum to 1".
            # Source: https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html
            # `.cpu()` moves the results from the GPU to system memory so they can be converted into Python values and saved as JSON.
            # https://docs.pytorch.org/docs/2.14/cpu.html
            probabilities = torch.softmax(logits, dim=1).cpu()

        # Process each headline alongside its three predicted probabilities
        for row, scores in zip(batch.itertuples(index=False), probabilities):
            predicted_index = int(scores.argmax())
            # Convert the highest-probability class index into its sentiment label
            predicted_label = model.config.id2label[predicted_index].lower()
            # Create a dictionary containing the probability assigned to every class
            # The `id2label` is loaded from finBERT's `config.json` to map:
            # - 0 to "positive"
            # - 1 to "negative"
            # - 2 to "neutral"
            # Link to finBERT's config.json file: https://huggingface.co/ProsusAI/finbert/blob/main/config.json
            # Therefore, this converts the highest-probability class index into its sentiment label using id2label
            finBERT_score_output = {
                model.config.id2label[index].lower(): round(float(score), 6)
                for index, score in enumerate(scores)
            }
            
            # Store the prediction the relevant data information
            predictions.append(
                {
                    HEADLINE_ID_COLUMN: row.headline_id,
                    MODEL_ID_COLUMN: FINBERT_MODEL_NAME,
                    KNOWLEDGE_LEVEL_COLUMN: NO_KNOWLEDGE_LEVEL,
                    LABEL_PRED_COLUMN: predicted_label,
                    "raw_output": json.dumps(finBERT_score_output, sort_keys=True),
                    "parse_ok": True,
                }
            )

        print(f"Processed {end_index:,} / {len(headlines):,} headlines")

    return pd.DataFrame(predictions)


def main():
    """Generate and save FinBERT predictions for every cleaned headline."""

    input_path = PROCESSED_DATA_DIR / HEADLINES_ALIGNED_PRICES_CSV_FILE_NAME
    output_path = PROCESSED_DATA_DIR / FINBERT_PREDICTIONS_CSV_FILE_NAME

    headlined_aligned_prices_df = pd.read_csv(input_path)

    # Choose "cuda"  if it is available, otherwise default to "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {FINBERT_MODEL_NAME}")

    # Create a single instance of the tokeniser and pass into the `predict_headlines` definition instead of creating multiple instances within the `predict_headlines` definition
    tokeniser = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model.to(device)
    model.eval()

    finBERT_predictions_df = predict_headlines(
        headlined_aligned_prices_df,
        tokeniser,
        model,
        device,
    )

    # Display the first five predictions in the terminal
    print("---------------------------------Prediction sample:---------------------------------")
    print(finBERT_predictions_df.head().to_string(index=False))

    # Ensure both dataframes are the same size
    assert len(finBERT_predictions_df) == len(headlined_aligned_prices_df)
    # Ensure that the below finBERT_predictions_df is valid
    assert set(finBERT_predictions_df[LABEL_PRED_COLUMN]).issubset(VALID_SENTIMENT_LABELS)

    finBERT_predictions_df.to_csv(output_path, index=False)

    print("Prediction counts:")
    print(finBERT_predictions_df[LABEL_PRED_COLUMN].value_counts().to_string())
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
