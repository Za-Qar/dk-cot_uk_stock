import json

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from uk_dkcot.config import PROCESSED_DATA_DIR


MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32


def predict_headlines(headlines, tokenizer, model, device):
    """Run FinBERT in batches and return one prediction per headline."""

    predictions = []

    for start_index in range(0, len(headlines), BATCH_SIZE):
        end_index = min(start_index + BATCH_SIZE, len(headlines))
        batch = headlines.iloc[start_index:end_index]

        model_inputs = tokenizer(
            batch["headline_text"].tolist(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            logits = model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=1).cpu()

        for row, scores in zip(batch.itertuples(index=False), probabilities):
            predicted_index = int(scores.argmax())
            predicted_label = model.config.id2label[predicted_index].lower()
            score_output = {
                model.config.id2label[index].lower(): round(float(score), 6)
                for index, score in enumerate(scores)
            }

            predictions.append(
                {
                    "headline_id": row.headline_id,
                    "model_id": MODEL_NAME,
                    "knowledge_level": "none",
                    "label_pred": predicted_label,
                    "raw_output": json.dumps(score_output, sort_keys=True),
                    "parse_ok": True,
                }
            )

        print(f"Processed {end_index:,} / {len(headlines):,} headlines")

    return pd.DataFrame(predictions)


def main():
    """Generate and save FinBERT predictions for every cleaned headline."""

    input_path = PROCESSED_DATA_DIR / "headlines_aligned_prices.csv"
    output_path = PROCESSED_DATA_DIR / "finbert_predictions.csv"

    headlines_df = pd.read_csv(input_path)

    required_columns = ["headline_id", "headline_text"]
    assert not headlines_df[required_columns].isna().any().any()
    assert headlines_df["headline_id"].is_unique

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    predictions_df = predict_headlines(
        headlines_df,
        tokenizer,
        model,
        device,
    )

    valid_labels = {"positive", "neutral", "negative"}
    assert len(predictions_df) == len(headlines_df)
    assert predictions_df["headline_id"].is_unique
    assert set(predictions_df["label_pred"]).issubset(valid_labels)

    predictions_df.to_csv(output_path, index=False)

    print("Prediction counts:")
    print(predictions_df["label_pred"].value_counts().to_string())
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
