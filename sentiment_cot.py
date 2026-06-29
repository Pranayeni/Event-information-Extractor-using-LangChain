"""
Task 1: Sentiment Analysis with Chain-of-Thought (CoT) Prompting
=================================================================
Uses few-shot CoT prompting via Google Gemini to classify customer
reviews as positive / neutral / negative with step-by-step reasoning.

Requirements:
    pip install google-genai python-dotenv
"""

import os
import re
import json
from dataclasses import dataclass
from dotenv import load_dotenv     
from google import genai           

# ---------------------------------------------------------------------------
# Configure Gemini
# ---------------------------------------------------------------------------
# 1. CRUCIAL: Call load_dotenv() to inject variables from .env into the environment
load_dotenv()

# 2. Initialize the standard modern Client (picks up the key from the environment)
client = genai.Client() 
MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# Dataset – 10 scraped e-commerce customer reviews
# ---------------------------------------------------------------------------
REVIEWS = [
    {
        "id": 1,
        "product": "Wireless Headphones",
        "rating": 5,
        "text": (
            "Absolutely love these headphones! The sound quality is crystal clear "
            "and the noise cancellation is outstanding. Battery lasts 30+ hours. "
            "Best purchase I've made this year!"
        ),
    },
    {
        "id": 2,
        "product": "Coffee Maker",
        "rating": 1,
        "text": (
            "Complete waste of money. Stopped working after 2 weeks. Customer service "
            "was unhelpful and rude. The carafe leaked and coffee tasted burnt every "
            "single time. Returning immediately."
        ),
    },
    {
        "id": 3,
        "product": "Running Shoes",
        "rating": 3,
        "text": (
            "Decent shoes for the price. The fit is true to size and they look nice. "
            "However, cushioning wears out quickly and they're not ideal for long runs. "
            "Average product overall."
        ),
    },
    {
        "id": 4,
        "product": "Laptop Stand",
        "rating": 5,
        "text": (
            "Incredible build quality! Sturdy aluminum, adjustable to any height, and "
            "folds flat for travel. My neck pain disappeared after just one day of use. "
            "Highly recommend to anyone working from home."
        ),
    },
    {
        "id": 5,
        "product": "Protein Powder",
        "rating": 2,
        "text": (
            "Terrible taste even with milk. Clumps badly and leaves a chalky aftertaste. "
            "The only positive is it mixes somewhat quickly. Would not buy again despite "
            "the good macros on paper."
        ),
    },
    {
        "id": 6,
        "product": "Smart Watch",
        "rating": 4,
        "text": (
            "Great fitness tracker with excellent heart rate monitoring. GPS is accurate "
            "and sleep tracking is useful. Knocking one star because battery only lasts "
            "2 days and the strap irritates my wrist slightly."
        ),
    },
    {
        "id": 7,
        "product": "Air Purifier",
        "rating": 3,
        "text": (
            "It works as described but nothing special. Noise level is acceptable on "
            "low setting but becomes annoying on high. Filter replacements are expensive. "
            "Neutral on whether I'd buy again."
        ),
    },
    {
        "id": 8,
        "product": "Mechanical Keyboard",
        "rating": 5,
        "text": (
            "A game changer for productivity! The tactile feedback is satisfying, RGB "
            "lighting looks stunning, and the build feels premium. Every keystroke is a "
            "joy. Worth every penny."
        ),
    },
    {
        "id": 9,
        "product": "Yoga Mat",
        "rating": 2,
        "text": (
            "Poor quality for the price. Tears at the edges after minimal use, provides "
            "almost no grip when sweating, and has a strong chemical smell that won't go "
            "away. Very disappointed."
        ),
    },
    {
        "id": 10,
        "product": "Bluetooth Speaker",
        "rating": 4,
        "text": (
            "Solid sound for its compact size. Waterproof design is great for outdoor use. "
            "Bass could be stronger and it lacks EQ controls, but overall a reliable "
            "portable speaker at a fair price."
        ),
    },
]


# ---------------------------------------------------------------------------
# Few-shot Chain-of-Thought prompt
# ---------------------------------------------------------------------------
COT_SYSTEM_INSTRUCTION = (
    "You are a sentiment analysis expert. Analyze customer reviews and classify "
    "them as positive, neutral, or negative using a strict step-by-step "
    "Chain-of-Thought reasoning process. Always follow the four steps shown in "
    "the examples and end every response with 'Label: <positive|neutral|negative>'."
)

COT_FEW_SHOT = """=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
Review: "This phone is amazing! The camera quality blows me away and battery life is superb. Best smartphone I've ever owned."

Step 1 - Positive phrases: "amazing", "blows me away", "superb", "Best smartphone I've ever owned"
Step 2 - Negative phrases: none detected
Step 3 - Contradictions/mixed sentiment: No contradictions. Uniformly positive throughout.
Step 4 - Final decision: POSITIVE — overwhelmingly enthusiastic with multiple strong positive indicators and zero negative phrases. Superlatives confirm very high satisfaction.
Label: positive

--- Example 2 ---
Review: "The product is okay. It does what it's supposed to do, nothing more. Arrived on time but packaging was a bit damaged."

Step 1 - Positive phrases: "does what it's supposed to do", "arrived on time"
Step 2 - Negative phrases: "packaging was a bit damaged", "nothing more" (implies no excitement)
Step 3 - Contradictions/mixed sentiment: Yes — mild positives (functional, timely) balanced by mild negatives (damaged packaging, underwhelmed). Neither dominates.
Step 4 - Final decision: NEUTRAL — positives and negatives cancel out; customer is satisfied at a baseline level but not impressed or notably disappointed.
Label: neutral

--- Example 3 ---
Review: "Terrible quality. Broke after 3 days. Waste of money and customer service ignored my refund request."

Step 1 - Positive phrases: none
Step 2 - Negative phrases: "Terrible quality", "Broke after 3 days", "Waste of money", "customer service ignored my refund request"
Step 3 - Contradictions/mixed sentiment: No contradictions. Entirely negative — covers product quality, durability, value, and support.
Step 4 - Final decision: NEGATIVE — multiple severe negative indicators across all dimensions with no redeeming phrases.
Label: negative

=== END OF EXAMPLES ===

Now analyze the review below using the SAME four-step format:
"""


def build_prompt(review: dict) -> str:
    return (
        COT_FEW_SHOT
        + f"\nProduct: {review['product']}\n"
        + f"Star Rating: {review['rating']}/5\n"
        + f"Review: \"{review['text']}\""
    )


def parse_label(text: str) -> str:
    match = re.search(r"Label:\s*(positive|neutral|negative)", text, re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


# ---------------------------------------------------------------------------
# Dataclass for results
# ---------------------------------------------------------------------------
@dataclass
class SentimentResult:
    review_id: int
    product: str
    rating: int
    predicted_label: str
    reasoning: str


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------
def analyze_review(review: dict) -> SentimentResult:
    prompt = build_prompt(review)
    
    # Migrated to the new client call convention
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "system_instruction": COT_SYSTEM_INSTRUCTION
        }
    )
    
    reasoning = response.text
    label = parse_label(reasoning)
    return SentimentResult(
        review_id=review["id"],
        product=review["product"],
        rating=review["rating"],
        predicted_label=label,
        reasoning=reasoning,
    )


def run_all_reviews(reviews: list[dict]) -> list[SentimentResult]:
    results = []
    print(f"\nAnalyzing {len(reviews)} reviews with Gemini ({MODEL_NAME}) + Chain-of-Thought...\n")

    for review in reviews:
        print(f"  → {review['product']} (#'s {review['id']})...", end="", flush=True)
        result = analyze_review(review)
        results.append(result)
        icon = {"positive": "✅", "neutral": "⚠️", "negative": "❌"}.get(result.predicted_label, "❓")
        print(f" {icon} {result.predicted_label.upper()}")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_result(result: SentimentResult) -> None:
    icon = {"positive": "✅", "neutral": "⚠️", "negative": "❌"}.get(result.predicted_label, "❓")
    print(f"\n{'='*70}")
    print(f"Review #{result.review_id} | {result.product} | Rating: {result.rating}/5")
    print(f"Predicted Sentiment: {icon} {result.predicted_label.upper()}")
    print(f"{'─'*70}")
    print(result.reasoning)


def print_summary(results: list[SentimentResult]) -> None:
    from collections import Counter
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Product':<22} {'Rating':<8} {'Label'}")
    print("─" * 55)
    for r in results:
        icon = {"positive": "✅", "neutral": "⚠️", "negative": "❌"}.get(r.predicted_label, "❓")
        print(f"{r.review_id:<4} {r.product:<22} {r.rating}/5     {icon} {r.predicted_label}")
    counts = Counter(r.predicted_label for r in results)
    print(f"\nDistribution → Positive: {counts['positive']} | Neutral: {counts['neutral']} | Negative: {counts['negative']}\n")


def save_results_json(results: list[SentimentResult], path: str = "sentiment_results.json") -> None:
    data = [
        {
            "review_id": r.review_id,
            "product": r.product,
            "rating": r.rating,
            "predicted_label": r.predicted_label,
            "reasoning": r.reasoning,
        }
        for r in results
    ]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_all_reviews(REVIEWS)

    for result in results:
        print_result(result)

    print_summary(results)
    save_results_json(results)