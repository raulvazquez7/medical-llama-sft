"""
Utility functions for data processing, formatting, and evaluation.
"""

import json
import random
from typing import Dict, List, Any
from pathlib import Path


def format_medical_prompt(example: Dict[str, Any]) -> str:
    """
    Format a medical example into the HuatuoGPT-o1 style prompt.

    Args:
        example: Dictionary with 'question', 'complex_cot', and 'response' keys

    Returns:
        Formatted prompt string
    """
    question = example.get('question', '')
    complex_cot = example.get('complex_cot', '')
    response = example.get('response', '')

    assistant_response = f"""## Thinking
{complex_cot}

## Final Response
{response}"""

    return assistant_response


def format_chat_template(example: Dict[str, Any], system_prompt: str = None) -> List[Dict[str, str]]:
    """
    Format example into chat template format for Llama 3.1.

    Args:
        example: Dictionary with 'question', 'complex_cot', and 'response'
        system_prompt: Optional system prompt (default medical expert prompt)

    Returns:
        List of message dictionaries
    """
    if system_prompt is None:
        system_prompt = (
            "You are a medical expert AI assistant. When answering medical questions, "
            "first provide your step-by-step reasoning in a '## Thinking' section, "
            "then provide your final answer in a '## Final Response' section."
        )

    assistant_response = format_medical_prompt(example)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['question']},
        {"role": "assistant", "content": assistant_response}
    ]

    return messages


def create_subset(data: List[Dict], size: int, random_seed: int = 42) -> List[Dict]:
    """
    Create a random subset of the data.

    Args:
        data: Full dataset
        size: Desired subset size
        random_seed: Random seed for reproducibility

    Returns:
        Subset of data
    """
    random.seed(random_seed)
    if len(data) <= size:
        return data
    return random.sample(data, size)


def split_dataset(data: List[Dict], train_ratio: float = 0.9, random_seed: int = 42) -> tuple:
    """
    Split dataset into train and test sets.

    Args:
        data: Full dataset
        train_ratio: Proportion for training (default 0.9)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(random_seed)
    data_copy = data.copy()
    random.shuffle(data_copy)

    split_idx = int(len(data_copy) * train_ratio)
    train_data = data_copy[:split_idx]
    test_data = data_copy[split_idx:]

    return train_data, test_data


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data in JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load data from JSONL format."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_token_stats(data: List[Dict], tokenizer) -> Dict[str, float]:
    """
    Calculate token statistics for the dataset.

    Args:
        data: Dataset
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with token statistics
    """
    token_counts = []

    for example in data:
        messages = format_chat_template(example)
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))

    return {
        'mean': sum(token_counts) / len(token_counts),
        'max': max(token_counts),
        'min': min(token_counts),
        'total_examples': len(token_counts)
    }


def extract_thinking_and_response(text: str) -> Dict[str, str]:
    """
    Extract thinking and final response sections from model output.

    Args:
        text: Model output text

    Returns:
        Dictionary with 'thinking' and 'response' keys
    """
    thinking = ""
    response = ""

    if "## Thinking" in text and "## Final Response" in text:
        parts = text.split("## Final Response")
        thinking_part = parts[0].replace("## Thinking", "").strip()
        response_part = parts[1].strip() if len(parts) > 1 else ""

        thinking = thinking_part
        response = response_part
    else:
        # Model didn't follow format, treat entire output as response
        response = text.strip()

    return {
        'thinking': thinking,
        'response': response
    }


def print_example(example: Dict[str, Any], idx: int = None):
    """Pretty print a medical example."""
    print("=" * 80)
    if idx is not None:
        print(f"Example {idx}")
    print("-" * 80)
    print("QUESTION:")
    print(example.get('question', 'N/A'))
    print("\n" + "-" * 80)
    print("THINKING:")
    print(example.get('complex_cot', 'N/A')[:500] + "..." if len(example.get('complex_cot', '')) > 500 else example.get('complex_cot', 'N/A'))
    print("\n" + "-" * 80)
    print("RESPONSE:")
    print(example.get('response', 'N/A'))
    print("=" * 80 + "\n")
