"""
Script to prepare medical dataset for fine-tuning.

This script:
1. Downloads the dataset from HuggingFace
2. Creates a subset for local training
3. Splits into train/test
4. Saves processed data
"""

import argparse
import sys
from pathlib import Path
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
import random

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import (
    create_subset,
    split_dataset,
    save_jsonl,
    print_example
)

console = Console()

def load_medical_dataset(dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"):
    """
    Download medical dataset from HuggingFace.

    Args:
        dataset_name: Dataset name on HuggingFace Hub

    Returns:
        List of examples (dictionaries)
    """
    console.print(f"[bold blue]📥 Downloading dataset: {dataset_name}[/bold blue]")

    try:
        dataset = load_dataset(dataset_name, "en", split="train")

        console.print(f"[green]✓ Dataset downloaded: {len(dataset)} examples[/green]")

        data = []
        for example in dataset:
            data.append({
                'question': example['Question'],
                'complex_cot': example['Complex_CoT'],
                'response': example['Response']
            })

        console.print(f"[green]✓ Data processed: {len(data)} examples[/green]")
        return data

    except Exception as e:
        console.print(f"[red]✗ Error downloading dataset: {e}[/red]")
        import traceback
        traceback.print_exc()
        raise

def analyze_dataset(data: list):
    """
    Analyze and display basic dataset statistics.

    Args:
        data: List of examples
    """
    console.print("\n[bold cyan]📊 Dataset Statistics[/bold cyan]")

    question_lengths = [len(ex['question']) for ex in data]
    cot_lengths = [len(ex['complex_cot']) for ex in data]
    response_lengths = [len(ex['response']) for ex in data]

    table = Table(title="Text lengths (characters)")
    table.add_column("Field", style="cyan")
    table.add_column("Average", style="magenta")
    table.add_column("Minimum", style="green")
    table.add_column("Maximum", style="red")

    table.add_row(
        "Question",
        f"{sum(question_lengths) / len(question_lengths):.0f}",
        f"{min(question_lengths)}",
        f"{max(question_lengths)}"
    )
    table.add_row(
        "Complex CoT",
        f"{sum(cot_lengths) / len(cot_lengths):.0f}",
        f"{min(cot_lengths)}",
        f"{max(cot_lengths)}"
    )
    table.add_row(
        "Response",
        f"{sum(response_lengths) / len(response_lengths):.0f}",
        f"{min(response_lengths)}",
        f"{max(response_lengths)}"
    )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Prepare medical dataset for fine-tuning")
    parser.add_argument(
        "--subset_size",
        type=int,
        default=500,
        help="Subset size for local training (default: 500)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train/test ratio (default: 0.9)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory (default: data)"
    )
    parser.add_argument(
        "--show_examples",
        type=int,
        default=2,
        help="Number of examples to show (default: 2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    random.seed(args.seed)

    console.print("\n[bold]🚀 STEP 1: Dataset Download[/bold]")
    full_data = load_medical_dataset()

    console.print("\n[bold]🚀 STEP 2: Full Dataset Analysis[/bold]")
    analyze_dataset(full_data)

    console.print(f"\n[bold]🚀 STEP 3: Subset Creation ({args.subset_size} examples)[/bold]")
    subset_data = create_subset(full_data, args.subset_size, random_seed=args.seed)
    console.print(f"[green]✓ Subset created: {len(subset_data)} examples[/green]")

    console.print(f"\n[bold]🚀 STEP 4: Train/Test Split ({args.train_ratio:.0%}/{1-args.train_ratio:.0%})[/bold]")
    train_data, test_data = split_dataset(subset_data, train_ratio=args.train_ratio, random_seed=args.seed)

    console.print(f"[green]✓ Train: {len(train_data)} examples[/green]")
    console.print(f"[green]✓ Test: {len(test_data)} examples[/green]")

    console.print("\n[bold]🚀 STEP 5: Saving Processed Data[/bold]")
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    train_file = output_path / "train_data.jsonl"
    test_file = output_path / "test_data.jsonl"

    save_jsonl(train_data, train_file)
    save_jsonl(test_data, test_file)

    console.print(f"[green]✓ Train saved: {train_file}[/green]")
    console.print(f"[green]✓ Test saved: {test_file}[/green]")

    console.print(f"\n[bold]🚀 STEP 6: Dataset Examples[/bold]")
    console.print("[yellow]Showing examples from train set:[/yellow]\n")

    for i in range(min(args.show_examples, len(train_data))):
        print_example(train_data[i], idx=i+1)

    console.print("\n[bold green]✅ PREPARATION COMPLETED![/bold green]")
    console.print("\n[bold]📁 Generated files:[/bold]")
    console.print(f"  • {train_file} ({len(train_data)} examples)")
    console.print(f"  • {test_file} ({len(test_data)} examples)")
    console.print("\n[bold]📝 Next step:[/bold]")
    console.print("  • Configure training in configs/training_local.yaml")
    console.print("  • Run src/train.py to start fine-tuning\n")


if __name__ == "__main__":
    main()
