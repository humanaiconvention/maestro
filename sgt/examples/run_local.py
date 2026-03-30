"""
Example usage of the Semantic Grounding Test Module with local QWEN/LLAMA models.

This example demonstrates how to create test cases and evaluate
semantic grounding using locally-hosted language models.

Usage:
    python examples/run_local.py --provider qwen --model-path /path/to/model
    python examples/run_local.py --provider llama --metrics-mode embedding
    python examples/run_local.py --help
"""

import os
import argparse
from semantic_grounding import (
    SemanticGroundingTestCase,
    SemanticGroundingEvaluator,
    GroundingType,
)


def create_sample_test_cases():
    """Create sample test cases for demonstration."""
    test_cases = [
        SemanticGroundingTestCase(
            test_id="obj_ref_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table next to the blue cup.",
            expected_grounding=["red ball", "table", "blue cup"],
            context="A scene description with objects and their attributes.",
        ),
        SemanticGroundingTestCase(
            test_id="spatial_001",
            grounding_type=GroundingType.SPATIAL_REASONING,
            prompt="The library is between the coffee shop and the park.",
            expected_grounding=["library", "coffee shop", "park", "between"],
            context="Understanding spatial relationships between locations.",
        ),
        SemanticGroundingTestCase(
            test_id="temporal_001",
            grounding_type=GroundingType.TEMPORAL_REASONING,
            prompt="After breakfast, she went to work before the meeting started.",
            expected_grounding=["breakfast", "work", "meeting", "after", "before"],
            context="Understanding temporal sequence of events.",
        ),
    ]
    return test_cases


def print_instructions():
    """Print instructions for setting up local models."""
    print("\n" + "=" * 70)
    print("SETUP INSTRUCTIONS FOR LOCAL MODEL INFERENCE")
    print("=" * 70)
    print("\n1. Download Model Weights:")
    print("   - For QWEN models:")
    print("     Visit: https://huggingface.co/Qwen")
    print("     Example: Qwen/Qwen-7B-Chat, Qwen/Qwen-14B-Chat")
    print("   - For LLAMA models:")
    print("     Visit: https://huggingface.co/meta-llama")
    print("     Example: meta-llama/Llama-2-7b-chat-hf")
    print("\n2. Set Environment Variables:")
    print("   export QWEN_MODEL_PATH=/path/to/qwen/model")
    print("   export LLAMA_MODEL_PATH=/path/to/llama/model")
    print("\n3. Install Dependencies:")
    print("   pip install -e .")
    print("   # For GPU support (recommended):")
    print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print("\n4. Optional: Use QWEN API")
    print("   export QWEN_API_KEY=your-api-key")
    print("   (Note: API integration is planned for future release)")
    print("\n" + "=" * 70 + "\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run semantic grounding evaluation with local models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with QWEN model using exact matching
  python examples/run_local.py --provider qwen --model-path /path/to/qwen-7b-chat

  # Run with LLAMA model using embedding-based matching
  python examples/run_local.py --provider llama --metrics-mode embedding

  # Use custom embedding model and threshold
  python examples/run_local.py --provider qwen --metrics-mode embedding \\
      --embedding-model all-mpnet-base-v2 --embedding-threshold 0.8

  # Enable structured output parsing
  python examples/run_local.py --provider qwen --structured-output
        """,
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["qwen", "llama"],
        help="Model provider (qwen or llama). If not specified, uses env vars to detect.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Path to local model weights. If not specified, "
            "uses QWEN_MODEL_PATH or LLAMA_MODEL_PATH env var."
        ),
    )
    parser.add_argument(
        "--metrics-mode",
        type=str,
        default="exact",
        choices=["exact", "embedding"],
        help=(
            "Metrics mode: 'exact' for n-gram matching or "
            "'embedding' for semantic similarity (default: exact)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embedding mode (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.7,
        help="Cosine similarity threshold for embedding mode (default: 0.7)",
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Expect model to return JSON with 'groundings' key",
    )
    return parser.parse_args()


def main():
    """Main example execution."""
    args = parse_args()

    print("\nSemantic Grounding Test Module - Local Model Examples")
    print("=" * 70)

    # Determine provider and model path
    provider = args.provider
    model_path = args.model_path

    # Auto-detect provider from environment if not specified
    if not provider:
        qwen_path = os.getenv("QWEN_MODEL_PATH")
        llama_path = os.getenv("LLAMA_MODEL_PATH")

        if qwen_path and not llama_path:
            provider = "qwen"
            model_path = model_path or qwen_path
        elif llama_path and not qwen_path:
            provider = "llama"
            model_path = model_path or llama_path
        elif qwen_path and llama_path:
            # Default to QWEN if both are set
            provider = "qwen"
            model_path = model_path or qwen_path
            print(f"⚠️  Both QWEN and LLAMA paths are set. Using {provider.upper()}.")
        else:
            print_instructions()
            # Show test cases for demonstration
            print("\nExample Test Cases (evaluation requires model setup):\n")
            test_cases = create_sample_test_cases()
            for i, test_case in enumerate(test_cases, 1):
                print(f"{i}. {test_case.test_id} ({test_case.grounding_type.value})")
                print(f"   Prompt: {test_case.prompt}")
                print(f"   Expected: {', '.join(test_case.expected_grounding)}\n")
            return

    # Set model path from env var if not specified
    if not model_path:
        if provider == "qwen":
            model_path = os.getenv("QWEN_MODEL_PATH")
        elif provider == "llama":
            model_path = os.getenv("LLAMA_MODEL_PATH")

    if not model_path:
        print(f"\n❌ Error: Model path not specified for {provider.upper()}")
        print(f"   Set {provider.upper()}_MODEL_PATH environment variable " "or use --model-path")
        return

    print("\nConfiguration:")
    print(f"  Provider: {provider.upper()}")
    print(f"  Model Path: {model_path}")
    print(f"  Metrics Mode: {args.metrics_mode}")
    if args.metrics_mode == "embedding":
        print(f"  Embedding Model: {args.embedding_model}")
        print(f"  Embedding Threshold: {args.embedding_threshold}")
    print(f"  Structured Output: {args.structured_output}")
    print("=" * 70)

    # Create evaluator with specified configuration
    try:
        print(f"\nInitializing {provider.upper()} evaluator...")
        evaluator = SemanticGroundingEvaluator(
            model_name=provider,
            model_path=model_path,
            success_threshold=0.7,
            metrics_mode=args.metrics_mode,
            embedding_model_name=args.embedding_model,
            embedding_threshold=args.embedding_threshold,
            structured_output=args.structured_output,
        )
    except Exception as e:
        print(f"\n❌ Error initializing evaluator: {e}")
        return

    # Create and evaluate test cases
    test_cases = create_sample_test_cases()

    print(f"\nEvaluating {len(test_cases)} test cases...")
    print("=" * 70)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(
            f"\n[{i}/{len(test_cases)}] {test_case.test_id} " f"({test_case.grounding_type.value})"
        )
        print(f"Prompt: {test_case.prompt}")

        try:
            result = evaluator.evaluate(test_case)
            results.append(result)

            print(f"Model Response: {result.model_response[:150]}...")
            print(f"Extracted: {', '.join(result.extracted_groundings[:5])}")
            print(
                f"Metrics: P={result.metrics.precision:.2f}, "
                f"R={result.metrics.recall:.2f}, "
                f"F1={result.metrics.f1_score:.2f}"
            )
            print(f"Success: {'✓' if result.success else '✗'}")
        except Exception as e:
            print(f"❌ Error evaluating test case: {e}")

    # Print summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        summary = evaluator.get_summary_statistics(results)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Average F1 Score: {summary['average_f1_score']:.2f}")
        print(f"Average Precision: {summary['average_precision']:.2f}")
        print(f"Average Recall: {summary['average_recall']:.2f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
