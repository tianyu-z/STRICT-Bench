import os
import json
import argparse
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

TOKENIZER = "CLIP"

if TOKENIZER == "CLIP":
    import core_clip as core
else:
    import core


def get_text_image_pairs(path: str) -> List[Tuple[str, str]]:
    """
    Find all text-image pairs in a directory or from a JSON file.
    (Copied from your provided script - assumed correct)
    """
    pairs = []
    base_dir = "."  # Default base directory

    # Check if path is a JSON file
    if path.endswith(".json") and os.path.isfile(path):
        # Paths in JSON might be relative to JSON location
        base_dir = os.path.dirname(path)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Assume JSON format has explicit pairs
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "text" in item and "image" in item:
                        # Resolve paths relative to the JSON file's directory
                        text_path = os.path.join(base_dir, item["text"])
                        image_path = os.path.join(base_dir, item["image"])
                        if os.path.exists(text_path) and os.path.exists(image_path):
                            pairs.append((text_path, image_path))
                        else:
                            print(
                                f"Warning: Skipping pair from JSON. File not found: {text_path if not os.path.exists(text_path) else image_path}")

            else:
                print(
                    f"Warning: JSON file '{path}' does not contain a list of objects with 'text' and 'image' keys.")
            return pairs

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON file '{path}'.")
            return []
        except Exception as e:
            print(f"Error processing JSON file '{path}': {e}")
            return []

    # Path is likely a directory
    if not os.path.isdir(path):
        print(f"Error: Path '{path}' is not a valid directory or JSON file.")
        return []  # Return empty list on error

    base_dir = path  # Paths are relative to this directory

    # Get all files in directory
    try:
        files = os.listdir(path)
    except OSError as e:
        print(f"Error listing directory '{path}': {e}")
        return []

    # Sort for consistency
    text_files = sorted([f for f in files if f.endswith(".txt")])

    # For each text file, look for a corresponding image file
    for text_file in text_files:
        # General way to remove extension
        base_name = os.path.splitext(text_file)[0]

        # Look for matching image files with various extensions
        found_match = False
        for ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:  # Added more extensions
            image_file = base_name + ext
            if image_file in files:
                text_path = os.path.join(base_dir, text_file)
                image_path = os.path.join(base_dir, image_file)
                pairs.append((text_path, image_path))
                found_match = True
                break  # Found the image for this text file
        # if not found_match:
        #     print(f"Warning: No corresponding image found for text file '{text_file}' in directory '{path}'")

    return pairs


def bootstrap_statistics(
    values: List[float], n_bootstrap: int = 1000, metric_name: str = "value"
) -> Optional[Dict[str, float]]:
    """
    Calculate bootstrap statistics for a list of values.

    Args:
        values: List of values to bootstrap.
        n_bootstrap: Number of bootstrap samples.
        metric_name: Name of the metric for printing warnings.

    Returns:
        Dictionary with mean, std_dev, and confidence intervals, or None if input is empty.
    """
    if not values:
        print(
            f"Warning: Cannot calculate bootstrap statistics for '{metric_name}' - no valid data points.")
        return None
    if len(values) == 1:
        print(
            f"Warning: Only one data point for '{metric_name}'. Reporting mean, std_dev=0, CI=mean.")
        mean_val = float(np.mean(values))
        return {"mean": mean_val, "std_dev": 0.0, "ci_lower": mean_val, "ci_upper": mean_val}

    # Original mean
    original_mean = np.mean(values)

    # Bootstrap means
    bootstrap_means = []
    # Use a fixed seed for reproducibility if needed, otherwise remove seed=
    # rng = np.random.default_rng(seed=42)
    rng = np.random.default_rng()
    n_values = len(values)

    for _ in range(n_bootstrap):
        # Handle potential numpy deprecation warning for non-tuple index
        indices = rng.integers(0, n_values, size=n_values)
        sample = np.array(values)[indices]
        bootstrap_means.append(np.mean(sample))

    # Calculate statistics
    # Use sample standard deviation (ddof=1) for the bootstrapped means distribution
    std_dev = np.std(bootstrap_means, ddof=1)
    sorted_means = np.sort(bootstrap_means)

    # Calculate 95% confidence interval using percentile method
    # Ensure indices are within bounds
    # -1 because of 0-based index
    lower_idx = max(0, int(0.025 * n_bootstrap) - 1)
    upper_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap) - 1)

    # Handle cases where bootstrap sample size is small
    if n_bootstrap < 40:  # Arbitrary threshold, adjust if needed
        print(
            f"Warning: Low number of bootstrap samples ({n_bootstrap}) for '{metric_name}'. CI might be unreliable.")
        # Fallback or different CI method could be implemented here if needed

    ci_lower = sorted_means[lower_idx]
    ci_upper = sorted_means[upper_idx]

    return {
        "mean": float(original_mean),
        "std_dev": float(std_dev),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def evaluate_pairs(
    pairs: List[Tuple[str, str]],
    ocr_lang: str,
    ocr_psm: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate all text-image pairs using the updated core evaluator
    and calculate statistics for multiple metrics.

    Args:
        pairs: List of (text_path, image_path) tuples.
        ocr_lang: Language for Tesseract OCR.
        ocr_psm: Page Segmentation Mode for Tesseract OCR.
        verbose: Whether to print verbose output from the core evaluator.

    Returns:
        Dictionary containing detailed per-pair results and aggregated statistics.
    """
    pair_results_list = []
    similarities = []
    similarities_truncated = []
    wers = []
    wers_truncated = []
    cers = []
    cers_truncated = []
    neds = []
    neds_truncated = []
    gt_char_counts = []
    gt_token_counts = []
    ocr_char_counts = []
    ocr_token_counts = []
    successful_evals = 0
    failed_evals = 0

    for i, (text_path, image_path) in enumerate(pairs):
        print(
            f"\n--- Evaluating Pair {i+1}/{len(pairs)} ---"
            f"\nText:  {os.path.basename(text_path)}"
            f"\nImage: {os.path.basename(image_path)}"
        )

        # Read ground truth text
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        except Exception as e:
            print(f"Error reading ground truth file {text_path}: {e}")
            failed_evals += 1
            continue  # Skip this pair

        # Evaluate the pair using the core function
        try:
            # Call the updated evaluation function
            eval_metrics = core.evaluate_document_image(
                ground_truth_text=ground_truth,
                generated_image_path=image_path,
                ocr_lang=ocr_lang,
                ocr_psm=ocr_psm,
                verbose=verbose  # Pass verbose flag down
            )

            if eval_metrics is None:
                print(
                    f"Skipping pair due to critical OCR failure for {image_path}")
                failed_evals += 1
                # Optionally store failure info
                pair_results_list.append({
                    "text_file": text_path,
                    "image_file": image_path,
                    "status": "OCR_FAILED",
                    "strict_sequence_similarity (Full)": None,
                    "strict_sequence_similarity (Truncated)": None,
                    "strict_wer (Full)": None,
                    "strict_wer (Truncated)"
                    "strict_cer (Full)": None,
                    "strict_cer (Truncated)": None,
                    "strict_ned (Full)": None,
                    "strict_ned (Truncated)": None,
                    "ground_truth_len_raw_chars": None,
                    "ocr_text_len_raw_chars": None,
                    "ground_truth_tokens": None,
                    "ocr_tokens": None,
                })
                continue

            # Store results for this pair
            pair_result = {
                "text_file": text_path,
                "image_file": image_path,
                "status": "Success" if eval_metrics['ocr_success'] else "OCR_EMPTY",
                "strict_sequence_similarity (Full)": eval_metrics['strict_sequence_similarity (Full)'],
                "strict_sequence_similarity (Truncated)": eval_metrics['strict_sequence_similarity (Truncated)'],
                "strict_wer (Full)": eval_metrics['strict_wer (Full)'],
                "strict_wer (Truncated)": eval_metrics['strict_wer (Truncated)'],
                "strict_cer (Full)": eval_metrics['strict_cer (Full)'],
                "strict_cer (Truncated)": eval_metrics['strict_cer (Truncated)'],
                "strict_ned (Full)": eval_metrics['strict_ned (Full)'],
                "strict_ned (Truncated)": eval_metrics['strict_ned (Truncated)'],
                "ground_truth_len_raw_chars": eval_metrics['ground_truth_len_raw_chars'],
                "ocr_text_len_raw_chars": eval_metrics['ocr_text_len_raw_chars'],
                "ground_truth_tokens": eval_metrics['ground_truth_tokens'],
                "ocr_tokens": eval_metrics['ocr_tokens'],
            }
            pair_results_list.append(pair_result)

            # Collect metrics for overall statistics if evaluation was successful
            # We calculate stats even if OCR output was empty but didn't critically fail
            similarities.append(
                eval_metrics['strict_sequence_similarity (Full)'])
            similarities_truncated.append(
                eval_metrics['strict_sequence_similarity (Truncated)'])

            wers.append(eval_metrics['strict_wer (Full)'])
            wers_truncated.append(eval_metrics['strict_wer (Truncated)'])

            cers.append(eval_metrics['strict_cer (Full)'])
            cers_truncated.append(eval_metrics['strict_cer (Truncated)'])

            neds.append(eval_metrics['strict_ned (Full)'])
            neds_truncated.append(eval_metrics['strict_ned (Truncated)'])

            gt_char_counts.append(eval_metrics['ground_truth_len_raw_chars'])
            gt_token_counts.append(eval_metrics['ground_truth_tokens'])

            if eval_metrics['ocr_text_len_raw_chars'] is not None:
                ocr_char_counts.append(eval_metrics['ocr_text_len_raw_chars'])
                ocr_token_counts.append(eval_metrics['ocr_tokens'])
            successful_evals += 1

            print(f"Metrics: Sim={eval_metrics['strict_sequence_similarity (Full)']:.4f}, Sim (Trunc)={eval_metrics['strict_sequence_similarity (Truncated)']:.4f}, "
                  f"WER={eval_metrics['strict_wer (Full)']:.4f}, WER (Trunc)={eval_metrics['strict_wer (Truncated)']:.4f}, "
                  f"CER={eval_metrics['strict_cer (Full)']:.4f}, CER (Trunc)={eval_metrics['strict_cer (Truncated)']:.4f}, "
                  f"NED={eval_metrics['strict_ned (Full)']:.4f}, NED (Trunc)={eval_metrics['strict_ned (Truncated)']:.4f}")

            print("-" * 50)

        except Exception as e:
            print(
                f"Error during evaluation process for pair {text_path} - {image_path}: {e}")
            # Optionally log the specific exception traceback here
            failed_evals += 1
            pair_results_list.append({
                "text_file": text_path,
                "image_file": image_path,
                "status": "EVALUATION_ERROR",
                "error_message": str(e),
                "strict_sequence_similarity (Full)": None,
                "strict_sequence_similarity (Truncated)": None,
                "strict_wer (Full)": None,
                "strict_wer (Truncated)": None,
                "strict_cer (Full)": None,
                "strict_cer (Truncated)": None,
                "strict_ned (Full)": None,
                "strict_ned (Truncated)": None,
                "ground_truth_len_raw_chars": None,
                "ocr_text_len_raw_chars": None,
                "ground_truth_tokens": None,
                "ocr_tokens": None,
            })
            continue

    print(
        f"\nEvaluation Summary: {successful_evals} successful, {failed_evals} failed/skipped.")

    print("\nCalculating overall statistics...")
    sim_stats = bootstrap_statistics(
        similarities, metric_name="Sequence Similarity")
    wer_stats = bootstrap_statistics(wers, metric_name="Word Error Rate")
    cer_stats = bootstrap_statistics(cers, metric_name="Character Error Rate")
    ned_stats = bootstrap_statistics(
        neds, metric_name="Normalized Edit Distance")
    sim_stats_truncated = bootstrap_statistics(
        similarities_truncated, metric_name="Sequence Similarity (Truncated)")
    wer_stats_truncated = bootstrap_statistics(
        wers_truncated, metric_name="Word Error Rate (Truncated)")
    cer_stats_truncated = bootstrap_statistics(
        cers_truncated, metric_name="Character Error Rate (Truncated)")
    ned_stats_truncated = bootstrap_statistics(
        neds_truncated, metric_name="Normalized Edit Distance (Truncated)")

    # Create final results dictionary
    final_results = {
        "evaluation_config": {
            "ocr_language": ocr_lang,
            "ocr_psm": ocr_psm,
        },
        "individual_results": pair_results_list,
        "aggregate_statistics": {
            "total_pairs_processed": len(pairs),
            "successful_evaluations": successful_evals,
            "failed_evaluations": failed_evals,
            "total_ground_truth_chars": sum(gt_char_counts),
            "total_ground_truth_tokens": sum(gt_token_counts),
            "total_ocr_chars": sum(ocr_char_counts) if ocr_char_counts else 0,
            "total_ocr_tokens": sum(ocr_token_counts) if ocr_token_counts else 0,
            "strict_sequence_similarity (Full)": sim_stats,
            "strict_sequence_similarity (Truncated)": sim_stats_truncated,
            "strict_wer (Full)": wer_stats,
            "strict_wer (Truncated)": wer_stats_truncated,
            "strict_cer (Full)": cer_stats,
            "strict_cer (Truncated)": cer_stats_truncated,
            "strict_ned (Full)": ned_stats,
            "strict_ned (Truncated)": ned_stats_truncated,
        },
    }

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate text-image pairs using OCR and order-preserving text comparison metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show default values
    )
    parser.add_argument(
        "--input_path", required=True,
        help="Path to a directory containing text (.txt) and image (.png, .jpg, etc.) "
             "files with matching base names, OR path to a JSON file listing pairs."
    )
    parser.add_argument(
        "-o", "--output", default="evaluation_results.json",
        help="Path to output JSON file to save detailed results and statistics."
    )

    # OCR Options (mirrored from core evaluator)
    parser.add_argument(
        "-l", "--lang", default='eng',
        help="Tesseract language code(s) (e.g., 'eng', 'eng+fra')."
    )
    parser.add_argument(
        "--psm", type=int, default=3, choices=range(0, 14),
        help="Tesseract Page Segmentation Mode (see tesseract --help-psm)."
    )

    # Preprocessing Options (mirrored from core evaluator)
    parser.add_argument(
        "--no-lowercase", action='store_true',
        help="Disable conversion to lowercase before comparison."
    )
    parser.add_argument(
        "--keep-punctuation", action='store_true',
        help="Disable removal of punctuation before comparison."
    )

    # Output Options
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="Enable verbose output from the core evaluation function for each pair."
    )
    parser.add_argument(
        "--n_bootstrap", type=int, default=1000,
        help="Number of bootstrap samples for calculating confidence intervals."
    )

    args = parser.parse_args()

    # --- Get Text-Image Pairs ---
    print(f"Searching for text-image pairs in: {args.input_path}")
    pairs = get_text_image_pairs(args.input_path)

    if not pairs:
        print("No text-image pairs found. Exiting.")
        return

    print(f"Found {len(pairs)} pairs.")
    if len(pairs) < 5:
        print("Warning: Very few pairs found, statistics might not be very meaningful.")

    print("\nStarting evaluation of pairs...")
    results = evaluate_pairs(
        pairs=pairs,
        ocr_lang=args.lang,
        ocr_psm=args.psm,
        verbose=args.verbose
    )

    # --- Save Results ---
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")
    except Exception as e:
        print(f"\nError saving results to {args.output}: {e}")

    print("\n======== AGGREGATE STATISTICS ========")
    stats = results["aggregate_statistics"]
    print(f"Total Pairs Processed: {stats['total_pairs_processed']}")
    print(f"Successful Evaluations: {stats['successful_evaluations']}")
    print(f"Failed/Skipped Evaluations: {stats['failed_evaluations']}")
    print(f"Total Ground Truth Chars: {stats['total_ground_truth_chars']}")
    print(f"Total OCR Chars (Successful): {stats['total_ocr_chars']}")
    print(f"Total Ground Truth Tokens: {stats['total_ground_truth_tokens']}")
    print(f"Total OCR Tokens (Successful): {stats['total_ocr_tokens']}")
    print("-" * 35)

    def print_metric_stats(metric_name: str, metric_stats: Optional[Dict[str, float]]):
        print(f"{metric_name}:")
        if metric_stats:
            print(f"  Mean:           {metric_stats['mean']:.4f}")
            print(f"  Std Dev (Boostrap): {metric_stats['std_dev']:.4f}")
            print(
                f"  95% CI (Bootstrap): [{metric_stats['ci_lower']:.4f}, {metric_stats['ci_upper']:.4f}]")
        else:
            print("  (No valid data to calculate statistics)")

    print_metric_stats("Sequence Similarity (Higher=Better)",
                       stats["strict_sequence_similarity (Full)"])
    print_metric_stats("Sequence Similarity (Higher=Better)",
                       stats["strict_sequence_similarity (Truncated)"])
    print_metric_stats("Word Error Rate (Lower=Better)",
                       stats["strict_wer (Full)"])
    print_metric_stats("Word Error Rate (Lower=Better)",
                       stats["strict_wer (Truncated)"])
    print_metric_stats("Character Error Rate (Lower=Better)",
                       stats["strict_cer (Full)"])
    print_metric_stats("Character Error Rate (Lower=Better)",
                       stats["strict_cer (Truncated)"])
    print_metric_stats("Normalized Edit Distance (Lower=Better)",
                       stats["strict_ned (Full)"])
    print_metric_stats("Normalized Edit Distance (Lower=Better)",
                       stats["strict_ned (Truncated)"])

    print("====================================")


if __name__ == "__main__":
    main()