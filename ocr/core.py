import pytesseract
from PIL import Image
import os
import argparse
import difflib  # For sequence comparison
import re  # For whitespace normalization
import jiwer
from jiwer import wer, cer  # For WER/CER calculation (pip install jiwer)
from typing import Tuple, List, Dict, Optional, Any  # Added for type hinting
from .ned import ned


def strict_text_processing(text: str) -> str:
    """
    Performs minimal processing for strict comparison:
    1. Replaces multiple whitespace characters (space, tab, newline, etc.)
       with a single space. Handles cases where OCR might add extra spaces/newlines.
    2. Removes leading/trailing whitespace.
    Keeps case and punctuation intact.
    """
    if not text:
        return ""
    # Replace runs of whitespace with a single space
    processed_text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return processed_text.strip()


def tokenize(text: str) -> List[str]:
    """Splits text into tokens based on spaces."""
    if not text:
        return []
    return text.split(' ')  # Simple space-based tokenization


def extract_text_from_image(
    image_path: str, lang: str = 'eng', psm: int = 3
) -> Optional[str]:
    """
    Uses Tesseract OCR to extract text from an image file.

    Args:
        image_path: Path to the image file.
        lang: Language code for Tesseract (e.g., 'eng', 'fra').
        psm: Page Segmentation Mode for Tesseract (e.g., 3 for auto, 6 for single block).

    Returns:
        The extracted text as a string, or None if a critical error occurs
        (file not found, Tesseract not found). Returns empty string if OCR runs
        but finds no text.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None  # Indicate critical failure: file missing
    try:
        img = Image.open(image_path)
        # Construct Tesseract configuration string
        config = f'-l {lang} --psm {psm}'
        print(f"Running Tesseract with config: '{config}'")
        # Use image_to_string which tries to preserve some layout initially
        text = pytesseract.image_to_string(img, config=config)
        # Note: Tesseract might return empty string ('') if no text is found,
        # or a string with only whitespace ('\n\x0c')
        # We return the raw output here; processing happens later.
        return text
    except pytesseract.TesseractNotFoundError:
        print("\nError: Tesseract is not installed or not in your PATH.")
        print("Please install Tesseract (https://github.com/tesseract-ocr/tesseract)")
        print("and ensure pytesseract can find it.")
        # Example: Set 'pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'' at the top
        return None  # Indicate critical failure: Tesseract missing
    except ImportError:
        print("\nError: Pillow or pytesseract not installed correctly.")
        print("Please install them: pip install Pillow pytesseract")
        return None  # Indicate critical failure: Dependency missing
    except Exception as e:
        print(f"Error during OCR processing for '{image_path}': {e}")
        # Return empty string here, as OCR ran but failed partially,
        # allowing evaluation metrics to show maximum error.
        return ""

# --- Evaluation Metrics ---


def calculate_comparison_metrics(
    ground_truth: str,
    hypothesis: str,
    label_suffix: str = ""  # e.g., " (Full)" or " (Truncated)"
) -> Dict[str, float]:
    """
    Calculates strict comparison metrics between two strings.
    Assumes inputs are already processed (e.g., whitespace normalized).
    Uses SequenceMatcher ratio, WER, and CER.

    Args:
        ground_truth: The reference text string.
        hypothesis: The text string to compare (e.g., OCR output).
        label_suffix: String to append to metric names in the output dict.

    Returns:
        A dictionary containing similarity ratio, WER, and CER.
        Handles empty input cases gracefully.
    """
    results = {}
    similarity_key = f'strict_sequence_similarity{label_suffix}'
    wer_key = f'strict_wer{label_suffix}'
    cer_key = f'strict_cer{label_suffix}'
    ned_key = f'strict_ned{label_suffix}'

    # --- Sequence Similarity (difflib) ---
    # Preserves order, case, and punctuation inherently if inputs are strict.
    if not ground_truth and not hypothesis:
        results[similarity_key] = 1.0
    elif not ground_truth or not hypothesis:
        results[similarity_key] = 0.0
    else:
        # Use SequenceMatcher directly on the strictly processed strings
        results[similarity_key] = difflib.SequenceMatcher(
            None, ground_truth, hypothesis
        ).ratio()

    # --- WER and CER (jiwer) ---
    # Calculate on the strictly processed strings to maintain sensitivity
    if not ground_truth and not hypothesis:
        results[wer_key] = 0.0
        results[cer_key] = 0.0
        results[ned_key] = 0.0
    elif not ground_truth or not hypothesis:
        # Assign maximum error if one string is empty and the other is not
        results[wer_key] = 1.0
        results[cer_key] = 1.0
        results[ned_key] = 1.0
    else:
        try:
            # jiwer computes measures based on edit distance.
            # Feeding it case/punctuation-sensitive strings makes the
            # metrics sensitive to these differences.
            wer_results = wer(ground_truth, hypothesis)
            cer_results = cer(ground_truth, hypothesis)
            ned_results = ned(ground_truth, hypothesis)
            results[wer_key] = wer_results
            results[cer_key] = cer_results
            results[ned_key] = ned_results
        except Exception as e:
            print(f"Error calculating WER/CER with jiwer{label_suffix}: {e}")
            # Return maximum error on failure
            results[wer_key] = 1.0
            results[cer_key] = 1.0
            results[ned_key] = 1.0

    return results

# --- Main Evaluation Function ---


def evaluate_document_image(
    ground_truth_text: str,
    generated_image_path: str,
    ocr_lang: str = 'eng',
    ocr_psm: int = 3,
    verbose: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Evaluates the generated document image against the ground truth text
    using strict, order-preserving metrics, comparing both full text and
    a truncated version based on OCR token count.

    Args:
        ground_truth_text: The original input text (string).
        generated_image_path: Path to the generated image file.
        ocr_lang: Language for Tesseract OCR.
        ocr_psm: Page Segmentation Mode for Tesseract OCR.
        verbose: Print detailed comparison info.

    Returns:
        A dictionary containing evaluation metrics for both full and
        truncated comparisons, or None if OCR fails critically.
    """
    print("\n--- Starting Evaluation ---")
    print(f"Ground Truth Length (raw chars): {len(ground_truth_text)}")
    print(f"Generated Image: '{generated_image_path}'")

    # 1. Extract text using OCR
    print("\nStep 1: Extracting text via OCR...")
    ocr_text_raw = extract_text_from_image(
        generated_image_path, lang=ocr_lang, psm=ocr_psm)

    if ocr_text_raw is None:
        print("OCR failed critically (Tesseract/File/Dependency Error). Cannot proceed.")
        return None  # Indicate critical failure

    print(f"OCR Extracted Raw Text Length: {len(ocr_text_raw)} characters")
    if not ocr_text_raw:
        print("Warning: OCR returned empty text. Evaluation metrics will reflect this (likely 0% similarity, 100% error).")

    if verbose:
        print("\n--- Raw OCR Output Snippet ---")
        print(ocr_text_raw[:500] + ("..." if len(ocr_text_raw) > 500 else ""))
        print("--- End Raw OCR Output ---\n")

    # 2. Perform Strict Text Processing (Whitespace normalization only)
    print("Step 2: Performing strict text processing (normalizing whitespace)...")
    gt_processed_strict = strict_text_processing(ground_truth_text)
    ocr_processed_strict = strict_text_processing(ocr_text_raw)

    if verbose:
        print("\n--- Processed Ground Truth Snippet (Strict) ---")
        print(gt_processed_strict[:500] +
              ("..." if len(gt_processed_strict) > 500 else ""))
        print("--- End Processed Ground Truth ---")
        print("\n--- Processed OCR Output Snippet (Strict) ---")
        print(ocr_processed_strict[:500] +
              ("..." if len(ocr_processed_strict) > 500 else ""))
        print("--- End Processed OCR Output ---\n")

    # 3. Calculate Metrics - Full Comparison
    print("Step 3a: Calculating evaluation metrics (Full Comparison)...")
    full_metrics = calculate_comparison_metrics(
        gt_processed_strict, ocr_processed_strict, label_suffix=" (Full)"
    )
    print(
        f"Calculated Strict Sequence Similarity (Full): {full_metrics.get('strict_sequence_similarity (Full)', 'N/A'):.4f}")
    print(
        f"Calculated Strict WER (Full): {full_metrics.get('strict_wer (Full)', 'N/A'):.4f}")
    print(
        f"Calculated Strict CER (Full): {full_metrics.get('strict_cer (Full)', 'N/A'):.4f}")
    print(
        f"Calculated Strict NED (Full): {full_metrics.get('strict_ned (Full)', 'N/A'):.4f}")

    if verbose and full_metrics.get('strict_wer (Full)', 0) > 0:
        print("\n--- Alignment Details (Full Comparison) ---")
        try:
            # Calculate detailed measures including alignment visualization
            # Need to recompute slightly redundantly to get the object for visualization
            detailed_measures = jiwer.compute_measures(
                gt_processed_strict, ocr_processed_strict)
            print(jiwer.visualize_alignment(
                detailed_measures, show_measures=False))
            print("-----------------------------------------")
        except Exception as e:
            print(f"Could not generate detailed alignment: {e}")

    # 4. Calculate Metrics - Truncated Comparison (based on OCR token count)
    print("\nStep 3b: Calculating evaluation metrics (Truncated Comparison)...")

    # Tokenize processed texts
    gt_tokens = tokenize(gt_processed_strict)
    ocr_tokens = tokenize(ocr_processed_strict)
    num_ocr_tokens = len(ocr_tokens)

    print(
        f"Ground Truth Tokens: {len(gt_tokens)}, OCR Tokens: {num_ocr_tokens}")

    truncated_metrics = {}
    if num_ocr_tokens == 0:
        print("OCR result has 0 tokens after processing. Truncated comparison metrics will reflect this.")
        # If OCR is empty, compare empty GT prefix to empty OCR
        gt_processed_strict_truncated = ""
        truncated_metrics = calculate_comparison_metrics(
            gt_processed_strict_truncated, ocr_processed_strict, label_suffix=" (Truncated)"
        )
    elif len(gt_tokens) < num_ocr_tokens:
        print("Warning: Ground truth has fewer tokens than OCR output. Truncated comparison will use full ground truth.")
        # Compare full GT to full OCR (same as full comparison in this edge case)
        gt_processed_strict_truncated = gt_processed_strict  # Use the full GT
        # Recalculate with the appropriate label
        truncated_metrics = calculate_comparison_metrics(
            gt_processed_strict_truncated, ocr_processed_strict, label_suffix=" (Truncated)"
        )
    else:
        # Truncate GT tokens and rejoin
        gt_tokens_truncated = gt_tokens[:num_ocr_tokens]
        gt_processed_strict_truncated = " ".join(gt_tokens_truncated)

        truncated_metrics = calculate_comparison_metrics(
            gt_processed_strict_truncated, ocr_processed_strict, label_suffix=" (Truncated)"
        )

    print(
        f"Calculated Strict Sequence Similarity (Truncated): {truncated_metrics.get('strict_sequence_similarity (Truncated)', 'N/A'):.4f}")
    print(
        f"Calculated Strict WER (Truncated): {truncated_metrics.get('strict_wer (Truncated)', 'N/A'):.4f}")
    print(
        f"Calculated Strict CER (Truncated): {truncated_metrics.get('strict_cer (Truncated)', 'N/A'):.4f}")
    print(
        f"Calculated Strict NED (Truncated): {truncated_metrics.get('strict_ned (Truncated)', 'N/A'):.4f}")

    if verbose and truncated_metrics.get('strict_wer (Truncated)', 0) > 0 and num_ocr_tokens > 0 and len(gt_tokens) >= num_ocr_tokens:
        print("\n--- Alignment Details (Truncated Comparison) ---")
        try:
            # Ensure we use the correctly truncated GT for visualization
            gt_viz = " ".join(tokenize(gt_processed_strict)[:num_ocr_tokens])
            detailed_measures_trunc = jiwer.compute_measures(
                gt_viz, ocr_processed_strict)
            print(jiwer.visualize_alignment(
                detailed_measures_trunc, show_measures=False))
            print("----------------------------------------------")
        except Exception as e:
            print(
                f"Could not generate detailed alignment for truncated comparison: {e}")

    print("\n--- Evaluation Complete ---")

    # Combine results
    results = {
        'ground_truth_len_raw_chars': len(ground_truth_text),
        'ocr_text_len_raw_chars': len(ocr_text_raw),
        'ground_truth_len_processed_chars': len(gt_processed_strict),
        'ocr_text_len_processed_chars': len(ocr_processed_strict),
        'ground_truth_tokens': len(gt_tokens),
        'ocr_tokens': num_ocr_tokens,
        # True if OCR didn't have critical failure
        'ocr_success': ocr_text_raw is not None,
        # True if OCR returned non-empty string
        'ocr_found_text': bool(ocr_text_raw)
    }
    results.update(full_metrics)
    results.update(truncated_metrics)

    return results


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a generated document image using OCR and strict, order-preserving text comparison metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-i", "--image", required=True,
        help="Path to the generated document image file."
    )

    # Group for ground truth input
    gt_group = parser.add_mutually_exclusive_group(required=True)
    gt_group.add_argument(
        "-t", "--text",
        help="Direct ground truth text string."
    )
    gt_group.add_argument(
        "-f", "--file",
        help="Path to the file containing the ground truth text (UTF-8 encoded)."
    )

    # OCR Options
    parser.add_argument(
        "-l", "--lang", default='eng',
        help="Tesseract language code(s) (e.g., 'eng', 'eng+fra')."
    )
    parser.add_argument(
        "--psm", type=int, default=3, choices=range(0, 14),
        help="Tesseract Page Segmentation Mode (see tesseract --help-psm)."
    )

    # Output Options
    parser.add_argument(
        "-v", "--verbose", action='store_true',
        help="Enable verbose output, including text snippets and alignment details."
    )

    args = parser.parse_args()

    # --- Get Ground Truth Text ---
    ground_truth = ""
    if args.text:
        ground_truth = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: Ground truth file not found at '{args.file}'")
            exit(1)
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        except Exception as e:
            print(f"Error reading ground truth file '{args.file}': {e}")
            exit(1)

    # --- Run Evaluation ---
    evaluation_results = evaluate_document_image(
        ground_truth_text=ground_truth,
        generated_image_path=args.image,
        ocr_lang=args.lang,
        ocr_psm=args.psm,
        verbose=args.verbose
    )

    # --- Print Results ---
    print("\n==============================================")
    print("            Strict Evaluation Results         ")
    print("==============================================")
    if evaluation_results:
        print("--- Full Comparison Metrics ---")
        sim_full = evaluation_results.get('strict_sequence_similarity (Full)')
        wer_full = evaluation_results.get('strict_wer (Full)')
        cer_full = evaluation_results.get('strict_cer (Full)')
        ned_full = evaluation_results.get('strict_ned (Full)')
        print(
            f"Strict Sequence Similarity (0-1, higher=better): {sim_full:.4f}" if sim_full is not None else "Strict Sequence Similarity: N/A")
        print(
            f"Strict Word Error Rate (WER) (0+, lower=better) : {wer_full:.4f}" if wer_full is not None else "Strict WER: N/A")
        print(
            f"Strict Character Error Rate (CER) (0+, lower=better): {cer_full:.4f}" if cer_full is not None else "Strict CER: N/A")
        print(
            f"Strict Normalized Edit Distance (NED) (0+, lower=better): {ned_full:.4f}" if ned_full is not None else "Strict NED: N/A")

        print("\n--- Truncated Comparison Metrics ---")
        print(
            f"(Comparing OCR output vs. Ground Truth truncated to {evaluation_results.get('ocr_tokens', 'N/A')} tokens)")
        sim_trunc = evaluation_results.get(
            'strict_sequence_similarity (Truncated)')
        wer_trunc = evaluation_results.get('strict_wer (Truncated)')
        cer_trunc = evaluation_results.get('strict_cer (Truncated)')
        ned_trunc = evaluation_results.get('strict_ned (Truncated)')
        print(
            f"Strict Sequence Similarity (0-1, higher=better): {sim_trunc:.4f}" if sim_trunc is not None else "Strict Sequence Similarity (Truncated): N/A")
        print(
            f"Strict Word Error Rate (WER) (0+, lower=better) : {wer_trunc:.4f}" if wer_trunc is not None else "Strict WER (Truncated): N/A")
        print(
            f"Strict Character Error Rate (CER) (0+, lower=better): {cer_trunc:.4f}" if cer_trunc is not None else "Strict CER (Truncated): N/A")
        print(
            f"Strict Normalized Edit Distance (NED) (0-1, lower=better): {ned_trunc:.4f}" if ned_trunc is not None else "Strict NED (Truncated): N/A")

        print("\n--- Text Statistics ---")
        print(
            f"Ground Truth Length (raw chars) : {evaluation_results['ground_truth_len_raw_chars']}")
        print(
            f"OCR Output Length (raw chars)   : {evaluation_results['ocr_text_len_raw_chars']}")
        print(
            f"Ground Truth Tokens (processed) : {evaluation_results['ground_truth_tokens']}")
        print(
            f"OCR Output Tokens (processed)   : {evaluation_results['ocr_tokens']}")
        print(
            f"OCR Success (Critical Check)    : {evaluation_results['ocr_success']}")
        print(
            f"OCR Found Any Text              : {evaluation_results['ocr_found_text']}")
    else:
        print("Evaluation could not be completed due to critical OCR failure.")
        print(f"(Check image path '{args.image}' and Tesseract installation)")
    print("==============================================")