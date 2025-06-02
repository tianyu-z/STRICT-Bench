## Prerequisites

1.  **Python 3:** Ensure you have Python 3 (tested with \>3.7) installed.
2.  **Tesseract OCR:** This tool relies on the Tesseract OCR engine.
      * **Installation:** Install Tesseract OCR for your operating system. Follow the official guide: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
      * **Language Packs:** Install the necessary language data packs for the languages you intend to OCR (e.g., `tesseract-ocr-eng` for English, `tesseract-ocr-deu` for German).
      * **PATH Configuration:** Make sure the `tesseract` executable is in your system's PATH. If not, you might need to configure the path within the `core.py` script (if it uses `pytesseract`) by setting `pytesseract.pytesseract.tesseract_cmd`.
3.  **Python Libraries:** Install the required Python libraries using pip:
    ```bash
    pip install Pillow pytesseract numpy jiwer python-Levenshtein
    ```
      * `Pillow`: For image handling.
      * `pytesseract`: Python wrapper for Tesseract.
      * `numpy`: For numerical operations (used in bootstrapping).
      * `jiwer`: For efficient WER and CER calculation.
      * `python-Levenshtein`: For calculating Normalized Edit Distance (NED).

## File Structure

You need three Python files, typically in the same directory:

1.  `core.py`: Contains the core function (`evaluate_document_image`) for OCR, text processing, and metric calculation for a single pair. (Assumed to exist).
2.  `main.py`: The main script to run the evaluation across multiple text/image pairs, handle file discovery, calculate aggregate statistics, and manage command-line arguments.
3.  `ned.py`: Contains the function to calculate Normalized Edit Distance.

You also need your evaluation data (ground truth text and generated images).

## Input Data Format

The script (`main.py`) can find text-image pairs in two ways:

1.  **Directory Method:**

      * Place ground truth `.txt` files and their corresponding generated image files (e.g., `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`) in the *same directory*.
      * The script expects pairs to share the same base filename (e.g., `document1.txt` and `document1.png`).
      * Example:
        ```
        evaluation_data/
        ├── report_page1.txt
        ├── report_page1.png
        ├── memo_final.txt
        └── memo_final.jpg
        ```

2.  **JSON File Method:**

      * Create a JSON file containing a list of objects, where each object specifies the paths to a `text` file and an `image` file. Paths can be absolute or relative *to the location of the JSON file*.
      * Example (`pairs.json`):
        ```json
        [
          {
            "text": "ground_truth/doc_a.txt",
            "image": "generated_images/doc_a_generated.png"
          },
          {
            "text": "../archive/texts/doc_b.txt",
            "image": "/output/images/doc_b.jpg"
          }
        ]
        ```

## How to Run

Use the `main.py` script to run the evaluation.

**Basic Command:**

```bash
python main.py --input_path <path_to_data_or_json>
```

  * `<path_to_data_or_json>`: This is the path to either:
      * The directory containing your `.txt` and image files (Directory Method).
      * The `.json` file listing the pairs (JSON File Method).

**Options:**

  * `--input_path`: **Required.** Path to the data directory or the input JSON file.
  * `-o <filename>`, `--output <filename>`: Specify the path for the output JSON file containing the evaluation results. (Default: `evaluation_results.json`)
  * `-l <lang>`, `--lang <lang>`: Tesseract language code(s) (e.g., 'eng', 'fra', 'eng+deu'). (Default: `eng`). Please find the language codes in the [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).
  * `--psm <mode>`: Tesseract Page Segmentation Mode (0-13). See `tesseract --help-psm` for details. (Default: `3` - Auto page segmentation with OSD)
  * `--no-lowercase`: Disable conversion to lowercase before comparison. Use if case sensitivity is important for your evaluation. (Default: Lowercase is enabled)
  * `--keep-punctuation`: Disable removal of punctuation before comparison. Use if punctuation accuracy is critical. (Default: Punctuation is removed)
  * `--n_bootstrap <N>`: Number of bootstrap samples for calculating confidence intervals for aggregate statistics. (Default: `1000`)
  * `-v`, `--verbose`: Print verbose output from the core evaluation function for each pair.

**Examples:**

1.  **Evaluate pairs in a directory `data/` and save results to default `evaluation_results.json`:**

    ```bash
    python main.py --input_path data/
    ```

2.  **Evaluate pairs listed in `config/pairs.json`, save to `run1_results.json`, use German language, keep punctuation, and enable verbose mode:**

    ```bash
    python main.py --input_path config/pairs.json -o run1_results.json -l deu --keep-punctuation -v
    ```

3.  **Evaluate English text assuming single text block (PSM 6), ignoring case and punctuation (default behavior):**

    ```bash
    python main.py --input_path documents/ --psm 6
    ```

**Console Output:**
After processing all pairs, the script prints a summary of the aggregate statistics to the console, similar to the `aggregate_statistics` section in the output JSON file.

## Output

The script generates a JSON file (e.g., `evaluation_results.json`) with the following structure:

```json
{
  "evaluation_config": {
    "ocr_language": "eng",
    "ocr_psm": 3,
  },
  "individual_results": [
    {
      "text_file": "data/doc1.txt",
      "image_file": "data/doc1.png",
      "status": "Success", // or "OCR_EMPTY", "OCR_FAILED", "EVALUATION_ERROR"
      "strict_sequence_similarity (Full)": 0.985,
      "strict_sequence_similarity (Truncated)": 0.987,
      "strict_wer (Full)": 0.021,
      "strict_wer (Truncated)": 0.020,
      "strict_cer (Full)": 0.008,
      "strict_cer (Truncated)": 0.007,
      "strict_ned (Full)": 0.015,
      "strict_ned (Truncated)": 0.014,
      "ground_truth_len_raw_chars": 512,
      "ocr_text_len_raw_chars": 509,    // Null if OCR failed critically
      "ground_truth_tokens": 95,
      "ocr_tokens": 94              // Null if OCR failed critically
    },
    // ... more pairs, potentially with different statuses or null metrics on failure
    {
      "text_file": "data/doc_fail.txt",
      "image_file": "data/doc_fail.png",
      "status": "OCR_FAILED",
      "strict_sequence_similarity (Full)": null,
      "strict_sequence_similarity (Truncated)": null,
      "strict_wer (Full)": null,
      "strict_wer (Truncated)": null,
      "strict_cer (Full)": null,
      "strict_cer (Truncated)": null,
      "strict_ned (Full)": null,
      "strict_ned (Truncated)": null,
      "ground_truth_len_raw_chars": null, // Or the length read from file
      "ocr_text_len_raw_chars": null,
      "ground_truth_tokens": null,      // Or the count from file
      "ocr_tokens": null
    }
  ],
  "aggregate_statistics": {
    "total_pairs_processed": 50,
    "successful_evaluations": 48,       // Count where core eval ran (incl. OCR_EMPTY)
    "failed_evaluations": 2,            // Count of OCR_FAILED or EVALUATION_ERROR
    "total_ground_truth_chars": 25600,  // Sum from successfully read GT files
    "total_ground_truth_tokens": 4800,  // Sum from successfully read GT files
    "total_ocr_chars": 25480,           // Sum from successful OCR runs (non-null OCR text)
    "total_ocr_tokens": 4750,           // Sum from successful OCR runs (non-null OCR text)
    "strict_sequence_similarity (Full)": {
      "mean": 0.9721,
      "std_dev": 0.0153,
      "ci_lower": 0.9680,
      "ci_upper": 0.9762
    },
    "strict_sequence_similarity (Truncated)": {
      "mean": 0.9755,
      "std_dev": 0.0148,
      "ci_lower": 0.9711,
      "ci_upper": 0.9790
    },
    "strict_wer (Full)": {
      "mean": 0.0352,
      "std_dev": 0.0181,
      "ci_lower": 0.0301,
      "ci_upper": 0.0403
    },
    "strict_wer (Truncated)": {
      "mean": 0.0339,
      "std_dev": 0.0175,
      "ci_lower": 0.0295,
      "ci_upper": 0.0388
    },
    "strict_cer (Full)": {
      "mean": 0.0124,
      "std_dev": 0.0092,
      "ci_lower": 0.0091,
      "ci_upper": 0.0155
    },
    "strict_cer (Truncated)": {
      "mean": 0.0118,
      "std_dev": 0.0088,
      "ci_lower": 0.0085,
      "ci_upper": 0.0149
    },
    "strict_ned (Full)": {
      "mean": 0.0180,
      "std_dev": 0.0090,
      "ci_lower": 0.0150,
      "ci_upper": 0.0210
    },
    "strict_ned (Truncated)": {
      "mean": 0.0170,
      "std_dev": 0.0085,
      "ci_lower": 0.0140,
      "ci_upper": 0.0200
    }
  }
}
```

  * **`evaluation_config`**: Records the settings used for the run (OCR lang, PSM, preprocessing flags).
  * **`individual_results`**: A list containing detailed metrics for each processed pair. `status` indicates if OCR succeeded (`Success`), returned empty text (`OCR_EMPTY`), failed critically (`OCR_FAILED`), or if an error occurred during the evaluation process (`EVALUATION_ERROR`). Metrics might be `null` if evaluation couldn't complete. Includes character and token counts for both ground truth and OCR output.
  * **`aggregate_statistics`**: Summary statistics calculated across all pairs where the core evaluation function ran successfully (i.e., status is `Success` or `OCR_EMPTY`). Includes total counts and bootstrapped mean, standard deviation (`std_dev`), and 95% confidence intervals (`ci_lower`, `ci_upper`) for each core metric (both Full and Truncated versions).

## Interpreting Results

  * **Sequence Similarity (Full / Truncated):** Ranges from 0.0 (no match) to 1.0 (perfect match). Higher is better. Based on `difflib`, sensitive to sequence changes. The "Truncated" version might adjust the comparison length.
  * **Word Error Rate (WER) (Full / Truncated):** Ranges from 0.0 (perfect match) upwards. Lower is better. Represents the percentage of words that need to be changed (substituted, inserted, deleted) to match the ground truth. A WER of 0.1 means 10% of words were incorrect relative to the ground truth length (for Full WER).
  * **Character Error Rate (CER) (Full / Truncated):** Ranges from 0.0 (perfect match) upwards. Lower is better. Similar to WER but at the character level. Useful for measuring fine-grained accuracy. A CER of 0.05 means 5% of characters were incorrect relative to the ground truth length (for Full CER).
  * **Normalized Edit Distance (NED) (Full / Truncated):** Ranges from 0.0 (perfect match) to 1.0. Lower is better. This metric calculates the Levenshtein (edit) distance between the ground truth and OCR output, and then normalizes it by the total number of operations (matches, substitutions, insertions, deletions) in the optimal alignment path. It provides a measure of dissimilarity that accounts for the overall complexity of transforming one string into another.
  * **Counts:** `total_ground_truth_chars/tokens` and `total_ocr_chars/tokens` give an idea of the volume of text processed and how much text the OCR successfully extracted across the dataset.
  * **Status:** Check individual results for `status` flags to understand specific failures (e.g., `OCR_FAILED` vs. `OCR_EMPTY`). Aggregate stats exclude pairs with `OCR_FAILED` or `EVALUATION_ERROR`.
  * **Confidence Intervals (CI):** The 95% CI [ci\_lower, ci\_upper] gives a range within which the true mean of the metric is likely to fall. Narrower intervals indicate more certainty, often seen with larger datasets or less variance.

Consider the `evaluation_config` when comparing results from different runs, as OCR settings (`lang`, `psm`) and preprocessing choices (`--no-lowercase`, `--keep-punctuation`) significantly impact the metrics. The aggregate statistics provide an overall measure of the generator's text rendering quality across your test set under the chosen configuration.
