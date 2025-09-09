# STRICT: Stress Test of Rendering Images Containing Text
[![Paper](https://img.shields.io/badge/Paper-000000.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.18985) 

An overview of the current SOTA-level image-generation model (lower -> better):

![image](https://github.com/tianyu-z/STRICT-Bench/blob/main/heatmap.png)

This project is designed to generate images containing text using various state-of-the-art image generation models and then evaluate the accuracy of the generated text using Optical Character Recognition (OCR). The main script, `main.py`, orchestrates the image generation process based on input text data and subsequently triggers an OCR evaluation pipeline to assess how well the models render text in images.

*Input Text (Ground Truth):*

```text
Asia (, ) is the largest continent in the world by both land area and population. It covers an area of more than 44 million square kilometres, about 30% of Earth's total land area and 8% of Earth's total surface area. The continent, which has long been home to the majority of the human population, was the site of many of the first civilisations. Its 4.7 billion people constitute roughly 60% of the world's population.
```

The generated images are

![image](https://github.com/tianyu-z/STRICT-Bench/blob/main/Figure.png)



## Project Overview

The primary goal is to test and benchmark the text rendering capabilities of different image generation models. This involves:
1.  **Image Generation**: Taking text snippets in various languages and lengths, and instructing different AI models to generate images depicting these texts.
2.  **OCR Evaluation**: Extracting text from the generated images using OCR and comparing it against the original ground truth text to measure accuracy.

This pipeline is useful for understanding how accurately models can render specific sequences of words, whether accuracy degrades with increasing text length, and how well the generated text layout survives OCR.

---

## `main.py` Functionality

The `main.py` script is the central component of this pipeline. Its key responsibilities include:

* **Image Generation**:
    * It supports multiple image generation services/models as defined in `AVAILABLE_SERVICES` (e.g., OpenAI's DALL-E, Google's Imagen & Gemini, Recraft, HiDream, Seedream).
    * It reads input text data from a JSON file specified by `INPUT_JSON_PATH` (e.g., `text_data/sample_data.json`). This file maps language codes (e.g., "en", "fr", "zh") to a list of records, each containing a `rowid` and the `text` to be rendered. If the input JSON file doesn't exist, a sample file is automatically created.
    * For each text record, it constructs a prompt using a `BASE_PROMPT` (e.g., "Produce an image of a typed document page with the following text: ") combined with a snippet of the original text. The length of this snippet is controlled by the `LENGTH_LIST` setting.
    * Generated images are saved in PNG format within the `OUTPUT_DIR` (default: `generated_output/`), organized by model, text length, and language code (e.g., `generated_output/<model_name_length>/<lang_code>/<rowid>.png`).
    * The corresponding ground truth text snippet used for generation is saved as a `.txt` file alongside the image (e.g., `<rowid>.txt`).
* **Concurrency and Retries**:
    * Manages concurrent image generation jobs using `MAX_CONCURRENT_IMAGE_JOBS`.
    * Implements a retry mechanism (`MAX_RETRIES_PER_IMAGE` attempts with `RETRY_DELAY_SECONDS` delay) for failed API calls.
* **OCR Orchestration**:
    * If `OCR_ENABLE` is true, `main.py` will automatically initiate the OCR evaluation process after image generation is complete.
    * It collects all successfully generated image-text pairs from the `OUTPUT_DIR`.
    * It then uses the OCR module (either `ocr/core.py` or `ocr/core_clip.py` based on `OCR_TOKENIZER` setting) to perform the evaluation.
    * The aggregated OCR evaluation results are saved as a JSON file (e.g., `ocr_evaluation_results_<timestamp>.json`) in the `OUTPUT_DIR`.

---

## Configuration

Key configurations for `main.py` are set directly within the script:

* **`AVAILABLE_SERVICES`**: A list of strings specifying which image generation services to use (e.g., `["gpt", "imagen", "recraft"]`).
* **`MAX_CONCURRENT_IMAGE_JOBS`**: Integer defining how many image generation tasks run in parallel.
* **`INPUT_JSON_PATH`**: String path to the input JSON file containing text data.
* **`OUTPUT_DIR`**: String path to the directory where generated files (images, texts, OCR results) will be saved.
* **`LENGTH_LIST`**: A list of integers specifying the character lengths of text snippets to be used for generation.
* **`BASE_PROMPT`**: The base instruction string prefixed to the text snippet for the image generation prompt.
* **API Keys**:
    * `OPENAI_API_KEY`, `RECRAFT_API_KEY`, `GOOGLE_API_KEY`, `HIDREAM_API_KEY`, `SEEDREAM_API_KEY`: API keys for the respective services.
    * Associated base URLs or model identifiers (e.g., `OPENAI_IMAGE_MODEL`, `IMAGEN_MODEL`, `HIDREAM_IMAGE_MODEL`).
* **OCR Settings**:
    * **`OCR_ENABLE`**: Boolean to enable/disable the OCR evaluation phase.
    * **`OCR_TOKENIZER`**: String ("CLIP" or "OTHER") to select the tokenizer and corresponding OCR core module.
    * **`OCR_DEFAULT_LANGUAGE_FOR_TESSERACT`**: Default Tesseract language (e.g., "eng"). Specific mappings for "zh" (Chinese) and "fr" (French) are handled internally.
    * **`OCR_PSM`**: Tesseract Page Segmentation Mode.
    * **`OCR_N_BOOTSTRAP`**: Number of bootstrap samples for OCR statistics.
    * **`OCR_VERBOSE`**: Boolean for verbose OCR output.
    * **`OCR_OUTPUT_FILENAME_PREFIX`**: Prefix for the OCR results JSON file.

---

## OCR Evaluation Module

After images are generated, `main.py` can automatically run an OCR evaluation. This process:
1.  Scans the `OUTPUT_DIR` for all generated image (`.png`) and ground truth text (`.txt`) pairs.
2.  For each pair, it uses Tesseract OCR (via `ocr/core.py` or `ocr/core_clip.py`) to extract text from the image.
3.  The extracted text is then compared against the ground truth text snippet.
4.  Various metrics are calculated, such as Sequence Similarity, Word Error Rate (WER), Character Error Rate (CER), and Normalized Edit Distance (NED), for both full and truncated text comparisons.
5.  The results, including individual pair metrics and aggregate statistics (with confidence intervals), are saved to a JSON file in the `OUTPUT_DIR`.

For more detailed information on the OCR metrics, preprocessing steps, and how to interpret the OCR results, please refer to the `ocr/readme.md` file.

---

## File Structure

The project is organized as follows:
```
strict_data_flow/
├── main.py                     # Main script for image generation and orchestrating OCR
├── merge_json_files.py         # Utility to combine language JSON files
├── readme.md                   # This README file
├── text_data/                  # Input text data for generation
│   ├── sample_data.json        # Example input file (or user-created)
│   └── ... (other lang json files e.g., en.json, fr.json)
├── generated_output/           # Output directory for generated images, texts, and OCR results
│   ├── &lt;model_name_length>/    # e.g., gpt-image-1_100
│   │   ├── &lt;lang_code>/        # e.g., en, fr, zh
│   │   │   ├── &lt;rowid>.png
│   │   │   └── &lt;rowid>.txt
│   └── ocr_evaluation_results_&lt;timestamp>.json # OCR results if enabled
└── ocr/                        # OCR evaluation module
├── core.py                 # Core OCR logic (standard tokenizer)
├── core_clip.py            # Core OCR logic (CLIP tokenizer)
├── ned.py                  # Normalized Edit Distance calculation
└── readme.md               # Detailed README for the OCR module
```
## Prerequisites

1.  **Python 3** (tested with >3.7).
2.  **API Keys**: Valid API keys for the image generation services you intend to use (configured in `main.py`).
3.  **Python Libraries**: Install the required libraries. You can typically install them using pip:
    ```bash
    pip install openai google-generativeai Pillow requests numpy
    ```
4.  **For OCR Evaluation**:
    * **Tesseract OCR Engine**: Must be installed and accessible in your system's PATH. See the [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html).
    * **Tesseract Language Packs**: Install language packs for the languages you'll be working with (e.g., `tesseract-ocr-eng`, `tesseract-ocr-chi-sim`).
    * Additional Python libraries for OCR (as listed in `ocr/readme.md`):
        ```bash
        pip install pytesseract jiwer python-Levenshtein transformers
        ```
        *(Note: `transformers` is needed if `OCR_TOKENIZER = "CLIP"`).*

---

## Setup & Running

1.  **Clone the Repository**:
    ```bash
    # git clone <repository_url>
    # cd strict_data_flow
    ```
2.  **Install Dependencies**:
    Install all necessary Python libraries as listed in the Prerequisites section.
3.  **Configure `main.py`**:
    * Open `main.py` and fill in your API keys for the desired services (e.g., `OPENAI_API_KEY`, `GOOGLE_API_KEY`).
    * Set `AVAILABLE_SERVICES` to include the models you want to test.
    * Adjust `INPUT_JSON_PATH`, `OUTPUT_DIR`, `LENGTH_LIST`, and other parameters as needed.
    * Configure OCR settings if you plan to use the OCR evaluation feature.
4.  **Prepare Input Data**:
    * Ensure your input text data is formatted correctly in a JSON file (e.g., `text_data/sample_data.json`). The structure should be:
        ```json
        {
          "en": [
            {"rowid": "en_sample1", "text": "This is an English sentence."},
            {"rowid": "en_sample2", "text": "Another example for testing."}
          ],
          "fr": [
            {"rowid": "fr_sample1", "text": "Ceci est une phrase en français."}
          ]
        }
        ```
    * You can use the `merge_json_files.py` script to combine multiple language-specific JSON files (e.g., `en.json`, `fr.json`) into a single file suitable for `INPUT_JSON_PATH`. To use it, configure the `language_files` map and `collated_output_path` in `merge_json_files.py` and run `python merge_json_files.py`.
5.  **Run the Script**:
    Execute `main.py` from the root directory (`strict_data_flow/`):
    ```bash
    python main.py
    ```
    The script will print progress messages to the console.

---

## Output

* **Generated Images & Texts**:
    * Images and their corresponding ground truth text files will be saved in subdirectories under `OUTPUT_DIR`, structured as: `OUTPUT_DIR/<model_name_text_length>/<language_code>/`. For example: `generated_output/gpt-image-1_100/en/en_sample1.png` and `generated_output/gpt-image-1_100/en/en_sample1.txt`.
* **OCR Evaluation Results**:
    * If `OCR_ENABLE` is true, a JSON file named `ocr_evaluation_results_<timestamp>.json` will be created in `OUTPUT_DIR`. This file contains detailed metrics for each image-text pair evaluated and aggregate statistics across all pairs.

---
## `merge_json_files.py` Utility

The `merge_json_files.py` script is a helper utility to combine multiple language-specific JSON files into a single JSON object. Each language file is expected to contain a list of records. The output is a JSON file where keys are language codes and values are the lists of records from the respective language files. This can be useful for preparing the `INPUT_JSON_PATH` file for `main.py`.

**Usage**:
1.  Modify the `language_files` dictionary in `merge_json_files.py` to map your language codes to their corresponding JSON file paths.
2.  Set the `collated_output_path` variable to your desired output file path.
3.  Run the script: `python merge_json_files.py`.

Example `language_files` map in `merge_json_files.py`:
```python
language_files = {
    "en": "text_data/en.json",
    "fr": "text_data/fr.json",
    "zh": "text_data/zh.json"
}
