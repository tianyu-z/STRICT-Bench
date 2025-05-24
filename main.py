import time
from openai import OpenAI
import os
import json
import base64
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from google import genai
from google.genai import types as google_types
from PIL import Image
from io import BytesIO
import requests
from urllib.request import urlretrieve
import random

# --- General Script Configuration ---
# specify model here, e.g. ["gpt","recraft"]
# available model choices ["gpt","recraft","imagen","gemini","hidream","seedream"]
AVAILABLE_SERVICES = ["imagen"] 
MAX_CONCURRENT_IMAGE_JOBS = 1
INPUT_JSON_PATH = "text_data/sample_data.json"
OUTPUT_DIR = "generated_output"
LENGTH_LIST = [5]
# prompt we use to generate image, as specified in the paper
BASE_PROMPT = "Produce an image of a typed document page with the following text: "

# --- Retry Configuration ---
MAX_RETRIES_PER_IMAGE = 3
RETRY_DELAY_SECONDS = 5

# --- OpenAI/GPT Configuration ---
OPENAI_API_KEY = "OPENAI_API_KEY"
OPENAI_BASE_URL = "GPT_BASE_URL" # in case you are using a custom endpoint from non-official OpenAI API providers
OPENAI_IMAGE_MODEL = "gpt-image-1"

# --- Recraft Configuration ---
RECRAFT_API_KEY = "RECRAFT_API_KEY"
RECRAFT_BASE_URL = "RECRAFT_BASE_URL"
RECRAFT_IMAGE_MODEL = "recraftv3"

# --- Google AI (Imagen & Gemini) Configuration ---
GOOGLE_API_KEY = "GOOGLE_API_KEY"
IMAGEN_MODEL = "imagen-3.0-generate-002"
GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"

# --- HiDream Configuration ---
HIDREAM_API_KEY = "HIDREAM_API_KEY"
HIDREAM_SUBMIT_URL = "HIDREAM_SUBMIT_URL"
HIDREAM_RESULT_BASE_URL = "HIDREAM_RESULT_BASE_URL"
HIDREAM_IMAGE_MODEL = "hidream-i1-full"
HIDREAM_POLLING_INTERVAL_SECONDS = 5
HIDREAM_MAX_POLLING_TIME_SECONDS = 300
HIDREAM_IMAGE_WIDTH = 1024
HIDREAM_IMAGE_HEIGHT = 1024

# --- Seedream Configuration ---
SEEDREAM_API_KEY = "SEEDREAM_API_KEY"
SEEDREAM_API_URL = "SEEDREAM_API_URL"
SEEDREAM_MODEL_IDENTIFIER = "SEEDREAM_MODEL_IDENTIFIER"
SEEDREAM_GUIDANCE_SCALE = 2.5
SEEDREAM_ENABLE_WATERMARK = True
SEEDREAM_IMAGE_WIDTH = 1024
SEEDREAM_IMAGE_HEIGHT = 1024

# --- OCR Configuration ---
OCR_ENABLE = True
OCR_TOKENIZER = "OTHER"
OCR_DEFAULT_LANGUAGE_FOR_TESSERACT = "eng"
OCR_PSM = 3
OCR_N_BOOTSTRAP = 1000
OCR_VERBOSE = False
OCR_OUTPUT_FILENAME_PREFIX = "ocr_evaluation_results_"

# --- OCR Core Module Import ---
# This assumes ocr/core_clip.py and ocr/core.py exist in an 'ocr' subdirectory
# relative to where main.py is executed.
ocr_core = None
if OCR_ENABLE:
    try:
        if OCR_TOKENIZER == "CLIP":
            from ocr import core_clip as ocr_core_module
            ocr_core = ocr_core_module
        else:
            from ocr import core as ocr_core_module
            ocr_core = ocr_core_module
        print(f"OCR Core module ({OCR_TOKENIZER}) loaded successfully.")
    except ImportError as e:
        print(f"ERROR: Failed to import OCR core module (tokenizer: {OCR_TOKENIZER}). OCR will be disabled. Details: {e}")
        OCR_ENABLE = False # Disable OCR if import fails
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while importing OCR core module. OCR will be disabled. Details: {e}")
        OCR_ENABLE = False

print_lock = threading.Lock()

def tprint(message):
    with print_lock:
        print(message)

def generate_image_via_api(client, prompt_text, image_model_name, service_name_for_log="UnknownService"):
    if not image_model_name:
        tprint(f"Error ({service_name_for_log}): No image_model_name provided to generate_image_via_api.")
        return None, 0

    begin = time.time()
    try:
        params = {
            "model": image_model_name,
            "prompt": prompt_text,
            "n": 1,
            "size": "1024x1024",
        }
        if "gpt" not in image_model_name:
            params["response_format"] = "b64_json"
        result = client.images.generate(**params)

        end = time.time()
        generation_time = end - begin
        image_base64 = result.data[0].b64_json
        return image_base64, generation_time
    except Exception as e:
        end = time.time()
        generation_time = end - begin
        tprint(f"({service_name_for_log}) API call to image_model '{image_model_name}' failed: {str(e)}, took {generation_time:.2f} seconds.")
        return None, generation_time

def generate_image_google_ai(google_api_key_param, prompt_text, model_name_google, service_name_for_log="GoogleAI"):
    """
    Call Google AI (Imagen or Gemini) to generate an image.
    Returns (PIL.Image, generation_time)
    """
    try:
        # It's important to use a distinct name for the genai client if it's initialized here
        # to avoid potential conflicts if 'client' is used elsewhere with a different type.
        google_client = genai.Client(api_key=google_api_key_param)
    except Exception as e:
        tprint(f"({service_name_for_log}) Failed to initialize Google AI client: {e}")
        return None, 0

    begin = time.time()
    pil_image = None
    
    try:
        if service_name_for_log.startswith("imagen"):
            # tprint(f"DEBUG: Imagen call with model {model_name_google}, prompt: {prompt_text[:60]}...")
            response = google_client.models.generate_images(
                model=model_name_google, 
                prompt=prompt_text,
                config=google_types.GenerateImagesConfig(
                    number_of_images=1,
                    include_rai_reason=True,
                    output_mime_type='image/jpeg',
                )
            )
            if response.generated_images and response.generated_images[0].image.image_bytes:
                pil_image = Image.open(BytesIO(response.generated_images[0].image.image_bytes))
            else:
                tprint(f"({service_name_for_log}) Image generation failed for model '{model_name_google}' - no image data returned.")

        elif service_name_for_log.startswith("gemini"):
            # tprint(f"DEBUG: Gemini call with model {model_name_google}, contents: {prompt_text[:60]}...")
            response = google_client.models.generate_content(
                model=model_name_google, 
                contents=prompt_text,
                config=google_types.GenerateContentConfig(
                    response_modalities=['IMAGE', 'TEXT']
                )
            )
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        pil_image = Image.open(BytesIO(part.inline_data.data))
                        break 
            else:
                 tprint(f"({service_name_for_log}) Gemini model '{model_name_google}' did not return any candidate content.")
        
        
        end = time.time()
        generation_time = end - begin
        if pil_image:
            return pil_image, generation_time
        else:
            return None, generation_time
            
    except Exception as err:
        end = time.time()
        generation_time = end - begin
        tprint(f"({service_name_for_log}) Exception calling Google AI model '{model_name_google}': {str(err)}, took {generation_time:.2f} seconds.")
        return None, generation_time

def generate_image_hidream_service(hidream_api_key_param: str, prompt_text: str, log_prefix: str = "HiDreamService") -> Tuple[Optional[str], float]:
    """Calls HiDream API to generate an image. Returns (image_url, total_duration_seconds)."""
    tprint(f"Start ({log_prefix}) HiDream image generation, prompt (first 50 characters): {prompt_text[:50]}...")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {hidream_api_key_param}",
    }
    payload = {
        "prompt": prompt_text,
        "size": f"{HIDREAM_IMAGE_WIDTH}x{HIDREAM_IMAGE_HEIGHT}",
        "seed": random.randint(1, 10000), # HiDream script used random.randint
        "enable_base64_output": False,
        "enable_safety_checker": True
    }

    overall_start_time = time.time()
    request_id = None
    image_url_final = None

    try:
        response = requests.post(HIDREAM_SUBMIT_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json().get("data", {})
        request_id = result.get("id")
        if not request_id:
            tprint(f"({log_prefix}) HiDream submit failed: No request_id returned. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        tprint(f"({log_prefix}) HiDream submit request failed: {str(e)}")
    except json.JSONDecodeError as e:
        tprint(f"({log_prefix}) HiDream submit response JSON decode failed: {str(e)}. Response text: {response.text if 'response' in locals() else 'N/A'}")
    except Exception as e:
        tprint(f"({log_prefix}) HiDream submit failed: {str(e)}")

    if not request_id:
        return None, time.time() - overall_start_time

    tprint(f"({log_prefix}) HiDream submit success. Request ID: {request_id}. Submit time: {time.time() - overall_start_time:.2f}s")

    result_url = f"{HIDREAM_RESULT_BASE_URL}/{request_id}/result"
    polling_headers = {"Authorization": f"Bearer {hidream_api_key_param}"}
    polling_start_time = time.time()

    while time.time() - polling_start_time < HIDREAM_MAX_POLLING_TIME_SECONDS:
        try:
            poll_response = requests.get(result_url, headers=polling_headers)
            poll_response.raise_for_status()
            result_data = poll_response.json().get("data", {})
            status = result_data.get("status")

            if status == "completed":
                tprint(f"({log_prefix}) HiDream task completed. Polling time: {time.time() - polling_start_time:.2f}s.")
                outputs = result_data.get("outputs")
                if outputs and len(outputs) > 0 and isinstance(outputs, list) and isinstance(outputs[0], str):
                    image_url_final = outputs[0]
                    break
                else:
                    tprint(f"({log_prefix}) HiDream task completed but no image URL found or invalid format. Result: {result_data}")
                    break
            elif status == "failed":
                error_message = result_data.get('error', 'Unknown error')
                tprint(f"({log_prefix}) HiDream task failed: {error_message}. Polling time: {time.time() - polling_start_time:.2f}s")
                break
            else:
                tprint(f"({log_prefix}) HiDream task status: {status}. Polling time: {time.time() - polling_start_time:.0f}s")
        except requests.exceptions.RequestException as e:
            tprint(f"({log_prefix}) HiDream polling request failed: {str(e)}. Retrying...")
        except json.JSONDecodeError as e:
            tprint(f"({log_prefix}) HiDream polling response JSON decode failed: {str(e)}. Response text: {poll_response.text if 'poll_response' in locals() else 'N/A'}")
        except Exception as e:
            tprint(f"({log_prefix}) HiDream polling failed: {str(e)}. Retrying...")

        time.sleep(HIDREAM_POLLING_INTERVAL_SECONDS)
    else:
        tprint(f"({log_prefix}) HiDream task timed out after {HIDREAM_MAX_POLLING_TIME_SECONDS} seconds. Request ID: {request_id}")

    total_duration = time.time() - overall_start_time
    if image_url_final:
        tprint(f"({log_prefix}) Image URL obtained. Total generation time: {total_duration:.2f}s")
    else:
        tprint(f"({log_prefix}) Failed to obtain image URL. Total time spent: {total_duration:.2f}s")
    return image_url_final, total_duration

def generate_image_seedream_service(seedream_api_key_param: str, prompt_text: str, log_prefix: str = "SeedreamService") -> Tuple[Optional[str], float]:
    """Calls Seedream API to generate an image. Returns (image_url, generation_time_seconds)."""
    tprint(f"({log_prefix}) Start Seedream image generation, prompt (first 50 characters): {prompt_text[:50]}...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {seedream_api_key_param}",
    }
    payload = {
        "model": SEEDREAM_MODEL_IDENTIFIER,
        "prompt": prompt_text,
        "response_format": "url",
        "size": f"{SEEDREAM_IMAGE_WIDTH}x{SEEDREAM_IMAGE_HEIGHT}",
        "seed": random.randint(1, 10000),
        "guidance_scale": SEEDREAM_GUIDANCE_SCALE,
        "watermark": SEEDREAM_ENABLE_WATERMARK
    }
    begin_time = time.time()
    image_url = None
    try:
        response = requests.post(SEEDREAM_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        if "data" in result and len(result.get("data", [])) > 0 and "url" in result["data"][0]:
            image_url = result["data"][0]["url"]
            tprint(f"({log_prefix}) Seedream task completed. Image URL: {prompt_text[:30]}...")
        else:
            error_detail = result.get("error", {}).get("message", "no error message") if isinstance(result.get("error"), dict) else response.text
            tprint(f"({log_prefix}) Seedream API error: Image URL not found. Status: {response.status_code}. Prompt: '{prompt_text[:30]}...'. Response: {error_detail[:200]}")
    except requests.exceptions.HTTPError as e:
        error_text = e.response.text if e.response else "no response text"
        tprint(f"({log_prefix}) Seedream API HTTP error: {e}. Prompt: '{prompt_text[:30]}...'. Response: {error_text[:200]}")
    except requests.exceptions.RequestException as e:
        tprint(f"({log_prefix}) Seedream API request exception: {str(e)}. Prompt: '{prompt_text[:30]}...'.")
    except json.JSONDecodeError as e_json:
        response_text_snippet = response.text[:100] if 'response' in locals() and hasattr(response, 'text') else 'N/A'
        tprint(f"({log_prefix}) Seedream API JSON decode error: {str(e_json)}. Prompt: '{prompt_text[:30]}...'. Response text snippet: {response_text_snippet}...")
    except Exception as e:
        tprint(f"({log_prefix}) Seedream generation failed: {str(e)}. Prompt: '{prompt_text[:30]}...'.")
    
    elapsed_time = time.time() - begin_time
    if image_url:
         tprint(f"({log_prefix}) Seedream task completed. Image URL for prompt: '{prompt_text[:30]}...'. Total time: {elapsed_time:.2f}s.")
    else:
         tprint(f"({log_prefix}) Seedream task failed or no URL obtained after {elapsed_time:.2f}s.")
    return image_url, elapsed_time

def save_text_to_file(text_content, file_path, service_name_for_log="UnknownService"):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        return True
    except Exception as e:
        tprint(f"({service_name_for_log}) Failed to save text to {file_path}: {str(e)}")
        return False

def save_base64_image(base64_string, file_path, service_name_for_log="UnknownService"):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        image_data = base64.b64decode(base64_string)
        with open(file_path, "wb") as f:
            f.write(image_data)
        return True
    except Exception as e:
        tprint(f"({service_name_for_log}) Failed to save base64 image to {file_path}: {str(e)}")
        return False

def save_image_from_url(url: str, file_path: str, service_name_for_log="UnknownService") -> bool:
    """Downloads an image from a URL and saves it to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # HiDream might require specific headers or use a library that handles them well.
        # For now, urlretrieve is simple. Consider requests.get(url, stream=True) for more control.
        urlretrieve(url, file_path)
        tprint(f"({service_name_for_log}) Image downloaded from URL saved to: {file_path}")
        return True
    except Exception as e:
        tprint(f"({service_name_for_log}) Failed to download image from URL ({url}) and save to {file_path}: {e}")
        return False

def process_single_image_task(service_name, api_key, base_url, image_model_for_api, lang_code, rowid, original_text, length):
    log_prefix = f"{service_name}/{lang_code}/{rowid}/{length}/{image_model_for_api}"
    tprint(f"Task processing started: {log_prefix}")

    safe_model_name_for_path = image_model_for_api.replace("/", "_").replace(":", "_")
    current_output_dir_path = os.path.join(OUTPUT_DIR, f"{safe_model_name_for_path}_{length}", lang_code)
    image_filename = f"{str(rowid)}.png"
    image_filepath = os.path.join(current_output_dir_path, image_filename)
    text_filename = f"{str(rowid)}.txt"
    text_filepath = os.path.join(current_output_dir_path, text_filename)

    if os.path.exists(image_filepath):
        tprint(f"({log_prefix}) Image already exists at: {image_filepath}. Skipping generation.")
        text_snippet_for_existing = original_text[:length]
        if not os.path.exists(text_filepath):
            save_text_to_file(text_snippet_for_existing, text_filepath, service_name_for_log=log_prefix)
            tprint(f"({log_prefix}) Created corresponding text file for existing image: {text_filepath}")
        return True

    openai_recraft_client = None

    if service_name.lower() in ["gpt", "recraft"]:
        if not api_key or api_key.startswith("YOUR_") or api_key.endswith("_HERE"):
            tprint(f"({log_prefix}) {service_name.upper()} API key not provided or still placeholder. Task aborted.")
            return False
        try:
            openai_recraft_client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            tprint(f"({log_prefix}) Failed to initialize {service_name.upper()} API client: {e}. Task aborted.")
            return False
    elif service_name.lower() in ["imagen", "gemini"]:
        if not api_key or api_key.startswith("YOUR_") or api_key.endswith("_HERE"): # This api_key is GOOGLE_API_KEY
            tprint(f"({log_prefix}) Google API key not provided or still placeholder. Task aborted.")
            return False
        # Google client is initialized within generate_image_google_ai, using the api_key (GOOGLE_API_KEY)
    elif service_name.lower() == "hidream":
        if not api_key or api_key.startswith("YOUR_") or api_key.endswith("_HERE"): # This api_key is HIDREAM_API_KEY
            tprint(f"({log_prefix}) HiDream API key not provided or still placeholder. Task aborted.")
            return False
        # HiDream client/API calls are direct, no specific client object to initialize here like OpenAI
    elif service_name.lower() == "seedream":
        if not api_key or api_key.startswith("YOUR_") or api_key.endswith("_HERE"): # This api_key is SEEDREAM_API_KEY
            tprint(f"({log_prefix}) Seedream API key not provided or still placeholder. Task aborted.")
            return False
        # Seedream API calls are direct, no specific client object
    else:
        tprint(f"({log_prefix}) Unknown service '{service_name}'. Task aborted.")
        return False

    text_snippet = original_text[:length]
    prompt = BASE_PROMPT + text_snippet

    b64_image = None
    # pil_image_result = None # Not needed here, conversion happens inside loop
    time_taken = 0
    generation_successful = False
    image_url_hidream = None # For HiDream service
    image_url_seedream = None # For Seedream service

    for attempt in range(1, MAX_RETRIES_PER_IMAGE + 1):
        tprint(f"({log_prefix}) Attempt {attempt}/{MAX_RETRIES_PER_IMAGE} to generate image...")
        temp_b64_image_direct = None
        temp_pil_image = None
        temp_image_url_hidream = None
        temp_image_url_seedream = None

        if service_name.lower() in ["gpt", "recraft"]:
            temp_b64_image_direct, temp_time_taken = generate_image_via_api(openai_recraft_client, prompt, image_model_name=image_model_for_api, service_name_for_log=log_prefix)
            if temp_b64_image_direct:
                b64_image = temp_b64_image_direct # Assign directly
                time_taken = temp_time_taken
                generation_successful = True
        elif service_name.lower() in ["imagen", "gemini"]:
            # api_key here is GOOGLE_API_KEY, image_model_for_api is IMAGEN_MODEL or GEMINI_IMAGE_MODEL
            temp_pil_image, temp_time_taken = generate_image_google_ai(google_api_key_param=api_key, prompt_text=prompt, model_name_google=image_model_for_api, service_name_for_log=log_prefix)
            if temp_pil_image:
                try:
                    buffered = BytesIO()
                    # Determine format based on service: Imagen uses JPEG, Gemini can use PNG
                    image_format = "JPEG" if service_name.lower() == "imagen" else "PNG"
                    temp_pil_image.save(buffered, format=image_format)
                    b64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    generation_successful = True
                except Exception as e_conv:
                    tprint(f"({log_prefix}) PIL Image to base64 conversion failed ({service_name}): {e_conv}")
                    b64_image = None
        elif service_name.lower() == "hidream":
            # api_key is HIDREAM_API_KEY
            temp_image_url_hidream, temp_time_taken = generate_image_hidream_service(hidream_api_key_param=api_key, prompt_text=prompt, log_prefix=log_prefix)
            if temp_image_url_hidream:
                image_url_hidream = temp_image_url_hidream
                time_taken = temp_time_taken
                generation_successful = True
        elif service_name.lower() == "seedream":
            # api_key is SEEDREAM_API_KEY
            temp_image_url_seedream, temp_time_taken = generate_image_seedream_service(seedream_api_key_param=api_key, prompt_text=prompt, log_prefix=log_prefix)
            if temp_image_url_seedream:
                image_url_seedream = temp_image_url_seedream
                time_taken = temp_time_taken
                generation_successful = True
        
        if generation_successful:
            tprint(f"({log_prefix}) Attempt {attempt} successful. Time taken: {time_taken:.2f}s.")
            break
        else:
            tprint(f"({log_prefix}) Attempt {attempt} failed.")
            if attempt < MAX_RETRIES_PER_IMAGE:
                tprint(f"({log_prefix}) Will retry in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                tprint(f"({log_prefix}) All {MAX_RETRIES_PER_IMAGE} attempts failed. Aborting this image.")

    if generation_successful and b64_image:
        text_saved = save_text_to_file(text_snippet, text_filepath, service_name_for_log=log_prefix)
        image_saved = save_base64_image(b64_image, image_filepath, service_name_for_log=log_prefix)
        
        if text_saved and image_saved:
            tprint(f"({log_prefix}): Image generated and saved successfully. Output to: {current_output_dir_path}")
            return True
        else:
            tprint(f"({log_prefix}): Image generated but save failed. Text saved: {text_saved}, Image saved: {image_saved}.")
            return False
    elif generation_successful and (image_url_hidream or image_url_seedream):
        # Handle saving for services that return a URL
        text_saved = save_text_to_file(text_snippet, text_filepath, service_name_for_log=log_prefix)
        
        image_saved_from_url = False
        if image_url_hidream:
            image_saved_from_url = save_image_from_url(image_url_hidream, image_filepath, service_name_for_log=log_prefix)
        elif image_url_seedream:
            image_saved_from_url = save_image_from_url(image_url_seedream, image_filepath, service_name_for_log=log_prefix)

        if text_saved and image_saved_from_url:
            tprint(f"({log_prefix}): Image generated and saved successfully from URL. Output to: {current_output_dir_path}")
            return True
        else:
            tprint(f"({log_prefix}): Image generated but save failed. Text saved: {text_saved}, Image saved from URL: {image_saved_from_url}.")
            return False
    else:
        tprint(f"({log_prefix}): Failed to generate image.")
        return False

# --- OCR Related Functions ---
def _ocr_bootstrap_statistics(
    values: List[float], n_bootstrap: int = 1000, metric_name: str = "value"
) -> Optional[Dict[str, float]]:
    """
    Calculate bootstrap statistics for a list of values.
    (Adapted from ocr/main.py)
    """
    if not values:
        tprint(
            f"OCR Stats Warning: Cannot calculate bootstrap for '{metric_name}' - no valid data.")
        return None
    if len(values) == 1:
        tprint(
            f"OCR Stats Warning: Only one data point for '{metric_name}'. Reporting mean, std_dev=0, CI=mean.")
        mean_val = float(np.mean(values))
        return {"mean": mean_val, "std_dev": 0.0, "ci_lower": mean_val, "ci_upper": mean_val}

    original_mean = np.mean(values)
    bootstrap_means = []
    rng = np.random.default_rng()
    n_values = len(values)

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_values, size=n_values)
        sample = np.array(values)[indices]
        bootstrap_means.append(np.mean(sample))

    std_dev = np.std(bootstrap_means, ddof=1)
    sorted_means = np.sort(bootstrap_means)
    lower_idx = max(0, int(0.025 * n_bootstrap) -1)
    upper_idx = min(n_bootstrap - 1, int(0.975 * n_bootstrap) -1)

    if n_bootstrap < 40:
        tprint(f"OCR Stats Warning: Low bootstrap samples ({n_bootstrap}) for '{metric_name}'. CI unreliable.")

    ci_lower = sorted_means[lower_idx]
    ci_upper = sorted_means[upper_idx]

    return {
        "mean": float(original_mean),
        "std_dev": float(std_dev),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }

def collect_all_generated_pairs(base_dir: str) -> List[Tuple[str, str, str]]: # Returns (txt_path, img_path, lang_code)
    pairs = []
    if not os.path.isdir(base_dir):
        tprint(f"OCR Pair Collection: Base directory '{base_dir}' does not exist.")
        return pairs

    tprint(f"OCR Pair Collection: Scanning '{base_dir}' for text-image pairs...")
    for model_len_dir_name in os.listdir(base_dir):
        model_len_dir_path = os.path.join(base_dir, model_len_dir_name)
        if not os.path.isdir(model_len_dir_path):
            continue

        for lang_code_dir_name in os.listdir(model_len_dir_path):
            lang_code_dir_path = os.path.join(model_len_dir_path, lang_code_dir_name)
            if not os.path.isdir(lang_code_dir_path):
                continue

            # lang_code_dir_name is our extracted language code, e.g., "en", "fr", "zh"
            current_lang_code = lang_code_dir_name 
            try:
                files_in_lang_dir = os.listdir(lang_code_dir_path)
                txt_files = sorted([f for f in files_in_lang_dir if f.endswith(".txt")])

                for txt_file_name in txt_files:
                    base_name = os.path.splitext(txt_file_name)[0]
                    # Main.py generates .png, but check for others just in case
                    for img_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
                        img_file_name = base_name + img_ext
                        if img_file_name in files_in_lang_dir:
                            txt_path = os.path.join(lang_code_dir_path, txt_file_name)
                            img_path = os.path.join(lang_code_dir_path, img_file_name)
                            pairs.append((txt_path, img_path, current_lang_code))
                            break 
            except OSError as e:
                tprint(f"OCR Pair Collection: Error listing directory '{lang_code_dir_path}': {e}")
    
    tprint(f"OCR Pair Collection: Found {len(pairs)} pairs.")
    return pairs

def run_ocr_evaluation_on_pairs(
    pairs_with_lang: List[Tuple[str, str, str]], # Expects (txt_path, img_path, lang_code)
    ocr_default_lang_tesseract: str, # Default tesseract lang (e.g., "eng")
    ocr_psm: int,
    ocr_n_bootstrap: int,
    ocr_verbose: bool
) -> Dict[str, Any]:
    """
    Evaluate all text-image pairs using the configured OCR core evaluator
    and calculate statistics for multiple metrics. (Adapted from ocr/main.py)
    """
    if not ocr_core:
        tprint("OCR Core module not available. Skipping OCR evaluation.")
        return {
            "evaluation_config": {"ocr_skipped_due_to_import_error": True},
            "individual_results": [],
            "aggregate_statistics": {"error": "OCR core module not loaded"}
        }

    pair_results_list = []
    # Initialize lists for all metrics mentioned in ocr/main.py
    metrics_collections: Dict[str, List[float]] = {
        'strict_sequence_similarity (Full)': [], 'strict_sequence_similarity (Truncated)': [],
        'strict_wer (Full)': [], 'strict_wer (Truncated)': [],
        'strict_cer (Full)': [], 'strict_cer (Truncated)': [],
        'strict_ned (Full)': [], 'strict_ned (Truncated)': [],
        'ground_truth_len_raw_chars': [], 'ocr_text_len_raw_chars': [],
        'ground_truth_tokens': [], 'ocr_tokens': []
    }
    successful_evals = 0
    failed_evals = 0

    for i, (text_path, image_path, lang_code) in enumerate(pairs_with_lang):
        # Determine Tesseract language string based on lang_code
        tesseract_lang_str = ocr_default_lang_tesseract
        if lang_code.lower() == 'zh':
            tesseract_lang_str = 'chi_sim+chi_tra'
        elif lang_code.lower() == 'fr':
            tesseract_lang_str = 'fra'
        # Add more language mappings here if needed
        # else: it uses the default

        tprint(
            f"""--- OCR Evaluating Pair {i+1}/{len(pairs_with_lang)} (Lang: {lang_code} -> Tesseract: '{tesseract_lang_str}') ---
Text:  {text_path}
Image: {image_path}"""
        )
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                ground_truth = f.read()
        except Exception as e:
            tprint(f"OCR Error: Reading ground truth file {text_path}: {e}")
            failed_evals += 1
            pair_results_list.append({"text_file": text_path, "image_file": image_path, "lang_code": lang_code, "status": "GT_READ_ERROR", "error_message": str(e)})
            continue

        try:
            eval_metrics = ocr_core.evaluate_document_image(
                ground_truth_text=ground_truth,
                generated_image_path=image_path,
                ocr_lang=tesseract_lang_str,
                ocr_psm=ocr_psm,
                verbose=ocr_verbose
            )

            if eval_metrics is None:
                tprint(f"OCR Critical Failure for {image_path}. Skipping.")
                failed_evals += 1
                pair_results_list.append({"text_file": text_path, "image_file": image_path, "lang_code": lang_code, "status": "OCR_CRITICAL_FAILURE"})
                continue
            
            # Corrected key for strict_wer (Truncated)
            current_pair_result = {
                "text_file": text_path, "image_file": image_path, "lang_code": lang_code,
                "status": "Success" if eval_metrics.get('ocr_success', False) else "OCR_EMPTY_OR_FAILED",
            }
            
            metric_keys_to_log = [
                'strict_sequence_similarity (Full)', 'strict_sequence_similarity (Truncated)',
                'strict_wer (Full)', 'strict_wer (Truncated)',
                'strict_cer (Full)', 'strict_cer (Truncated)',
                'strict_ned (Full)', 'strict_ned (Truncated)',
                'ground_truth_len_raw_chars', 'ocr_text_len_raw_chars',
                'ground_truth_tokens', 'ocr_tokens'
            ]

            for key in metric_keys_to_log:
                value = eval_metrics.get(key)
                current_pair_result[key] = value
                if value is not None and key in metrics_collections:
                    if isinstance(value, (int, float)):
                        metrics_collections[key].append(value)
                    elif key.endswith("_tokens") and isinstance(value, list):
                        metrics_collections[key].append(len(value))


            pair_results_list.append(current_pair_result)
            successful_evals += 1
            sim_full = current_pair_result.get('strict_sequence_similarity (Full)', float('nan'))
            wer_full = current_pair_result.get('strict_wer (Full)', float('nan'))
            tprint(f"OCR Metrics (Pair {i+1}): Sim={sim_full:.4f}, WER={wer_full:.4f}")


        except Exception as e:
            tprint(f"OCR Error: Evaluation process for {image_path}: {e}")
            failed_evals += 1
            pair_results_list.append({"text_file": text_path, "image_file": image_path, "lang_code": lang_code, "status": "EVALUATION_ERROR", "error_message": str(e)})
            continue
    
    tprint("OCR Evaluation Summary: {successful_evals} successful, {failed_evals} failed/skipped.")
    
    aggregate_stats: Dict[str, Any] = {
        "total_pairs_input_to_ocr": len(pairs_with_lang),
        "ocr_successful_evaluations": successful_evals,
        "ocr_failed_evaluations": failed_evals,
    }
    for metric_name, values_list in metrics_collections.items():
        if metric_name not in ['ground_truth_len_raw_chars', 'ocr_text_len_raw_chars', 'ground_truth_tokens', 'ocr_tokens']:
            stats = _ocr_bootstrap_statistics(values_list, ocr_n_bootstrap, metric_name)
            aggregate_stats[metric_name] = stats
    
    aggregate_stats["total_ground_truth_chars"] = sum(metrics_collections.get('ground_truth_len_raw_chars', []))
    aggregate_stats["total_ocr_chars"] = sum(metrics_collections.get('ocr_text_len_raw_chars', []))
    aggregate_stats["total_ground_truth_tokens"] = sum(metrics_collections.get('ground_truth_tokens', []))
    aggregate_stats["total_ocr_tokens"] = sum(metrics_collections.get('ocr_tokens', []))


    final_results = {
        "ocr_evaluation_config": {
            "ocr_tokenizer": OCR_TOKENIZER,
            "ocr_default_language_for_tesseract": ocr_default_lang_tesseract,
            "ocr_psm": ocr_psm,
            "ocr_n_bootstrap": ocr_n_bootstrap,
            "ocr_verbose": ocr_verbose
        },
        "ocr_individual_pair_results": pair_results_list,
        "ocr_aggregate_statistics": aggregate_stats,
    }
    return final_results


if __name__ == "__main__":
    script_start_time = time.time()
    tprint(f"--- Starting Script ---")
    tprint(f"Available Services: {AVAILABLE_SERVICES}")
    tprint(f"Max concurrent image jobs: {MAX_CONCURRENT_IMAGE_JOBS}")
    tprint(f"Max retries per image: {MAX_RETRIES_PER_IMAGE}, retry delay: {RETRY_DELAY_SECONDS}s")
    tprint(f"OCR enabled: {'Enabled' if OCR_ENABLE else 'Disabled'}. Default Tesseract lang: '{OCR_DEFAULT_LANGUAGE_FOR_TESSERACT}'")


    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        tprint(f"Created output directory: {OUTPUT_DIR}")
    
    if not os.path.exists(INPUT_JSON_PATH):
        sample_dir = os.path.dirname(INPUT_JSON_PATH)
        if sample_dir and not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        tprint(f"Input file {INPUT_JSON_PATH} not found. Creating sample input file...")
        sample_json_content = {
            "en": [{"rowid": "en_demo_01", "text": "This is a sample English text for demonstration purposes."}],
            "fr": [{"rowid": "fr_demo_01", "text": "Ceci est un exemple de texte français pour démonstration."}]
        }
        try:
            with open(INPUT_JSON_PATH, 'w', encoding='utf-8') as sf:
                json.dump(sample_json_content, sf, ensure_ascii=False, indent=2)
            tprint(f"Created sample input file: {INPUT_JSON_PATH}. Please modify or replace it as needed.")
        except IOError as e:
            tprint(f"Failed to create sample input file {INPUT_JSON_PATH}: {e}. Script will exit.")
            exit(1)


    all_image_tasks_params = []
    for service_name_from_list in AVAILABLE_SERVICES:
        service_name_to_process = service_name_from_list.lower().strip()
        current_api_key, current_base_url, current_image_model_for_api = None, None, None
        tprint(f"Collecting tasks for service '{service_name_to_process}'...")
        if service_name_to_process == "gpt":
            current_api_key, current_base_url, current_image_model_for_api = OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_IMAGE_MODEL
        elif service_name_to_process == "recraft":
            current_api_key, current_base_url, current_image_model_for_api = RECRAFT_API_KEY, RECRAFT_BASE_URL, RECRAFT_IMAGE_MODEL
        elif service_name_to_process == "imagen":
            current_api_key, current_base_url, current_image_model_for_api = GOOGLE_API_KEY, "N/A (Google AI SDK)", IMAGEN_MODEL
        elif service_name_to_process == "gemini":
            current_api_key, current_base_url, current_image_model_for_api = GOOGLE_API_KEY, "N/A (Google AI SDK)", GEMINI_IMAGE_MODEL
        elif service_name_to_process == "hidream":
            current_api_key, current_base_url, current_image_model_for_api = HIDREAM_API_KEY, HIDREAM_SUBMIT_URL, HIDREAM_IMAGE_MODEL
            # base_url for hidream is actually the submit_url. Result url is separate.
        elif service_name_to_process == "seedream":
            current_api_key, current_base_url, current_image_model_for_api = SEEDREAM_API_KEY, SEEDREAM_API_URL, SEEDREAM_MODEL_IDENTIFIER
        else:
            tprint(f"Warning: Unknown service '{service_name_to_process}'. Skipping.")
            continue
        if not current_api_key or current_api_key.startswith("YOUR_") or current_api_key.endswith("_HERE"):
            tprint(f"Warning: API key for service '{service_name_to_process}' is not configured. Skipping.")
            continue
        if not current_image_model_for_api:
            tprint(f"Warning: Image model for service '{service_name_to_process}' is not configured. Skipping.")
            continue
        api_key_display = f"****{current_api_key[-4:]}" if len(current_api_key) > 4 else "****"
        tprint(f"Service '{service_name_to_process}' configuration: API Key (last 4 digits): {api_key_display}, URL: {current_base_url}, Model: {current_image_model_for_api}")
        try:
            with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f: data = json.load(f)
        except Exception as e:
            tprint(f"Error ({service_name_to_process}): Failed to process input JSON {INPUT_JSON_PATH}: {e}. Skipping this service.")
            continue
        for lang_code, records in data.items():
            if not isinstance(records, list): continue
            for record in records:
                rowid, original_text = record.get("rowid"), record.get("text")
                if not rowid or not original_text: continue
                for length in LENGTH_LIST:
                    all_image_tasks_params.append({
                        "service_name": service_name_to_process, "api_key": current_api_key,
                        "base_url": current_base_url, "image_model_for_api": current_image_model_for_api,
                        "lang_code": lang_code, "rowid": str(rowid),
                        "original_text": original_text, "length": length
                    })

    if not all_image_tasks_params:
        tprint("No valid image generation tasks collected.")
    else:
        num_workers = max(1, MAX_CONCURRENT_IMAGE_JOBS)
        tprint(f"Using {num_workers} worker threads to process {len(all_image_tasks_params)} image generation tasks.")
        successful_tasks_count = 0
        failed_tasks_count = 0
        results_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_single_image_task, **params) for params in all_image_tasks_params]
            for i, future in enumerate(futures):
                # Basic progress update
                if (i + 1) % 10 == 0 or (i+1) == len(futures):
                     tprint(f"Image generation progress: {i+1}/{len(futures)} futures processed...")
                try:
                    if future.result(): successful_tasks_count += 1
                    else: failed_tasks_count += 1
                except Exception as e:
                    tprint(f"An image generation task failed with a critical error: {e}")
                    failed_tasks_count += 1
        tprint(f"--- All image generation tasks completed ---")
        tprint(f"Total tasks: {len(all_image_tasks_params)}")
        tprint(f"Successful tasks: {successful_tasks_count}")
        tprint(f"Failed tasks: {failed_tasks_count}")

    # --- OCR Evaluation Step ---
    if OCR_ENABLE and ocr_core: # Ensure OCR is enabled and core module loaded
        tprint("--- Starting OCR evaluation ---")
        # Scan all subdirectories of OUTPUT_DIR for pairs
        generated_pairs_for_ocr = collect_all_generated_pairs(OUTPUT_DIR)

        if generated_pairs_for_ocr:
            tprint(f"Will evaluate OCR for {len(generated_pairs_for_ocr)} image-text pairs.")
            ocr_results_data = run_ocr_evaluation_on_pairs(
                pairs_with_lang=generated_pairs_for_ocr,
                ocr_default_lang_tesseract=OCR_DEFAULT_LANGUAGE_FOR_TESSERACT,
                ocr_psm=OCR_PSM,
                ocr_n_bootstrap=OCR_N_BOOTSTRAP,
                ocr_verbose=OCR_VERBOSE
            )
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ocr_output_filepath = os.path.join(OUTPUT_DIR, f"{OCR_OUTPUT_FILENAME_PREFIX}{timestamp}.json")
            try:
                with open(ocr_output_filepath, "w", encoding="utf-8") as f_ocr_out:
                    json.dump(ocr_results_data, f_ocr_out, indent=2, ensure_ascii=False)
                tprint(f"OCR evaluation results saved to: {ocr_output_filepath}")
            except Exception as e:
                tprint(f"Failed to save OCR results to {ocr_output_filepath}: {e}")
        else:
            tprint("No generated image-text pairs found for OCR evaluation.")
    elif OCR_ENABLE and not ocr_core:
        tprint("OCR is enabled but the core module failed to load. Skipping OCR evaluation.")
    else:
        tprint("OCR evaluation is disabled. Skipping.")
    
    script_end_time = time.time()
    tprint(f"--- Script ended (total time: {script_end_time - script_start_time:.2f} seconds) ---")
