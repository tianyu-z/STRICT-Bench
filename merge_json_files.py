import json
import os

def merge_json_files(lang_file_paths_map, output_filepath):
    """
    Merges content from multiple language-specific JSON files into a single
    JSON object. The keys of the object will be the language codes, and the values
    will be the list of records from the respective language file.

    Args:
        lang_file_paths_map (dict): A dictionary where keys are language codes
                                    (e.g., "en", "fr", "zh") and values are the
                                    paths to the JSON files. Each JSON file is
                                    expected to contain a list of records.
        output_filepath (str): Path to save the merged JSON object.
    """
    collated_data = {}

    for lang_code, filepath in lang_file_paths_map.items():
        try:
            with open(filepath, 'r', encoding='utf-8') as f_lang:
                lang_data = json.load(f_lang)  # lang_data is expected to be a list of records
            
            if not isinstance(lang_data, list):
                print(f"警告: 文件 {filepath} (语言: {lang_code}) 的内容不是一个列表。将跳过此文件。")
                continue

            collated_data[lang_code] = lang_data
            print(f"成功加载并添加了语言 '{lang_code}' 的数据 (来自 {filepath})，共 {len(lang_data)} 条记录。")

        except FileNotFoundError:
            print(f"错误：文件 {filepath} (语言: {lang_code}) 未找到。此语言将不会包含在输出中。")
        except json.JSONDecodeError:
            print(f"错误：文件 {filepath} (语言: {lang_code}) 不是有效的JSON。此语言将不会包含在输出中。")
        except Exception as e:
            print(f"处理文件 {filepath} (语言: {lang_code}) 时发生未知错误: {e}。此语言将不会包含在输出中。")

    if not collated_data:
        print("没有成功加载任何语言文件的数据。输出文件将为空对象或不创建。")
        # 你可以选择创建一个空文件或不创建
        # with open(output_filepath, 'w', encoding='utf-8') as f_out:
        #     json.dump({}, f_out, ensure_ascii=False, indent=2)
        return

    try:
        output_dir = os.path.dirname(output_filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(collated_data, f_out, ensure_ascii=False, indent=2)
        print(f"成功将所有语言数据合并到 {output_filepath}")
    except IOError:
        print(f"错误：无法写入到文件 {output_filepath}")
    except Exception as e:
        print(f"写入输出文件 {output_filepath} 时发生未知错误: {e}")

if __name__ == "__main__":
    language_files = {
        "en": "text_data/en.json",
        "fr": "text_data/fr.json",
        "zh": "text_data/zh.json"
    }
    
    collated_output_path = "text_data/collated_languages_object.json" # Changed filename for clarity

    # 确保示例文件存在以便测试
    for lang, path in language_files.items():
        if not os.path.exists(path):
            print(f"提示: 示例文件 {path} (语言: {lang}) 不存在。将创建一个空的JSON列表文件用于演示。")
            try:
                dir_path = os.path.dirname(path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                with open(path, 'w', encoding='utf-8') as f_sample:
                    # 创建一个包含特定于语言的示例数据的列表
                    sample_content = []
                    if lang == "en":
                        sample_content = [{"rowid": 1, "text": "Hello from en.json"}]
                    elif lang == "fr":
                        sample_content = [{"rowid": 10, "text": "Bonjour de fr.json"}]
                    elif lang == "zh":
                        sample_content = [{"rowid": 100, "text": "你好来自 zh.json"}]
                    json.dump(sample_content, f_sample, ensure_ascii=False, indent=2)
                print(f"已创建示例文件: {path}，包含 {len(sample_content)} 条记录。")
            except IOError:
                print(f"无法创建示例文件 {path}。")

    merge_json_files(language_files, collated_output_path) 