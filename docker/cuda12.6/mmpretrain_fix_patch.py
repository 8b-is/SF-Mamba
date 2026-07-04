import os
import sys
import transformers

def fix_imports(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # 置換処理
                new_content = content.replace(
                    "from transformers.modeling_utils import (GenerationConfig, GenerationMixin,",
                    "from transformers import (GenerationConfig, GenerationMixin,"
                )


                # 変更があればファイルを上書き
                if new_content != content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated: {filepath}")
                

                # transformersのversionが4.19以降でのimport修正
                # transformersのバージョンチェック
                version_parts = transformers.__version__.split('.')
                major = int(version_parts[0])
                minor = int(version_parts[1]) if len(version_parts) > 1 else 0
                if (major, minor) >= (4, 19):
                    old_import = (
                        "from transformers.modeling_utils import (PreTrainedModel,\n"
                        "                                             apply_chunking_to_forward,\n"
                        "                                             find_pruneable_heads_and_indices,\n"
                        "                                             prune_linear_layer)"
                    )
                    new_import = (
                        "from transformers.modeling_utils import PreTrainedModel\n"
                        "    from transformers.pytorch_utils import apply_chunking_to_forward, prune_linear_layer"
                    )
                    new_content2 = new_content.replace(old_import, new_import)
                    # 変更があればファイルを上書き
                    if new_content2 != content:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(new_content2)
                        print(f"Updated: {filepath}")


                    old_import = (
                        "from transformers.modeling_utils import (PreTrainedModel,\n"
                        "                                         apply_chunking_to_forward,\n"
                        "                                         find_pruneable_heads_and_indices,\n"
                        "                                         prune_linear_layer)"
                    )
                    new_import = (
                        "from transformers.modeling_utils import PreTrainedModel\n"
                        "from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer"
                    )
                    new_content2 = new_content.replace(old_import, new_import)
                    # 変更があればファイルを上書き
                    if new_content2 != content:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(new_content2)
                        print(f"Updated: {filepath}")

                    # 置換処理
                    new_content = content.replace(
                        "from transformers.modeling_utils import apply_chunking_to_forward",
                        "from transformers.pytorch_utils import apply_chunking_to_forward"
                    )

                    # 変更があればファイルを上書き
                    if new_content != content:
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"Updated: {filepath}")





if __name__ == "__main__":
    package_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "mmpretrain")
    if os.path.exists(package_path):
        fix_imports(package_path)
    else:
        print("mmpretrain package not found. Check your installation path.")