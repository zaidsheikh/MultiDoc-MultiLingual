import os
import glob
import json
import shutil
import argparse
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import click

MAX_TOKENS = 1024
TOKENIZER = AutoTokenizer.from_pretrained('google/mt5-base')


def get_text_oracle(references, references_aug):
    num_links = len(references) + len(references_aug)
    num_tokens = int(MAX_TOKENS / num_links)
    text = ""  # TODO: tokenizer
    for ref in references + references_aug:
        tokens = TOKENIZER.encode(ref['text'])[:num_tokens]
        text += TOKENIZER.decode(tokens)
    return text


def get_contents(line):
    obj = json.loads(line)
    summary = obj["summary"]
    references = obj["references"]
    references_aug = obj["references_aug"]
    text = get_text_oracle(references, references_aug)
    return summary, text


def extract_data(input_dir, output_dir, language):
    multilingual_dir = os.path.join(
        output_dir,
        "multilingual"
    )
    os.makedirs(multilingual_dir, exist_ok=True)

    f_iterator = glob.glob(
        os.path.join(
            input_dir,
            "{}_*.jsonl".format(language)
        )
    )

    for input_file in tqdm(f_iterator):
        input_file_name = os.path.basename(input_file)
        lang = "_".join(input_file_name.split("_")[:-1])
        lang_dir = os.path.join(output_dir, "individual", lang)
        os.makedirs(lang_dir, exist_ok=True)

        source_file = os.path.join(lang_dir, input_file_name.split("_")[
            -1].replace(".jsonl", ".source"))
        target_file = os.path.join(lang_dir, input_file_name.split("_")[
            -1].replace(".jsonl", ".target"))

        with open(input_file) as inpf:
            with open(source_file, 'w') as srcf, \
                    open(target_file, 'w') as tgtf:

                for line in inpf:
                    summary, text = get_contents(line)
                    json.dump(text, srcf)
                    srcf.write("\n")
                    json.dump(summary, tgtf)
                    tgtf.write("\n")

        if source_file.endswith("train.source"):
            shutil.copy(
                source_file,
                os.path.join(
                    multilingual_dir,
                    lang + "_" + os.path.basename(source_file)
                )
            )

            shutil.copy(
                target_file,
                os.path.join(
                    multilingual_dir,
                    lang + "_" + os.path.basename(target_file)
                )
            )
        if source_file.endswith("val.source"):
            shutil.copy(
                source_file,
                os.path.join(
                    multilingual_dir,
                    lang + "_" + os.path.basename(source_file)
                )
            )

            shutil.copy(
                target_file,
                os.path.join(
                    multilingual_dir,
                    lang + "_" + os.path.basename(target_file)
                )
            )

    val_source_files = glob.glob(os.path.join(multilingual_dir, "*_val.source"))
    with open(os.path.join(multilingual_dir, "val.source"), 'w') as fout:
        for val_source_file in val_source_files:
            with open(val_source_file, 'r') as fin:
                fout.write(fin.read())

    val_target_files = glob.glob(os.path.join(multilingual_dir, "*_val.target"))
    with open(os.path.join(multilingual_dir, "val.target"), 'w') as fout:
        for val_target_file in val_target_files:
            with open(val_target_file, 'r') as fin:
                fout.write(fin.read())


@click.command()
@click.option('--input-dir', required=False, type=str, help='Input directory.')
@click.option('--output-dir', required=False, type=str, help='Output directory.')
@click.option('--language', required=False, type=str, help='Language.')
def main(input_dir, output_dir, language):
    extract_data(input_dir, output_dir, language)


if __name__ == '__main__':
    main()
