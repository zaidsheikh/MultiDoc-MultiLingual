import click
import json
import requests
import time
import os
from tqdm import tqdm
import csv
import re
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT


def tokenize_zh(text):
    words = jieba.lcut(text)
    return words


def extract_keywords(text, lang):
    if lang == 'EN':
        #kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        kw_model = KeyBERT(model='paraphrase-mpnet-base-v2')
    else:
        #kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
        kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')

    if lang == "EN":
        # keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 4),stop_words='english',
        #                        use_mmr=True, diversity=0.7, top_n=5)
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english',
                                             use_mmr=True, diversity=0.8, top_n=4)
        print(keywords)
        #keywords = [keyword[0] for keyword in keywords if keyword[1]>0.2]
        keys = [keyword[0] for keyword in keywords[:2]]
        keys.extend([keyword[0]
                     for keyword in keywords[2:] if keyword[1] > 0.08])
        keywords = keys
        print(keywords)
    else:
        if lang == "chinese" or lang == "cantonese":
            vectorizer = CountVectorizer(tokenizer=tokenize_zh)
            vectorizer = CountVectorizer(tokenizer=tokenize_zh)
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 4),
                                                 use_mmr=True, diversity=0.7, top_n=5, vectorizer=vectorizer)
            # print(keywords)
            keys = [keyword[0] for keyword in keywords[:3]]
            keys.extend([keyword[0]
                         for keyword in keywords[3:] if keyword[1] > 0.2])
            keywords = keys
        elif lang == "RU" or lang == "PL" or lang == "CZ" or lang == "VI" or lang == "UA":
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3),
                                                 use_mmr=True, diversity=0.8, top_n=4)
            # print(keywords)
            keys = [keyword[0] for keyword in keywords[:2]]
            keys.extend([keyword[0]
                         for keyword in keywords[2:] if keyword[1] > 0.08])
            keywords = keys
            #keywords = [keyword[0] for keyword in keywords if keyword[1]>0.3]
        else:
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 4),
                                                 use_mmr=True, diversity=0.7, top_n=5)

            keywords = [keyword[0] for keyword in keywords if keyword[1] > 0.2]
    # print(keywords)
    return keywords


def preprocess(orig_text):
    pos = []
    for match in re.finditer(r"\[\d+\]|（.*?）|\(.*?\)", orig_text):
        pos.append(match.span())
    if len(pos) != 0:
        text = ""
        for i in range(len(pos)):
            if i == 0:
                text += orig_text[:pos[0][0]]
            else:
                text += orig_text[pos[i-1][1]:pos[i][0]]
        text += orig_text[pos[-1][1]:]
    else:
        text = orig_text
    text = text.replace("\n", ' ')
    return text


lang_2_summary_dict = {
    "EN": "text",
    "CZ": "summary",
    "PL": "summary",
    "RU": "summary",
    "UA": "summary",
    "VI": "summary",
    "cantonese": "summary",
    "chinese": "text",
}


@click.command()
@click.option('--data_dir', required=True, default=None)
@click.option('--output_dir', required=True, default=None)
@click.option('--file_name', required=True, default=None)
def main(data_dir: str, output_dir: str, file_name: str):
    start_id = 0
    end_id = None
    lang = file_name.split("_")[0]

    data = []
    for line in open(os.path.join(data_dir, file_name), 'r'):
        data.append(json.loads(line))

    if os.path.exists(os.path.join(output_dir, lang+'_keywords.csv')):
        output_file = open(os.path.join(output_dir, lang+'_keywords.csv'), 'r')
        for row in output_file:
            start_id += 1
        output_file.close()

    output_file = open(os.path.join(output_dir, lang+'_keywords.csv'), 'a')
    writer = csv.writer(output_file, delimiter='\t')

    for i in tqdm(range(start_id, len(data) if end_id is None else end_id)):
        orig_text = data[i][lang_2_summary_dict[lang]]
        text = preprocess(orig_text)
        print(text)
        keywords = extract_keywords(text, lang=lang)
        orig_text = orig_text.replace("\n", ' ')
        row = [i, orig_text, keywords]
        writer.writerow(row)
        # print(row)
        print("")

    output_file.close()


if __name__ == "__main__":
    main()
