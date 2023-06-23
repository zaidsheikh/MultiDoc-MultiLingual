# MultiDoc-MultiLingual
## Setup
### Install Rouge Package
Follow the [README.md](multilingual_rouge_scoring/README.md) under the multilingual_rouge_scoring directory.

## Dataset Augmentation
We use Google Search Engine to find more input articles for each event from the WCEP with the following steps:
### Extract Keywords
We use keyBERT to extract keywords by running:
```bash
$ python dataset_collection/keywords_extraction_keyBERT.py \
    --file_name "cantonese_crawl.jsonl" \
    --data_dir "./Multi-Doc-Sum/Mtl_data" \
    --output_dir "./Multi-Doc-Sum/keywords_extraction_keyBERT"
```

The meaning of the arguments are as follows:
- data_dir dataset directory of the original crawled data from WCEP.
- file_name a specific file of a certain language under the dataset directory.
- output_dir output directory of the extracted keywords.


### Google Search
We run the following command to search Google:
```bash
$ python dataset_collection/google_search.py \
    --my_api_key $GOOGLE_SEARCH_API_KEY \
    --my_cse_id $CUSTOM_SEARCH_ENGINE_ID \
    --file_name "cantonese_crawl.jsonl" \
    --data_dir "./Multi-Doc-Sum/Mtl_data" \
    --keywords_dir "./Multi-Doc-Sum/keywords_extraction_keyBERT" \
    --data_aug_dir "./Multi-Doc-Sum/Mtl_data_aug" 
```
The meaning of the arguments are as follows:
- MY_API_KEY your google search API key.
- MY_CSE_ID your Custom Search Engine ID.
- data_dir dataset directory of the original crawled data from WCEP.
- file_name a specific file of a certain language under the dataset directory.
- output_dir output directory of the extracted keywords.


### Dataset Cleaning
A clean dataset is generated by filtering the source documents using the ORACLE method. The first step is to calculate the ORACLE score of each source document with the summary:
```bash
$ python dataset_collection/filter_source_documents/filter_oracle_get_score.py \
    --data-dir "./Multi-Doc-Sum/Mtl_data/doc_extraction" \
    --output-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/scored" \
    --input-file-name "cantonese_extracted.jsonl" \
    --output-file-name "cantonese_scored.jsonl"
``` 

The second step is to filter out the source docuemnts with ORACLE score bellow a threshold:
```bash
$ python dataset_collection/filter_source_documents/filter_oracle.py \
    --threshold 7 \
    --data-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/scored" \
    --output-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/filtered" \
    --input-file-name "cantonese_scored.jsonl" \
    --output-file-name "cantonese_filtered.jsonl"
``` 

### Split Dataset
Both the noisy and the clean datasets are randomly split into 80%, 10%, and 10% training, validation, and test sets, respectively:

```bash
$ python dataset_collection/split.py \
    --lang cantonese\
    --data-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/orig" \
    --output-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/split/orig" \
    --input-file-name "cantonese_scored.jsonl" 

$ python dataset_collection/split.py \
    --lang cantonese \
    --data-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/filtered" \
    --output-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/split/filtered" \
    --input-file-name "cantonese_filtered.jsonl" 
``` 

## Baselines

### Heuristic Baseline
Heuristic Baselines are calculated by:
```bash
$ python baselines/heuristic/get_heuristic_baseline_result.py \
    --input-file-name "cantonese_test.jsonl" \
    --data-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/split/filtered/" \
    --lang cantonese \
    --output-dir "./baseline_results/clean_dataset"
```

### TextRank Baseline
TextRank Baselines are calculated by:


### Mt5 Baseline
- Prepare the dataset to train mt5 models:

```bash
$ python baselines/mt5/prepare_dataset.py \
    --input-dir "./Multi-Doc-Sum/Mtl_data_aug_filtered/split/filtered" \
    --output-dir "data/first_sentences" \
    --language "cantonese"
```

- Here is an example [trainer_cantonese.sh](baselines/mt5/examples/trainer_cantonese.sh) to train single langauge mt5 model. 

- Here is an example [train_multilingual.sh](baselines/mt5/examples/train_multilingual.sh) to train multilingual mt5 model. Our trained multilingual model can be downloaded from [Multilingual-mt5](https://drive.google.com/drive/u/1/folders/1JJ1XvAeL7JFCxPv5IHPhyVmjDfRetpW7). 


## Evaluation
Here is an example [evaluation.py](evaluation/run_evaluation.py) to use evaluation metrics: BERTScore and T5Score. To run T5Score, a T5Score model should be downloaded from [T5Score-summ](https://drive.google.com/drive/u/1/folders/1VrVWRbXZRBDnl4pGvcfvfLzF4P4JRO_m) to directory ./model/T5Score/.

### Docker image

- We have also provided a docker image with all dependencies pre-installed in order to make it easier to run the above scripts. Here's how to run the training pipeline inside a docker container:
```
docker pull zs12/multidoc_multilingual:v0.2

# train single langauge mt5 model
./dockerfiles/docker_train_mt5.sh prepared_dataset/individual/EN/ output/

# train multilingual mt5 model
./dockerfiles/docker_train_mt5.sh prepared_dataset/multilingual/ output/ multi

# Run inference/prediction using a trained multilingual mt5 model
./dockerfiles/docker_predict_with_generate.sh model_dir/ data_dir/ output_dir/
```

- To run the docker container via ClearML:

```
# train single langauge mt5 model
./clearml_scripts/clearml_train_mt5.sh prepared_dataset/individual/EN/ output/

# train multilingual mt5 model
./clearml_scripts/clearml_train_mt5.sh prepared_dataset/multilingual/ output/ multi
```
