import json


def load_metrics_no_mask(dataset_names:list, merging_method_name:str, language_model_name:str):
    save_merged_model_path = f"D:/programFiles/data/lsq/MergeLM/save_merge_models/{dataset_names[0]}_{dataset_names[-1]}/{merging_method_name}/{language_model_name}.json"
    with open(save_merged_model_path, "r") as file:
        results_dict = json.load(file)
        print(results_dict)


def load_metrics_with_mask(dataset_names:list, mask_apply_method:str, language_model_name:str):
    save_merged_model_path = f"D:/programFiles/data/lsq/MergeLM/save_merge_models/{dataset_names[0]}_{dataset_names[-1]}/mask_merging/{mask_apply_method}/{language_model_name}.json"


dataset_names = ["cola", "sst2", "stsb", "qqp", "mnli", "rte"]
merging_method_names = ["average_merging", "task_arithmetic", "fisher_merging", "regmean_merging", "ties_merging"]
language_model_name = "roberta-base"

glue_data_metrics_map = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "averaged_scores",   # average of accuracy and f1
    "stsb": "averaged_scores",   # average of pearson and spearmanr
    "qqp": "averaged_scores",    # average of accuracy and f1
    "mnli": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy"
}

if __name__ == "__main__":
    for merging_method_name in merging_method_names:
        for source_dataset_name in dataset_names:
            for target_dataset_name in dataset_names:
                if source_dataset_name == target_dataset_name:
                    continue
                load_metrics_no_mask([source_dataset_name, target_dataset_name], merging_method_name, language_model_name)