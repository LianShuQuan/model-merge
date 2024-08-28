import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from inference_llms_instruct_math_code import create_llm, test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp
from utils.load_config import cache_dir


task_model_mapping_dict = {
    "instruct": "deepseek-ai/deepseek-llm-7b-chat",
    "math": "deepseek-ai/deepseek-math-7b-base",
    "code": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
}
finetuned_model_backbone_mapping_dict = {
    "deepseek-ai/deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-math-7b-base": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": "deepseek-ai/deepseek-llm-7b-base"
}


save_model_base = "/data/lsq"


def get_merge_performance(args: argparse.Namespace, finetuned_model_names: list, merge_task_names: list, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizers: list, layers_to_merge: list = None):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param merge_task_names: list, names of tasks that need to be merged
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizers: list of tokenizers
    :return:
    """
    logger.info(f"configuration is {args}")

    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(args.pretrained_model_name, device_map="auto")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    except:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir, device_map="auto")
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir)
    if not pretrained_tokenizer.pad_token:
        pretrained_tokenizer.pad_token = pretrained_tokenizer.unk_token
    if not pretrained_tokenizer.pad_token:
        pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token    


    # get include_param_names_regex
    if args.merge_norm or args.merge_lm_head or args.merge_embedding or args.layers_to_merge is not None:
        include_param_names_regex = []
    if args.merge_norm:
        include_param_names_regex.append(r"model.norm.weight")
    if args.merge_lm_head:
        include_param_names_regex.append(r".*lm_head.*")
    if args.merge_embedding:
        include_param_names_regex.append(r".*embed_tokens.*")
    if args.layers_to_merge is not None:
        for layer_id in args.layers_to_merge:
            include_param_names_regex.append(r"model\.layers\." + str(layer_id) + r"\.")

    # set random seed to guarantee reproducibility
    set_random_seed(seed=0)
    merged_model = pretrained_model
    merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                   models_to_merge=models_to_merge,
                                                   exclude_param_names_regex=[],
                                                   include_param_names_regex=include_param_names_regex,
                                                   trainers=trainers,
                                                   scaling_coefficient=args.scaling_coefficient,
                                                   nums_fisher_examples=None,
                                                   fisher_scaling_coefficients=None,
                                                   normalize_fisher_weight=None,
                                                   minimal_fisher_weight=None,
                                                   nums_regmean_examples=None,
                                                   reduce_non_diagonal_ratio=None,
                                                   param_value_mask_rate=None,
                                                   weight_format=args.weight_format,
                                                   weight_mask_rates=args.weight_mask_rates,
                                                   use_weight_rescale=args.use_weight_rescale,
                                                   mask_strategy=args.mask_strategy,
                                                   mask_apply_method=args.mask_apply_method,
                                                   models_use_deepcopy=False,
                                                   layers_to_merge = layers_to_merge)


    # only save the merged model,and backbone tokenizer
    save_model_path = f"{save_model_base}/save_merge_models/{args.save_model_name}"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    logger.info(f"saving models at {save_model_path}...")
    merged_model.save_pretrained(save_directory=save_model_path)
    tokenizer_from_path = finetuned_model_backbone_mapping_dict[task_model_mapping_dict["instruct"]]
    tok = AutoTokenizer.from_pretrained(tokenizer_from_path)
    tok.save_pretrained(save_directory=save_model_path)


    # save_instruct_model_path = save_math_model_path = save_code_model_path = None
    # if args.merge_instruct:
    #     save_instruct_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/instruct/{args.save_model_name}"
    # if args.merge_math:
    #     save_math_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math/{args.save_model_name}"
    # if args.merge_code:
    #     save_code_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/code/{args.save_model_name}"

    # # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    # save_model_paths = [save_instruct_model_path, save_math_model_path, save_code_model_path]
    # index = 0
    # for save_model_path in save_model_paths:
    #     if save_model_path is not None:
    #         logger.info(f"saving models at {save_model_path}...")
    #         merged_model.save_pretrained(save_directory=save_model_path)
    #         tokenizers[index].save_pretrained(save_directory=save_model_path)
    #         index += 1
    logger.info(f"models are saved")
    del merged_model, tokenizers

    # if save_instruct_model_path is not None:
    #     logger.info(f"evaluating merged model on instruct task...")
    #     llm = create_llm(finetuned_model_name=save_instruct_model_path, pretrained_model_name=args.pretrained_model_name,
    #                      args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
    #                      just_inference=True, save_model_path=None)
    #     save_gen_results_folder = f"./save_gen_instruct_responses_results/{'_'.join(merge_task_names)}/alpaca_eval/{args.save_model_name}"
    #     test_alpaca_eval(llm=llm, finetuned_model_name=save_instruct_model_path,
    #                      args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
    #                      save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    # if save_math_model_path is not None:
    #     logger.info(f"evaluating merged model on math task...")
    #     llm = create_llm(finetuned_model_name=save_math_model_path, pretrained_model_name=args.pretrained_model_name,
    #                      args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
    #                      just_inference=True, save_model_path=None)
    #     test_data_path = "math_code_data/gsm8k_test.jsonl"
    #     test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
    #                start_index=args.start_index, end_index=args.end_index, save_model_path=None)
    #     test_data_path = "math_code_data/MATH_test.jsonl"
    #     test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
    #                         start_index=args.start_index, end_index=args.end_index, save_model_path=None)

    # if save_code_model_path is not None:
    #     logger.info(f"evaluating merged model on code task...")
    #     llm = create_llm(finetuned_model_name=save_code_model_path, pretrained_model_name=args.pretrained_model_name,
    #                      args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
    #                      just_inference=True, save_model_path=None)
    #     save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/human_eval/{args.save_model_name}"
    #     test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
    #                     save_model_path=None, save_gen_results_folder=save_gen_results_folder)
    #     save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/mbpp/{args.save_model_name}"
    #     test_data_path = "math_code_data/mbpp.test.jsonl"
    #     test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
    #               start_index=args.start_index, end_index=args.end_index,
    #               save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    # for save_model_path in save_model_paths:
    #     if save_model_path is not None:
    #         shutil.rmtree(save_model_path, ignore_errors=True)
    # logger.info(f"inference of merging method {args.merging_method_name} is completed")


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--merge_instruct", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge code model")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "mask_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.1, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "ties_merging"])
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=sys.maxsize)
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
parser.add_argument('--layers_to_merge', nargs='+')
parser.add_argument("--merge_embedding", action="store_true", default=False, help="whther to merge embedding")
parser.add_argument("--merge_norm", action="store_true", default=False, help="whther to merge last norm layer")
parser.add_argument("--merge_lm_head", action="store_true", default=False, help="whther to merge lm head")


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()


if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    assert sum([args.merge_instruct, args.merge_math, args.merge_code]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_instruct, args.merge_math, args.merge_code], ["instruct", "math", "code"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        elif args.mask_apply_method == "task_arithmetic":
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        else:
            assert args.mask_apply_method == "ties_merging"
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"

        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]

        args.save_model_name = (
                f"deepseek"
                f"{'_instruct' if args.merge_instruct else '_'}"
                f"{'_math' if args.merge_math else '_'}"
                f"{'_code' if args.merge_code else '_'}"
                f"_{args.merging_method_name}_{mask_apply_method_name}_{args.weight_mask_rate}"
            )
        if args.layers_to_merge is not None:
            args.save_model_name += f"_merge_layers_{'_'.join(args.layers_to_merge)}"
        if args.merge_norm:
            args.save_model_name += "_norm"
        if args.merge_lm_head:
            args.save_model_name += "_lmhead"
        if args.merge_embedding:
            args.save_model_name += "_embedding"
        
            


    save_merge_log_path = f"./save_merge_llm_logs/{'_'.join(merge_task_names)}/{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    models_to_merge = []
    finetuned_tokenizers = []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_name, device_map="auto")
        finetuned_tokenizer = AutoTokenizer.from_pretrained(finetuned_model_name)
        if not finetuned_tokenizer.pad_token:
            finetuned_tokenizer.pad_token = finetuned_tokenizer.unk_token
        if not finetuned_tokenizer.pad_token:
            finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token   
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)


    get_merge_performance(args=args, finetuned_model_names=finetuned_model_names, merge_task_names=merge_task_names, models_to_merge=models_to_merge,
                          trainers=[None for _ in range(len(finetuned_model_names))], logger=logger, merging_method=merging_method, tokenizers=finetuned_tokenizers, layers_to_merge=args.layers_to_merge)

    sys.exit()
