import argparse
import sys
import logging
import os
import time
from vllm import LLM, SamplingParams

from inference_llms_instruct_math_code import test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp




if __name__ == "__main__":

    parser = argparse.ArgumentParser("Interface for direct inference merged LLMs")
    parser.add_argument("--model_path", type=str)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=sys.maxsize)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
    parser.add_argument("--evaluate_task", type=str, help="task to be evaluated")
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)



    llm = LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size)

    if args.evaluate_task == "alpaca_eval":
        logger.info(f"evaluating merged model on alpaca_eval task...")
        save_gen_results_folder = f"./save_gen_instruct_responses_results/alpaca_eval/{args.save_model_name}"
        test_alpaca_eval(llm=llm, finetuned_model_name=args.model_path.split("/")[-1],
                         args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                         save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    if args.evaluate_task == "gsm8k":
        logger.info(f"evaluating merged model on gsm8k task...")
        test_data_path = "math_code_data/gsm8k_test.jsonl"
        test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                   start_index=args.start_index, end_index=args.end_index, model_name=args.model_path.split("/")[-1])
        

    if args.evaluate_task == "MATH":
        logger.info(f"evaluating merged model on MATH task...")
        test_data_path = "math_code_data/MATH_test.jsonl"
        test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                            start_index=args.start_index, end_index=args.end_index, model_name=args.model_path.split("/")[-1])

    if args.evaluate_task == "human_eval":
        logger.info(f"evaluating merged model on code task...")
        save_gen_results_folder = f"./save_gen_codes_results/human_eval/{args.save_model_name}"
        test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                        save_model_path=None, save_gen_results_folder=save_gen_results_folder)


    if args.evaluate_task == "mbpp":
        save_gen_results_folder = f"./save_gen_codes_results/mbpp/{args.save_model_name}"
        test_data_path = "math_code_data/mbpp.test.jsonl"
        test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                  start_index=args.start_index, end_index=args.end_index,
                  save_model_path=None, save_gen_results_folder=save_gen_results_folder)

    logger.info(f"inference of merging method {args.merging_method_name} is completed")

    sys.exit()
