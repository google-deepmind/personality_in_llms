"""Runs social media status update task on a given vLLM or OpenAI model.

This script works for both HuggingFace models on vLLM and OpenAI API models. 
`asyncio` is used to run bulk inference on OpenAI models, which can be
rate-limited using `--rpm_limit` (50 RPM will be set by default). 

Use `python generate_status_updates.py -h` for detailed usage information. 

Example vLLM usage:
    python inference_scripts/generate_status_updates.py \
        --admin_session='admin_sessions/generate_updates_ablation_01_admin_session_100.json' \
        --model_pointer='meta-llama/Llama-2-7b-chat-hf' \
        --n_gpus=2 \
        --vllm \
        --sample

Example OpenAI usage:
    python inference_scripts/generate_status_updates.py \
        --admin_session='generate_updates_ablation_01_admin_session_25.json' \
        --model_pointer='gpt-3.5-turbo-0125' \
        --payload='admin_sessions/generate_updates_ablation_01_admin_session_25_payloads/generate_updates_ablation_01_admin_session_25_payload_0.pkl' \
        --openai # \
        # sample only 1,000 rows from payload
        # --sample \
        # output as csv
        # --csv \
        # specify results directory
        # --results_path='<YOUR_DIR_HERE>'
"""

# load dependencies
import pandas as pd

import os
import sys

# point system to a local clone of PsyBORGS
PATH = "../"
sys.path.append(PATH)

from psyborgs import survey_bench_lib
from inference_scripts.run_hf_inference import generate_payload_df

# from typing import Union, Callable, List
from typing import Union, List, Optional
from collections.abc import Callable

# vLLM integration
import vllm

# OpenAI integration
from functools import singledispatch
import openai

# MAIN
import argparse


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError, openai.APITimeoutError),
):
    """Retry a function with exponential backoff.
    
    Adapted from
        https://platform.openai.com/docs/guides/rate-limits/error-mitigation.
    """
    import random
    import time


    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise RuntimeError(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    ) from e

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay + 60)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

def generate_status_updates(payload_df: pd.DataFrame,
                            model: Union[Callable, str],
                            temperature: float=1.0,
                            max_tokens: int=1024,
                            rpm_limit: Optional[int]=None) -> pd.DataFrame:
    """Generate status updates using a given model."""
    # extract prompts
    prompts = list(payload_df['prompt_text'])


    print(f"`temperature` is set to {temperature}.")
    print(f"`max_tokens` is set to {max_tokens}.")

    # run bulk inference
    print(f"Starting bulk inference for {len(prompts)} prompts.")
    model_answers = bulk_inference(
        model, prompts, temperature, max_tokens, rpm_limit)

    # attach model answers to original dataframe
    payload_df['model_output'] = model_answers

    return payload_df

@singledispatch
def bulk_inference(model,
                   prompts: List[str],
                   temperature: float=1.0,
                   max_tokens: int=1024,
                   rpm_limit: Optional[int]=None) -> List[str]:
    """Run bulk inference."""
    raise NotImplementedError(f"Unsupported type for model: {type(model)}")

@bulk_inference.register
def bulk_inference_vllm(model: Callable,
                        prompts: List[str],
                        temperature: float=1.0,
                        max_tokens: int=1024,
                        rpm_limit: Optional[int]=None) -> List[str]:
    """Run bulk inference for HF Transformers models via vLLM."""
    # set vLLM model sampling parameters
    sampling_params = vllm.SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens)
    outputs = model.generate(prompts, sampling_params=sampling_params)
    print("Bulk inference complete.")

    # extract model answers from outputs
    model_answers = []
    for output in outputs:
        # generated text is located in outputs object
        answer = output.outputs[0].text
        model_answers.append(answer)

    return model_answers

@bulk_inference.register
def bulk_inference_openai(model: str,
                          prompts: List[str],
                          temperature: float=1.0,
                          max_tokens: int=1024,
                          rpm_limit: Optional[int]=None) -> List[str]:
    """Run bulk inference for OpenAI models via API."""
    from openai import AsyncOpenAI
    import asyncio
    from aiolimiter import AsyncLimiter
    from tqdm.asyncio import tqdm

    client = AsyncOpenAI(max_retries=10)

    print("Beginning async OpenAI API inference.")
    @retry_with_exponential_backoff
    async def call_api(prompt, temperature, max_tokens):
        user_message = [{"role": "user", "content": prompt}]

        # send prompts in bulk
        async with limiter:
            response = await client.chat.completions.create(
                model=model,
                messages=user_message,
                max_tokens=max_tokens,
                seed=42,
                temperature=temperature
            )

        return response.choices[0].message.content

    async def _main():
        tasks = [call_api(prompt, temperature, max_tokens) for prompt in prompts]
        model_answers = await tqdm.gather(*tasks)
        print("OpenAI inference complete.")
        return model_answers

    # limit inference to 50 RPMs if no `rpm_limit` is provided
    rpm_limit = rpm_limit or 50

    limiter = AsyncLimiter(rpm_limit)

    asyncio.run(_main())


def parse_args():
    """Registers arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--admin_session', type=str, help='`admin_session` path', required=True)
    parser.add_argument(
        '--model_pointer', type=str, 
        help=('HuggingFace model repo ID (make sure model is already '
              'downloaded to cache) or OpenAI model ID'), required=True)
    parser.add_argument(
        '--job_name', type=str, default='vllm-job', 
        help='experiment name (to be used in output filenames)')
    parser.add_argument(
        '--job_id', type=str, default='0',
        help='job ID (for tracking if using HPC)')
    parser.add_argument(
        '--n_gpus', type=int, default=1, 
        help='number of GPUs to use for distributed tensor-parallel inference')
    parser.add_argument(
        '--temperature', type=float, default=0.70,
        help='float controlling randomness of sampling')
    parser.add_argument(
        '--sample', action='store_true',
        help=('if True, only sample 1,000 rows of the `admin_session` used '
              '(for testing)'))
    parser.add_argument(
        '--vllm', action='store_true',
        help=('run inference using vLLM (requires at least one GPU with a CUDA '
              'compute capability of 7.0 or higher)'))
    parser.add_argument(
        '--openai', action='store_true',
        help=('run inference using OpenAI\'s API'))
    parser.add_argument(
        '--rpm_limit', type=int, default=None,
        help='requests per minute limit for OpenAI API')
    parser.add_argument(
        '--payload', type=str, default=None, 
        help=('generative payload pickle path, if running on smaller payload '
              'chunks'))
    parser.add_argument(
        '--csv', action='store_true',
        help='also write results to a CSV file')
    parser.add_argument(
        '--results_path', type=str, default=None,
        help='directory path for outputed results file(s)')

    return parser.parse_args()


def main():
    # register arguments
    args = parse_args()

    # validate inference options: only select vllm, OpenAI, Transformers
    if sum(map(bool, [args.vllm, args.openai])) > 1:
        raise ValueError(
            'Please choose only one flag for inference (e.g., `--vllm`). '
            'Multiple flags detected :)')

    # validate inference options: either vLLM or OpenAI must be elected
    if sum(map(bool, [args.vllm, args.openai])) == 0:
        raise ValueError(
            'Please choose a method for inference, `--vllm` or `--openai`.')

    # set default results path
    if args.results_path is None:
        # create a results directory if it doesn't already exist
        if not os.path.exists('results'):
            os.makedirs('results')
        args.results_path = 'results'

    # get just the model ID for output filename
    # i.e., text to the right of the backslash for HF models
    if args.model_pointer.__contains__('/'):
        MODEL_ID = str(args.model_pointer).split('/')[1]
    else:
        MODEL_ID = args.model_pointer

    # load admin_session
    admin_session = survey_bench_lib.load_admin_session(
        args.admin_session)

    # create optional chunk number
    chunk_n = ""

    # create generative payload by default
    if not args.payload:
        # create regular payload (n = prompts x continuations)
        payload_df = generate_payload_df(admin_session=admin_session, 
                            model_id=args.model_pointer)
    # use payload chunk if specified
    elif args.payload:
        payload_df = pd.read_pickle(args.payload)

        # set model_id
        payload_df['model_id'] = args.model_pointer

        # record chunk number that can help name output file,
        # which should be after the last underscore
        chunk_n = "_chunk_" + str(args.payload).split('_')[-1]

    # sample 1000 prompts
    if args.sample:
        print("Sampling only 1,000 prompts.")
        payload_df = payload_df.sample(1000, random_state=42)

    # initialize model to None
    model = None

    # input validation guarantees one of the following options:
    if args.vllm:
        model = vllm.LLM(model=args.model_pointer,
                         seed=42,
                         # set number of GPUs to use in parallel
                         tensor_parallel_size=args.n_gpus)
    elif args.openai:
        model = args.model_pointer

    # code to run bulk inference
    results = generate_status_updates(
        payload_df,
        model=model,
        temperature=args.temperature,
        # if `rpm_limit` isn't used
        **({"rpm_limit": args.rpm_limit} if hasattr(args, 'rpm_limit') else {}))

    if args.csv:
        results.to_csv(f"{args.results_path}/results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}.csv")
    results.to_pickle(f"{args.results_path}/results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}.pkl")

if __name__=='__main__':
    main()
