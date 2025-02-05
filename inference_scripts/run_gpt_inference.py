"""Administers PsyBORGS tests to OpenAI API models via Outlines.

This script supports async inference using `--bulk`. Adjust chunk size to stay
within your desired rate limit. Make sure `OPENAI_API_KEY` is set in your
environment.

Use `python run_gpt_inference.py -h` for detailed usage information.

Example usage:
    python inference_scripts/run_gpt_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='gpt-3.5-turbo-0125' \
        --sample

Usage for processing one payload chunk:
    python inference_scripts/run_gpt_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='gpt-3.5-turbo-0125' \
        --payload='admin_sessions/prod_run_01_external_rating_payload_0.pkl'
"""

# load dependencies
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Callable

import pandas as pd
from tqdm.auto import tqdm
import outlines
from openai import AsyncOpenAI
from outlines.models.openai import OpenAIConfig
import tiktoken

# point system to a local clone of PsyBORGS
PATH = "../"
sys.path.append(PATH)
from psyborgs import survey_bench_lib

# CONSTANTS

SPID = ['item_preamble_id',
        'item_postamble_id',
        'response_scale_id',
        'response_choice_postamble_id',
        'model_id']


# FUNCTIONS

def generate_payload_df(admin_session: survey_bench_lib.AdministrationSession,
                        model_id: str) -> pd.DataFrame:
    """Returns sorted df of prompts, continuations, and info to be scored."""
    # accumulate payloads in a list to be sent to LLM endpoints in parallel
    payload_list = []

    # iterate through all measures and scale combinations
    for measure_iteration in survey_bench_lib.measure_generator(admin_session):

        # iterate through all prompt combinations
        for prompt_iteration in survey_bench_lib.prompt_generator(
                measure_iteration, admin_session):

            # iterate through all continuation combinations
            for continuation_iteration in survey_bench_lib.continuation_generator(
                    measure_iteration, admin_session):

                # generate payload spec with null scores and set model_id
                payload_spec = survey_bench_lib.generate_payload_spec(
                    measure_iteration, prompt_iteration, continuation_iteration,
                    0, model_id)
                payload_list.append(payload_spec)

    # dataframe is sorted by prompt, continuation
    return pd.DataFrame(payload_list).sort_values(
        ['prompt_text', 'continuation_text'])

def to_generative_payload(df: pd.DataFrame) -> pd.DataFrame:
    """ Converts a regular PsyBORGS payload to one for generative scoring.
    
    Regular PsyBORGS payloads contain N rows for N continuations for each unique
    prompt. This function takes this payload and collapses the continuations
    into a list so that there is only one row per prompt, and that row contains 
    a list of its N continuations.

    Args:
        df: A regular PsyBORGS payload df from `generate_layload_df()`.
    Returns:
        A dataframe with one row per prompt–SPID combination.
    """
    converted_df = df \
        .sort_index() \
        .groupby(['prompt_text', 
                  'measure_id', 
                  'measure_name', 
                  'scale_id', 
                  'item_id'] + SPID) \
        .agg({'continuation_text': list,
              'response_value': list,
              'response_choice': list}) \
        .reset_index()

    return converted_df


def process_payload_group(prompts: Sequence[str], 
                          continuations: List[str], 
                          model: Callable,
                          bulk: bool=False,
                          debug: bool=False) -> Sequence[str]:
    """Returns Outlines-constrained model responses for a sequence of prompts.

    Responses are constrained to a static list of continuations.

    Args:
        prompts: A sequence of prompts in string format (likely a dataframe 
            column).
        continuations: A list of possible (string) continuations. Model must 
            choose (i.e., generate) only one of these continuations.
    Returns:
        A sequence of model outputs (strings).
    """
    if debug:
        return prompts.index

    # set sampler (OpenAI models only support multinomial)
    sampler = outlines.samplers.multinomial()

    # if in bulk mode, process each prompt in parallel
    if bulk:

        def generate_answer(prompt):
            # function to generate answer for a single prompt
            generator = outlines.generate.choice(
                model, continuations, sampler=sampler)
            return generator(prompt)

        # create list to preserve order
        model_answers = [None] * len(prompts)

        with ThreadPoolExecutor() as executor:
            future_to_idx = {}
            for idx, prompt in enumerate(prompts):
                future_to_idx[executor.submit(generate_answer, prompt)] = idx

            for future in tqdm(
                as_completed(future_to_idx), total=len(prompts), leave=True):
                idx = future_to_idx[future]
                try:
                    model_answers[idx] = future.result()
                except Exception as exc:
                    print(f"GPT inference script error for prompt '{prompts[idx]}': {exc}")

    elif not bulk:
        # create generator
        generator = outlines.generate.choice(model, continuations, sampler=sampler)

        # create list. since answers are generated in order, we can just append
        model_answers = []
        for prompt in tqdm(
            prompts,
            leave=True):
            answer = generator(prompt)
            model_answers.append(answer)

    return model_answers


def administer_session_via_outlines(payload_df: pd.DataFrame, 
                                    model: Callable,
                                    bulk: bool=False,
                                    vllm: bool=False,
                                    debug: bool=False) -> pd.DataFrame:
    """Send prompts to Outlines, grouped by response scale.

    Args:
        payload_df: A pandas dataframe from `generate_payload_df()`.
        model: An Outlines model object.
        bulk: If True, prompts are sent to Outlines as a full list, where
            they are batched and parallelized as async calls. This might crash
            the kernel if there isn't enough memory. If False (e.g., for 
            debugging on a local / low memory machine), prompts are
            processed serially.
        vllm: 
        debug: If True, models are not called. Instead, the output for each
            prompt is returned as its original index within the converted
            `payload_df`. Good for checking if results are coming back in order.
    Returns:
        A dataframe containing the metadata, item information, and model
            output for each prompt.
    """
    # print memory usage
    # print("converted payload_df memory: ")
    # print(pd.io.formats.info.DataFrameInfo(payload_df).memory_usage_string.strip())

    # get continuations (response scales) to separate payload by
    response_scales = list(payload_df['response_scale_id'].unique())

    # create a list to accumulate grouped results
    scored_dfs = []

    # create and score subpayloads based on common continuations
    for response_scale in tqdm(
        response_scales, 
        leave=True, 
        desc="Iterating through subpayloads grouped by continuations"):
        grouped_df = payload_df.loc[
            payload_df['response_scale_id'] == response_scale].copy()

        # print grouped_df memory
        # print("grouped_df memory: ")
        # print(pd.io.formats.info.DataFrameInfo(grouped_df).memory_usage_string.strip())

        # cache the first set of continuations for reuse
        continuations = grouped_df['continuation_text'].iloc[0]
        print("Working on continuations [" + ', ' \
            .join(continuations) + "]")

        # send subpayload for batch inference
        # record model outputs in new column

        # process using Outlines by default
        if not vllm:
            grouped_df['model_output'] = process_payload_group(
                grouped_df['prompt_text'],
                continuations,
                model=model,
                bulk=bulk,
                debug=debug)
        # call separate function for vLLM
        else:
            grouped_df['model_output'] = process_payload_group_vllm(
                prompts=grouped_df['prompt_text'],
                continuations=continuations,
                model=model,
                bulk=bulk)

        # add scored subpayload to list
        scored_dfs.append(grouped_df)

    # merge list of result dfs
    return pd.concat(scored_dfs).sort_index()


# MAIN

def parse_args():
    """Registers arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--admin_session', type=str, help='`admin_session` path', required=True)
    parser.add_argument(
        '--model_pointer', type=str, 
        help=('pointer or repo string for model (e.g., "gpt-3.5-turbo-0125" '
              'for OpenAI; "mistralai/Mistral-7B-Instruct-v0.2" for '
              'HuggingFace)'),
        default='gpt-3.5-turbo-0125')
    parser.add_argument(
        '--api_key', type=str,
        help='OpenAI API key', default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        '--job_name', type=str, default='gpt-run', 
        help='experiment name (to be used in output filenames)')
    parser.add_argument(
        '--job_id', type=str, default='0',
        help='job ID (for tracking if using HPC)')
    parser.add_argument(
        '--sample', type=bool, default=False,
        help=('if True, only sample 1,000 rows of the `admin_session` used '
              '(for testing)'))
    parser.add_argument(
        '--bulk', type=bool, default=False,
        help='if True, makes parallel calls to OpenAI API. Note rate limits.')
    parser.add_argument(
        '--payload', type=str, default=None, 
        help=('generative payload pickle path, if running on smaller payload '
              'chunks'))

    return parser.parse_args()


def main():
    # register arguments
    args = parse_args()

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

        # convert to generative payload 
        # (continuation choices stored in same row)
        gen_payload = to_generative_payload(payload_df)
    # use payload chunk if specified
    elif args.payload:
        gen_payload = pd.read_pickle(args.payload)

        # set model_id
        gen_payload['model_id'] = args.model_pointer

        # record chunk number for output file,
        # which should be after the last underscore
        chunk_n = "_chunk_" + str(args.payload).split('_')[-1]

    # sample 1000 prompts
    if args.sample:
        print("Sampling only 1,000 prompts.")
        gen_payload = gen_payload.sample(1000, random_state=42)

    # print payload length
    print(f"Payload size: {len(gen_payload)}")

    # retrieve tokenizer
    tokenizer = tiktoken.encoding_for_model(args.model_pointer)

    # create client
    client = AsyncOpenAI(api_key=args.api_key)

    # create configuration
    config = OpenAIConfig(
        model=args.model_pointer,
        seed=42)

    # create model instance
    model = outlines.models.openai(client, config, tokenizer=tokenizer)

    # run inference
    results = administer_session_via_outlines(
        gen_payload, model=model, bulk=args.bulk)

    # show model usage
    print(f"\nPrompt tokens used: {model.prompt_tokens}\n")
    print(f"Completion tokens used: {model.completion_tokens}\n")

    # create a results directory if it doesn't already exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # results.to_csv(
    #     f"results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}.csv")
    results.to_pickle(
        f"results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}.pkl")

if __name__=='__main__':
    main()
