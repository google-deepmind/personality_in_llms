"""Administers PsyBORGS tests to HuggingFace models via Outlines.

This script carries out constrained batch inference using Outlines. Basic local
inference is supported by default. If memory permits, it is more efficient run
batched inference, using the `--bulk` flag.

We recommend running accelerated / distributed inference for larger models using
vLLM (`--vllm`). This option requires at least one CUDA-capable GPU.

Run `python run_hf_inference.py -h` for detailed usage information.

Example usage:
    python inference_scripts/run_hf_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='meta-llama/Llama-2-7b-chat-hf'

Usage for processing one payload chunk:
    python inference_scripts/run_hf_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='meta-llama/Llama-2-7b-chat-hf' \
        --payload='admin_sessions/prod_run_01_external_rating_payloads/prod_run_01_external_rating_payload_0.pkl'

Small test for local usage, without vLLM:
    python inference_scripts/run_hf_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='meta-llama/Llama-2-7b-chat-hf' \
        --sample

Small test for systems with vLLM support:
    python inference_scripts/run_hf_inference.py \
        --admin_session='admin_sessions/prod_run_01_external_rating.json' \
        --model_pointer='meta-llama/Llama-2-7b-chat-hf' \
        --sample \
        --vllm
"""

# load dependencies
import argparse
import sys
import os
from typing import List, Sequence, Callable

import pandas as pd
from tqdm.auto import tqdm
import outlines

# point system to a local clone of PsyBORGS
PATH = "../"
sys.path.append(PATH)
from psyborgs import survey_bench_lib

# adjust this line if you are using CPU cores, without vLLM
# import torch
# torch.set_num_threads(55)


# CONSTANTS

# Simulated Participant IDs (SPIDs) for linking test responses to a particular
# prompt set and model
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
            column.
        continuations: A list of possible (string) continuations. Model must 
            choose (i.e., generate) only one of these continuations.
    Returns:
        A sequence of model outputs (strings).
    """
    if debug:
        return prompts.index

    # set sampler
    # use greedy sampling to ensure determinism
    sampler = outlines.samplers.greedy()
    # sampler = outlines.samplers.multinomial()

    generator = outlines.generate.choice(model, continuations, sampler=sampler)

    # bulk inference
    if bulk:
        model_answers = generator(list(prompts))

    # serial inference
    elif not bulk:
        model_answers = []
        for prompt in tqdm(
            prompts,
            leave=True):
            answer = generator(prompt)
            model_answers.append(answer)

    return model_answers

def process_payload_group_vllm(prompts: Sequence[str],
                               continuations: List[str],
                               model: Callable,
                               temperature: float=1.0) -> Sequence[str]:
    """Returns Outlines-constrained, vLLM model responses for prompts in bulk.

    Responses are constrained to a static list of continuations.

    Args:
        prompts: A sequence of prompts in string format (likely a dataframe 
            column.
        continuations: A list of possible (string) continuations. Model must 
            choose (i.e., generate) only one of these continuations.
        model: A `vllm.LLM` model instance.
    Returns:
        A sequence of model outputs (strings).
    """
    # load vLLM-specific logits processor
    from outlines.serve.vllm import RegexLogitsProcessor

    # convert continuations to Outlines-friendly regex
    regex_str = r"(" + r"|".join(continuations) + r")"

    # compile the finite state machine (FSM); biases the logits before sampling
    logits_processor = RegexLogitsProcessor(regex_string=regex_str,
                                            llm=model.llm_engine)

    # set vLLM sampling parameters
    vllm_sampling_params = vllm.SamplingParams(
        # this seed encourages reproducibility across requests
        seed=42,
        temperature=temperature,
        logits_processors=[logits_processor])

    # run bulk inference
    outputs = model.generate(
        list(prompts),
        sampling_params=vllm_sampling_params)

    # extract model answers from outputs
    model_answers = []
    for output in outputs:
        # generated text is located in outputs object
        answer = output.outputs[0].text
        model_answers.append(answer)

    return model_answers


def administer_session_via_outlines(payload_df: pd.DataFrame,
                                    model: Callable,
                                    temperature: float=1.0,
                                    bulk: bool=False,
                                    use_vllm: bool=False,
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
        if not use_vllm:
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
                temperature=temperature)

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
        help=('pointer or repo string for model (e.g., '
              '"mistralai/Mistral-7B-Instruct-v0.2" for HuggingFace)'),
        default='meta-llama/Llama-2-7b-chat-hf')
    parser.add_argument(
        '--job_name', type=str, default='hf-run', 
        help='experiment name (to be used in output filenames)')
    parser.add_argument(
        '--job_id', type=str, default='999',
        help='job ID (for tracking if using HPC)')
    parser.add_argument(
        '--n_gpus', type=int, default=1, 
        help='number of GPUs to use for distributed tensor-parallel inference')
    parser.add_argument(
        '--temperature', type=float, default=1.00,
        help='float controlling randomness of sampling')
    parser.add_argument(
        '--sample', action='store_true',
        help=('if True, only sample 1,000 rows of the `admin_session` used '
              '(for testing)'))
    parser.add_argument(
        '--bulk', action='store_true',
        help=('if True, run using Outlines\' default implementation of batched'
             'inference (not using vLLM)'))
    parser.add_argument(
        '--vllm', action='store_true',
        help=('run model on vLLM (requires at least one GPU with a CUDA '
              'compute capability of 7.0 or higher)'))
    parser.add_argument(
        '--enforce_eager', action='store_true',
        help='execute the model in eager mode')
    parser.add_argument(
        '--payload', type=str, default=None, 
        help=('generative payload pickle path, if running on smaller payload '
              'chunks'))

    return parser.parse_args()


def main():
    """Registers arguments, runs inference, saves results."""
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

        # convert to generative payload (continuation choices are stored in same
        # row)
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

    # create model instance
    if args.vllm:
        # import vLLM outside toplevel only if used. otherwise, this script will
        # not work on machines without vLLM-supported GPUs
        import vllm
        global vllm

        model = vllm.LLM(
            model=args.model_pointer,
            # seed for `vllm.EngineArgs`
            seed=42,
            # set number of GPUs to use in parallel
            tensor_parallel_size=args.n_gpus)
    else:
        model = outlines.models.transformers(
            model_name=args.model_pointer,
            device="auto")

    # print temperature
    print(f"Sampling temperature: {args.temperature}")

    # run inference
    results = administer_session_via_outlines(
        gen_payload,
        model=model,
        temperature=args.temperature,
        bulk=args.bulk,
        use_vllm=args.vllm)

    # create a results directory if it doesn't already exist
    if not os.path.exists('results'):
        os.makedirs('results')

    # results.to_csv(
    #     f"results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}")
    results.to_pickle(
        f"results/results_{args.job_name}_{MODEL_ID}_{args.job_id}{chunk_n}")

if __name__=='__main__':
    main()
