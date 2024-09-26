from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import sys
from math import floor
from tqdm import tqdm
import datasets
import re
import textdistance
from pvg.utils.apps_metric import check_correctness
import requests
import multiprocessing
import argparse
from openai import OpenAI
from pvg.constants import CV_DATA_DIR, OPENAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN

HF_TOKEN = "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"

def suppress_output(func):
    def wrapper(*args, **kwargs):
        with open(os.devnull, 'w') as fnull:
            sys.stdout = fnull
            sys.stderr = fnull
            result = func(*args, **kwargs)
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        return result
    return wrapper

def solution_generation_wrapper(
    result,
    datum,
    model,
    openrouter_api_key,
    system_prompt,
    max_modifications,
    max_attempts=None,
    debug=False,
):

    try:
        buggy_solutions = generate_buggy_solutions(
            datum,
            model,
            openrouter_api_key,
            system_prompt,
            max_modifications,
            max_attempts=max_attempts,
            debug=debug,
        )
        return result.append(buggy_solutions)
    except Exception as e:
        result.append([None])


def get_openai_response(
    model,
    messages,
    api_key=OPENAI_API_KEY,
    temperature=1.0,
    log_probs=False,
    top_logprobs=None,
    num_responses=1,
):

    client = OpenAI(api_key=api_key)

    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        logprobs=log_probs,
        top_logprobs=top_logprobs,
        n=num_responses,
    ).choices


def get_openrouter_response(
    model,
    messages,
    api_key=OPENROUTER_API_KEY,
    temperature=1.0,
    log_probs=False,
    top_logprobs=None,
    num_responses=1,
    force_multiple_generations=False,
):
    """
    Sends a POST request to the OpenRouter API to get responses from a chat model.

    Parameters:
    - model (str): The name of the chat model to use.
    - messages (list): A list of dictionaries representing the chat messages. Each dictionary should have a "role" key with the value "user" or "assistant", and a "content" key with the content of the message.

    Returns:
    - response (Response): The response object returned by the API.

    Raises:
    - requests.exceptions.RequestException: If there was an error sending the request.

    """
    responses = []

    # Crazily, the openrouter API doesn't support multiple completions in a single request, so we have to make multiple requests
    # Also, we don't seem to have access to the the o1 models via the OpenAI API atb the moment?
    if "openai" not in model or "o1" in model or force_multiple_generations:
        completions = [
            requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                data=json.dumps(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "logprobs": log_probs,
                        "top_logprobs": top_logprobs,
                    }
                ),
            ).json()["choices"][0]
            for _ in range(num_responses)
        ]

    else:
        completions = get_openai_response(
            model.split("/")[-1],
            messages,
            temperature=temperature,
            log_probs=log_probs,
            top_logprobs=top_logprobs,
            num_responses=num_responses,
        )
        completions = [c.model_dump() for c in completions]

    for completion in completions:

        responses.append(
            {
                "message": completion["message"]["content"],
                "log_probs": (
                    [
                        {k: c[k] for k in ["token", "logprob"]}
                        for c in completion["logprobs"]["content"]
                    ]
                    if log_probs and "log_probs" in completion and completion["logprobs"] is not None
                    else None
                ),
                "top_logprobs": (
                    [
                        [
                            {k: cc[k] for k in ["token", "logprob"]}
                            for cc in c["top_logprobs"]
                        ]
                        for c in completion["logprobs"]["content"]
                    ]
                    if top_logprobs and "log_probs" in completion and completion["logprobs"] is not None
                    else None
                ),
            }
        )

    return responses


@dataclass
class CodeValidationDatasetConfig:
    """
    CodeValidationDatasetConfig is a configuration class for generating datasets used in code validation systems.

    Attributes:
        model (str): The model to be used, default is "openai/gpt-4o-mini".
        difficulties (List[str]): List of difficulty levels, default is ["interview"].
        split (str | None): The data split, default is None.
        fraction_to_modify (float): Fraction of data to modify, default is 0.5.
        max_modifications (int): Maximum number of modifications, default is 10.
        num_data (Optional[int]): Number of data points per split per difficulty level, default is 100.
        num_problematic_inputs (int): Number of problematic inputs to request, default is 0.
        system_prompt (Optional[str]): System prompt for generating incorrect solutions, default is None.
        local_dir (str): Local directory for data storage.
        pull_repo (Optional[str]): Repository to pull data from, default is 'lrhammond/buggy-apps'.
        push_repo (Optional[str]): Repository to push data to, default is 'lrhammond/buggy-apps'.
        save_after (Optional[int]): Number of operations after which to save data, default is 10.

    Methods:
        __post_init__: Initializes the system prompt based on the number of problematic inputs.
    """

    model: str = "openai/gpt-4o-mini"
    difficulties: List[str] = field(default_factory=lambda: ["interview","competition","introductory"])
    split: str | None = None
    fraction_to_modify: float = 0.5
    max_modifications: int = 1
    num_data: Optional[int] = 10000  # per split per difficulty level
    num_problematic_inputs: int = 0
    system_prompt: Optional[str] = None
    max_attempts: int = 10
    local_dir: str = CV_DATA_DIR
    openrouter_api_key: Optional[str] = OPENROUTER_API_KEY
    token: Optional[str] = HF_TOKEN
    pull_repo: Optional[str] = "lrhammond/buggy-apps"
    push_repo: Optional[str] = "lrhammond/buggy-apps"
    save_after: Optional[int] = 10

    def __post_init__(self):
        """
        Post-initialization method to set the system prompt based on the number of problematic inputs.

        This method sets the `system_prompt` attribute based on the value of `num_problematic_inputs`.
        If `system_prompt` is already provided, it does nothing. Otherwise, it generates a prompt
        for generating incorrect solutions and problematic inputs for a code validation system.

        - If `num_problematic_inputs` is less than 0 or greater than 10, it raises a ValueError.
        - If `num_problematic_inputs` is 0, it generates a prompt for modifying a solution without
          providing problematic inputs.
        - If `num_problematic_inputs` is between 1 and 10, it generates a prompt for modifying a solution
          and provides placeholders for the specified number of problematic inputs.

        Raises:
            ValueError: If `num_problematic_inputs` is not between 0 and 10.
        """

        if self.system_prompt is None:
            if self.num_problematic_inputs < 0 or self.num_problematic_inputs > 10:
                return ValueError("num_problematic_inputs must be between 0 and 10")
            elif self.num_problematic_inputs == 0:
                self.system_prompt = f"I am developing a new code validation system. In order test this system, I need to provide it with a set of correct and incorrect solutions for a given coding problem. I already have the correct solutions, and I need your help to generate some incorrect solutions by modifying the correct solutions. In what follows, I will provide you with a QUESTION and a SOLUTION. Your task is to subtly modify this solution by introducing an error that causes the code to fail on a small fraction of inputs. To test the system thoroughly, the error in this MODIFIED SOLUTION should be as difficult to spot as possible (while making sure the code remains similar to the original solution) and should not be a simple syntax error. Please only modify the solution given to you and preserve all other aspects of the solution (such as the formatting and variable names). You must *not* include any natural language explanation of your modifications or any comments in the code. The format of your answer should be:\n\n MODIFIED SOLUTION:\n ```<your modified code here>```"
            else:
                indices = {
                    1: "first",
                    2: "second",
                    3: "third",
                    4: "fourth",
                    5: "fifth",
                    6: "sixth",
                    7: "seventh",
                    8: "eighth",
                    9: "ninth",
                    10: "tenth",
                }
                problematic_inputs = "\n\n".join(
                    [
                        f"PROBLEMATIC INPUT {i+1}:\n<your {indices[i+1]} problematic input here>"
                        for i in range(self.num_problematic_inputs)
                    ]
                )
                self.system_prompt = f"I am developing a new code validation system. In order test this system, I need to provide it with a set of correct and incorrect solutions for a given coding problem. I already have the correct solutions, and I need your help to generate some incorrect solutions by modifying the correct solutions. In what follows, I will provide you with a QUESTION and a SOLUTION. Your task is to subtly modify this solution by introducing an error that causes the code to fail on a small fraction of inputs, and to provide {self.num_problematic_inputs} PROBLEMATIC INPUTs on which the original solution *works*, but your MODIFIED SOLUTION *fails*. To test the system thoroughly, the error in this MODIFIED SOLUTION should be as difficult to spot as possible (while making sure the code remains similar to the original solution) and should not be a simple syntax error. Please only modify the solution given to you and preserve all other aspects of the solution (such as the formatting and variable names). You must *not* include any natural language explanation of your modifications or any comments in the code. The format of your answer should be:\n\n MODIFIED SOLUTION:\n ```<your modified code here>```\n\n{problematic_inputs}\n"


def extract_code_and_input(model_output: str) -> Tuple[str, List[str], bool]:
    """
    Extracts the modified solution and problematic inputs from the model output.

    This function uses regular expressions to parse the model output and extract:
    1. Problematic inputs.
    2. The modified solution.
    3. A safety check to ensure the modified solution does not contain dangerous functions.

    Args:
        model_output (str): The output from the model containing problematic inputs and the modified solution.

    Returns:
        tuple: A tuple containing:
            - modified_solution (str or None): The modified solution extracted from the model output, or None if not found.
            - problematic_inputs (list): A list of problematic inputs extracted from the model output.
            - safe (bool): A boolean indicating whether the modified solution is safe (i.e., does not contain dangerous functions).
    """

    # Extract the problematic input using regex
    pattern = re.compile(
        r"PROBLEMATIC INPUT \d+:\n(.*?)(?=\nPROBLEMATIC INPUT \d+:|\Z)", re.DOTALL
    )
    problematic_inputs = [
        input.replace("```", "").strip() for input in pattern.findall(model_output)
    ]

    # Extract the modified solution using regex
    if problematic_inputs != []:
        pattern = re.compile(
            r"MODIFIED SOLUTION:\s*(.*?)\s*PROBLEMATIC INPUT 1:", re.DOTALL
        )
    else:
        pattern = re.compile(r"MODIFIED SOLUTION:\s*(.*)", re.DOTALL)
    match = pattern.search(model_output)
    if match:
        modified_solution = (
            match.group(1).replace("python", "").replace("```", "").strip()
        )
    else:
        modified_solution = None

    # Check for dangerous functions
    dangerous_functions = [
        "exit",
        "quit",
        "kill",
        "system",
        "putenv",
        "remove",
        "removedirs",
        "rmdir",
        "fchdir",
        "setuid",
        "fork",
        "forkpty",
        "killpg",
        "rename",
        "renames",
        "truncate",
        "replace",
        "unlink",
        "fchmod",
        "fchown",
        "chmod",
        "chown",
        "chroot",
        "lchflags",
        "lchmod",
        "lchown",
        "getcwd",
        "chdir",
        "rmtree",
        "move",
        "chown",
        "Popen",
        "__builtins__",
        "ipdb",
        "joblib",
        "resource",
        "psutil",
        "tkinter",
    ]
    safe = not any(func in modified_solution for func in dangerous_functions)

    return modified_solution, problematic_inputs, safe


def test_buggy_solution(
    buggy_solution: str,
    solution: str,
    problematic_inputs: List[str],
    datum: dict,
    debug: bool = False,
    ignore_invalid_outputs: bool = False,
) -> tuple[bool, List, List]:
    """
    Tests a buggy solution against a correct solution using provided inputs and datum.
    Args:
        buggy_solution (str): The buggy solution code to be tested.
        solution (str): The correct solution code for comparison.
        problematic_inputs (List[str]): A list (possibly empty) of inputs that are predicted to cause issues.
        datum (dict): The corresponding datum from the original APPS dataset.
        debug (bool, optional): If True, prints debug information. Defaults to False.
        ignore_invalid_outputs (bool, optional): If True, ignores invalid outputs when checking correctness. Defaults to False.
    Returns:
        tuple[bool, List, List]: A tuple containing:
            - A boolean indicating if the buggy solution is incorrect.
            - A list of results from the correctness checks.
            - A list of mismatched input-output pairs for the problematic_inputs if produced, otherwise None.
    """

    # Test the buggy solution against the correctness checks
    results, buggy_outputs = check_correctness(
        datum, buggy_solution, timeout=5, debug=False, return_outputs=True
    )

    if results is None:
        return False, results, None

    fraction_correct = float(results.count(True) / len(results))
    fraction_errors = float(buggy_outputs.count(None) / len(buggy_outputs))
    if debug:
        print(f"{int(fraction_errors*100)}% errors")
        print(f"{int(fraction_correct*100)}% checks passed")

    if (
        fraction_correct < 1
        if ignore_invalid_outputs
        else fraction_correct < (1 - fraction_errors)
    ):
        return True, results, None

        # in_outs = json.loads(datum["input_output"])
        # mismatched = []
        # for i in range(len(results)):
        #     buggy = False
        #     if results[i] is not True:
        #         if ignore_invalid_outputs:
        #             buggy = True
        #         else:
        #             buggy = buggy_outputs[i] is not None
        #         if buggy:
        #             mismatched.append(
        #                 {
        #                     "input": in_outs['inputs'][i],
        #                     "output": in_outs['outputs'][i],
        #                     "buggy_output": buggy_outputs[i],
        #                 }
        #             )

    # Generate a new outputs based on the problematic inputs and test the buggy solution
    if problematic_inputs == []:
        return False, results, None

    _, outputs = check_correctness(
        datum, solution, evaluate=problematic_inputs, timeout=5, debug=False
    )

    if outputs is not None:
        _, buggy_outputs = check_correctness(
            datum, buggy_solution, evaluate=problematic_inputs, timeout=5, debug=False
        )
    else:
        buggy_outputs = outputs

    # Check for matching solutions
    if len(outputs) != len(buggy_outputs):
        raise ValueError("Both lists must have the same length")

    numer = 0
    denom = 0
    mismatched_indices = []

    for i in range(len(outputs)):
        buggy = False
        # Ignore invalid inputs
        if outputs[i] is not None:
            denom += 1
            # Check for a a mismatch
            if outputs[i] != buggy_outputs[i]:
                if ignore_invalid_outputs:
                    buggy = True
                else:
                    buggy = buggy_outputs[i] is not None
        if buggy:
            numer += 1
            mismatched_indices.append(i)

    fraction_matching = (denom - numer) / denom if denom > 0 else 1

    if debug:
        print(f"{(denom / len(outputs))*100:.2f}% valid inputs")
        print(f"{fraction_matching*100:.2f}% matching outputs")

    if fraction_matching == 1.0:
        return False, results, None

    else:
        mismatched = []
        for i in mismatched_indices:
            mismatched.append(
                {
                    "input": problematic_inputs[i],
                    "output": outputs[i],
                    "buggy_output": buggy_outputs[i],
                }
            )

    return True, results, mismatched


def generate_buggy_solutions(
    datum: dict,
    model: str,
    openrouter_api_key: str,
    system_prompt: str,
    max_modifications: int,
    fraction_to_modify: float = 0.5,
    max_attempts: int = 10,
    multiple_completions: bool = True,
    existing_buggy_solutions: Optional[List[dict]] = None,
    debug: Optional[bool] = False,
) -> List[str | None]:
    """
    Generate buggy solutions for a given datum by modifying a fraction of the provided solutions.
    Args:
        datum (dict): The corresponding datum from the original APPS dataset.
        model (str): The model to use for generating buggy solutions.
        system_prompt (str): The system prompt to provide context to the model.
        fraction_to_modify (float): The fraction of solutions to modify.
        max_modifications (int): The maximum number of solutions to modify.
        max_attempts (int, optional): The maximum number of attempts to generate a valid buggy solution. Defaults to 10.
        existing_buggy_solutions (Optional[List[dict]], optional): A list of existing buggy solutions to update or extend. Defaults to None.
        debug (Optional[bool], optional): If True, print debug information. Defaults to False.
    Returns:
        List[str | None]: A list of buggy solutions or None if a valid buggy solution could not be generated.
    """

    if debug:
        print(f"Problem: {datum['problem_id']}")

    if existing_buggy_solutions is None:
        buggy_solutions = [None] * len(json.loads(datum["solutions"]))

    solutions = json.loads(datum["solutions"])
    if len(solutions) != len(buggy_solutions):
        raise ValueError("Both lists of solutions must have the same length")

    num_to_modify = min(floor(fraction_to_modify * len(solutions)), max_modifications)
    modified = 0

    for s in range(len(solutions)):

        if modified >= num_to_modify or buggy_solutions[s] is not None:
            continue

        # Check that original solution is valid
        elif not all(check_correctness(datum, solutions[s])):
            continue

        else:
            solution = solutions[s]

            user_prompt = (
                "QUESTION:\n\n\n" + datum["question"] + "\n\nSOLUTION:\n\n\n" + solution
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            valid_buggy_solution = False
            attempts = 0

            if debug:
                print(f"Solution: {s}/{len(solutions)}")

            while not valid_buggy_solution and attempts < max_attempts:

                if debug:
                    print(f"Attempt: {attempts}")

                if multiple_completions:
                    model_outputs = get_openrouter_response(
                        model, messages, openrouter_api_key, num_responses=max_attempts
                    )
                    model_outputs = [m["message"] for m in model_outputs]
                else:
                    model_outputs = [
                        get_openrouter_response(model, messages, openrouter_api_key)[0][
                            "message"
                        ]
                    ]

                if debug:
                    print("Model output(s) received")

                for model_output in model_outputs:

                    attempts += 1

                    buggy_solution, problematic_inputs, safe = extract_code_and_input(
                        model_output
                    )

                    if debug:
                        print("Answer extracted")

                    if not safe:
                        if debug:
                            print("Unsafe solution generated")
                        continue

                    valid_buggy_solution, input_output_checks, _ = test_buggy_solution(
                        buggy_solution, solution, problematic_inputs, datum
                    )

                    if valid_buggy_solution:
                        break

            # if not (True in input_output_checks or False in input_output_checks):
            #     valid_buggy_solution = False

            if valid_buggy_solution:
                if debug:
                    print(f"Valid buggy solution generated in {attempts} attempts\n")

                # Calculate the Levenshtein distance
                levenshtein_distance = textdistance.levenshtein(
                    solution, buggy_solution
                )
                normalised_levenstein_distance = levenshtein_distance / max(
                    len(solution), len(buggy_solution)
                )
                levenshtein_distance = {
                    "normalised": normalised_levenstein_distance,
                    "raw": levenshtein_distance,
                }

            if attempts == max_attempts:
                if debug:
                    print("Failed to generate buggy solution\n")
                buggy_solutions[s] = None
            else:
                passed = input_output_checks.count(True)
                failed = input_output_checks.count(False)

                buggy_solutions[s] = {
                    "apps_solution_number": s,
                    "solution": buggy_solution,
                    "levenshtein_distance": levenshtein_distance,
                    "checks": {
                        "pass": passed,
                        "fail": failed,
                        "error": len(input_output_checks) - passed - failed,
                    },
                    "generation_attempts": attempts,
                }
                modified += 1

    return buggy_solutions

def create_empty_dataset():

    return datasets.Dataset.from_dict(
                {
                    k: []
                    for k in [
                        "apps_split",
                        "apps_problem_id",
                        "difficulty",
                        "question",
                        "solutions",
                        "buggy_solutions",
                    ]
                }
            )

def load_buggy_data(config: CodeValidationDatasetConfig, split: List[str]) -> datasets.Dataset:

    # Load existing buggy data (the only split is train, so we specify that)
    if config.pull_repo is not None:
        buggy_data = datasets.load_dataset(config.pull_repo, split="train")
        Path(config.local_dir).mkdir(parents=True, exist_ok=True)
    else:
        if config.local_dir is None:
            raise ValueError(
                "local_dir must be specified if pull_repo is not specified"
            )
        elif all(
            [
                os.path.exists(os.path.join(config.local_dir, f"{s}.jsonl"))
                for s in split
            ]
        ):
            buggy_data = datasets.load_dataset(config.local_dir)
        else:
            Path(config.local_dir).mkdir(parents=True, exist_ok=True)
            buggy_data = create_empty_dataset()

    return buggy_data

def generate_cv_dataset(
    config: CodeValidationDatasetConfig | dict,
    manager: Optional[multiprocessing.Manager] = None,
):

    if isinstance(config, dict):
        config = CodeValidationDatasetConfig(**config)

    # Create process manager in case we get stuck on any of the problems
    if manager is None:
        manager = multiprocessing.Manager()
    else:
        process_results = manager.list()

    split = ["train", "test"] if config.split is None else [config.split]

    # Load existing data
    data = datasets.load_dataset("codeparrot/apps", trust_remote_code=True)
    buggy_data = load_buggy_data(config, split)

    start_time = datetime.now()

    for s in split:

        problematic_problems_filename = os.path.join(
            config.local_dir, f"problematic_{s}_problems.csv"
        )
        with open(problematic_problems_filename, mode="r") as file:
            line = file.readline().strip()
            problematic_problems = [
                int(value) for value in line.split(",") if value != ""
            ]

        for d in config.difficulties:

            data_slice = (
                data[s].filter(lambda x: x["difficulty"] == d).sort("problem_id")
            )
            buggy_data_slice = buggy_data.filter(
                lambda x: x["difficulty"] == d and x["apps_split"] == s
            ).sort("apps_problem_id")

            num_buggy_data_to_generate = max(min(config.num_data, len(data_slice)) - len(
                buggy_data_slice
            ), 0)
            num_data_added = 0
            print(
                f"Generating {num_buggy_data_to_generate} buggy data for the {d} problems in the {s} split"
            )

            new_data = create_empty_dataset()

            for datum in tqdm(data_slice, colour="green"):

                if (
                    datum["solutions"] == ""
                    or int(datum["problem_id"]) in problematic_problems
                ):
                    continue
                # print(datum["problem_id"])
                solutions = json.loads(datum["solutions"])

                # Check if we've already generated the buggy data or if there is only one solution given
                if (
                    len(buggy_data_slice) >= num_buggy_data_to_generate
                    or num_data_added >= num_buggy_data_to_generate
                ):
                    break
                elif (
                    datum["problem_id"] in buggy_data_slice["apps_problem_id"]
                    or len(solutions) <= 1
                ):
                    continue

                buggy_datum = {
                    "apps_split": s,
                    "apps_problem_id": datum["problem_id"],
                    "difficulty": d,
                    "question": datum["question"],
                    "solutions": [],
                    "buggy_solutions": [],
                }

                # Generate new buggy solutions
                timeout = min(20 + (5 * len(solutions)), 120)
                p = multiprocessing.Process(
                    target=solution_generation_wrapper,
                    args=(
                        process_results,
                        datum,
                        config.model,
                        config.openrouter_api_key,
                        config.system_prompt,
                        config.max_modifications,
                        config.max_attempts,
                        False,
                    ),
                )
                p.start()
                p.join(timeout)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    print(
                        f"Generating buggy solutions for problem {datum['problem_id']} ({s}) timed out after {timeout} seconds"
                    )
                    buggy_solutions = [None]
                else:
                    buggy_solutions = process_results[-1]

                # buggy_solutions = generate_buggy_solutions(
                #     datum,
                #     config.model,
                #     config.openrouter_api_key,
                #     config.system_prompt,
                #     config.max_modifications,
                #     max_attempts=config.max_attempts,
                #     debug=True,
                # )

                # If we didn't manage to geerate any buggy solutions, we make a note of this problem so we can skip it in the future
                num_buggy_solutions_generated = len(
                    buggy_solutions
                ) - buggy_solutions.count(None)
                if num_buggy_solutions_generated == 0:
                    with open(problematic_problems_filename, "a") as f:
                        f.write(f"{datum['problem_id']},")
                    continue

                solutions_added = 0
                num_checks = len(json.loads(datum["input_output"])["inputs"])
                for i in range(len(buggy_solutions)):
                    if buggy_solutions[i] is None:
                        if solutions_added < num_buggy_solutions_generated:
                            solution = {
                                "apps_solution_number": i,
                                "solution": solutions[i],
                                "levenshtein_distance": {"normalised": 0.0, "raw": 0},
                                "checks": {"pass": num_checks, "fail": 0, "error": 0},
                                "generation_attempts": 0,
                            }
                            buggy_datum["solutions"].append(solution)
                            solutions_added += 1
                    else:
                        buggy_datum["buggy_solutions"].append(buggy_solutions[i])

                new_data = new_data.add_item(buggy_datum)
                num_data_added += 1

                # Occasionally save if required
                if config.save_after is None:
                    continue
                elif num_data_added % config.save_after == 0 and num_data_added > 0:
                    # buggy_data.to_json(os.path.join(config.local_dir, s, f"{d}.jsonl"))
                    buggy_data = load_buggy_data(config, split)
                    combined_buggy_data = datasets.concatenate_datasets([buggy_data, new_data])
                    new_data = create_empty_dataset()
                    combined_buggy_data.save_to_disk(
                        os.path.join(config.local_dir, "code_validation.data")
                    )
                    if config.push_repo is not None:
                        combined_buggy_data.push_to_hub(config.push_repo, token=config.token)

            # Calculate the elapsed time, rounding microseconds down
            elapsed_time = datetime.now() - start_time
            elapsed_time = timedelta(
                days=elapsed_time.days, seconds=elapsed_time.seconds
            )
            message = f"Added {num_data_added} buggy data overall in {elapsed_time} for the {d} problems in the {s} split"
            print(message)

            # Save data
            if num_data_added > 0:
                # buggy_data.to_json(os.path.join(config.local_dir, s, f"{d}.jsonl"))
                buggy_data = load_buggy_data(config, split)
                combined_buggy_data = datasets.concatenate_datasets([buggy_data, new_data])
                new_data = create_empty_dataset()
                combined_buggy_data.save_to_disk(
                    os.path.join(config.local_dir, "code_validation.data")
                )
                if config.push_repo is not None:
                    combined_buggy_data.push_to_hub(
                        config.push_repo, commit_message=message, token=config.token
                    )


def change_to_new_cv_dataset(max_solutions=1):

    data = datasets.load_dataset(
        "codeparrot/apps",
        trust_remote_code=True,
    )

    buggy_data = datasets.load_dataset("lrhammond/buggy-apps")

    new_dataset = datasets.Dataset.from_dict(
        {
            k: []
            for k in [
                "apps_split",
                "apps_problem_id",
                "difficulty",
                "question",
                "solutions",
                "buggy_solutions",
            ]
        }
    )

    # for split in ["train", "test"]:
    for split in ["train", "test"]:

        indices = buggy_data[split]["problem_id"]
        sliced_data = data[split].filter(lambda x: x["problem_id"] in indices)

        print("Split: {split}")

        for buggy_datum, datum in tqdm(zip(buggy_data[split], sliced_data)):

            # continue

            if buggy_datum["problem_id"] != datum["problem_id"]:
                raise ValueError("The data is not aligned.")

            solutions = json.loads(datum["solutions"])
            buggy_solutions = json.loads(buggy_datum["solutions"])

            # Don't add items for problems with only one solution
            if len(solutions) <= 1:
                continue

            new_datum = {
                "apps_split": split,
                "apps_problem_id": datum["problem_id"],
                "difficulty": datum["difficulty"],
                "question": datum["question"],
                "solutions": [],
                "buggy_solutions": [],
            }

            num_checks = len(json.loads(datum["input_output"])["inputs"])

            # Add at most 10 pairs of solutions, and keep the dataset balanced
            num_pairs_to_add = min(
                max_solutions,
                len(buggy_solutions) - buggy_solutions.count(None),
                buggy_solutions.count(None),
            )
            added = {"solutions": 0, "buggy_solutions": 0}

            for i in range(len(solutions)):

                if buggy_solutions[i] is not None:

                    if added["buggy_solutions"] >= num_pairs_to_add:
                        continue

                    buggy_solution = json.loads(buggy_solutions[i])
                    passed = buggy_solution["input_output_checks"].count(True)
                    failed = buggy_solution["input_output_checks"].count(False)
                    # if max(passed, failed) == 0:
                    #     raise ValueError("No checks were performed")

                    new_solution = {
                        "apps_solution_number": i,
                        "solution": buggy_solution["solution"],
                        "levenshtein_distance": buggy_solution["levenshtein_distance"],
                        "checks": {
                            "pass": passed,
                            "fail": failed,
                            "error": num_checks - passed - failed,
                        },
                        "generation_attempts": buggy_solution["generation_attempts"],
                    }

                    new_datum["buggy_solutions"].append(new_solution)
                    added["buggy_solutions"] += 1

                else:

                    if added["solutions"] >= num_pairs_to_add:
                        continue

                    new_solution = {
                        "apps_solution_number": i,
                        "solution": solutions[i],
                        "levenshtein_distance": {"normalised": 0.0, "raw": 0},
                        "checks": {"pass": num_checks, "fail": 0, "error": 0},
                        "generation_attempts": 0,
                    }

                    new_datum["solutions"].append(new_solution)
                    added["solutions"] += 1

            new_dataset = new_dataset.add_item(new_datum)

    new_dataset.save_to_disk(os.path.join(CV_DATA_DIR, "new_cv_dataset.data"))
    new_dataset.to_json(os.path.join(CV_DATA_DIR, "new_cv_dataset.jsonl"))

    return new_dataset


# TODO (work in progress, original version that is useful for more intelligently updating or extending the buggy dataset)
def update_cv_dataset(config: CodeValidationDatasetConfig | dict):

    return

    # return

    if isinstance(config, dict):
        config = CodeValidationDatasetConfig(**config)

    split = ["train", "test"] if config.split is None else [config.split]

    # Load original data
    data = datasets.load_dataset(
        "codeparrot/apps",
        trust_remote_code=True,
        split=config.split,
    )

    # Load existing buggy data
    if config.pull_repo is not None:
        buggy_data = datasets.load_dataset(config.pull_repo, split=config.split)
    else:
        if config.local_dir is None:
            raise ValueError(
                "local_dir must be specified if pull_repo is not specified"
            )
        elif os.path.exists(config.local_dir):
            buggy_data = datasets.load_dataset(
                "json", data_dir=config.local_dir, split=config.split
            )
        else:
            all_data = (
                data
                if config.split is None
                else datasets.load_dataset("codeparrot/apps", trust_remote_code=True)
            )
            buggy_data = datasets.DatasetDict(
                {
                    s: datasets.Dataset.from_dict(
                        {k: all_data[s][k] for k in ["problem_id", "difficulty"]}
                        | {"solutions": [""] * len(all_data[s])}
                    )
                    for s in ["train", "test"]
                }
            )

    # To help with indexing
    if config.split is not None:
        buggy_data = {s: buggy_data for s in split}
        data = {s: data for s in split}

    for s in split:
        if len(data[s]) != len(buggy_data[s]):
            raise ValueError("The lengths of the original and buggy data do not match")

    num_data_added = 0
    num_data_updated = 0
    start_time = datetime.now()

    for s in split:
        # print(f"Generating {len(config.data_range)} buggy data for the {s} split")
        counters = {d: 0 for d in config.difficulties}
        for i in range(len(data[s])):

            datum = data[s][i]
            d = datum["difficulty"]

            # Skip difficulties not in the list
            if d not in config.difficulties:
                continue
            # Skip if we have already generated enough buggy data for this difficulty level
            if counters[d] >= config.num_data:
                continue

            # Sanity check matching problem IDs and difficulties
            buggy_datum = buggy_data[s][i]
            if buggy_datum["problem_id"] != datum["problem_id"]:
                raise ValueError("Problem IDs do not match")
            if buggy_datum["difficulty"] != d:
                raise ValueError("Difficulties do not match")

            # Check if have already generated buggy solutions
            if buggy_datum["solutions"] == "":
                buggy_solutions = [None] * len(json.loads(datum["solutions"]))
                num_data_added += 1
            else:
                buggy_solutions = json.loads(buggy_datum["solutions"])
                if (
                    buggy_solutions.count(None) / len(buggy_solutions)
                    <= 1 - config.fraction_to_modify
                ):
                    continue
                num_data_updated += 1

            # Generate new buggy solutions
            new_buggy_solutions = generate_buggy_solutions(
                datum,
                buggy_solutions,
                config.model,
                config.system_prompt,
                config.fraction_to_modify,
            )
            # buggy_data.map(lambda x: x["solutions"] = json.dumps(new_buggy_solutions))
            buggy_data[s][i]["solutions"] = json.dumps(new_buggy_solutions)

            # Update the counters
            counters[d] += 1

            # Save if required
            if config.save_after is None:
                continue
            elif sum(counters.values()) % config.save_after == 0:
                Path(config.local_dir).mkdir(parents=True, exist_ok=True)
                buggy_data[s].to_json(os.path.join(config.local_dir, f"{s}.jsonl"))
                if config.push_repo is not None:
                    buggy_data[s].push_to_hub(config.push_repo, token=config.token)

    # Calculate the elapsed time, rounding microseconds down
    elapsed_time = datetime.now() - start_time
    elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)
    print(f"Done in {elapsed_time}")

    # Add new buggy data to the existing buggy data
    # for s in config.split:
    #     with open(os.path.join(config.local_dir, f"{s}.jsonl"), 'a') as file:
    #         for d in new_buggy_data[s]:
    #             # Convert the dictionary to a JSON string and write it to the file
    #             file.write(json.dumps(d) + '\n')

    # Save
    if sum(counters.values()) > 0:
        Path(config.local_dir).mkdir(parents=True, exist_ok=True)
        for s in split:
            buggy_data[s].to_json(os.path.join(config.local_dir, f"{s}.jsonl"))
            if config.push_repo is not None:
                buggy_data.push_to_hub(config.push_repo, split=s)

            new_datum = {
                "apps_split": split,
                "apps_problem_id": datum["problem_id"],
                "difficulty": datum["difficulty"],
                "question": datum["question"],
                "solutions": [],
                "buggy_solutions": [],
            }

            num_checks = len(json.loads(datum["input_output"])["inputs"])

            # Add at most 10 pairs of solutions, and keep the dataset balanced
            num_pairs_to_add = min(
                max_solutions,
                len(buggy_solutions) - buggy_solutions.count(None),
                buggy_solutions.count(None),
            )
            added = {"solutions": 0, "buggy_solutions": 0}

            for i in range(len(solutions)):

                if buggy_solutions[i] is not None:

                    if added["buggy_solutions"] >= num_pairs_to_add:
                        continue

                    buggy_solution = json.loads(buggy_solutions[i])
                    passed = buggy_solution["input_output_checks"].count(True)
                    failed = buggy_solution["input_output_checks"].count(False)
                    if max(passed, failed) == 0:
                        raise ValueError("No checks were performed")

                    new_solution = {
                        "apps_solution_number": i,
                        "solution": buggy_solution["solution"],
                        "levenshtein_distance": buggy_solution["levenshtein_distance"],
                        "checks": {
                            "pass": passed,
                            "fail": failed,
                            "error": num_checks - passed - failed,
                        },
                        "generation_attempts": buggy_solution["generation_attempts"],
                    }


if __name__ == "__main__":

    # Set up the arg parser
    parser = argparse.ArgumentParser(
        description="Generate a code validation dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Whether to draw problems from the train or test split of the APPS dataset",
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=100,
        help="How many problems the dataset should contain (per split per difficulty level)",
    )
    parser.add_argument(
        "--save_after",
        type=int,
        default=10,
        help="The number of problems added after which to save (and possibly push) the dataset",
    )
    # parser.add_argument(
    #     "--openrouter_api_key",
    #     type=str,
    #     default=OPENROUTER_API_KEY,
    #     help="The OpenRouter API key to use for generating responses",
    # )

    # Get the arguments
    cmd_args = parser.parse_args()

    # Create the config
    config = CodeValidationDatasetConfig(
        split=cmd_args.split,
        num_data=cmd_args.num_data,
        save_after=cmd_args.save_after,
        # openrouter_api_key=cmd_args.openrouter_api_key,
    )

    manager = multiprocessing.Manager()

    # Generate the dataset
    generate_cv_dataset(config, manager=manager)
