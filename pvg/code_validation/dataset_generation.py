from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from agents import get_response
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
import re

from apps_metric import utils

# import

# Eventually add to pvg.constants
from pvg.constants import DATA_DIR

CV_DATA_DIR = os.path.join(DATA_DIR, "code_validation")


@dataclass
class CodeValidationDatasetConfig:
    """
    Configuration class for code validation dataset generation.
    Attributes:
        model (str): The LLM to be used for generating the dataset.
        difficulties (list[str]): The difficulties of the generated dataset.
        split (str | None): The split of the dataset.
        fraction_to_modify (float): The fraction of solutions to be modified.
        remove_original (bool): Whether to remove the original solutions.
        system_prompt (Optional[str]): The system prompt for the LLM.
    """

    model: str = "openai/gpt-4o-mini"
    difficulties: List[str] = field(default_factory=lambda: ["interview"])
    split: str | None = None
    fraction_to_modify: float = 0.5
    remove_original: bool = True
    # system_prompt: Optional[str] = "I am developing a new code validation system. In order test this system, I need to provide it with a set of correct and incorrect solutions for a given coding problem. I already have the correct solutions, and I need your help to generate some incorrect solutions by modifying the correct solutions. In what follows, I will provide you with a QUESTION and a SOLUTION. Your task is to subtly modify this solution by introducing an error that causes the code to fail on a small fraction of inputs. To test the system thoroughly, the error in this MODIFIED SOLUTION should be as difficult to spot as possible (while making sure the code remains similar to the original solution) and should not be a simple syntax error. Please only modify the solution given to you and preserve all other aspects of the solution (such as the formatting and variable names). Do not include any natural language explanation of your modifications or any comments in the code. In other words, please only include the modified code itself in your response."
    system_prompt: Optional[str] = (
        "I am developing a new code validation system. In order test this system, I need to provide it with a set of correct and incorrect solutions for a given coding problem. I already have the correct solutions, and I need your help to generate some incorrect solutions by modifying the correct solutions. In what follows, I will provide you with a QUESTION and a SOLUTION. Your task is to subtly modify this solution by introducing an error that causes the code to fail on a small fraction of inputs, and to provide a PROBLEMATIC INPUT on which the original solution works, but your MODIFIED SOLUTION fails. To test the system thoroughly, the error in this MODIFIED SOLUTION should be as difficult to spot as possible (while making sure the code remains similar to the original solution) and should not be a simple syntax error. Please only modify the solution given to you and preserve all other aspects of the solution (such as the formatting and variable names). Do not include any natural language explanation of your modifications or any comments in the code. The format of your answer should be:\
    \n\
    \n\
    MODIFIED SOLUTION:\n\
    ```<your modified code here>```\n\n\
    PROBLEMATIC INPUT: <your problematic input here>"
    )


def extract_code_and_input(model_output: str):

    # Extract the problematic input using regex
    match = re.search(r"PROBLEMATIC INPUT:\s*(.*)", model_output, re.DOTALL)
    if match:
        problematic_input = match.group(1).replace("```", "").strip()
    else:
        problematic_input = None

    # Remove the "PROBLEMATIC INPUT: ..." text from the original string
    model_output = re.sub(
        r"PROBLEMATIC INPUT:\s*.*", "", model_output, flags=re.DOTALL
    ).strip()

    # Extract the modified solution code
    match = re.search(r"MODIFIED SOLUTION:\s*(.*)", model_output, re.DOTALL)
    if match:
        modified_solution = (
            match.group(1).replace("python", "").replace("```", "").strip()
        )
    else:
        modified_solution = None

    # Check for dangerous functions
    dangerous_functions = [
        "builtins.exit",
        "builtins.quit",
        "os.kill",
        "os.system",
        "os.putenv",
        "os.remove",
        "os.removedirs",
        "os.rmdir",
        "os.fchdir",
        "os.setuid",
        "os.fork",
        "os.forkpty",
        "os.killpg",
        "os.rename",
        "os.renames",
        "os.truncate",
        "os.replace",
        "os.unlink",
        "os.fchmod",
        "os.fchown",
        "os.chmod",
        "os.chown",
        "os.chroot",
        "os.lchflags",
        "os.lchmod",
        "os.lchown",
        "os.getcwd",
        "os.chdir",
        "shutil.rmtree",
        "shutil.move",
        "shutil.chown",
        "subprocess.Popen",
        '__builtins__["help"]',
        'sys.modules["ipdb"]',
        'sys.modules["joblib"]',
        'sys.modules["resource"]',
        'sys.modules["psutil"]',
        'sys.modules["tkinter"]',
    ]
    safe = not any(func in modified_solution for func in dangerous_functions)

    return modified_solution, problematic_input, safe


def test_anti_solution(
    anti_solution: str,
    solution: str,
    problematic_input: str,
    datum: dict,
    max_correctness: float = 1.0,
) -> bool:

    # Generate a new output based on the problematic input and test the anti-solution
    output = (
        None
        if problematic_input is None
        else utils.check_correctness(
            datum, solution, evaluate=problematic_input, timeout=10, debug=True
        )
    )
    if output is not None:
        anti_output = utils.check_correctness(
            datum, anti_solution, evaluate=problematic_input, timeout=10, debug=True
        )
    else:
        anti_output = output

    # Test the anti-solution against the correctness checks
    results = utils.check_correctness(datum, anti_solution, timeout=10, debug=True)
    fraction_correct = float(sum(results) / len(results))

    return (
        "\n".join(output) != "\n".join(anti_output)
        or fraction_correct < max_correctness
    )


def generate_anti_solutions(
    datum: dict,
    model: str,
    system_prompt: str,
    remove_original: bool,
    fraction_to_modify: float,
) -> str:
    """
    Generates anti-solutions based on the given datum.
    Args:
        datum: The input datum containing the question and solutions.
        model (str): The LLM to use for generating incorrect ("anti-") solutions.
        system_prompt (str): The system prompt to use during generation.
        remove_original (bool): Whether to remove the original solutions from the list.
        fraction_to_modify (float): The fraction of solutions to modify.
    Returns:
        str: The generated anti-solutions.
    Raises:
        None
    """

    solutions = json.loads(datum["solutions"])
    anti_solutions = []

    for solution in solutions[: int(fraction_to_modify * len(solutions))]:

        user_prompt = (
            "QUESTION:\n\n\n" + datum["question"] + "\n\nSOLUTION:\n\n\n" + solution
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        valid_anti_solution = False
        attempts = 0

        while not valid_anti_solution and attempts < 10:
            attempts += 1
            model_output = get_response(model, messages)
            anti_solution, problematic_input, safe = extract_code_and_input(
                model_output
            )
            if not safe:
                continue
            valid_anti_solution = test_anti_solution(
                anti_solution, solution, problematic_input, datum
            )

        if attempts == 10:
            print("Failed to generate anti-solution")
            continue
        else:
            anti_solutions.append(anti_solution)
            if remove_original:
                solutions.remove(solution)

    datum["anti_solutions"] = json.dumps(anti_solutions)

    return datum["anti_solutions"]


def generate_cv_dataset(config: CodeValidationDatasetConfig | dict):

    start_time = datetime.now()

    if isinstance(config, dict):
        config = CodeValidationDatasetConfig(**config)

    data = load_dataset(
        "codeparrot/apps",
        trust_remote_code=True,
        split=config.split,
        difficulties=config.difficulties,
    )
    # metric = load('vendor/apps_metric')

    split = ["train", "test"] if config.split is None else [config.split]

    for s in split:
        # for datum in tqdm(data[s]):
        for datum in data[s]:
            generate_anti_solutions(
                datum,
                config.model,
                config.system_prompt,
                config.remove_original,
                config.fraction_to_modify,
            )

    Path(CV_DATA_DIR).mkdir(parents=True, exist_ok=True)
    removed_or_kept = "removed" if config.remove_original else "kept"
    dataset_filename = os.path.join(
        CV_DATA_DIR,
        "-".join(sorted(config.difficulties)),
        f"{config.fraction_to_modify}-modifed-originals-{removed_or_kept}.pt",
    )
    data.save_to_disk(dataset_filename)

    # Calculate the elapsed time, rounding microseconds down
    elapsed_time = datetime.now() - start_time
    elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)

    print(f"Done in {elapsed_time}")


generate_cv_dataset(CodeValidationDatasetConfig())
