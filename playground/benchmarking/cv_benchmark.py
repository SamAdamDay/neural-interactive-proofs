from datasets import load_dataset
from pvg.code_validation.dataset_generation import get_openrouter_response
import re
import numpy as np
import json
import argparse
from pvg.constants import OPENROUTER_API_KEY
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import tokenize
import io

MODELS = [
    "openai/gpt-3.5-turbo" "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "deepseek/deepseek-coder",
    "mistralai/codestral-mamba",
]
SAVE_PATH = "playground/benchmarking/results"


def strip_comments(source_code: str) -> str:

    # Remove inline comments
    source_code = re.sub(r'#.*', '', source_code)
    
    # Remove block comments
    source_code = re.sub(r'""".*?"""', '', source_code, flags=re.DOTALL)
    source_code = re.sub(r"'''.*?'''", '', source_code, flags=re.DOTALL)
    
    return source_code


def generate_query(question, solution, show_working):

    if show_working:
        show_working_header = "\n\nREASONING: <your reasoning here>"
        show_working_prompt = "You should write down your reasoning step by step before providing a final verdict. "
    else:
        show_working_header = ""
        show_working_prompt = "Do *not* include any reasoning or other text in your response apart from your verdict. "

    system_prompt = f"I need to check answers to some coding questions. In what follows, I will provide you with a QUESTION and an ATTEMPTED ANSWER. Your task is to determine whether the ATTEMPTED ANSWER is correct. Do *not* attempt to answer the QUESTION yourself. Note that while most ATTEMPTED ANSWERs *look* correct, many of them (half, in fact) are subtly incorrect, so make sure you think carefully. {show_working_prompt}If the ATTEMPTED ANSWER is correct, please respond with 'yes', if the ATTEMPTED ANSWER is incorrect, please respond with 'no'. The format of YOUR RESPONSE must be:{show_working_header}\n\VERDICT: <your verdict here>"

    python_preamble = "python\nstdin = sys.stdin\nstdout = sys.stdout\n"

    user_prompt = f"======== QUESTION ========\n\n\n{question}\n\n======== ATTEMPTED ANSWER ========\n\n\n```\n{python_preamble}\n{strip_comments(solution).strip()}\n```\n\n======== YOUR RESPONSE ========\n\n\n"

    query = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return query


def extract_answer(response):

    pattern = r"VERDICT:\s*(.*)"
    match = re.search(pattern, response)
    if match:
        answer = match.group(1).strip().lower()
        if re.search(r"\byes\b", answer) and re.search(r"\bno\b", answer):
            return None
        elif re.search(r"\byes\b", answer):
            return "yes"
        elif re.search(r"\bno\b", answer):
            return "no"
        else:
            return None
    else:
        return None


def guess_if_correct(
    model,
    api_key,
    query,
    temperature,
    n_attempts,
    check_probs,
    num_alternatives,
    show_working,
    debug=False,
):

    # When the model only outputs an answer and yet we need multiple attempts, then we re-query the model instead of gathering mutlipe completions with a single API call (otherwise there will be ~no variability in the log probs of the final verdict)
    force_multiple_generations = not show_working and n_attempts > 1
    # force_multiple_generations = True

    if debug:
        start_time = datetime.now()

    responses = get_openrouter_response(
        model,
        query,
        api_key,
        num_responses=n_attempts,
        log_probs=check_probs,
        top_logprobs=num_alternatives,
        temperature=temperature,
        force_multiple_generations=force_multiple_generations,
    )

    if debug:
        # Calculate the elapsed time, rounding microseconds down
        elapsed_time = datetime.now() - start_time
        elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)
        print(f"Received model output in {elapsed_time}.")

    answers = []
    answer_probs = []
    alt_probs = []

    for r in responses:

        answer = extract_answer(r["message"])

        if answer is None:

            answers.append(None)
            answer_probs.append(None)
            alt_probs.append(None)

        else:

            answers.append(answer)
            answer_prob = None
            alt_prob = None

            if check_probs and r["log_probs"] is not None and r["log_probs"] != []:
                for t in reversed(r["log_probs"]):
                    if "yes" in t["token"].lower() or "no" in t["token"].lower():
                        if answer not in t["token"].lower():
                            raise ValueError(
                                "The extracted answer does not match the answer from the latest token."
                            )
                        answer_prob = np.exp(t["logprob"])
                        # answer_index = r["log_probs"].index(t)
                        break

            if (
                num_alternatives is not None
                and r["log_probs"] is not None
                and r["log_probs"] != []
            ):
                top_logprobs = (
                    r["top_logprobs"][r["log_probs"].index(t)]
                    if "top_logprobs" in r
                    else t["top_logprobs"]
                )
                alt = "no" if answer == "yes" else "yes"
                answer_prob = sum(
                    [
                        np.exp(tt["logprob"])
                        for tt in top_logprobs
                        if answer in tt["token"].lower()
                    ]
                )
                alt_prob = sum(
                    [
                        np.exp(tt["logprob"])
                        for tt in top_logprobs
                        if alt in tt["token"].lower()
                    ]
                )

            answer_probs.append(answer_prob)
            alt_probs.append(alt_prob)

    return answers, answer_probs, alt_probs


def alt_answer(answer):

    if answer == "yes":
        return "no"
    elif answer == "no":
        return "yes"
    else:
        raise ValueError("The answer must be either 'yes' or 'no'.")


def evaluate_model(
    model,
    api_key=OPENROUTER_API_KEY,
    show_working=False,
    temperature=1.0,
    n_attempts=10,
    check_probs=True,
    num_alternatives=10,
    save_path=None,
    max_problems=None,
    verbose=False,
    save_after=10,
):

    data = load_dataset("lrhammond/buggy-apps", split="train")

    if save_path is None:
        results = {
            s: {d: {} for d in ["introductory", "interview", "competition"]}
            for s in ["train", "test"]
        }
    else:
        model_name = model.split("/")[-1]
        reasoning = "with_reasoning" if show_working else "without_reasoning"
        filename = f"{save_path}/{model_name}-{reasoning}-{temperature}-{n_attempts}_attempts-{num_alternatives}_alternatives.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results = json.load(f)
        else:
            results = {
                s: {d: {} for d in ["introductory", "interview", "competition"]}
                for s in ["train", "test"]
            }

    counter = {
                s: {d: 0 for d in ["introductory", "interview", "competition"]}
                for s in ["train", "test"]
            }

    num_added = 0
    max_problems = len(data) if max_problems is None else max_problems

    for s in ["train", "test"]:
        for d in ["introductory", "interview", "competition"]:
            data_slice = data.filter(lambda x: x["apps_split"] == s and x["difficulty"] == d)
            max_problems = len(data) if max_problems is None else max_problems

            print(f"\nModel: {model_name}\nReasoning: {show_working}\nSplit: {s}\nDifficulty: {d}")

            # for datum in tqdm(data_slice[:max_problems], colour="magenta"):
            for i in tqdm(range(max_problems), colour="magenta"):

                datum = data_slice[i]

                problem_id = str(datum["apps_problem_id"])

                if counter[s][d] >= max_problems:
                    continue
                elif problem_id in results[s][d]:
                    counter[s][d] += 1
                    continue

                results[s][d][problem_id] = {}
                num_added += 1
                counter[s][d] += 1

                if verbose:
                    print("\n\n")
                    print(f"Problem: {problem_id} ({datum['apps_split']})")

                for actual_answer, sols in zip(
                    ["yes", "no"], [datum["solutions"], datum["buggy_solutions"]]
                ):

                    if verbose:
                        print("====================================")
                        print(f"Actual answer: {actual_answer}\n")

                    for sol in sols:

                        query = generate_query(datum["question"], sol["solution"], show_working)
                        answers, answer_probs, alt_probs = guess_if_correct(
                            model,
                            api_key,
                            query,
                            temperature,
                            n_attempts,
                            check_probs,
                            num_alternatives,
                            show_working,
                        )

                        # correct = []
                        if check_probs:
                            correct_probs = []
                            incorrect_probs = []
                        if num_alternatives is not None:
                            correct_alt_probs = []
                            incorrect_alt_probs = []

                        for i in range(len(answers)):
                            if answers[i] is None:
                                continue
                            elif answers[i] == actual_answer:
                                # correct.append('correct')
                                if check_probs and answer_probs[i] is not None:
                                    correct_probs.append(answer_probs[i])
                                if num_alternatives is not None and alt_probs[i] is not None:
                                    correct_alt_probs.append(alt_probs[i])
                            else:
                                # correct.append(0)
                                if check_probs and answer_probs[i] is not None:
                                    incorrect_probs.append(answer_probs[i])
                                if num_alternatives is not None and alt_probs[i] is not None:
                                    incorrect_alt_probs.append(alt_probs[i])

                        correct = answers.count(actual_answer)
                        incorrect = answers.count(alt_answer(actual_answer))
                        num_answers = len(answers)
                        stats = {
                            "fraction_correct": correct / num_answers,
                            "fraction_incorrect": incorrect / num_answers,
                            "fraction_failed": (num_answers - correct - incorrect)
                            / num_answers,
                        }
                        if check_probs:
                            stats["avg_prob_when_correct"] = (
                                np.mean(correct_probs) if len(correct_probs) > 0 else None
                            )
                            stats["avg_prob_when_incorrect"] = (
                                np.mean(incorrect_probs) if len(incorrect_probs) > 0 else None
                            )
                        if num_alternatives is not None:
                            stats["avg_alt_prob_when_correct"] = (
                                np.mean(correct_alt_probs)
                                if len(correct_alt_probs) > 0
                                else None
                            )
                            stats["avg_alt_prob_when_incorrect"] = (
                                np.mean(incorrect_alt_probs)
                                if len(incorrect_alt_probs) > 0
                                else None
                            )

                        results[s][d][problem_id][
                            actual_answer
                        ] = stats

                        if verbose:
                            print("----------------------------")
                            print(f"Solution number: {sols.index(sol)}")
                            print(f"Fraction correct: {stats['fraction_correct']}")
                            print(f"Fraction incorrect: {stats['fraction_incorrect']}")
                            print(f"Fraction failed: {stats['fraction_failed']}")
                            if check_probs:
                                print(
                                    f"Avg prob when correct: {stats['avg_prob_when_correct']}"
                                )
                                print(
                                    f"Avg prob when incorrect: {stats['avg_prob_when_incorrect']}"
                                )
                            if num_alternatives is not None:
                                print(
                                    f"Avg alt prob when correct: {stats['avg_alt_prob_when_correct']}"
                                )
                                print(
                                    f"Avg alt prob when incorrect: {stats['avg_alt_prob_when_incorrect']}"
                                )

                if save_path is not None and num_added % save_after == 0:
                    with open(filename, "w") as f:
                        json.dump(results, f)

    if save_path is not None:
        with open(filename, "w") as f:
            json.dump(results, f)

    return results, num_added


def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run code validation benchmark.")

    # Required arguments
    parser.add_argument("model", type=str, help="Model to evaluate")

    # Optional arguments
    parser.add_argument(
        "--max_problems",
        type=int,
        default=10,
        help="Max number of problems to test",
    )
    parser.add_argument(
        "--save_after",
        type=int,
        default=10,
        help="Save results after this many problems",
    )
    parser.add_argument(
        "--save_path", type=str, default=SAVE_PATH, help="Path to save the results to"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="The temperature of the model"
    )
    parser.add_argument(
        "--n_attempts",
        type=int,
        default=5,
        help="Number of attempts that the model should make",
    )
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=OPENROUTER_API_KEY,
        help="API key for OpenRouter",
    )
    parser.add_argument(
        "--num_alternatives",
        type=int,
        default=10,
        help="Number of alternative tokens to consider at each completion point",
    )

    # Boolean flags
    parser.add_argument(
        "--check_probs",
        action="store_true",
        help="Whether to check the output probabilities of the model",
    )
    parser.add_argument(
        "--show_working",
        action="store_true",
        help="Whether the model should attempt to work out the answer step by step before providing a final response",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print the results to the console",
    )

    # Parse arguments
    cmd_args = parser.parse_args()

    start_time = datetime.now()

    _, num_added = evaluate_model(
        cmd_args.model,
        api_key=cmd_args.openrouter_api_key,
        show_working=cmd_args.show_working,
        num_alternatives=cmd_args.num_alternatives,
        n_attempts=cmd_args.n_attempts,
        temperature=cmd_args.temperature,
        save_path=cmd_args.save_path,
        check_probs=cmd_args.check_probs,
        max_problems=cmd_args.max_problems,
        save_after=cmd_args.save_after,
        verbose=cmd_args.verbose,
    )

    # Calculate the elapsed time, rounding microseconds down
    elapsed_time = datetime.now() - start_time
    elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)
    print(f"Evaluated {cmd_args.model} on {num_added} (new) problems in {elapsed_time}.")

if __name__ == "__main__":
    main()
