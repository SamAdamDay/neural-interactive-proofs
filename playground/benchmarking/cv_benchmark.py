from datasets import load_dataset
from pvg.code_validation.agents import get_openrouter_response
import re
import numpy as np
import json
import argparse
from pvg.constants import OPENROUTER_API_KEY
from datetime import datetime, timedelta
import os

MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "deepseek/deepseek-coder",
    "mistralai/codestral-mamba",
]
SAVE_PATH = "/workspaces/nip/playground/benchmarking/results"


def generate_query(question, solution, show_working):

    if show_working:
        show_working_header = "\n\nREASONING: <your reasoning here>"
        show_working_prompt = "You should write down your reasoning step by step before providing a final response. "
    else:
        show_working_header = ""
        show_working_prompt = "Do *not* include any reasoning or other text in your response apart from your verdict. "

    system_prompt = f"I need to check answers to some coding challenges. In what follows, I will provide you with a QUESTION and an ATTEMPED ANSWER. Your task is to determine whether the ATTEMPED ANSWER is correct. Do *not* attempt to answer the QUESTION yourself. Note that while most ATTEMPED ANSWERs *look* correct, several of them are subtly inccorrect, so make sure you think carefully. {show_working_prompt}If the ATTEMPED ANSWER is correct, please respond with 'yes', if the ATTEMPED ANSWER is incorrect, please respond with 'no'. The format of YOUR VERDICT must be:{show_working_header}\n\nCORRECT: <your verdict here>"

    python_preamble = "python\nstdin = sys.stdin\nstdout = sys.stdout\n"

    user_prompt = f"======== QUESTION ========\n\n\n{question}\n\n======== ATTEMPED ANSWER ========\n\n\n```\n{python_preamble}\n{solution.strip()}\n```\n\n======== YOUR RESPONSE ========\n\n\n"

    query = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return query


def extract_answer(response):

    pattern = r"CORRECT:\s*(.*)"
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
):

    # When the model only outputs an answer and yet we need multiple attempts, then we re-query the model instead of gathering mutlipe completions with a single API call (otherwise there will be ~no variability in the log probs)
    force_multiple_generations = not show_working and n_attempts > 1

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

            if check_probs:
                for t in reversed(r["log_probs"]):
                    if "yes" in t["token"].lower() or "no" in t["token"].lower():
                        if answer not in t["token"].lower():
                            raise ValueError(
                                "The extracted answer does not match the answer from the latest token."
                            )
                        answer_prob = np.exp(t["logprob"])
                        # answer_index = r["log_probs"].index(t)
                        break

            if num_alternatives is not None:
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


def evaluate_model(
    model,
    api_key=OPENROUTER_API_KEY,
    show_working=False,
    temperature=1.0,
    n_attempts=10,
    check_probs=True,
    num_alternatives=None,
    save_path=None,
    max_problems=None,
    max_solutions=10,
    verbose=False,
    save_after=10,
):

    data = load_dataset("codeparrot/apps", trust_remote_code=True)
    buggy_data = load_dataset("lrhammond/buggy-apps")

    if save_path is None:
        results = {split: {} for split in ["train", "test"]}
    else:
        model_name = model.split("/")[-1]
        reasoning = "with_reasoning" if show_working else "without_reasoning"
        filename = f"{save_path}/{model_name}-{reasoning}-{temperature}-{n_attempts}_attempts-{num_alternatives}_alternatives.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                results = json.load(f)
        else:
            results = {split: {} for split in ["train", "test"]}

    for split in ["train", "test"]:

        indices = buggy_data[split]["problem_id"]
        sliced_data = data[split].filter(lambda x: x["problem_id"] in indices)

        problems_tested = 0
        num_added = 0

        for buggy_datum, datum in zip(buggy_data[split], sliced_data):

            if buggy_datum["problem_id"] != datum["problem_id"]:
                raise ValueError("The data is not aligned.")
            elif problems_tested >= max_problems:
                break
            elif str(datum["problem_id"]) in results[split]:
                problems_tested += 1
                continue

            question = datum["question"]
            solutions = json.loads(datum["solutions"])
            buggy_solutions = json.loads(buggy_datum["solutions"])

            results[split][str(datum["problem_id"])] = {}
            num_added += 1
            problems_tested += 1

            if verbose:
                print("\n\n")
                print(f"Split: {split}")
                print(f"Problem ID: {buggy_datum['problem_id']}")

            for actual_answer, sols in zip(["yes", "no"], [solutions, buggy_solutions]):

                if verbose:
                    print("====================================")
                    print(f"Actual answer: {actual_answer}\n")

                for sol in sols:

                    # Ignore solutions where we don't have a buggy and non-buggy pair
                    if buggy_solutions[sols.index(sol)] == None:
                        continue
                    # elif solutions_tested >= max_solutions:
                    #     break
                    query = generate_query(question, sol, show_working)
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

                    correct = []
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
                            correct.append(1)
                            if check_probs and answer_probs[i] is not None:
                                correct_probs.append(answer_probs[i])
                            if (
                                num_alternatives is not None
                                and alt_probs[i] is not None
                            ):
                                correct_alt_probs.append(alt_probs[i])
                        else:
                            correct.append(0)
                            if check_probs:
                                incorrect_probs.append(answer_probs[i])
                            if (
                                num_alternatives is not None
                                and alt_probs[i] is not None
                            ):
                                incorrect_alt_probs.append(alt_probs[i])

                    stats = {
                        "fraction_correct": (
                            np.mean(correct) if len(correct) > 0 else None
                        ),
                        "fraction_failed": 1 - (len(correct) / len(answers)),
                    }
                    if check_probs:
                        stats["avg_prob_when_correct"] = (
                            np.mean(correct_probs) if len(correct_probs) > 0 else None
                        )
                        stats["avg_prob_when_incorrect"] = (
                            np.mean(incorrect_probs)
                            if len(incorrect_probs) > 0
                            else None
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

                    results[split][str(datum["problem_id"])][actual_answer] = stats

                    if verbose:
                        print("----------------------------")
                        print(f"Solution number: {sols.index(sol)}")
                        print(f"Fraction correct: {stats['fraction_correct']}")
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

            if save_path is not None and problems_tested % save_after == 0:
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
        default=None,
        help="Max number of problems to test per split",
    )
    # parser.add_argument('--max_solutions', type=int, default=10, help='Max number of solutions to test per problem')
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
        default=None,
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
        #    max_solutions=cmd_args.max_solutions,
        save_after=cmd_args.save_after,
        verbose=cmd_args.verbose,
    )

    # Calculate the elapsed time, rounding microseconds down
    elapsed_time = datetime.now() - start_time
    elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)
    print(f"Evaluated {cmd_args.model} on {num_added} problems in {elapsed_time}.")


if __name__ == "__main__":
    main()
