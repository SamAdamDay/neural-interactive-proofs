### The code below is modified from the APPS metric at https://huggingface.co/spaces/codeparrot/apps_metric

from enum import Enum
from unittest.mock import patch, mock_open
from io import StringIO
import platform
import sys
import signal
import faulthandler
import numpy as np
from typing import Optional, List
from datetime import datetime
import json
import multiprocessing

from nip.utils.runtime_module import RuntimeModule


class CODE_TYPE(Enum):
    call_based = 0
    standard_input = 1


# Define a custom exception for timeout
class TimeoutException(Exception):
    pass


# Define the signal handler
def handler(signum, frame):
    raise TimeoutException("Function call timed out")


# Register the signal handler
signal.signal(signal.SIGALRM, handler)


def wrapper(result, problem, solution, evaluate, debug):

    # warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*getargspec.*")
    try:
        results, outputs = run_test(
            problem, test=solution, evaluate=evaluate, debug=debug
        )
        result.append((results, outputs))
    except Exception:
        result.append((None, None))


def check_correctness(
    problem, solution, evaluate=None, timeout=30, debug=False, return_outputs=False
):
    """Check correctness of code solution with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside ``run_test``"""

    # warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*getargspec.*")

    # manager = multiprocessing.Manager()
    # result = manager.list()
    # p = multiprocessing.Process(target=wrapper, args=(result, problem, solution, evaluate, debug))
    # p.start()
    # p.join(timeout)

    # if p.is_alive():
    #     p.terminate()
    #     p.join()
    #     print(f"The function run_test timed out after {timeout} seconds.")
    #     results, outputs = None, None
    # else:
    #     results, outputs = result[0]

    # def _temp_run(problem, solution, debug, result):
    #     result.append(run_test(problem, test=solution, debug=debug))
    # manager = multiprocessing.Manager()
    # result = manager.list()
    # p = multiprocessing.Process(target=_temp_run, args=(problem, solution, debug, result))
    # p.start()
    # p.join(timeout=timeout + 1)
    # if p.is_alive():
    #     p.kill()

    ### PREVIOUS CODE BLOCK ###
    try:
        # start_time = datetime.now()
        signal.alarm(timeout)
        results, outputs = run_test(
            problem, test=solution, evaluate=evaluate, debug=debug
        )
        # elapsed_time = datetime.now() - start_time
        # elapsed_time = timedelta(days=elapsed_time.days, seconds=elapsed_time.seconds)
        # print(f"Correctness check took {elapsed_time}.")
        signal.alarm(0)
    except TimeoutException:
        print(f"The function run_test timed out after {timeout} seconds.")  # noqa: T201
        results, outputs = None, None

    # if results is None and evaluate is None:
    #     in_outs = json.loads(problem["input_output"])
    #     # consider that all tests failed
    #     results = [[None for _ in range(len(in_outs["inputs"]))]]
    #     if debug:
    #         print(f"global timeout")

    if return_outputs:
        return results, outputs
    else:
        return results


def run_test(
    sample, test=None, debug=False, evaluate: Optional[List[str]] = None, timeout=5
):
    """
    if test(generated_code) is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    if debug:
        print(f"start = {datetime.now().time()}")  # noqa: T201

    if evaluate is not None:
        in_outs = {"inputs": evaluate, "outputs": [None for _ in evaluate]}
    else:
        try:
            in_outs = json.loads(sample["input_output"])
        except ValueError:
            in_outs = None

    if in_outs:
        original_in_outs = json.loads(sample["input_output"])
        if original_in_outs.get("fn_name") is None:
            which_type = CODE_TYPE.standard_input  # Standard input
            method_name = None
        else:
            which_type = CODE_TYPE.call_based  # Call-based
            method_name = original_in_outs["fn_name"]

    if debug:
        print(f"loaded input_output = {datetime.now().time()}")  # noqa: T201

    if test is None:
        return in_outs, None
    elif test is not None:
        results = []
        outputs = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"
        if debug:
            print(f"loading test code = {datetime.now().time()}")  # noqa: T201

        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug:
                print(f"sol = {sol}")  # noqa: T201
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 0 compilation error = {e}")  # noqa: T201
                results.append(-2)
                return None, None
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("\t" + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test

            new_test = ""
            started = False
            for i in tmp_test:
                if i.startswith("\t") and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                    new_test += "\t" + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")  # noqa: T201
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")  # noqa: T201
                results.append(-2)
                outputs.append(None)
                return None, None
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")  # noqa: T201

        try:
            method = getattr(tmp, method_name)  # get_attr second arg must be str
        except Exception:
            signal.alarm(0)
            e = sys.exc_info()
            print(f"unable to get function error = {e}")  # noqa: T201
            results.append(-2)
            return None, None

        for index, inputs in enumerate(in_outs["inputs"]):
            # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index].items()}
                    ]
            except Exception:
                pass
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [
                        {int(k): v for k, v in in_outs["outputs"][index][0].items()}
                    ]
            except Exception:
                pass

            if debug:
                print(  # noqa: T201
                    f"time: {datetime.now().time()} testing index = {index}  inputs = {inputs}, {type(inputs)}. type = {which_type}"
                )
            if which_type == CODE_TYPE.call_based:  # Call-based
                signal.alarm(timeout)
                faulthandler.enable()
                try:
                    output = method(*inputs)

                    # ground truth sequences are not tuples
                    if isinstance(output, tuple):
                        output = list(output)

                    outputs.append(("\n").join(output))
                    if evaluate is not None:
                        results.append(None)
                        continue

                    tmp_result = output == in_outs["outputs"][index]
                    if (
                        isinstance(in_outs["outputs"][index], list)
                        and in_outs["outputs"][index]
                    ):
                        tmp_result = tmp_result or (
                            output == in_outs["outputs"][index][0]
                        )

                    # ground truth sequences are not tuples
                    try:
                        if isinstance(output[0], tuple):
                            tmp_result = tmp_result or (
                                [list(x) for x in output]
                                == in_outs["outputs"][index][0]
                            )
                    except Exception:
                        pass

                    results.append(tmp_result)

                    # reset the alarm
                    signal.alarm(0)
                except Exception as e:
                    signal.alarm(0)
                    faulthandler.disable()
                    if debug:
                        print(  # noqa: T201
                            f"Standard input runtime error or time limit exceeded error = {e}"
                        )
                    outputs.append(None)
                    results.append(None)

                    continue
                faulthandler.disable()
                signal.alarm(0)
                if debug:
                    print(  # noqa: T201
                        f"outputs = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                    )

            elif which_type == CODE_TYPE.standard_input:  # Standard input
                faulthandler.enable()
                signal.alarm(timeout)
                passed = False

                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs["outputs"][index], list):
                    in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

                with Capturing() as output:
                    try:
                        call_method(method, inputs)
                        # reset the alarm
                        signal.alarm(0)
                        passed = True
                    except Exception as e:
                        # runtime error or took too long
                        signal.alarm(0)
                        print(  # noqa: T201
                            f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}"
                        )
                        outputs.append(None)
                        results.append(None)
                    signal.alarm(0)

                if not passed:
                    if debug:
                        nl = "\n"
                        if not isinstance(inputs, list):
                            print(  # noqa: T201
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )
                        else:
                            print(  # noqa: T201
                                f"not passed output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                            )

                    continue

                # ground truth sequences are expressed as lists not tuples
                if isinstance(output, tuple):
                    output = list(output)

                outputs.append(("\n").join(output))
                if evaluate is not None:
                    results.append(None)
                    continue

                if passed and debug:
                    print(  # noqa: T201
                        f"==> output = {output}, test outputs = {in_outs['outputs'][index]}"
                    )

                if custom_compare_(output, in_outs["outputs"][index]):
                    tmp_result = True
                    results.append(tmp_result)
                    continue

                tmp_result = False
                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        if isinstance(output[0], str):
                            tmp_result = tmp_result or (
                                [e.strip() for e in output] == in_outs["outputs"][index]
                            )
                except Exception as e:
                    if debug:
                        print(f"Failed check1 exception = {e}")  # noqa: T201
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try one more time without \n
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = i.split("\n")
                        in_outs["outputs"][index][tmp_index] = [
                            x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                        ]
                else:
                    in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                    in_outs["outputs"][index] = list(
                        filter(len, in_outs["outputs"][index])
                    )
                    in_outs["outputs"][index] = list(
                        map(lambda x: x.strip(), in_outs["outputs"][index])
                    )

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check2 exception = {e}")  # noqa: T201
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    output = list(filter(len, output))

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(  # noqa: T201
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(  # noqa: T201
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                try:
                    tmp_result = output == [in_outs["outputs"][index]]
                    if isinstance(in_outs["outputs"][index], list):
                        tmp_result = tmp_result or (output == in_outs["outputs"][index])
                except Exception as e:
                    if debug:
                        print(f"Failed check3 exception = {e}")  # noqa: T201
                    pass

                try:
                    output_float = [float(e) for e in output]
                    gt_float = [float(e) for e in in_outs["outputs"][index]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float))
                        and np.allclose(output_float, gt_float)
                    )
                except Exception:
                    pass
                try:
                    if isinstance(output[0], list):
                        output_float = [float(e) for e in output[0]]
                        gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                        tmp_result = tmp_result or (
                            (len(output_float) == len(gt_float))
                            and np.allclose(output_float, gt_float)
                        )
                except Exception:
                    pass

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the stuff into split up list
                if isinstance(in_outs["outputs"][index], list):
                    for tmp_index, i in enumerate(in_outs["outputs"][index]):
                        in_outs["outputs"][index][tmp_index] = set(i.split())
                else:
                    in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                try:
                    tmp_result = output == in_outs["outputs"][index]
                except Exception as e:
                    if debug:
                        print(f"Failed check4 exception = {e}")  # noqa: T201
                    continue

                if tmp_result == True:
                    results.append(tmp_result)
                    continue

                # try by converting the output into a split up list too
                if isinstance(output, list):
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = i.split()
                    output = list(filter(len, output))
                    for tmp_index, i in enumerate(output):
                        output[tmp_index] = set(i)
                else:
                    output = output.split()
                    output = list(filter(len, output))
                    output = set(output)

                try:
                    tmp_result = set(frozenset(s) for s in output) == set(
                        frozenset(s) for s in in_outs["outputs"][index]
                    )
                except Exception as e:
                    if debug:
                        print(f"Failed check5 exception = {e}")  # noqa: T201

                # if they are all numbers, round so that similar numbers are treated as identical
                try:
                    tmp_result = tmp_result or (
                        set(frozenset(round(float(t), 3) for t in s) for s in output)
                        == set(
                            frozenset(round(float(t), 3) for t in s)
                            for s in in_outs["outputs"][index]
                        )
                    )
                except Exception as e:
                    if debug:
                        print(f"Failed check6 exception = {e}")  # noqa: T201

                if tmp_result == True and debug:
                    print("PASSED")  # noqa: T201

                results.append(tmp_result)

                if debug:
                    nl = "\n"
                    if not isinstance(inputs, list):
                        print(  # noqa: T201
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl,' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )
                    else:
                        print(  # noqa: T201
                            f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}"
                        )

    return results, outputs


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    # sys.setrecursionlimit(10000)

    # @patch('builtins.input', side_effect=inputs.split("\n"))
    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    # @patch('sys.stdout.write', print)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def custom_compare_(output, ground_truth):

    stripped_ground_truth = [
        g.lstrip().rstrip() for g in ground_truth.strip("\n").split("\n")
    ]
    stripped_output = [o.lstrip().rstrip() for o in output]

    return "\n".join(stripped_ground_truth) == "\n".join(stripped_output).strip("\n")
