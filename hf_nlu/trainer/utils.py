import os
import json


def batch(iterable: iter, n: int = 1) -> iter:
    '''
    Batch an iterable into n elements.

    Inputs:
        - iterable: iterable list (iter).
        - n: batch size (int).

    Output:
        - list of element from iterable of length n.

    '''
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def rev_dict(dictionary: dict) -> dict:
    """Revert dictionary as value/key (assuming it is bijective)."""
    return {v: k for k, v in dictionary.items()}


def _check_and_make(path: str):
    """Check if file exist in path and create it if not."""
    if not os.path.exists(path):
        os.mkdir(path)


def save_evaluation(eval_dict: dict, output_dir: str, eval_folder: str = "results", filename: str = "eval.json"):
    """Save evaluation dictionary into folder as json file."""
    model_output_name = os.path.join(output_dir, eval_folder)
    _check_and_make(model_output_name)

    with open(os.path.join(model_output_name, filename), "w") as fp:
        json.dump(eval_dict, fp)
