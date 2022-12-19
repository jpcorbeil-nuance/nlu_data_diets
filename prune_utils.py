import os
import re

import numpy as np
import pandas as pd


class PruneConfig:
    """
    Configuration for pruning.
    """
    def __init__(self, mode: str, size: float, epoch: float,
                offset: float, avg_mode: str, avg_window_size: int,
                frequency: int, ema_alpha: float, ema_var_coef: float,
                ema_use_std: float):
        self.mode = mode
        self.epoch = epoch
        self.size = size
        self.offset = offset
        self.frequency = frequency
        self.avg_mode = avg_mode
        self.avg_window_size = avg_window_size
        self.ema_alpha = ema_alpha
        self.ema_var_coef = ema_var_coef
        self.ema_use_std = ema_use_std


def list_score_files(output_dir: str, prefix: str) -> iter:
    all_files = os.listdir(output_dir)
    score_files = filter(lambda f: f.startswith(prefix), all_files)
    return score_files


def extract_epochs(files: iter, prefix: str) -> iter:
    pattern = re.compile(rf"{prefix}_(\d+.\d+).tsv")
    return map(lambda s: float(pattern.findall(s)[0]), files)


def get_epochs_from_score_files(output_dir: str, prune_mode: str) -> iter:
    """
    Get all epochs from score filenames.
    """
    score_files = list_score_files(output_dir, prune_mode)
    epochs = extract_epochs(score_files, prune_mode)
    epochs = sorted(epochs)
    return epochs


def fetch_latest_score_file(output_dir: str, prune_mode: str) -> pd.DataFrame:
    """
    Get only last score file in pandas' dataframe.
    """
    epochs = get_epochs_from_score_files(output_dir, prune_mode)
    return pd.read_csv(os.path.join(output_dir, f"{prune_mode}_{epochs[-1]}.tsv"), sep="\t")


def load_epoch_score_files(output_dir: str, prune_mode: str, epochs: list) -> pd.DataFrame:
    """
    Load score files in pandas' dataframe.
    """
    df_lst = []
    for epoch in epochs:
        df = pd.read_csv(os.path.join(output_dir, f"{prune_mode}_{epoch}.tsv"), sep="\t")
        df["Epoch"] = epoch
        df_lst.append(df)
    return pd.concat(df_lst)


def fetch_last_k_average(output_dir: str, prune_mode: str, k: int = 1) -> pd.DataFrame:
    """
    Get scores and average uniformly on last k reports.
    """
    epochs = get_epochs_from_score_files(output_dir, prune_mode)
    df = load_epoch_score_files(output_dir, prune_mode, epochs[-k:])
    mean_df = df.groupby("Id")["Score"].mean()
    return mean_df.reset_index()


def fetch_ema(output_dir: str, prune_mode: str, alpha: float = 0.8, c: float = 1.0, use_std: bool = True) -> pd.DataFrame:
    """
    Get scores and apply EMA with added variance.
    """
    epochs = get_epochs_from_score_files(output_dir, prune_mode)
    df = load_epoch_score_files(output_dir, prune_mode, epochs)
    pdf = pd.pivot_table(df, values="Score", index="Epoch", columns="Id")
    mean = pdf.ewm(alpha=alpha, adjust=False).mean()
    s = mean
    if c > 0.0:
        emw = pdf.ewm(alpha=alpha, adjust=False)
        if use_std:
            deviation = emw.std()
        else:
            deviation = emw.var()
        s += (c * deviation)
    s = s.loc[epochs[-1], :]
    s = s.reset_index().rename({epochs[-1]: "Score"}, axis=1)
    return s



class PruneScoreManager:
    """
    Handle pruning management of scores.
    """
    def __init__(self, output_dir: str, config: PruneConfig):
        self.output_dir = output_dir
        self.config = config

    def get_scores_from_files(self):
        output_dir = self.output_dir
        prune_mode = self.config.mode
        prune_avg_mode = self.config.avg_mode
        prune_window_size = self.config.avg_window_size

        if prune_avg_mode == "ema":
            scores = fetch_ema(output_dir, prune_mode, alpha=self.config.ema_alpha,
                                c=self.config.ema_var_coef, use_std=self.config.ema_use_std)
        elif prune_avg_mode == "avg":
            scores = fetch_last_k_average(output_dir, prune_mode, prune_window_size)
        else:
            scores = fetch_latest_score_file(output_dir, prune_mode)

        return scores

    def generate_random(self, sample_ids: list):
        return {i: np.random.rand() for i in sample_ids}

    def get_scores_and_bounds(self, sample_ids: list = None):
        prune_mode = self.config.mode
        prune_size = self.config.size
        prune_offset = self.config.offset

        # LOAD SCORES
        if prune_mode in ["el2n", "loss", "grand"]:
            scores = self.get_scores_from_files()
            norm_top_percentile = float(scores["Score"].quantile(1.0 - prune_offset))
            norm_down_percentile = float(scores["Score"].quantile(1.0 - prune_offset - prune_size))
            score_dict = {i: n for i, n in scores[["Id", "Score"]].values}
        elif prune_mode in ["random"] and sample_ids is not None:
            score_dict = self.generate_random(sample_ids)
            norm_top_percentile = 1.0
            norm_down_percentile = 1.0 - prune_size

        return score_dict, (norm_down_percentile, norm_top_percentile)
