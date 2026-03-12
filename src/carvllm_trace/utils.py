#   Copyright 2025 - 2026 Christos Kozanitis, FORTH, Greece
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pandas as pd

def read_events_csv(
    path: str,
    start_col: str,
    end_col: str,
    method_col: str,
    delimiter: str
) -> pd.DataFrame:
    """
    Read a pipe-delimited CSV and normalize it to columns:
    ["start", "end", "type"]
    """
    df = pd.read_csv(path, sep=delimiter)
    result_df = (
        df
        .rename(columns={start_col: "start", end_col: "end", method_col: "type"})
        .assign(
            type=lambda d: d["type"].astype(str).str.startswith(("NCCL", "nccl")).map({True: "nccl", False: "no_nccl"})
        )
        [["start", "end", "type"]]
        .copy() #evaluate if needed
    )
    return result_df



REQUIRED_COLUMNS = {"start", "end", "type"}


def find_type_overlaps(
    df: pd.DataFrame,
    type_a: str = "nccl",
    type_b: str = "no_nccl",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with many columns.
    start_col : str
        Column name for interval start.
    end_col : str
        Column name for interval end.
    type_a, type_b : str
        Types to compare (defaults: "nccl" and "no_nccl").
    Returns
    -------
    pd.DataFrame with columns:
        ["a_start", "a_end", "b_start", "b_end"]
    """
    # ---- 2) Merge same-type overlapping intervals ----
    merged = _merge_same_type(df)
    #print('merged: ', merged)

    # ---- 3) Split by types ----
    df_a = merged[merged["type"] == type_a][["start", "end"]].reset_index(drop=True)
    df_b = merged[merged["type"] == type_b][["start", "end"]].reset_index(drop=True)
    #print('df_a: ', df_a)
    #print('df_b: ', df_b)

    if df_a.empty or df_b.empty:
        return _empty_overlap_df()

    return _find_overlaps_interval_index(df_a, df_b)


def _empty_overlap_df():
    return pd.DataFrame(columns=['a_start', 'a_end', 'b_start', 'b_end'])


def _find_overlaps_interval_index(df_a, df_b):
    idx = pd.IntervalIndex.from_arrays(df_a["start"], df_a["end"], closed="left")
    hits = []
    for _, row in df_b.iterrows():
        overlapping = df_a[idx.overlaps(pd.Interval(row["start"], row["end"], closed="left"))]
        for _, arow in overlapping.iterrows():
            hits.append((arow["start"], arow["end"], row["start"], row["end"]))

    if not hits:
        return _empty_overlap_df()

    ret =  pd.DataFrame(hits, columns=["a_start", "a_end", "b_start", "b_end"])
    #print ('Returning ', ret)
    return ret



def _merge_same_type(df):
    df = df.sort_values(["type", "start"])

    group = (df["start"] > df["end"].shift()).cumsum()

    return (
        df.assign(_g=group)
          .groupby(["type", "_g"], as_index=False)
          .agg(start=("start", "min"), end=("end", "max"))
          .drop(columns="_g")
    )

#simple but inefficient
def _find_overlaps_cross_product(df_a, df_b):
    pairs = df_a.merge(df_b, how="cross", suffixes=("_a", "_b"))

    overlaps = pairs[
        (pairs.start_a < pairs.end_b) &
        (pairs.start_b < pairs.end_a)
    ]

    return overlaps.rename(
        columns={
            "start_a": "a_start",
            "end_a": "a_end",
            "start_b": "b_start",
            "end_b": "b_end",
        }
    )[["a_start", "a_end", "b_start", "b_end"]]


def measure_percentage_overlapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: ["a_start", "a_end", "b_start", "b_end"]

    Returns
    -------
    pd.DataFrame with columns:
        ["a_start", "a_end", "overlapping"]
    One row per unique A interval. "overlapping" is the fraction of the A interval
    covered by the union of overlapping B intervals.
    """

    if df.empty:
        return pd.DataFrame(columns=["a_start", "a_end", "overlapping"])

    # Ensure B intervals are clipped to A intervals
    df = df.copy()
    df["b_start_clipped"] = df[["a_start", "b_start"]].max(axis=1)
    df["b_end_clipped"] = df[["a_end", "b_end"]].min(axis=1)
    df = df[df["b_start_clipped"] < df["b_end_clipped"]]

    if df.empty:
        a_intervals = df[["a_start", "a_end"]].drop_duplicates()
        a_intervals["overlapping"] = 0.0
        return a_intervals

    # Sort by A interval and B start
    df = df.sort_values(["a_start", "a_end", "b_start_clipped"]).reset_index(drop=True)

    # Group by each unique A interval
    def compute_fraction(group):
        # Merge overlapping B intervals
        b_intervals = group[["b_start_clipped", "b_end_clipped"]].values
        merged = []
        cur_start, cur_end = b_intervals[0]
        for s, e in b_intervals[1:]:
            if s <= cur_end:
                cur_end = max(cur_end, e)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

        a_len = group["a_end"].iloc[0] - group["a_start"].iloc[0]
        overlap_len = sum(e - s for s, e in merged)
        fraction = overlap_len / a_len if a_len > 0 else 0.0
        return pd.Series({"overlapping": fraction})

    result = (
        df.groupby(["a_start", "a_end"], sort=False)
          .apply(compute_fraction)
          .reset_index()
    )

    # Include A intervals that had no overlapping B
    all_a_intervals = df[["a_start", "a_end"]].drop_duplicates()
    result = pd.merge(
        all_a_intervals,
        result,
        on=["a_start", "a_end"],
        how="left"
    ).fillna({"overlapping": 0.0})

    return result[["a_start", "a_end", "overlapping"]]

# def _find_overlapping_intervals(df_a, df_b):
#     # Convert to Interval Series (assuming closed='right' by default)
#     intervals_a = pd.IntervalIndex.from_arrays(df_a['start'], df_a['end'], closed='right')
#     intervals_b = pd.IntervalIndex.from_arrays(df_b['start'], df_b['end'], closed='right')
#
#     # Create DataFrames with interval columns and original indices
#     df_a_intervals = df_a.copy()
#     df_a_intervals['interval'] = intervals_a
#     df_a_intervals['df_name'] = 'A'
#
#     df_b_intervals = df_b.copy()
#     df_b_intervals['interval'] = intervals_b
#     df_b_intervals['df_name'] = 'B'
#
#     # Find all pairwise overlaps efficiently using broadcasted comparison
#     overlaps_mask = intervals_a[:, None].overlaps(intervals_b)
#
#     # Get indices of overlapping pairs
#     a_indices, b_indices = np.nonzero(overlaps_mask)
#
#     # Create result DataFrame with overlapping pairs
#     result = pd.DataFrame({
#         'df_a_index': a_indices,
#         'df_b_index': b_indices,
#         'a_start': df_a['start'].iloc[a_indices],
#         'a_end': df_a['end'].iloc[a_indices],
#         'b_start': df_b['start'].iloc[b_indices],
#         'b_end': df_b['end'].iloc[b_indices]
#     })
#
#     return result
