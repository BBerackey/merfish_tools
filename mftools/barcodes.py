"""A collection of functions to work with the barcodes decoded by MERlin.

Functions
---------
process_merlin_barcodes
    Determines error bit/type for barcodes.
expand_codebook
    Creates a new codebook with additional barcodes representing all possible
    single bit flips for all genes. Used to assign error correction information
    to barcodes.
normalize_codebook
    Returns a codebook with L2 normalized barcodes.
set_barcode_stats
    Adds barcode statistics to the global stats object (see stats.py)
make_table
    Given a MERlin analysis folder and codebook, creates a table of all decoded
    barcodes with error correction information.
calculate_global_coordinates
    Adds the global coordinates to a barcode table.
assign_to_cells
    Determines the cell IDs for each barcode.
link_cell_ids
    Renames cell IDs to unify overlapping cells in adjacent FOVs.
mark_barcodes_in_overlaps
    Sets the status of barcodes to "edge" for those barcodes which are in the
    overlapping regions of FOVs.
create_cell_by_gene_table
    Given a barcode table with assigned cell IDs, returns a cell by gene matrix.
    Barcodes marked with "edge" status are not counted.
count_unfiltered_barcodes
    Gets the number of barcodes MERlin decoded before applying adaptive filtering.
get_per_bit_stats
    Calculates the per-bit error rates for a given gene.
get_per_gene_error
    Calculates the overall error rates for each gene.
per_bit_error
    Calculates the average error rate for each bit across all genes.
per_fov_error
    Calculates the error rate within each FOV.
"""
from collections import defaultdict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import norm

from . import fileio
from . import stats
from . import config


# global variable to hold number of bits
no_bits = 0

def process_merlin_barcodes(
    barcodes: pd.DataFrame,
    neighbors: faiss.IndexFlatL2,
    expanded_codebook: pd.DataFrame,
) -> pd.DataFrame:
    """Process the barcodes for a single field of view.

    The error type and bit are determined using an expanded codebook and returned as a
    DataFrame with extraneous columns removed.
    """
    X = np.ascontiguousarray(barcodes.filter(like="intensity_").to_numpy(), dtype=np.float32)
    indexes = neighbors.search(X, k=1)

    res = expanded_codebook.iloc[indexes[1].flatten()].copy()
    res = res.set_index(["name", "id"]).filter(like="bit").sum(axis=1)
    res = pd.DataFrame(res, columns=["bits"]).reset_index()

    df = barcodes[["barcode_id", "fov", "x", "y", "z"]]
    df = df.reset_index(drop=True)
    df["gene"] = res["name"]
    df["error_type"] = res["bits"] - 4 # the reason we are subtracting four is because each barcode(bit pattern) is suppose to have 4 onbits
    try:
        df["error_bit"] = res["id"].str.split("flip", expand=True)[1].fillna(0).astype(int)
    except KeyError:
        df["error_bit"] = 0
    return df


def expand_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    """Add codes for every possible bit flip to a codebook."""
    #TODO: modify to accomodate code for vizgen codebook
    # tentative solution: rename columns
    global no_bits
    col_mapper = {x:f'bit{i}' for i,x in enumerate(codebook.columns)
                  if (i !=0) and (i != codebook.shape[1]-1)}
    col_mapper.update({codebook.columns[0]:codebook.columns[0],
                       codebook.columns[-1]:codebook.columns[-1]})
    codebook = codebook.rename(col_mapper,axis = 'columns')

    books = [codebook]
    bits = len(codebook.filter(like="bit").columns)
    no_bits = bits # update the global variables

    for bit in range(1, bits + 1):
        flip = codebook.copy()
        flip[f"bit{bit}"] = (~flip[f"bit{bit}"].astype(bool)).astype(int)
        flip["id"] = flip["id"] + f"_flip{bit}"
        books.append(flip)
    return pd.concat(books)


def normalize_codebook(codebook: pd.DataFrame) -> pd.DataFrame:
    """L2 normalize a codebook."""
    codes = codebook.filter(like="bit")
    normcodes = codes.apply(lambda row: row / norm(row), axis=1)
    return normcodes


def set_barcode_stats(merlin_result: fileio.MerlinOutput, bcs: pd.DataFrame, colors: list) -> pd.DataFrame:
    stats.set("Unfiltered barcode count", count_unfiltered_barcodes(merlin_result))
    stats.set("Filtered barcode count", len(bcs))
    stats.set(
        "Unfiltered barcodes per FOV",
        stats.get("Unfiltered barcode count") / stats.get("FOVs"),
    )
    stats.set(
        "Filtered barcodes per FOV",
        stats.get("Filtered barcode count") / stats.get("FOVs"),
    )
    stats.set(
        "% barcodes kept",
        stats.get("Filtered barcode count") / stats.get("Unfiltered barcode count"),
    )
    stats.set("Exact barcode count", len(bcs[bcs["error_type"] == 0]))
    stats.set(
        "Corrected barcode count",
        stats.get("Filtered barcode count") - stats.get("Exact barcode count"),
    )
    stats.set(
        "% exact barcodes",
        stats.get("Exact barcode count") / stats.get("Filtered barcode count"),
    )
    stats.set(
        "0->1 error rate",
        len(bcs[bcs["error_type"] == 1]) / stats.get("Filtered barcode count"),
    )
    stats.set(
        "1->0 error rate",
        len(bcs[bcs["error_type"] == -1]) / stats.get("Filtered barcode count"),
    )
    try:
        error = per_bit_error(bcs, colors)
        stats.set(
            "Average per-bit 0->1 error rate",
            error[error["Error type"] == "0->1"]["Error rate"].mean(),
        )
        stats.set(
            "Average per-bit 1->0 error rate",
            error[error["Error type"] == "1->0"]["Error rate"].mean(),
        )
        return error
    except ValueError:  # Temporary hack to avoid issue
        return None


def make_table(merlin_result: fileio.MerlinOutput, codebook: pd.DataFrame) -> pd.DataFrame:
    """Create a table of all barcodes with error correction information."""
    codebook = expand_codebook(codebook)
    X = np.ascontiguousarray(normalize_codebook(codebook).to_numpy(), dtype=np.float32)
    neighbors = faiss.IndexFlatL2(X.shape[1])
    neighbors.add(X)
    dfs = []
    fov_list = merlin_result.load_fov_list()
    # for fov in tqdm(range(merlin_result.n_fovs()), desc="Preparing barcodes"):
    # Note: it is better to calculate number of fov based on length of positions.csv file
    # rather than number of .dax files, since there could be fovs not images due to focus lock problem
    for fov in tqdm(fov_list, desc="Preparing barcodes"):
        try:
            #TODO: comeup with a format the generaizes or improve way to load
            # different file naming scheme
            # for vizgen dataset, .00{fov}
            barcodes = merlin_result.load_filtered_barcodes(fov) # added by bereket
        except FileNotFoundError:
            continue
        dfs.append(process_merlin_barcodes(barcodes, neighbors, codebook))
    df = pd.concat(dfs, ignore_index=True)
    df["status"] = "unprocessed"
    return df


def calculate_global_coordinates(barcodes: pd.DataFrame, positions: pd.DataFrame,fov_size_pxl = 2048,fov_size = 220) -> None:
    """Add global_x and global_y columns to barcodes."""

    def convert_to_global(group: pd.DataFrame) -> pd.DataFrame:
        """Calculate the global coordinates for a single FOV."""
        # note the inner function can access the namespace of the outer function
        fov = int(group.iloc[0]["fov"])
        ypos = positions.loc[fov][1] # this is actual pos['y']
        xpos = positions.loc[fov][0] # this is actual pos['x']
        # group["global_y"] = fov_size * (fov_size_pxl - group["y"]) / fov_size_pxl + ypos
        # group["global_x"] = fov_size * group["x"] / fov_size_pxl + xpos

        # import pdb
        # pdb.set_trace()
        micron_pxl_ratio = fov_size/fov_size_pxl
        # # group["global_y"] = micron_pxl_ratio * group["y"] + (ypos - fov_size/2)
        group['global_y'] = micron_pxl_ratio*group["y"] + (ypos + fov_size/2)
        group["global_x"] = micron_pxl_ratio * (fov_size_pxl - group["x"]) + (xpos + fov_size/2)
        return group[["global_x", "global_y"]]

    barcodes[["global_x", "global_y"]] = barcodes.groupby("fov", group_keys=False).apply(convert_to_global)


def assign_to_cells(barcodes, masks, drifts=None, transpose=False, flip_x=False, flip_y=False,fov_size_pxl = 2048):
    for fov in tqdm(np.unique(barcodes["fov"]), desc="Assigning barcodes to cells"):
        group = barcodes.loc[barcodes["fov"] == fov]
        if drifts is not None:
            xdrift = drifts.loc[fov]["X drift"]
            ydrift = drifts.loc[fov]["Y drift"]
        else:
            xdrift, ydrift = 0, 0
        x = (group["x"] + xdrift).round() // config.get("scale")
        y = (group["y"] + ydrift).round() // config.get("scale")
        if flip_x:
            x = fov_size_pxl - x
        if flip_y:
            y = fov_size_pxl - y
        if transpose:
            x, y = y, x
        x = x.clip(upper=fov_size_pxl-1).astype(int)
        y = y.clip(upper=fov_size_pxl- 1).astype(int)
        if len(masks[int(fov)].shape) == 3: # note fov is a string not an int in this for loop
                                       # but i don't think it will affect the loading process
                                       # also since we are using it in arthmatic operation below
                                       # better to convert it to int
            # TODO: Remove hard-coding of scale
            # z = (group["z"].round() / 6.333333).astype(int)
            z = group["z"].round().astype(int)
            barcodes.loc[barcodes["fov"] == fov, "cell_id"] = masks[int(fov)][z, y, x] + 10000 * int(fov)
        else:
            barcodes.loc[barcodes["fov"] == fov, "cell_id"] = masks[int(fov)][y, x] + 10000 * int(fov)
    barcodes.loc[barcodes["cell_id"] % 10000 == 0, "cell_id"] = 0
    barcodes["cell_id"] = barcodes["cell_id"].astype(int)
    stats.set("Barcodes assigned to cells", len(barcodes[barcodes["cell_id"] != 0]))
    stats.set(
        "% barcodes assigned to cells",
        stats.get("Barcodes assigned to cells") / len(barcodes),
    )


def link_cell_ids(barcodes, cell_links):
    link_map = {cell: list(group)[0] for group in cell_links for cell in group}
    barcodes["cell_id"] = barcodes["cell_id"].apply(lambda cid: link_map[cid] if cid in link_map else cid)


def trim_barcodes_in_overlaps(barcodes, trim_overlaps,fov_size_pxl = 2048):
    xstarts = defaultdict(list)
    xstops = defaultdict(list)
    ystarts = defaultdict(list)
    ystops = defaultdict(list)

    # import pdb
    # pdb.set_trace()

    for pair in trim_overlaps:
        for overlap in pair:
            if overlap.xslice.start:
                xstarts[overlap.xslice.start].append(overlap.fov)
            if overlap.xslice.stop:
                xstops[fov_size_pxl // config.get("scale") + overlap.xslice.stop].append(overlap.fov)
            if overlap.yslice.start:
                ystarts[overlap.yslice.start].append(overlap.fov)
            if overlap.yslice.stop:
                ystops[fov_size_pxl // config.get("scale") + overlap.yslice.stop].append(overlap.fov)
    for xstart, fovs in xstarts.items():
        barcodes = barcodes[(barcodes["fov"].isin(fovs)) & (barcodes["y"] // config.get("scale") <= xstart)]
    for ystart, fovs in ystarts.items():
        barcodes = barcodes[(barcodes["fov"].isin(fovs)) & (barcodes["x"] // config.get("scale") <= ystart)]
    for xstop, fovs in xstops.items():
        barcodes = barcodes[(barcodes["fov"].isin(fovs)) & (barcodes["y"] // config.get("scale") >= xstop)]
    for ystop, fovs in ystops.items():
        barcodes = barcodes[(barcodes["fov"].isin(fovs)) & (barcodes["x"] // config.get("scale") >= ystop)]
    return barcodes


def create_cell_by_gene_table(barcodes, drop_blank=False) -> pd.DataFrame:
    # Create cell by gene table
    incells = barcodes[barcodes["cell_id"] != 0]
    ctable = pd.crosstab(index=incells["cell_id"], columns=incells["gene"])
    # Drop blank barcodes
    if drop_blank:
        drop_cols = [col for col in ctable.columns if "notarget" in col or "blank" in col.lower()]
        ctable = ctable.drop(columns=drop_cols)
    return ctable


def count_unfiltered_barcodes(merlin_result: fileio.MerlinOutput) -> int:
    """Count the total number of barcodes decoded by MERlin before adaptive filtering."""
    fov_list = merlin_result.load_fov_list()
    raw_count = 0
    for fov in tqdm(fov_list, desc="Counting unfiltered barcodes"):
        # try__except statement to handle missing fov data
        try:
            fov_raw_count = merlin_result.count_raw_barcodes(fov)
        except FileNotFoundError:
            continue
        raw_count += fov_raw_count

    return raw_count


def get_per_bit_stats(gene: str, barcodes: pd.DataFrame) -> pd.DataFrame:
    k0 = len(barcodes[barcodes["error_type"] == 0])
    total = len(barcodes)
    rows = []
    for bit in range(1, no_bits + 1): # no_bits is a global variable that is updated when expanding the codebook
                                      # should contain the correct total number of bits
        errors = barcodes[barcodes["error_bit"] == bit]
        k1 = len(errors)
        if k1 > 0 and k0 > 0:
            err_type = "1->0" if errors.iloc[0]["error_type"] == -1 else "0->1"
            rate = (k1 / k0) / (1 + (k1 / k0))
            rows.append([gene, bit, k1, err_type, rate, rate * total, total])
    return pd.DataFrame(
        rows,
        columns=[
            "gene",
            "Bit",
            "count",
            "Error type",
            "Error rate",
            "weighted",
            "total",
        ],
    ).reset_index()


def per_gene_error(barcodes) -> pd.DataFrame:
    """Get barcode error statistics per gene."""
    error = (
        barcodes.groupby(["gene", "error_type"])
        .count()
        .reset_index()
        .pivot(index="gene", columns="error_type", values="barcode_id")
    )
    error.columns = ["1->0 errors", "Exact barcodes", "0->1 errors"]
    error["Total barcodes"] = error["1->0 errors"] + error["0->1 errors"] + error["Exact barcodes"]
    error["% exact barcodes"] = error["Exact barcodes"] / error["Total barcodes"]
    return error


def per_bit_error(barcodes, colors) -> pd.DataFrame:
    """Get barcode error statistics per bit."""
    error = pd.concat(get_per_bit_stats(gene, group) for gene, group in barcodes.groupby("gene"))
    error["Color"] = error.apply(lambda row: colors[(row["Bit"] - 1) % len(colors)], axis=1)
    error["Hybridization round"] = error.apply(lambda row: (row["Bit"] - 1) // len(colors) + 1, axis=1)
    return error


def per_fov_error(barcodes) -> pd.DataFrame:
    """Get barcode error statistics per FOV."""
    error = barcodes.groupby(["fov", "error_type"]).count()["gene"] / barcodes.groupby("fov").count()["gene"]
    error = error.reset_index()
    error.columns = ["FOV", "Error type", "Error rate"]
    error["Error type"] = error["Error type"].replace([0, -1, 1], ["Correct", "1->0", "0->1"])
    return error
