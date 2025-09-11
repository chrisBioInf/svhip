
"""
Created on Tue Aug 16 23:15:23 2022

@author: christopher
"""


__author__ = "Christopher Klapproth"
__institution__= "University Leipzig"
__credits__ = []
__license__ = "GPLv2"
__version__="1.0.6"
__maintainer__ = "Christopher Klapproth"
__email__ = "christopher.klapproth@protonmail.com"
__status__ = "Release"


import os
import sys
from pathlib import Path
import optparse
from optparse import OptionParser
import logging
import numpy as np
import pandas as pd
import time
import shlex
import subprocess
import shutil
from random import shuffle
from subprocess import call
import tempfile
import re
import pickle
from itertools import product, combinations
from math import factorial
import random
from joblib import parallel_backend, Parallel, delayed

#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from Bio import AlignIO, SeqIO, Phylo
# from Bio.Align.Applications import ClustalwCommandline
from Bio.Align.AlignInfo import SummaryInfo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import blosum
import RNA as RNA

from typing import Optional


#########################Default parameters#############################

DEFAULT_THREADS = max(os.cpu_count()-1, 1)
RANDOM_SEED = random.randint(0, 2**31)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RNAz_DIR = os.path.join(THIS_DIR, "RNAz_toolkit")
hexamer_backup = os.path.join(os.path.join(THIS_DIR, "hexamer_models"), "Human_hexamer.tsv")

class_dict = {
    1 : "other",
    -1: "non-coding",
    0 : "coding",
    "other": 1,
    "non-coding": -1,
    "coding": 0,
    }
strand_dict = {
    "forward": "+",
    "reverse": "-",
    }

rng = np.random.default_rng()

nucleotides = ["A", "C", "G", "U", "-"]
dinucleotides = ["AA", "AC", "AG", "AU",
                 "CA", "CC", "CG", "CU",
                 "GA", "GC", "GG", "GU",
                 "UA", "UC", "UG", "UU"
                 ]

svr_feature_names = ['C_to_GC', 'A_to_AU', 'AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG',
                     'CU', 'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU', 'length']

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
model_std_dict, model_mean_dict = {}, {}

for f in ["20-30", "30-40", "40-50",
          "50-60", "60-70", "70-80"]:
    model_std_dir = "z_score_model_SVR/std_model_%s.model" % f
    model_mean_dir = "z_score_model_SVR/mean_model_%s.model" % f
    model_std_dict[f] = pickle.load(open(os.path.join(THIS_DIR, model_std_dir), 'rb'))
    model_mean_dict[f] = pickle.load(open(os.path.join(THIS_DIR, model_mean_dir), 'rb'))

bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

min_len = 50
max_len = 200



################################################
#
#               File handling
#
################################################

def is_fasta(filename: str) -> bool:
    with open(filename, "r") as handle:
        try:
            fasta = SeqIO.parse(handle, "fasta")
        except Exception:
            return False
    return any(fasta) 
    
    
def is_alignment(filename: str) -> bool:
    try:
        alignment = AlignIO.read(filename, "clustal")
    except Exception:
        return False
    return any(alignment)


def ungap_sequence(seq: str) -> str:
    return re.sub(r'-', "", seq)


################################################
#
#              External function calls
#
################################################


def clustalo_cmdline(infile: str, outfile: str, threads: int) -> list[str]:
    print(f"Now aligning: {infile}")
    cmd = f"clustalo --threads {threads} --output-order=input-order --infmt=fa --outfmt=clu --force -i {infile} -o {outfile}"
    return shlex.split(cmd)


def generate_control_alignment(inpath: str, outpath: str, sissiz_shuffle: bool=False) -> None:
    if sissiz_shuffle:
        with open(outpath, "w+") as outf:
            cmd = "SISSIz -n 1 -s --flanks 1750 %s" % inpath
            subprocess.call(shlex.split(cmd), stdout=open(outpath, "w"))
    else:
        shuffle_alignment_file(input_file=inpath, output_file=outpath)



###############################################
#
#               Altschulerickson Algorithm
#
###############################################

# !!!!!  EDITED FROM:
# altschulEriksonDinuclShuffle.py
# P. Clote, Oct 2003
# NOTE: One cannot use function "count(s,word)" to count the number
# of occurrences of dinucleotide word in string s, since the built-in
# function counts only nonoverlapping words, presumably in a left to
# right fashion.

def computeCountAndLists(s: str) -> tuple[dict[str, int], dict[str, dict[str, int]], dict[str, list[str]]]:
    #WARNING: Use of function count(s,'UU') returns 1 on word UUU
    #since it apparently counts only nonoverlapping words UU
    #For this reason, we work with the indices.

    #Initialize lists and mono- and dinucleotide dictionaries
    ls = {} #List is a dictionary of lists
    ls['A'] = []; ls['C'] = []
    ls['G'] = []; ls['U'] = []
    nuclList   = ["A","C","G","U"]
    s       = s.upper()
    s       = s.replace("T","U")
    nuclCnt    = {}  #empty dictionary
    dinuclCnt  = {}  #empty dictionary
    for x in nuclList:
        nuclCnt[x]=0
        dinuclCnt[x]={}
        for y in nuclList:
            dinuclCnt[x][y]=0

    #Compute count and lists
    nuclCnt[s[0]] = 1
    nuclTotal     = 1
    dinuclTotal   = 0
    for i in range(len(s)-1):
        x = s[i]; y = s[i+1]
        ls[x].append( y )
        nuclCnt[y] += 1; nuclTotal  += 1
        dinuclCnt[x][y] += 1; dinuclTotal += 1
    assert (nuclTotal==len(s))
    assert (dinuclTotal==len(s)-1)
    return nuclCnt,dinuclCnt,ls
 
 
def chooseEdge(x: str, dinuclCnt: dict[str, dict[str, int]]) -> str:
    numInList = 0
    for y in ['A','C','G','U']:
        numInList += dinuclCnt[x][y]
    z = random.random()
    denom=dinuclCnt[x]['A']+dinuclCnt[x]['C']+dinuclCnt[x]['G']+dinuclCnt[x]['U']
    numerator = dinuclCnt[x]['A']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['A'] -= 1
        return 'A'
    numerator += dinuclCnt[x]['C']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['C'] -= 1
        return 'C'
    numerator += dinuclCnt[x]['G']
    if z < float(numerator)/float(denom):
        dinuclCnt[x]['G'] -= 1
        return 'G'
    dinuclCnt[x]['U'] -= 1
    return 'U'

def connectedToLast(edgeList: list[list[str]], nuclList: list[str], lastCh: str) -> int:
    D = {}
    for x in nuclList: D[x]=0
    for edge in edgeList:
        a = edge[0]; b = edge[1]
        if b==lastCh: D[a]=1
    for i in range(2):
        for edge in edgeList:
            a = edge[0]; b = edge[1]
            if D[b]==1: D[a]=1
    ok = 0
    for x in nuclList:
        if x!=lastCh and D[x]==0: return 0
    return 1


def eulerian(s: str) -> tuple[int, list[list[str]], list[str], str]:
    nuclCnt,dinuclCnt,ls = computeCountAndLists(s)
    #compute nucleotides appearing in s
    nuclList = []
    for x in ["A","C","G","U"]:
        if x in s: nuclList.append(x)
    #compute numInList[x] = number of dinucleotides beginning with x
    numInList = {}
    for x in nuclList:
        numInList[x]=0
        for y in nuclList:
            numInList[x] += dinuclCnt[x][y]
    #create dinucleotide shuffle L
    firstCh = s[0]  #start with first letter of s
    lastCh  = s[-1]
    edgeList = []
    for x in nuclList:
        if x!= lastCh: edgeList.append( [x,chooseEdge(x,dinuclCnt)] )
    ok = connectedToLast(edgeList,nuclList,lastCh)
    return ok,edgeList,nuclList,lastCh


def shuffleEdgeList(L: list[str]) -> list[str]:
    n = len(L); barrier = n
    for i in range(n-1):
        z = int(random.random() * barrier)
        tmp = L[z]
        L[z]= L[barrier-1]
        L[barrier-1] = tmp
        barrier -= 1
    return L


def dinuclShuffle(s: str) -> str:
    ok = 0
    while not ok:
        ok,edgeList,nuclList,lastCh = eulerian(s)
    nuclCnt,dinuclCnt,ls = computeCountAndLists(s)

    #remove last edges from each vertex list, shuffle, then add back
    #the removed edges at end of vertex lists.
    for [x,y] in edgeList: ls[x].remove(y)
    for x in nuclList: shuffleEdgeList(ls[x])
    for [x,y] in edgeList: ls[x].append(y)

    #construct the eulerian path
    L = [s[0]]; prevCh = s[0]
    for i in range(len(s)-2):
        ch = ls[prevCh][0] 
        L.append( ch )
        del ls[prevCh][0]
        prevCh = ch
    L.append(s[-1])
    t = str().join(L)
    return t


################################################
#
#              Helper functions
#
################################################


bp_dict = {
    'A': 'U',
    'U': 'A',
    'T': 'A',
    'G': 'C',
    'C': 'G',
}

def get_reverse_complement(seq: str) -> str: 
    return str().join([bp_dict.get(x, x) for x in reversed(seq)])


def calculate_pairwise_identity(seq1: str, seq2: str) -> float:
    """
    Calculate pairwise sequence identity between two sequences.
    
    Args:
        seq1, seq2: Bio.Seq objects or strings
        
    Returns:
        float: Identity as fraction (0.0 to 1.0)
    """
    seq1_str = str(seq1).upper()
    seq2_str = str(seq2).upper()
    
    if len(seq1_str) != len(seq2_str):
        raise ValueError("Sequences must be of equal length for alignment")
    
    if len(seq1_str) == 0:
        return 0.0
    
    # Count matches (excluding gaps)
    matches = 0
    valid_positions = 0
    
    for c1, c2 in zip(seq1_str, seq2_str):
        # Skip positions where either sequence has a gap
        if c1 != '-' and c2 != '-':
            valid_positions += 1
            if c1 == c2:
                matches += 1
    
    return matches / valid_positions if valid_positions > 0 else 0.0


def get_max_identity_in_set(records: list[SeqRecord]) -> float:
    """
    Calculate the maximum pairwise identity within a set of records.
    
    Args:
        records: List of SeqRecord objects
        
    Returns:
        float: Maximum pairwise identity in the set
    """
    if len(records) < 2:
        return 0.0
    
    max_identity = 0.0
    for rec1, rec2 in combinations(records, 2):
        identity = calculate_pairwise_identity(rec1.seq, rec2.seq)
        max_identity = max(max_identity, identity)
    
    return max_identity


def calculate_mean_distance(seq_list: list[str]) -> float:
    if len(seq_list) < 2:
        return 0
        
    distance_vector =[]
    for i in range(0, len(seq_list)-1):
        for a in range(i+1, len(seq_list)):    
            seq_struc1, seq_struc2 = RNA.fold_rna(seq_list[i])[0], RNA.fold_rna(seq_list[a])[0]
            seq_tree1, seq_tree2 = RNA.make_tree(seq_struc1), RNA.make_tree(seq_struc2)
            distance_vector.append(RNA.tree_edit_distance(seq_tree1, seq_tree2))
            
    return float(np.mean(a=distance_vector))



################################################
#
#              Alignment methods (shuffling, sequence selection, windows)
#
################################################


def shuffle_alignment_columns(alignment: MultipleSeqAlignment) -> MultipleSeqAlignment:
    """
    Read a CLUSTAL alignment, shuffle its columns, and write the result.
    
    Args:
        input_file (str): Path to input CLUSTAL alignment file
        output_file (str): Path to output CLUSTAL alignment file  
    """
    # Get alignment dimensions
    num_sequences = len(alignment)
    alignment_length = alignment.get_alignment_length()
    
    # Create list of column indices and shuffle them
    column_indices = list(range(alignment_length))
    random.shuffle(column_indices)
    
    # Create new shuffled sequences
    shuffled_records = []
    for i, record in enumerate(alignment):
        # Build new sequence by taking columns in shuffled order
        new_seq = ''.join(str(record.seq)[col_idx] for col_idx in column_indices)
        
        # Create new SeqRecord with shuffled sequence
        shuffled_record = SeqRecord(
            Seq(new_seq),
            id=record.id,
            description=record.description
        )
        shuffled_records.append(shuffled_record)
    
    # Create new alignment with shuffled sequences
    shuffled_alignment = MultipleSeqAlignment(shuffled_records)

    return shuffled_alignment
    
    

def shuffle_alignment_file(input_file: str, output_file: str):
    # Read the alignment
    alignment = AlignIO.read(input_file, "clustal")

    # Randomize alignment columns
    shuffled_alignment = shuffle_alignment_columns(alignment)

    # Write the shuffled alignment
    AlignIO.write(shuffled_alignment, output_file, "clustal")
    print(f"Shuffled alignment written to {output_file}")



def select_diverse_records(alignment: MultipleSeqAlignment, n_records: int=6, max_identity: float=0.9, max_attempts: int=1000) -> Optional[list[SeqRecord]]:
    """
    Select N records from alignment with maximum pairwise identity below threshold.

    Args:
        alignment: Bio.Align.MultipleSeqAlignment object
        n_records (int): Number of records to select
        max_identity (float): Maximum allowed pairwise identity (0.0-1.0)
        max_attempts (int): Maximum random sampling attempts

    Returns:
        list: Selected SeqRecord objects, or None if no valid set found
    """
    all_records = list(alignment)

    if len(all_records) <= n_records:
        # If we have fewer records than requested, check if they meet criteria
        max_id = get_max_identity_in_set(all_records)
        if max_id <= max_identity:
            return all_records
        else:
            print(f"Warning: Only {len(all_records)} records available, but max identity ({max_id:.3f}) exceeds threshold ({max_identity})")
            return all_records  # Return what we have

    # Try random sampling to find a diverse set
    for attempt in range(max_attempts):
        selected = random.sample(all_records, n_records)
        max_id = get_max_identity_in_set(selected)

        if max_id <= max_identity:
            print(f"Found diverse set after {attempt + 1} attempts (max identity: {max_id:.3f})")
            return selected

    # If no valid set found, try greedy selection
    print(f"Random sampling failed after {max_attempts} attempts.")
    return None



def select_seqs(input_file, output_file, n_records=6, max_identity=0.95, max_attempts=1000):
    """
    Main function to filter alignment by selecting diverse records.
    
    Args:
        input_file (str): Path to input CLUSTAL alignment
        output_file (str): Path to output CLUSTAL alignment
        n_records (int): Number of records to select
        max_identity (float): Maximum pairwise identity threshold
    """
    # Read alignment
    print(f"Reading alignment from {input_file}...")
    alignment = AlignIO.read(input_file, "clustal")
    print(f"Original alignment: {len(alignment)} sequences, {alignment.get_alignment_length()} positions")
    
    # Select diverse records
    selected_records = select_diverse_records(alignment, n_records, max_identity, max_attempts)
    
    if not selected_records:
        print("Error: No records could be selected")
        return False
    
    # Create new alignment with selected records
    filtered_alignment = MultipleSeqAlignment(selected_records)
    
    # Write output
    AlignIO.write(filtered_alignment, output_file, "clustal")
    
    return True



def slice_alignment_into_windows(input_file: str, window_length: int = 120, step_size: int = 40) -> list[MultipleSeqAlignment]:
    """
    Read a CLUSTAL alignment and slice it into overlapping windows.

    Args:
        input_file (str): Path to input CLUSTAL alignment file
        window_length (int): Length of each window
        step_size (int): Step size between windows

    Returns:
        List[MultipleSeqAlignment]: List of alignment windows
    """
    # Read the alignment
    alignment = AlignIO.read(input_file, "clustal")
    alignment_length = alignment.get_alignment_length()

    windows = []
    start_pos = 0

    while start_pos < alignment_length:
        # Calculate end position for this window
        end_pos = min(start_pos + window_length, alignment_length)

        # Create new records for this window
        window_records = []
        for record in alignment:
            # Extract the sequence slice for this window
            window_seq = str(record.seq)[start_pos:end_pos]

            # Create new SeqRecord with the sliced sequence
            window_record = SeqRecord(
                Seq(window_seq),
                id=record.id,
                description=record.description
            )
            window_records.append(window_record)

        # Create MultipleSeqAlignment for this window
        window_alignment = MultipleSeqAlignment(window_records)
        windows.append(window_alignment)

        # Move to next window position
        start_pos += step_size

        # Stop if we've reached the end
        if end_pos == alignment_length:
            break

    print(f"Created {len(windows)} windows from alignment of length {alignment_length}")
    print(f"Window length: {window_length}, Step size: {step_size}")

    return windows



################################################
#
#               Generate Hexamer Model
#
################################################


def hexamer_permutations() -> list[str]:
    return [str().join(x) for x in product(*(["ATGC"] * 6))]


def get_hexamers_from_seq(seq: str) -> list[str]:
    hexamers = []
    i = 0
    while i < len(seq)+6:
        hexamers.append(seq[i:i+6])
        i += 3
    return hexamers


def get_fasta_sequence(filename: str) -> list[str]:
    handle = SeqIO.parse(handle=open(filename, 'r'), format="fasta")
    records = [str(record.seq) for record in handle]
    return records


def hexamer_model_calibration(options: optparse.Values) -> None:
    coding_file = options.in_coding
    noncoding_file = options.in_noncoding
    outfile = options.out_file

    if not outfile.endswith(".tsv"):
        outfile = outfile + ".tsv"

    coding_seqs = get_fasta_sequence(coding_file)
    noncoding_seqs = get_fasta_sequence(noncoding_file)
    hex_combinations = hexamer_permutations()

    noncoding_dict = {}
    coding_dict = {}
    coding_hexamers, noncoding_hexamers = [], []
    n_seqs_coding = len(coding_seqs)
    n_seqs_noncoding = len(noncoding_seqs)
    count = 0

    for seq in coding_seqs:
        count += 1
        print("Screen coding sequences: %s/%s" % (count, n_seqs_coding))
        coding_hexamers += get_hexamers_from_seq(seq)
    count = 0

    for seq in noncoding_seqs:
        count += 1
        print("Screen noncoding sequences: %s/%s" % (count, n_seqs_noncoding))
        noncoding_hexamers += get_hexamers_from_seq(seq)

    n_coding = len(coding_hexamers)
    n_noncoding = len(noncoding_hexamers)
    count = 0

    for hex_ in hex_combinations:
        count += 1
        print("Scan Hexamers: %s/%s" % (count, 4096))
        coding_dict[hex_] = round(coding_hexamers.count(hex_) / n_coding, 8)
        noncoding_dict[hex_] = round(noncoding_hexamers.count(hex_) / n_noncoding, 8)

    with open(outfile, "w") as f:
        for hex_ in hex_combinations:
            f.write("%s\t%s\t%s\n" % (hex_, coding_dict.get(hex_), noncoding_dict.get(hex_)))
    print("Hexamer model written to %s." % outfile)




################################################
#
#               Calculate Features
#
################################################

######################### Misc. ####################################

def ungap_sequence(seq: str) -> str:
    return re.sub(r'-', "", seq)


def get_GC_content(seq: str) -> float:
    Gs = seq.count("G")
    Cs = seq.count("C")
    return (Gs + Cs) / len(seq)

def average_GC_content(seqs: list[str]) -> float:
    return np.mean([get_GC_content(s) for s in seqs])


def normalize_seq_length(length: int, min_len: int, max_len: int) -> float:
    return (length - min_len) / (max_len - min_len)


def calculate_sequence_features(seq: str, GC_content: float, min_len: int, max_len: int) -> pd.DataFrame:
    length = len(seq)
    Gs, Cs = seq.count("G"), seq.count("C")
    As, Us = seq.count("A"), seq.count("U")
    mono_features = [
                    # GC_content,
                    Cs / (Cs + Gs),
                    As / (As + Us),
                    ]
    doublets = length -1
    di_features = [(seq.count(dinucleotide) / doublets) for dinucleotide in dinucleotides]
    normalized_length = normalize_seq_length(length, min_len, max_len)
    values = mono_features + di_features + [normalized_length]
    features_ = pd.DataFrame(data=[values], columns=svr_feature_names)
    return features_


######################### Shannon entropy ##########################

def log_2(x: float) -> float:
    if x == 0:
        return 0
    return np.log2(x)


def column_entropy(dinucleotide: str, column: list[str]) -> float:
    col_string = str().join(column)
    l = len(column)
    f = col_string.count(dinucleotide) / l
    return f *log_2(f)


def shannon_entropy(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return 0
    columns = np.asarray([list(seq.strip("\n")) for seq in seqs]).transpose()
    entropy = 0.0

    if len(columns) < 4:
        return 0

    for column in columns:
        entropy += sum([column_entropy(n, column) for n in nucleotides])

    return float(round(entropy *(-1/len(columns)), 4))


########################### Structural conservation index ############

def fold_single(seq: str) -> tuple[str, float]:
    try:
        fc = RNA.fold_compound(seq)
        mfe_structure, mfe = fc.mfe()
    except Exception:
        print("\nCould not fold sequence for some reason...")
        print(f"Offending sequence: {seq}\n")
        mfe = 0.0
        mfe_structure = str().join(['.' for i in range(len(seq))])
    return mfe_structure, mfe


def call_alifold(seqs: list[str]) -> tuple[str, float]:
    consensus_structure, mfe = RNA.alifold(seqs)
    return consensus_structure, mfe


def structural_conservation_index(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return 0
    e_consensus = call_alifold(seqs)[1]
    seqs_ungapped = [ungap_sequence(seq) for seq in seqs]
    single_energies = sum([fold_single(s)[1] for s in seqs_ungapped])
    if single_energies == 0:
        return 0
    return round(e_consensus / ( (1/len(seqs))*single_energies), 4)


########################### z-score of MFE ###########################

def get_bin(GC: float) -> str:
    low, high = 0, 0
    for i in range(1, len(bins)):
        lower_bound = bins[i]
        if GC <= lower_bound:
            low = int(bins[i-1]*100)
            high = int(lower_bound*100)
            break

    return "%s-%s" % (low, high)


def predict_std(seq: str, GC_bin: str, feature_vector: list[float]) -> float:
    m = model_std_dict.get(GC_bin)
    return m.predict(feature_vector)[0]


def predict_mean(seq: str, GC_bin: str, feature_vector: list[float]) -> float:
    m = model_mean_dict.get(GC_bin)
    return m.predict(feature_vector)[0]


def get_distribution(mfes: list[float]) -> tuple[float, float]:
    std = np.std(mfes)
    mu = np.mean(mfes)
    return mu, std


def z_(x: float, std: float, mu: float) -> float:
    return (x - mu) / std


def random_permutation_generator(seq: str):
    while True:
        nucls = list(seq) 
        random.shuffle(nucls)
        yield ''.join(nucls)


def shuffle_distribution(seq: str) -> list[str]:
    references = []
    if "N" in seq:
        print("SHUFFLE WARNING: Unknown nucleotides 'N' in input sequene...")
        generator = random_permutation_generator(seq)
        for i, perm in enumerate(generator):
            references.append(perm)
            if i >= 99:
                break
    else:
        for i in range(0, 100):
            references.append(dinuclShuffle(seq))
    
    return references


def z_score_of_mfe(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return 0
    zs = []
    distributions = []
    lengths = [len(seq) for seq in seqs]

    # def shuffle_seq(s):
    #     return s
    
    for seq in seqs:
        GC_content = get_GC_content(seq)
        
        if (GC_content > 0.8) or (GC_content < 0.2):
            print("GC content outside of training range. Shuffling sequence instead...")
            print(seq)
            references = shuffle_distribution(seq)
            mfes = [fold_single(s)[1] for s in references]
            mu, std = get_distribution(mfes)
            distributions.append((std, mu))
        else:
            seq_features = calculate_sequence_features(seq, GC_content, min_len, max_len)
            GC_bin = get_bin(GC_content)
            if GC_bin == "0-0":
                continue
            
            std = predict_std(seq, GC_bin, seq_features)
            mu = predict_mean(seq, GC_bin, seq_features)
            distributions.append((std, mu))
    
    for i in range(0, len(seqs)):
        seq = seqs[i]
        std, mu = distributions[i]
        zs.append(z_(fold_single(seq)[1], std, mu))
    
    return float(round(np.mean(zs), 4))


########################### Hexamer Score ###########################


def get_background_model(hexamer_model: dict) -> tuple[dict, dict]:
    return hexamer_model.get("coding"), hexamer_model.get("noncoding")


def load_hexamer_model(filename: str) -> dict:
    coding, noncoding = {}, {}
    
    with open(filename, 'r') as f:
        for line in f.readlines():
            hexamer, c, nc = line.split("\t")
            coding[hexamer] = float(c)
            noncoding[hexamer] = float(nc)  
    return {
        "coding": coding,
        "noncoding": noncoding
    }


def word_generator(seq: str, word_size: int, step_size: int, frame: int = 0):
    for i in range(frame,len(seq),step_size):
        word =  seq[i:i+word_size]
        if len(word) == word_size:
            yield word


def kmer_ratio(seq: str, word_size: int, step_size: int, hexamer_model: dict) -> list[float]:
    if len(seq) < word_size:
        return 0
    scores = []
    coding, noncoding = get_background_model(hexamer_model)
    
    for frame in [0, 1, 2]:
        sum_of_log_ratio_0 = 0.0
        frame0_count=0.0
        for k in word_generator(seq=seq, word_size = word_size, step_size=step_size,frame=frame):    
            if (k not in coding) or (k not in noncoding):
                continue
            if coding.get(k)>0 and noncoding.get(k) >0:
                sum_of_log_ratio_0  +=  np.log2( coding.get(k) / noncoding.get(k))
            elif coding.get(k)>0 and noncoding.get(k) == 0:
                sum_of_log_ratio_0 += 1
            elif coding.get(k) == 0 and noncoding.get(k) == 0:
                continue
            elif coding.get(k) == 0 and noncoding.get(k) >0:
                sum_of_log_ratio_0 -= 1
            else:
                continue
            frame0_count += 1
        try:
            scores.append(sum_of_log_ratio_0/frame0_count)
        except:
            scores.append(-1)
    
    return scores
    
    
def hexamer_score(seqs: list[str], hexamer_model: dict) -> float:
    if len(seqs) < 2:
        return 0
    
    coding, noncoding = get_background_model(hexamer_model)
    frame_matrix = np.asarray([kmer_ratio(seq, 6, 3, hexamer_model) for seq in seqs]).T
    frame_scores = [sum(frame_matrix[0]), sum(frame_matrix[1]), sum(frame_matrix[2])]
    frame = frame_scores.index(max(frame_scores))
    
    sum_of_log_ratio_0 = 0.0
    score = 0.0
    frame0_count=0.0
    for seq in seqs:
        for k in word_generator(seq=seq, word_size = 6, step_size=3,frame=frame):    
            if (k not in coding) or (k not in noncoding):
                continue
            if coding.get(k)>0 and noncoding.get(k) >0:
                sum_of_log_ratio_0  +=  np.log2( coding.get(k) / noncoding.get(k))
            elif coding.get(k)>0 and noncoding.get(k) == 0:
                sum_of_log_ratio_0 += 1
            elif coding.get(k) == 0 and noncoding.get(k) == 0:
                continue
            elif coding.get(k) == 0 and noncoding.get(k) >0 :
                sum_of_log_ratio_0 -= 1
            else:
                continue
            frame0_count += 1
        try:
            score += sum_of_log_ratio_0/frame0_count 
        except:
            score = -1
    
    score = score / len(seqs)
    return round(float(score), 4)


########################## Codon conservation #######################

stop_score = 0.0
framegap_score = -12.0
blosum_62 = blosum.BLOSUM(62)
nucleotides = ["A", 
               "C",
               "G",
               "U"]

known_nucs = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "U",
    "U": "U",
    }


def generate_kmers() -> list[tuple[str, ...]]:
    k = 3
    nucleotides = ['A', 'C', 'G', 'U']
    k_comb = product(nucleotides, repeat=k)
    return list(k_comb)

codons = generate_kmers()


def maxSubArraySum(arr: list[float]) -> float:
    if len(arr) == 0:
        return 0.0
    size = len(arr)
    max_till_now = arr[0]
    max_ending = 0
    
    for i in range(0, size):
        max_ending = max_ending + arr[i]
        if max_ending < 0:
            max_ending = 0
        elif (max_till_now < max_ending):
            max_till_now = max_ending
            
    return max_till_now


def empirical_substitution_rates(ref_seq: str, seq: str) -> dict:
    transitions = {
        "A_A": 0.0, "A_C": 0.0, "A_T": 0.0, "A_G":0.0,
        "C_A": 0.0, "C_C": 0.0, "C_T": 0.0, "C_G":0.0,
        "T_A": 0.0, "T_C": 0.0, "T_T": 0.0, "T_G":0.0,
        "G_A": 0.0, "G_C": 0.0, "G_T": 0.0, "G_G":0.0,
        }
    
    for i in range(0, len(ref_seq)):
        if seq[i]=="-" or ref_seq[i]=="-":
            continue
        n_n = ref_seq[i] + "_" + seq[i]
        
        if not n_n in transitions:
            continue
        
        transitions[n_n] += 1
    
    for k in transitions.keys():
        transitions[k] = transitions.get(k) / (ref_seq.count(k[0])+1/len(ref_seq))
    
    return transitions


def get_stationary_frequencies(seq: str) -> dict:
    s = seq.replace("-", "")
    length = len(s)+1
    frequencies = {
        "A": s.count("A") / length,
        "U": s.count("U") / length,
        "G": s.count("G") / length,
        "C": s.count("C") / length,
        }
    return frequencies


def hamming_distance(codon1: str, codon2: str) -> int:
    d = 3
    for i in range(0, 3):
        if codon1[i] == codon2[i]:
            d -= 1
    return d


def statistical_penalty(codon1: str, codon2: str, aa_1: str, aa_2: str, frequencies: dict, distance: int, ref_seq: str, seq: str, transitions: dict) -> float:
    hamming = hamming_distance(codon1, codon2)
    s = 0
    
    for codon in codons:
        if hamming_distance(codon1, codon) != hamming:
            continue
            
        a1_b1 = transitions.get(codon1[0] + "_" + codon[0], 0)
        a2_b2 = transitions.get(codon1[1] + "_" + codon[1], 0)
        a3_b3 = transitions.get(codon1[2] + "_" + codon[2], 0)
        
        try:
            s += blosum_62[aa_1][aa_2]*(a1_b1/(distance+1) ) *(a2_b2/(1+distance)) *(a3_b3/(1+distance)) 
        except Exception:
            continue
    return s


def gap_score(gap: int) -> float:
    if gap == 3:
        return 0.0
    else:
        return -2.0
    

def calculate_offset(seq: str, index: int) -> int:
    nucleotides = []
    old_index = index
    while len(nucleotides) < 3 and index < len(seq):
        if seq[index] != "-":
            nucleotides.append(seq[index])
        index += 1
    return index - old_index 


def sanitize_alignment(aln: MultipleSeqAlignment) -> MultipleSeqAlignment:
    for record in aln:
        record.id = str(record.id).split(".")[0]
    set_id = set()
    aln_ = []
    for record in aln:
        if record.id not in set_id:
            aln_.append(record)
        set_id.add(record.id)
    return MultipleSeqAlignment(records=aln_)


def codon_conservation(alignment: MultipleSeqAlignment, tree_path: str=None) -> float:
    seqs = [str(record.seq) for record in alignment]
    names = [str(record.id).split(".")[0] for record in alignment]
    
    if tree_path:
        tree = Phylo.read(tree_path, "newick")
    else:
        alignment_ = sanitize_alignment(alignment)
        calculator = DistanceCalculator('identity')
        constructor = DistanceTreeConstructor(calculator, 'nj')
        tree = constructor.build_tree(alignment_)

    frames = [0, 1, 2]
    codon_length = 3
    frame_scores = []
    ref_seq = seqs[0]
    frequencies = get_stationary_frequencies(ref_seq)

    for frame in frames:
        pair_scores = []
        ref_name = names[0]
        
        for n in range(1, len(seqs)):
            score = 0
            local_score = 0
            seq = seqs[n]
            name = names[n]
            distance = tree.distance(ref_name, name)
            transitions = empirical_substitution_rates(ref_seq, seq)
            single_scores = []
            ref_index = frame
            seq_index = frame
            ref_frame_offset = 0
            seq_frame_offset = 0
            offset_total = 0
            
            while (seq_index + 2 < len(seq)) and (ref_index + 2 < len(ref_seq)):
                codon1 = Seq(ref_seq[ref_index] + ref_seq[ref_index+1] + ref_seq[ref_index+2])
                codon2 = Seq(seq[seq_index] + seq[seq_index+1] + seq[seq_index+2])
    
                if not "-" in codon2 and not "-" in codon1:
                    aa_1 = codon1.translate() 
                    aa_2 = codon2.translate()
                    if aa_1 == "*":
                        local_score = stop_score
                        break
                    else:
                        try:
                            local_score = blosum_62[aa_1][aa_2] - statistical_penalty(codon1, codon2, aa_1, aa_2, frequencies, distance, ref_seq, seq, transitions)
                        except Exception:
                            print(aa_1, aa_2, blosum_62[aa_1][aa_2])
                            print("Local score error...")
                            pass
                    
                    codon1_gaps = codon1.count("-")
                    codon2_gaps = codon2.count("-")
                    
                    if (codon1_gaps > 0) and (codon1_gaps == codon2_gaps):
                        local_score = gap_score(codon1_gaps)
                        ref_index += 1
                        seq_index += 1
                        continue
                    elif codon1_gaps > codon2_gaps:
                        ref_frame_offset = calculate_offset(ref_seq, ref_index)
                        local_score = framegap_score
                    elif codon2_gaps > 0:
                        local_score = framegap_score
                    if ref_frame_offset > 0:
                        ref_index += ref_frame_offset # + codon_length
                        seq_index += ref_frame_offset # + codon_length
                        # offset_total += ref_frame_offset
                    else:
                        ref_index += codon_length
                        seq_index += codon_length
    
                    if (offset_total == 3):
                        seq_index += 0
                        offset_total = 0
                    else:
                        pass
    
                    score += local_score
                    single_scores.append(local_score)
                    ref_frame_offset = 0
                    seq_frame_offset = 0
    
                else:
                    ref_index += codon_length
                    seq_index += codon_length
        
        frame_scores.append(maxSubArraySum(single_scores))

    normalized_score = round(max(frame_scores) / len(seqs[0]), 4)
    return normalized_score


def get_feature_set(window: MultipleSeqAlignment, hexamer_model, reverse: bool = False, tree_path: Optional[str] = None, stdout: bool = True, starts: list[int] = [0], ends: list[int] = [0], sequences: list[int] = [0]) -> Optional[dict]:
    seqs = [str(record.seq).upper().replace("T", "U") for record in window]
    seqs = [seq for seq in seqs if len(seq) > 0]

    if len(seqs) < 2:
        print("Not a valid alignment window - less than 2 non-empty sequences...")
        return None
    
    if reverse:
        seqs = [get_reverse_complement(seq) for seq in seqs]

    try:
        sci = structural_conservation_index(seqs)
        sh_entroy = shannon_entropy(seqs)
        codon_conserv = codon_conservation(window, tree_path)

        seqs = [ungap_sequence(seq) for seq in seqs]

        z_score = z_score_of_mfe(seqs) 
        hexamers = hexamer_score(seqs, hexamer_model)
    except Exception as e:
        print("Could not calculate a full feature set. Point of failure:\n")
        print(e, "\n")
        return None
    #print({
    #    "SCI": sci,
    #    "z-score of MFE": z_score,
    #    "Shannon-entropy": sh_entroy,
    #    "Hexamer Score": hexamers,
    #    "Codon conservation": codon_conserv,
    #})
    return {
        "SCI": sci,
        "z-score of MFE": z_score,
        "Shannon-entropy": sh_entroy,
        "Hexamer Score": hexamers,
        "Codon conservation": codon_conserv,
    }


################################################
#
#               Data generation
#
################################################


def init_feature_data() -> dict[str, list]:
    return  {
        "SCI": [],
        "z-score of MFE": [],
        "Shannon-entropy": [],
        "Hexamer Score": [],
        "Codon conservation": [],
        "Seqs": [],
        "Class": [],
        "Name": [],
    }

def create_training_set(filepath: str, label: str, options: dict) -> pd.DataFrame:
    path = Path(filepath)
    sample_path = path.parent / f"{path.stem}_samples"
    sample_path.mkdir(exist_ok=True)
    name = str(path.stem)

    hexamer_model = load_hexamer_model(options.hexamer_model)

    # Select X samples of N in [2..12] sequences 
    for n_seqs in range(2, 12+1, 1):
        for x in range(1, options.n_samples+1):
            this_sample_file = sample_path / f"{n_seqs}_seqs_sample_{x}.aln"
            select_seqs(input_file=filepath, output_file=this_sample_file, n_records=n_seqs, max_identity=options.max_id, max_attempts=options.sampling_attempts)
    
    # Prepate Filter by tree edit distance  -  if requested
    print("Now preparing background distribution of tree edit distances... ")
    if options.structure_filter:
        reference_distances = []

        # Slice alignments into overlapping windows
        for f in sample_path.glob("*.aln"):
            windows = slice_alignment_into_windows(input_file=f, window_length=options.window_length, step_size=options.slide)

            for window in windows:
                shuffled_window = shuffle_alignment_columns(window)
                reference_distances.append(calculate_mean_distance([ungap_sequence(seq) for seq in shuffled_window]))

        # We assume that tree edit distances follow an approximately gaussian distribution
        std = np.std(a=reference_distances)
        mu = np.mean(a=reference_distances)
        cutoff = mu - 1.645*std

    feature_data = init_feature_data()

    # Slice alignments into overlapping windows
    for f in sample_path.glob("*.aln"):
        # Iff we want structure filtering we now sample each window by it's average tree edit distance vs. the cutoff estimated from the shuffled background distribution
        if options.structure_filter and (label != 1):
            windows = []
            candidates = slice_alignment_into_windows(input_file=f, window_length=options.window_length, step_size=options.slide)

            for candidate in candidates:
                if calculate_mean_distance([ungap_sequence(seq) for seq in candidate]) > cutoff:
                    continue
                windows.append(candidate)
        else:
            windows = slice_alignment_into_windows(input_file=f, window_length=options.window_length, step_size=options.slide)

        # Calculate feature vectors for each window
        for window in windows:
            # Calculate single features
            features = get_feature_set(window, hexamer_model=hexamer_model, 
                                       tree_path=options.tree_path, stdout=False)
            if not features:
                continue
            
            for k, v in features.items():
                feature_data[k].append(v)
            feature_data["Seqs"].append(len(window))
            feature_data["Class"].append(class_dict.get(label))
            feature_data["Name"].append(name)

    for k, v in feature_data.items():
        print(k, len(v))

    return pd.DataFrame(data=feature_data)


def build_data(options: optparse.Values, sissiz_shuffle: bool = False):
    """
    Build training data for the classifier.
        
    Returns:
        Dictionary containing the built training data
    """    
    worktree = {}

    # Helper function to pass alignment to ClustalO - if necessary
    def handle_clustalo(infile, threads, realign=False):
        entry = str(infile)
        outpath = entry.replace(".fa", ".aln")
        if (not Path(outpath).exists()) or realign:
            cline = clustalo_cmdline(infile=entry, outfile=outpath, threads=options.threads)
            subprocess.call(cline)
        return outpath
    
    # ncRNA training data alignment
    if options.noncoding:
        path = Path(options.noncoding)
        for entry in path.glob("*.fa"):
            outpath = handle_clustalo(infile=entry, threads=options.threads, realign=options.realign)
            worktree[outpath] = -1
    
    # coding training data alignment
    if options.coding:
        path = Path(options.coding)
        for entry in path.glob("*.fa"):
            outpath = handle_clustalo(infile=entry, threads=options.threads, realign=options.realign)
            worktree[outpath] = 0
    
    # If a negative set is supplied it will be scanned now
    if options.other:
        path = Path(options.other)
        for entry in path.glob("*.fa"):
            outpath = handle_clustalo(infile=entry, threads=options.threads, realign=options.realign)
            worktree[outpath] = 1

    # Otherwise, one will be auto-generated using SISSIz
    else:
        control_path = Path("Control")
        control_path.mkdir(exist_ok=True)
        generated_files = []

        for f, v in worktree.items():
            outpath = str(control_path / Path(f).name)
            generate_control_alignment(inpath=f, outpath=outpath, sissiz_shuffle=sissiz_shuffle)
            if not is_alignment(outpath):
                print("Failed to create a valid control alignment for %s." % f)
                continue
            generated_files.append(outpath)
        
        for f in generated_files:
            worktree[f] = 1
            
    if len(worktree) == 0:
        print("You supplied a directory, but I could not find readable fasta-files in it. Could you double-check the format?")
        sys.exit()

    def process_single_item(f: str, label: str, options: dict) -> Optional[pd.DataFrame]:
        """Process a single worktree item and return the feature dataframe"""
        try:
            return create_training_set(f, label, options)
        except Exception as e:
            print(f"Failed to create feature data for: {f}")
            return None

    # Initialize the output tsv file
    feature_df = pd.DataFrame(data=init_feature_data())
    feature_df.to_csv(options.out_file, sep="\t", index=False)
    files = []
    labels = []
    n_data_points = 0

    # Iterate over items until 
    for n, (f, label) in enumerate(worktree.items()):
        files.append(f)
        labels.append(label)

        if n+1 % options.threads == 0:
            # Parallelize the feature calculation
            results = Parallel(n_jobs=options.threads, backend="threading", verbose=1)(
                delayed(process_single_item)(f, label, options)
                for f, label in zip(files, labels)
            )
            # Filter out None results (failed processing)
            valid_results = [result for result in results if result is not None]
            files = []
            labels = []

            if valid_results:
                # Combine all results into a single dataframe
                combined_df = pd.concat(valid_results, ignore_index=True)
                n_data_points += len(combined_df)
            
                # Attach to the initialized output file
                combined_df.to_csv(options.out_file, sep="\t", index=False, mode='a', header=False)
            
                print(f"Successfully processed {len(valid_results)} out of {len(worktree)} items.")
            else:
                print("No valid results to write.")
    
    # Process the leftovers
    if len(files) > 0:
        results = Parallel(n_jobs=options.threads, backend="threading", verbose=1)(
                delayed(process_single_item)(f, label, options)
                for f, label in zip(files, labels)
            )
        valid_results = [result for result in results if result is not None]
        if valid_results:
                combined_df = pd.concat(valid_results, ignore_index=True)
                combined_df.to_csv(options.out_file, sep="\t", index=False, mode='a', header=False)
                n_data_points += len(combined_df)

    print(f"Data generation process exited normally. {n_data_points} data points written to {options.out_file}.")



################################################
#
#               Model building
#
################################################


###################### Parameter Optimization ###############################


def get_rf_hyperparameters(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    parameters = {
        "n_estimators": 100,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        }
    if options.optimize == True:
        cv_results = rf_hyperparameter_search(x, y, options)
        cv_results.sort_values(by="rank_test_score", ascending=True, inplace=True)
        for ft in ["param_n_estimators", "param_min_samples_split", "param_min_samples_leaf"]:
            parameters[ft.replace("param_", "")] = cv_results[ft].iloc[0]

        cv_results.to_csv(f"{options.out_file}_rf_hyperparameters.tsv", index=False, sep="\t")
            
    return parameters


def get_svm_hyperparameters(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    parameters = {
            "cost": 2.0,
            "gamma": 1.0,
            }
    if options.optimize == True:
        cv_results = svm_hyperparameter_search(x, y, options)
        cv_results.sort_values(by="rank_test_score", ascending=True, inplace=True)
        parameters["cost"] = cv_results["param_C"].iloc[0]
        parameters["gamma"] = cv_results["param_gamma"].iloc[0]

        cv_results.to_csv(f"{options.out_file}_svm_hyperparameters.tsv", index=False, sep="\t")
                
    return parameters


def get_lr_hyperparameters(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    parameters = {
            "cost": 2.0,
            "tolerance": 1.0,
            }
    if options.optimize == True:
        cv_results = lr_hyperparameter_search(x, y, options)
        cv_results.sort_values(by="rank_test_score", ascending=True, inplace=True)
        parameters["cost"] = cv_results["param_C"].iloc[0]
        parameters["tolerance"] = cv_results["param_tol"].iloc[0]

        cv_results.to_csv(f"{options.out_file}_lr_hyperparameters.tsv", index=False, sep="\t")

    return parameters


def rf_hyperparameter_search(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    m = RandomForestClassifier()
    
    stepsize_estimators = int(round( (options.high_estimators - options.low_estimators) / options.grid_steps , 0))
    stepsize_split = int(round( (options.high_split - options.low_split) / options.grid_steps, 0))
    stepsize_leaf = int(round( (options.high_leaf - options.low_leaf) / options.grid_steps, 0))
    
    parameter_grid = {
            "n_estimators": range(options.low_estimators, options.high_estimators, stepsize_estimators),
            "min_samples_split": range(options.low_split, options.high_split, stepsize_split),
            "min_samples_leaf": range(options.low_leaf, options.high_leaf, stepsize_leaf),
            }
    
    with parallel_backend('threading', n_jobs=options.threads):
        if options.optimizer == "gridsearch":
            searcher = GridSearchCV(m, 
                                    parameter_grid, 
                                    n_jobs=-1, cv=5,
                                    verbose=2)
            searcher.fit(x, y)
            
        elif options.optimizer == "randomwalk":
            searcher = RandomizedSearchCV(m, 
                                        parameter_grid, 
                                        n_jobs=-1, 
                                        cv=5,
                                        n_iter=50,
                                        verbose=2)
            searcher.fit(x, y)
        
    return pd.DataFrame(data=searcher.cv_results_)


def svm_hyperparameter_search(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    m = svm.SVC(verbose=1, probability=True)
    
    if options.logscale:
        parameter_grid = {
            "C": np.logspace(start=options.low_c, stop=options.high_c, num=options.grid_steps, base=options.logbase),
            "gamma": np.logspace(start=options.low_g, stop=options.high_g, num=options.grid_steps, base=options.logbase),
            }
    else:
        stepsize_c = round( (options.high_c - options.low_c) / options.grid_steps , 0)
        stepsize_g = round( (options.high_g - options.low_g) / options.grid_steps , 0)
        
        parameter_grid = {
            "C": np.arange(start=options.low_c, stop=options.high_c, step=stepsize_c),
            "gamma": np.arange(start=options.low_g, stop=options.high_g, step=stepsize_g),
            }
    
    with parallel_backend('threading', n_jobs=options.threads):
        if options.optimizer == "gridsearch":
            searcher = GridSearchCV(m, 
                                    parameter_grid, 
                                    n_jobs=-1, cv=5)
            searcher.fit(x, y)
            
        elif options.optimizer == "randomwalk":
            searcher = RandomizedSearchCV(m, 
                                        parameter_grid, 
                                        n_jobs=-1, 
                                        cv=5,
                                        n_iter=0.25*(len(parameter_grid.get("C")) *len(parameter_grid.get("gamma")) ))
            searcher.fit(x, y)

    return pd.DataFrame(data=searcher.cv_results_)


def lr_hyperparameter_search(x: np.typing.NDArray[np.float64], y: np.typing.NDArray[np.float64], options: optparse.Values) -> dict[str, int]:
    m = LogisticRegression()
    
    if options.logscale:
        parameter_grid = {
            "C": np.logspace(start=options.low_c, stop=options.high_c, num=options.grid_steps, base=options.logbase),
            "tol": np.logspace(start=options.low_g, stop=options.high_g, num=options.grid_steps, base=options.logbase),
            }
    else:
        stepsize_c = round( (options.high_c - options.low_c) / options.grid_steps , 0)
        stepsize_g = round( (options.high_g - options.low_g) / options.grid_steps , 0)
        
        parameter_grid = {
            "C": np.arange(start=options.low_c, stop=options.high_c, step=stepsize_c),
            "tol": np.arange(start=options.low_g, stop=options.high_g, step=stepsize_g),
            }
    
    with parallel_backend('threading', n_jobs=options.threads):
        if options.optimizer == "gridsearch":
            searcher = GridSearchCV(m, 
                                    parameter_grid, 
                                    n_jobs=-1, cv=5)
            searcher.fit(x, y)
            
        elif options.optimizer == "randomwalk":
            searcher = RandomizedSearchCV(m, 
                                        parameter_grid, 
                                        n_jobs=-1, 
                                        cv=5,
                                        n_iter=0.25*(len(parameter_grid.get("C")) *len(parameter_grid.get("tol")) ))
            searcher.fit(x, y)

    return pd.DataFrame(data=searcher.cv_results_)


######################## Create models ##################################

def scale_features(x: np.typing.NDArray[np.float64], threads: int=DEFAULT_THREADS) -> tuple[np.typing.NDArray[np.float64], MinMaxScaler]:
    scaler = MinMaxScaler()
    with parallel_backend('threading', n_jobs=threads):
        X = scaler.fit_transform(X=x)
    return X, scaler


def create_sklearn_svm_model(X: np.typing.NDArray[np.float64], Y: np.typing.NDArray[np.float64], parameters: dict[str, float], filepath: str, threads: int=DEFAULT_THREADS):
    X, scaler = scale_features(X)
    m = svm.SVC(C=parameters.get("cost"), gamma=parameters.get("gamma"), 
                verbose=1, kernel="rbf", probability=True)
    with parallel_backend('threading', n_jobs=threads):
        m.fit(X, Y)
    pickle.dump(m, open(f"{filepath}.model", "wb"))
    pickle.dump(scaler, open(f"{filepath}.scaler", "wb"))


def create_sklearn_rf_model(X: np.typing.NDArray[np.float64], Y: np.typing.NDArray[np.float64], parameters: dict[str, float], filepath: str, threads: int=DEFAULT_THREADS):
    X, scaler = scale_features(X)
    m = RandomForestClassifier(**parameters)
    with parallel_backend('threading', n_jobs=threads):
        m.fit(X, Y)
    pickle.dump(m, open(f"{filepath}.model", "wb"))
    pickle.dump(scaler, open(f"{filepath}.scaler", "wb"))


def create_sklearn_lr_model(X: np.typing.NDArray[np.float64], Y: np.typing.NDArray[np.float64], parameters: dict[str, float], filepath: str, threads: int=DEFAULT_THREADS):
    X, scaler = scale_features(X)
    m = LogisticRegression(C=parameters.get("cost"), tol=parameters.get("tolerance"))
    with parallel_backend('threading', n_jobs=threads):
        m.fit(X, Y)
    pickle.dump(m, open(f"{filepath}.model", "wb"))
    pickle.dump(scaler, open(f"{filepath}.scaler", "wb"))

    

################### Model Core #################################################

def unpack_features(df: pd.DataFrame) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
    if "coding" in df["Class"].unique():
        x = df[["SCI", "z-score of MFE", "Shannon-entropy",]].to_numpy()
    else:    
        x = df[["SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"]].to_numpy()

    y = df["Class"].to_numpy()
    return x, y


def build_model(options: optparse.Values, filename: str) -> None:
    df = pd.read_csv(filename, sep="\t")
    base_length = len(df)
    df.dropna(inplace=True)
    nan_length = len(df)

    if base_length > nan_length:
        print(f"WARNING: Table contained NaN values. {base_length - nan_length} rows were dropped.")

    df["Label"] = [class_dict.get(a) for a in df["Class"]]
    x, y = unpack_features(df)
    
    if options.model == "SVM":
        parameters = get_svm_hyperparameters(x, y, options)
        create_sklearn_svm_model(x, y, parameters, filepath=options.out_file, threads=options.threads)

    elif options.model == "RF":
        parameters = get_rf_hyperparameters(x, y, options)
        create_sklearn_rf_model(x, y, parameters, filepath=options.out_file, threads=options.threads)
        
    elif options.model == "LR":
        parameters = get_lr_hyperparameters(x, y, options)
        create_sklearn_lr_model(x, y, parameters, filepath=options.out_file, threads=options.threads)



################################################
#
#               Windows
#
################################################


def slice_alignment_windows(
    alignment: MultipleSeqAlignment,
    L: int = 120,
    W: int = 40,
    min_identity: float = 0.5,
    max_identity: float = 0.95,
    N: int = 6,
    O: float = 0.8,
    G: float = 0.75
) -> list[MultipleSeqAlignment]:
    """
    Slice a MultipleSeqAlignment into overlapping windows with filtering.

    Parameters:
    - alignment: Input MultipleSeqAlignment object
    - L: Window length (default: 120)
    - W: Step size (default: 40)
    - min_identity: Minimum pairwise identity threshold (default: 0.5)
    - max_identity: Maximum pairwise identity threshold (default: 0.95)
    - N: Maximum number of sequences per window (default: 6)
    - O: Target identity for greedy selection (default: 0.8)
    - G: Maximum gap fraction for reference sequence (default: 0.75)

    Returns:
    - List of MultipleSeqAlignment objects representing windows
    """
    def calculate_pairwise_identity(seq1: str, seq2: str) -> float:
        """Calculate pairwise identity between two sequences, ignoring gaps."""
        valid_positions = [(c1, c2) for c1, c2 in zip(seq1, seq2) if c1 != '-' and c2 != '-']
        if not valid_positions:
            return 0.0
        matches = sum(1 for c1, c2 in valid_positions if c1 == c2)
        return matches / len(valid_positions)

    def get_ungapped_coordinates(sequence: str, start_pos: int, end_pos: int) -> tuple:
        """Get actual sequence coordinates accounting for gaps."""
        ungapped_start = 0
        ungapped_end = 0
        ungapped_pos = 0

        for i, char in enumerate(sequence):
            if char != '-':
                if i == start_pos:
                    ungapped_start = ungapped_pos
                if i == end_pos - 1:
                    ungapped_end = ungapped_pos + 1
                    break
                ungapped_pos += 1
        else:
            # Handle case where end_pos is at the very end
            ungapped_end = ungapped_pos

        return ungapped_start, ungapped_end

    def has_annotations(record: SeqRecord) -> bool:
        """Check if record has MAF-style annotations."""
        return (hasattr(record, 'annotations') and
                'start' in record.annotations and
                'size' in record.annotations and
                'strand' in record.annotations)
    
    def remove_consensus_gaps(sequences: list[str]) -> list[str]:
        """Remove columns that are all gaps."""
        if not sequences:
            return sequences
        
        # Find columns that are not all gaps
        non_gap_columns = []
        for col_idx in range(len(sequences[0])):
            column = [seq[col_idx] for seq in sequences]
            if not all(char == '-' for char in column):
                non_gap_columns.append(col_idx)
        
        # Extract non-gap columns
        filtered_sequences = []
        for seq in sequences:
            filtered_seq = ''.join(seq[col_idx] for col_idx in non_gap_columns)
            filtered_sequences.append(filtered_seq)
        
        return filtered_sequences
    
    def calculate_gap_fraction(sequence: str) -> float:
        """Calculate the fraction of gaps in a sequence."""
        if not sequence:
            return 0.0
        return sequence.count('-') / len(sequence)

    if len(alignment) == 0:
        return []

    alignment_length = alignment.get_alignment_length()
    
    # Drop alignments shorter than target window size
    if alignment_length < L:
        return []
    
    reference_seq = str(alignment[0].seq)
    windows = []

    # Generate windows
    for start in range(0, alignment_length - L + 1, W):
        end = start + L

        # Extract window sequences
        window_records = []
        for record in alignment:
            window_seq = str(record.seq)[start:end]
            window_records.append((record, window_seq))

        # Check reference gap fraction
        reference_window = window_records[0][1]
        if calculate_gap_fraction(reference_window) > G:
            continue
        
        # Filter by pairwise identity to reference
        filtered_records = [window_records[0]]  # Always include reference

        for record, window_seq in window_records[1:]:
            identity = calculate_pairwise_identity(reference_window, window_seq)
            if min_identity <= identity <= max_identity:
                filtered_records.append((record, window_seq))

        # Skip if less than 2 sequences
        if len(filtered_records) < 2:
            continue

        # Greedy selection based on target identity O
        if len(filtered_records) > N:
            # Keep reference, select others greedily
            selected = [filtered_records[0]]
            candidates = filtered_records[1:]

            # Sort by distance to target identity O
            candidates.sort(key=lambda x: abs(calculate_pairwise_identity(reference_window, x[1]) - O))
            selected.extend(candidates[:N-1])
            filtered_records = selected

        # Create new alignment window
        new_records = []
        for record, window_seq in filtered_records:
            # Handle annotations or add coordinates
            if has_annotations(record):
                # Update MAF annotations
                ungapped_start, ungapped_end = get_ungapped_coordinates(str(record.seq), start, end)

                new_record = SeqRecord(
                    Seq(window_seq),
                    id=record.id,
                    description=record.description
                )

                # Update annotations
                new_record.annotations = record.annotations.copy()
                original_start = record.annotations.get('start', 0)
                strand = record.annotations.get('strand', 1)

                if strand == 1:
                    new_record.annotations['start'] = original_start + ungapped_start
                else:
                    # For negative strand, coordinates are more complex
                    original_size = record.annotations.get('size', len(str(record.seq).replace('-', '')))
                    new_record.annotations['start'] = original_start + (original_size - ungapped_end)

                new_record.annotations['size'] = ungapped_end - ungapped_start

            else:
                # Add coordinates to identifier
                ungapped_start, ungapped_end = get_ungapped_coordinates(str(record.seq), start, end)
                new_id = f"{record.id}/{ungapped_start}-{ungapped_end}"

                new_record = SeqRecord(
                    Seq(window_seq),
                    id=new_id,
                    description=record.description
                )

            new_records.append(new_record)

        # Create new MultipleSeqAlignment
        if len(new_record):
            window_alignment = MultipleSeqAlignment(new_records)
            windows.append(window_alignment)

    return windows


def process_windows(options: optparse.Values):
    """
    Slice a MultipleSeqAlignment into overlapping windows with filtering.

    Parameters (as passed on to the slice_alignment_windows function):
    - alignment: Input MultipleSeqAlignment object
    - L: Window length (default: 120)
    - W: Step size (default: 40)
    - min_identity: Minimum pairwise identity threshold (default: 0.5)
    - max_identity: Maximum pairwise identity threshold (default: 0.95)
    - N: Maximum number of sequences per window (default: 6)
    - O: Target identity for greedy selection (default: 0.8)
    - G: Maximum gap fraction for reference sequence (default: 0.75)

    Returns:
        None, result is written to output file.
    """
    in_file = options.in_file
    out_file = options.out_file
    file_handle = open(in_file, 'r')

    # Loading a MAF or a clustal alignment?
    if in_file.endswith(".maf"):
        handle = AlignIO.parse(handle=file_handle, format="maf")
        is_maf = True
    else:
        try:
            handle = AlignIO.parse(handle=file_handle, format="clustal")
            is_maf = False
        except Exception:
            print("Is this supposed to be a MAF alignment? If yes, it should have the .maf file ending.")

    for alignment in handle:
        windows = slice_alignment_windows(
            alignment=alignment,
            L=options.length,
            W=options.slide,
            min_identity=options.min_id,
            max_identity=options.max_id,
            N=options.n_seqs,
            O=options.opt_id,
            G=options.max_gaps,
        )
        if is_maf:
            outfmt = "maf"
        else:
            outfmt = "clustal"

        if len(windows) > 0:
            AlignIO.write(windows, handle=open(out_file, 'a'), format=outfmt)
            print(f"{len(windows)} windows written to {out_file}.")

    print("Process finished.")


################################################
#
#               Prediction
#
################################################


def load_model(filename: str):
    model = pickle.load(open(filename, 'rb'))
    scaler = None
    try: 
        scaler = pickle.load(open(filename.replace("model", "scaler"), 'rb'))
    except Exception:
        print("Could not load corresponding scaler.")

    return model, scaler


def init_result_dict() -> dict[str, list]:
    return {
        "Name": [],
        "SCI": [],
        "z-score of MFE": [],
        "Shannon-entropy": [],
        "Hexamer Score": [],
        "Codon conservation": [],
        "seqs": [],
        "strand": [],
        "prediction": [],
        "probability": [],
    }


def process_single_window(window: str, hexamer_model, options: optparse.Values, reverse: bool=False) -> Optional[pd.DataFrame]:
    """Process a window and return the feature dataframe"""
    try:
        return get_feature_set(window, hexamer_model, reverse=reverse, tree_path=options.tree_path)
    except Exception as e:
        print(f"Failed to create feature data.")
        return None


def process_windows_block(blocks: list[MultipleSeqAlignment], hexamer_model, options: optparse.Values, reverse: bool=False) -> pd.DataFrame:
    results = Parallel(n_jobs=options.threads, backend="multiprocessing", verbose=1)(
        delayed(process_single_window)(window, hexamer_model, options, reverse)
        for window in blocks
    )
    return results


def predict_and_append_to_result_frame(result_data: dict[str, list], model, scaler, block_to_process: list[MultipleSeqAlignment], feature_dict_list: list[dict], is_maf: bool=False, starts: Optional[list[int]]=None, ends: Optional[list[int]]=None):
    for window, features in zip(block_to_process, feature_dict_list): 
        length = len(window)
        if is_maf: 
            start = window[0].annotations["start"]
            end = start + window[0].annotations["size"]
        ref_name = window[0].id

        if features:
            vector = np.zeros(shape=(1,5))
            i = 0
            for k in ("SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"):
                vector[0][i] = features.get(k)
                i += 1
            if scaler:
                vector = scaler.transform(vector)

            if np.isnan(np.sum(vector)):
                print("Caught NaN value in feature vector...")
                continue
            try:
                y = model.predict(vector)[0]
                y_prob = round(max(model.predict_proba(vector)[0]), 4)
            except ValueError:
                print("Invalid input vector -")
                continue
                        
            for k in ("SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"):
                result_data[k].append(features.get(k))
            result_data["Name"].append(ref_name)
            result_data["seqs"].append(length)
            result_data["strand"].append('-')
            result_data["prediction"].append(y)
            result_data["probability"].append(y_prob)
            print(f"Reverse strand prediction: {y} with probability {y_prob}.")

            if is_maf:
                starts.append(start)
                ends.append(end)


def handle_parallel_window_block(block_to_process: list[MultipleSeqAlignment], options: optparse.Values, model, scaler, hexamer_model, is_maf: bool=False, window_count: int = 0):
    # Initialite the result frame to append
    result_data = init_result_dict()

    # Track alignment coordinates (if MAF alignment)
    starts = []
    ends = []

    # Calculate feature vectors for all windows on forward strand
    feature_dict_list = process_windows_block(block_to_process, hexamer_model, options)

    # Process results from forward direction 
    predict_and_append_to_result_frame(
        result_data, 
        model, 
        scaler, 
        block_to_process, 
        feature_dict_list, 
        is_maf,
        starts, ends
    )
    if options.both_strands:
        # Calculate feature vectors for reverse strand
        reverse_feature_dict_list = process_windows_block(block_to_process, hexamer_model, options, reverse=True)

        # Process results for reverse strand
        predict_and_append_to_result_frame(
            result_data, 
            model, 
            scaler, 
            block_to_process, 
            reverse_feature_dict_list, 
            is_maf,
            starts, ends
        )

    # Continously append the result data frame to the output file
    df = pd.DataFrame(data=result_data)
    if is_maf:
        df["start"] = starts
        df["end"] = ends 
        df = df[["Name", "SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation", "seqs", "start", "end", "strand", "prediction", "probability"]]
        starts, ends = [], []
    df.to_csv(options.out_file, sep="\t", index=False, mode="a", header=None)


def svhip_prediction(options: optparse.Values):
    """
    Perform RNA prediction using a previously trained model.
    
    This function loads pre-trained machine learning models and processes input sequences to predict
    identity of aligned transcripts or genome regions. 
    Args:
        options (optparse.Values): Command-line options object containing:
            - model_path: Path to the trained prediction model file
            - hexamer_model: Path to the hexamer scoring model file  
            - in_file: Path to input file containing sequences to analyze
            - out_file: Path to output file for results
    
    Returns:
        None: Results are written to the specified output file
    """    
    # Load the prediction and hexamer score models from file
    model, scaler = load_model(options.model_path)
    hexamer_model = load_hexamer_model(options.hexamer_model)

    file_handle = open(options.in_file, 'r')

    # Loading a MAF or a clustal alignment?
    if options.in_file.endswith(".maf"):
        handle = AlignIO.parse(handle=file_handle, format="maf")
        is_maf = True
    else:
        try:
            handle = AlignIO.parse(handle=file_handle, format="clustal")
            is_maf = False
        except Exception:
            print("Is this supposed to be a MAF alignment? If yes, it should have the .maf file ending.")

    # Initialize the result data frame and outfile
    result_data = init_result_dict()
    df = pd.DataFrame(data=result_data)

    # Track alignment coordinates (if MAF alignment)
    if is_maf:
        df["start"] = []
        df["end"] = []
        df = df[["Name", "SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation", "seqs", "start", "end", "strand", "prediction", "probability"]]
    df.to_csv(options.out_file, sep="\t", index=False)
    window_count = 0
    block_to_process = []

    for window in handle:
        print("\n",window)
        length = len(window)

        if length < 2:
            continue
        window_count += 1
        block_to_process.append(window)
        # Every N windows, process as a block - this done to (A) facilitate parallelization and (B) to prevent loss of data in case of an Exception / loss of power / whatever
        if len(block_to_process) % options.n_blocks == 0:
            handle_parallel_window_block(block_to_process, options, model, scaler, hexamer_model, is_maf, window_count)
            # result_data = init_result_dict()
            block_to_process = []

    # Cleanup any leftover from input stream
    if len(block_to_process) > 0:
        handle_parallel_window_block(block_to_process, options, model, scaler, hexamer_model, is_maf)

    print(f"Output written to: {options.out_file}")

    if options.bed: 
        if not is_maf:
            print("Writing a bed file requires genome coordinates (i.e. a MAF input), though.")
            return 
        df = df[["Name", "start", "end", "prediction", "probability", "strand"]]
        df = df[df["prediction"] != "other"]

        if options.out_file.endswith(".tsv"):
            bedname = options.out_file.replace(".tsv", ".bed")
        else:
            bedname = options.out_file + ".bed"

        df.to_csv(bedname, sep="\t", index=False, header=None)
        print(f"Bed file written to: {bedname}")



################################################
#
#               Core functions
#
################################################


def has_input_options(parser, options) -> bool:
    if not options.coding and not options.noncoding:
        return False
    return True


def has_output_options(parser, options) -> bool:
    if not options.out_file:
        return False
    return True


def is_clustalo_available() -> None:
    if not shutil.which("clustalo"):
        print("Clustal Omega ('clustalo') command not found. Is it installed in your PATH?")
        sys.exit()


def is_sissiz_available() -> bool:
    if not shutil.which("SISSIz"):
        print("SISSIz command not found. Is it installed in your PATH?")
        print("Falling back to manual column shuffling...")
        return False
    
    return True


def data(parser):
    parser.add_option("--noncoding",action="store",type="string", dest="noncoding",help="Input directory, containing fasta file(s) of noncoding sequences (Requires at least 1 of 'noncoding', 'coding').")
    parser.add_option("--coding",action="store",type="string", dest="coding",help="Input directory, containing fasta file(s) of coding sequences (Requires at least 1 of 'noncoding', 'coding').")
    parser.add_option("--other",action="store",type="string", dest="other",help="Input directory, containing fasta file(s) of random (intergenic) sequences (will o auto-generated via randomization of input).")
    parser.add_option("-o","--outfile",action="store",type="string", dest="out_file",help="Name for the output file (Required).")
    parser.add_option("-N", "--negative",action="store",type="string",dest="negative",help="Should a specific negative data set be supplied for data generation? If this field is EMPTY it will be auto-generated based on the data at hand (This will be the desired option for most uses).")
    parser.add_option("-d", "--max-id",action="store",type="float",dest="max_id",default=0.95,help="During data preprocessing, sequences above identity threshold (in percent) will be removed. Default: 95.")
    parser.add_option("-n", "--num-sequences",action="store",type="int",dest="n_seqs",default=100,help="Number of sequences input alignments will be optimized towards. Default: 100.")
    parser.add_option("-l", "--window-length",action="store",type="int",dest="window_length",default=120,help="Length of overlapping windows that alignments will be sliced into. Default: 120.")
    parser.add_option("-w", "--windowslide",action="store",type="int",dest="slide",default=40,help="Controls the step size during alignment slicing and thereby the overlap of each window.")
    parser.add_option("-s", "--samples",action="store",type="int",dest="n_samples",default=10,help="The number of times samples should be drawn from each input alignment and number of sequences - for variation (Default: 10).")
    parser.add_option("-a", "--sample-attempts",action="store",type="int",dest="sampling_attempts",default=1000,help="The number of times sampling should be attempted on a given alignment. Mostly a question of computation time (Default: 1000)")
    # parser.add_option("-g", "--generate-control", action="store",default=True,dest="generate_control",help="Flag to determine if a negative set should be auto-generated (Default: True, if no 'other' data set is supplied).")
    parser.add_option("-c", "--shuffle-control", action="store_true", default=False, dest="shuffle_control",help="Use a simpler column-based shuffling approach instead of SISSIz (Default: False).")
    # parser.add_option("-p", "--positive-label", action="store", default="ncRNA",dest="pos_label",help="The label that should be assigned to the feature vectors generated from the (non-control) input data. Can be CDS (for protein coding sequences) or ncRNA. (Default: ncRNA).")
    parser.add_option("-H", "--hexamer-model", action="store",dest="hexamer_model",default=os.path.join(os.path.join(THIS_DIR, "hexamer_models"), "Human_hexamer.tsv"),help="The Location of the statistical Hexamer model to use. An example file is included with the download as Human_hexamer.tsv, which will be used as a fallback.")
    parser.add_option("-S", "--no-structural-filter", action="store", default=False,dest="structure_filter",help="Set this flag to True if no filtering of alignment windows for statistical significance of structure should occur (Default: False).")
    parser.add_option("-T", "--tree", action="store", default=None, dest="tree_path", help="If an evolutionary tree of species in the alignment is available in Newick format, you can pass it here. Names have to be identical. If None is passed, one will be estimated based on sequences at hand." )
    parser.add_option("-f", "--force-realignment", action="store_true", default=False, dest="realign", help="Force realignment of input sequences even if alignments already exist (Default: False).")
    options, args = parser.parse_args()
    
    if not has_input_options(parser, options):
        print("Output path is missing. Supply it with -o, --output. \n")
        sys.exit()
    if not has_output_options(parser, options):
        parser.print_help()
        print("Output path is missing. Supply it with -o, --output. \n")
        sys.exit()

    is_clustalo_available()

    if options.shuffle_control:
        sissiz_shuffle = False
    else:
        sissiz_shuffle = is_sissiz_available()

    random.seed(options.seed)
    build_data(options, sissiz_shuffle)


def training(parser):
    # Model training specific options
    parser.add_option("-i","--input",action="store",type="string", dest="in_file",help="The input file generated with 'data' (Required).")
    parser.add_option("-o","--outfile",action="store",type="string", dest="out_file",help="Prefix for the model file (Required).")
    # parser.add_option("-S", "--structure",action="store",dest="structure",default=False,help="Flag determining if only secondary structure conservation features should be considered. If True, protein coding features will be included (Default: False).")
    parser.add_option("-M", "--model",action="store",type="string",dest="model",default="RF",help="The model type to be trained. You can choose LR (Logistic regression), SVM (Support vector machine) or RF (Random Forest). (Default: RF)")
    parser.add_option("--optimize-hyperparameters",action="store_true",dest="optimize",default=False,help="Flag: If a parameter optimization should be performed for the ML model. Default is off.")
    parser.add_option("--optimizer",action="store",type="string",dest="optimizer",default="randomwalk",help="Select the optimizer for hyperparameter search. Search will be conducted with 5-fold crossvalidation and either of 'gridsearch' (exhaustive) or 'randomwalk' (default, faster).")
    
    parser.add_option("--low-c",action="store",type="int",dest="low_c",default=1,help="SVM hyperparameter search: Lowest value of the cost (C) parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--high-c",action="store",type="int",dest="high_c",default=100,help="SVM hyperparameter search: Highest value of the cost (C) parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--low-gamma",action="store",type="int",dest="low_g",default=1,help="SVM hyperparameter search: Lowest value of the gamma parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--high-gamma",action="store",type="int",dest="high_g",default=100,help="SVM hyperparameter search: Highest value of the gamma parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--hyperparameter-steps",action="store",type="int",dest="grid_steps",default=10,help="Number of values to try out for EACH hyperparameter. Values will be evenly spaced. Default: 10")
    parser.add_option("--logscale",action="store_true",dest="logscale",default=False,help="Flag that decides if a logarithmic scale should be used for the hyperparameter grid. If set, a log base can be set with --logbase.")
    parser.add_option("--logbase",action="store",type="int",dest="logbase",default=2,help="The logarithmic base if a log scale is used in hyperparameter search. Default: 2.")
    
    # Random Forest specific options 
    parser.add_option("--min-trees",action="store",type="int",dest="low_estimators",default=100,help="Random Forest hyperparameter search: Minimum number of trees before optimization. Does nothing if no RF classifier is used (Default: 100).")
    parser.add_option("--max-trees",action="store",type="int",dest="high_estimators",default=500,help="Random hyperparameter search: Maximum number of trees before optimization. Does nothing if no RF classifier is used (Default: 500).")
    parser.add_option("--min-samples-split",action="store",type="int",dest="low_split",default=2,help="Random Forest hyperparameter search: Minimum number of samples for splitting an internal node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--max-samples-split",action="store",type="int",dest="high_split",default=16,help="Random hyperparameter search:  Maximum number of samples for splitting an internal node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--min-samples-leaf",action="store",type="int",dest="low_leaf",default=1,help="Random Forest hyperparameter search: Minimum number of samples for splitting a leaf node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--max-samples-leaf",action="store",type="int",dest="high_leaf",default=16,help="Random hyperparameter search: Maximum number of samples for splitting a leaf node in the forest. Does nothing if no RF classifier is used.")
    options, args = parser.parse_args()
    
    if not options.in_file:
        print("You did not provide a file with features. Maybe you need to create one with 'svhip.py data'?")
        sys.exit()
    if not options.out_file:
        print("A name prefix for the output model needs to be provided.")
        sys.exit()
    if options.model not in ["RF", "SVM", "LR"]:
        print("Unknown model type. Valid choices are: RF (Random forest), LR (Logistic regression), SVM (Support vector machine).")
        sys.exit()

    random.seed(options.seed)
    filename = options.in_file
    build_model(options, filename)


def svhip_windows(parser):
    parser.add_option("-i","--input",action="store",type="string", dest="in_file",help="The input alignment file (Required).")
    parser.add_option("-o","--outfile",action="store",type="string", dest="out_file",help="Name for the output file (Required).")
    parser.add_option("-l", "--length", action="store", type="int", default=120, dest="length", help="Length of windows to cut into (Default: 120).")
    parser.add_option("-s", "--slide", action="store", type="int", default=80, dest="slide", help="Slide step size for overlapping windows (Default: 80).")
    parser.add_option("--min-id", action="store", type="float", default=0.5, dest="min_id", help="Minimum pairwise identity of sequences to keep (Default: 0.5).")
    parser.add_option("--max-id", action="store", type="float", default=0.95, dest="max_id", help="Maximum pairwise identity of sequences to keep (Default: 0.95).")
    parser.add_option("--opt-id", action="store", type="float", default=0.8, dest="opt_id", help="Pairwise identity of sequences to optimize for (Default: 0.8)")
    parser.add_option("-n", "--num-seqs", action="store", type="int", default=6, dest="n_seqs", help="Maximum number of sequences in the alignment window (Default: 6).")
    parser.add_option("-g", "--max-gaps", action="store", type="float", default=0.75, dest="max_gaps", help="Maximum fraction of gaps in reference sequence (Default: 0.75).")
    options, args = parser.parse_args()
    
    if not options.in_file:
        print("Please add an input alignment with -i.")
        sys.exit()
    if not options.out_file:
        print("Missing output file argument -o.")
        sys.exit()
    
    random.seed(options.seed)
    process_windows(options)


def predict(parser):
    # Test / prediction specific options
    parser.add_option("-i","--input",action="store",type="string", dest="in_file",help="The input alignment file (Required).")
    parser.add_option("-o","--outfile",action="store",type="string", dest="out_file",help="Name for the output file (Required).")
    parser.add_option("-M", "--model-path",action="store",type="string",dest="model_path",default="",help="The path to the trained model (Required).")
    parser.add_option("-T", "--tree", action="store", default=None, dest="tree_path", help="If an evolutionary tree of species in the alignment is available in Newick format, you can pass it here. Names have to be identical. If None is passed, one will be estimated based on sequences at hand." )
    parser.add_option("-H", "--hexamer-model", action="store",dest="hexamer_model",default=os.path.join(os.path.join(THIS_DIR, "hexamer_models"), "Human_hexamer.tsv"),help="The Location of the statistical Hexamer model to use. An example file is included with the download as Human_hexamer.tsv, which will be used as a fallback.")
    parser.add_option("--both-strands", action="store_true", default=False, dest="both_strands", help="Should forward and reverse strands be screened? Default: False.")
    # parser.add_option("--column-label",action="store",type="string",dest="prediction_label",default="Prediction",help="Column name for the prediction in the output.")
    # parser.add_option("--structure", action="store", dest="ncrna", default=False, help="Set to True if only features for conservation of secondary structure should be used. Depends on type of model.")
    parser.add_option("--bed", action="store_true", dest="bed", default=False, help="Set to True if you want overlapping annotations to be merged and written as BED file. IMPORTANT: Requires genomic coordinates in input - i.e. MAF fomat.")
    parser.add_option("--windows-per-block", action="store", type="int", default=50, dest="n_blocks", help="Define how many windows are processed as a parallelized block and written to output before loading the next block (Default: 50).")
    # parser.add_option("--probability", action="store", dest="probability", default=False, help="Set to True if you want class probabilities assigned in final output. Warning: Requires model to be trained with probability flag.")
    options, args = parser.parse_args()
    
    if not options.in_file:
        print("Please add an input alignment with -i.")
        sys.exit()
    if not options.out_file:
        print("Missing output file argument -o.")
        sys.exit()
    if not options.out_file:
        print("Missing model file argument -M.")
        sys.exit()

    random.seed(options.seed)
    svhip_prediction(options)


def hexamer_calibrator(parser):
    parser.add_option("-c","--coding",action="store", dest="in_coding", help="Should point towards a Fasta-file of coding transcripts (transcripts HAVE to be in-frame).")
    parser.add_option("-n","--noncoding",action="store", dest="in_noncoding", help="Fasta-file with transcripts or sequences that are NOT coding.")
    parser.add_option("-o","--outfile",action="store", dest="out_file", help="Name or path of the file to write. Will be a tab-delimited text file (.tsv).")
    options, args = parser.parse_args()
    hexamer_model_calibration(options)


################################################
#
#               Main
#
################################################


def main():
    args = sys.argv
    
    # Setup a command line parser, replacing the outdated "currysoup" module
    usage = "\n%prog  [options]"
    parser = OptionParser(usage,version="%prog " + __version__)
    parser.add_option("--threads",action="store",type="int", dest="threads", default=DEFAULT_THREADS, help="Number of threads to allocate (Default: max(CPU_COUNT-1, 1)).")
    parser.add_option("--seed",action="store",type="int", dest="seed", default=RANDOM_SEED, help="Integer seed to control certain randomized behavior (in particular sequence and non-SISSIz alignment shuffling). Optional.")

    t = time.time()
    
    if len(args) < 2:
        print("Usage: svhip [Task] [Options] with Task being one of 'data', 'train', 'windows', 'predict','hexcalibrate'.")
        sys.exit()
    if args[1] == "data":
        data(parser)
    elif args[1] == "train":
        training(parser)
    elif args[1] == "windows":
        svhip_windows(parser)
    elif args[1] == "predict":
        predict(parser)
    elif args[1] == "hexcalibrate":
        hexamer_calibrator(parser)
    
    else:
        print("Usage: svhip [Task] [Options] with Task being one of 'data', 'train', 'windows', 'predict','hexcalibrate'.")
        sys.exit()

    print("\nProgram ran for %s seconds." % round(time.time() - t, 2))
    

if __name__=='__main__':
    main()

