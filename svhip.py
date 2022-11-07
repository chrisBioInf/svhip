#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 23:15:23 2022

@author: christopher
"""


__author__ = "Christopher Klapproth"
__institution__= "University Leipzig"
__credits__ = []
__license__ = "GPLv2"
__version__="1.0.0"
__maintainer__ = "Christopher Klapproth"
__email__ = "christopher@bioinf.uni-leipzig.de"
__status__ = "Development"


import os
import sys
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
from itertools import product
from math import factorial

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from Bio import AlignIO, SeqIO, Phylo
from Bio.Align.Applications import ClustalwCommandline
from Bio.Align.AlignInfo import SummaryInfo
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.Data import CodonTable
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from itertools import combinations, product
import blosum
import RNA as vienna


#########################Default parameters#############################

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RNAz_DIR = os.path.join(THIS_DIR, "RNAz_toolkit")
hexamer_backup = os.path.join(os.path.join(THIS_DIR, "hexamer_models"), "Human_hexamer.tsv")

class_dict = {
    1 : "OTHER",
    -1: "RNA",
    0 : "CDS",
    "OTHER": 1,
    "ncRNA": -1,
    "RNA": -1,
    "CDS": 0,
    }

rng = np.random.default_rng()

nucleotides = ["A", "C", "G", "U", "-"]
dinucleotides = ["AA", "AC", "AG", "AU",
                 "CA", "CC", "CG", "CU",
                 "GA", "GC", "GG", "GU",
                 "UA", "UC", "UG", "UU"
                 ]

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

def is_fasta(filename):
    with open(filename, "r") as handle:
        fasta = SeqIO.parse(handle, "fasta")
        return any(fasta) 
    
    
def is_alignment(filename):
    alignment = AlignIO.read(filename, "clustal")
    return any(alignment)


def ungap_sequence(seq):
    return re.sub(r'-', "", seq)


def filter_windows(filename, edit_distances, cutoff):
    alignments = AlignIO.parse(filename, "clustal")
    buffer_file = filename + "_tmp"
    writer = AlignIO.ClustalIO.ClustalWriter(open(buffer_file, "a"))
    alignment_count = 0
        
    for a in alignments:
        alignment_count += 1
        if edit_distances.get(alignment_count) > cutoff:
            continue
        writer.write_alignment(a)
            
    os.remove(filename)
    os.rename(buffer_file, filename)
    

def get_consensus_sequence(alignment):
    summary = SummaryInfo(alignment)
    return summary.gap_consensus(threshold=0.4).replace("X", "-")


###############################################
#
#               Altschulerickson Algorithm
#
###############################################


def multiset_coefficient(num_elements, cardinality):
    """Return multiset coefficient.

      num_elements      num_elements + cardinality - 1
    ((            )) = (                              )
      cardinality                cardinality
    """
    if num_elements < 1:
        raise ValueError('num_elements has to be an int >= 1')
    ret = factorial(num_elements+cardinality-1)
    ret //= factorial(cardinality)
    ret //= factorial(num_elements-1)
    return ret

def multiset_multiplicities(num_elements, cardinality):
    """Generator function for element multiplicities in a multiset.

    Arguments:
    - num_elements -- number of different elements the multisets are chosen from
    - cardinality -- cardinality of the multisets
    """
    if cardinality < 0:
        raise ValueError('expected cardinality >= 0')
    if num_elements == 1:
        yield (cardinality,)
    elif num_elements > 1:
        for count in range(cardinality+1):
            for other in multiset_multiplicities(num_elements-1,
                                                 cardinality-count):
                yield (count,) + other
    else:
        raise ValueError('expected num_elements >= 1')

def multinomial_coefficient(*args):
    """Return multinomial coefficient.

     sum(k_1,...,k_m)!
    (                 )
      k_1!k_2!...k_m!
    """
    if len(args) == 0:
        raise TypeError('expected at least one argument')
    ret = factorial(sum(args))
    for arg in args:
        ret //= factorial(arg)
    return ret

"""Custom error classes."""

class SequenceTypeError(Exception):
    """Raised if a sequence type is neither RNA or DNA.

    RNA and DNA are string constants defined in zscore/literals.py.
    """
    pass

class DefaultValueError(Exception):
    """Raised if a value in zscore/defaults.py fails a validity check."""
    pass

# Type of nucleic acid.
DNA = 'DNA'
RNA = 'RNA'

# Nucleotide symbols.
A = 'A'
C = 'C'
G = 'G'
T = 'T'
U = 'U'

# DNA nucleotides.
DNA_NUCLEOTIDES = (A, C, G, T)

# RNA nucleotides.
RNA_NUCLEOTIDES = (A, C, G, U)

# Dinucleotides.
AA = 'AA'
AC = 'AC'
AG = 'AG'
AT = 'AT'
AU = 'AU'
CA = 'CA'
CC = 'CC'
CG = 'CG'
CT = 'CT'
CU = 'CU'
GA = 'GA'
GC = 'GC'
GG = 'GG'
GT = 'GT'
GU = 'GU'
TA = 'TA'
TC = 'TC'
TG = 'TG'
TT = 'TT'
UA = 'UA'
UC = 'UC'
UG = 'UG'
UU = 'UU'

# DNA dinucleotides.
DNA_DINUCLEOTIDES = (AA, AC, AG, AT, CA, CC, CG, CT, GA, GC, GG, GT, TA, TC, TG, TT)

# RNA dinucleotides.
RNA_DINUCLEOTIDES = (AA, AC, AG, AU, CA, CC, CG, CU, GA, GC, GG, GU, UA, UC, UG, UU)

from re import (
    escape,
    findall
)

def doublet_counts(sequence, doublets):
    """Return dict of observed doublet counts in given sequence."""
    counts = dict()
    for d in doublets:
        counts[d] = len(findall('(?={0})'.format(escape(d)), sequence))
    return counts

class ZGraphs():
    """
    Container for valid Z graphs.

    Arguments:
    -- z_vertices: elements in sequence not equal to last element in sequence
    -- last: last element in sequence
    """

    def __init__(self, vertices, last):
        if not (2 <= len(vertices) <= 4):
            raise TypeError('expected sequence of 2, 3 or 4 unique elements '
                            'but got {0} {1}'.format(
                                len(vertices),sorted(list(vertices))))
        if last not in vertices:
            raise ValueError('expected last to be in vertices')
        self.vertices = vertices
        self.last = last
        self.nonlast_vertices = sorted(set(vertices).difference(self.last))
        self.valid = self._create_valid(self.vertices, self.nonlast_vertices,
                                        self.last)

    def _create_valid(self, vertices, nonlast_vertices, last):
        z_graphs = []
        for ends in product(*(vertices for _ in range(len(nonlast_vertices)))):
            z_graph = dict(e for e in zip(nonlast_vertices, ends))
            if self._valid(z_graph, last):
                z_graphs.append(z_graph)
        return tuple(z_graphs)

    def _valid(self, z_graph, last):
        if len(z_graph) > 3:
            raise TypeError('expected z_graph to be a dict representing at '
                            'most 3 last edges')
        connected = True
        for tail, head in z_graph.items():
            if head != last:
                head2 = z_graph.get(head)
                if head2 != last:
                    head3 = z_graph.get(head2)
                    if head3 != last:
                        connected = False
                        break
        return connected

    def check(self, z_graph):
        """Returns true if a Z graph is valid."""
        return z_graph in self.valid


class DoubletGraph:
    """
    Doublet Graph of a sequence.

    A doublet of a sequence is any pair of successive elements in the sequence.
    A doublet graph of a sequence is an ordered directed multigraph.
    In such a graph the vertices are the unique elements of the sequence and
    the directed edges are the doublets in that sequence with direction from
    sequence start to sequence end.

    Arguments:
    -- sequence: string representation of the sequence
    """

    def __init__(self, rng, sequence):
        self.rng = rng
        try:
            self.first = sequence[0]
        except IndexError:
            raise TypeError('expected seqeunce to be a non-empty string')
        self.last = sequence[-1]
        self.edges = tuple(e for e in zip(sequence, sequence[1:]))
        self.vertices = sorted(set(sequence))
        self.z_graphs = ZGraphs(self.vertices, self.last)
        self.nonlast_vertices = self.z_graphs.nonlast_vertices
        self.edgelists = self._create_edgelists(self.vertices, self.edges)
        self.edgemultiplicities = self._create_edgemultiplicites(
            self.vertices, self.edgelists)
        self.possible_z_graphs = self._create_possible_z_graphs()

    def _create_edgelists(self, vertices, edges):
        ret = dict((v,[]) for v in vertices)
        for start, end in edges:
            ret.get(start).append(end)
        for v, l in ret.items():
            ret[v] = tuple(l)
        return ret

    def _create_edgemultiplicites(self, vertices, edgelists):
        ret = dict()
        for v, l in edgelists.items():
            m = dict((v, l.count(v)) for v in vertices)
            ret[v] = m
        return ret

    def _create_possible_z_graphs(self):
        ret = []
        for z in self.z_graphs.valid:
            possible = True
            for v in self.nonlast_vertices:
                if z.get(v) not in self.edgelists.get(v):
                    possible = False
                    break
            if possible:
                ret.append(z)
        return ret

    def num_permutations(self):
        total = 0
        for z in self.possible_z_graphs:
            subtotal = 1
            for v in z:
                last = z.get(v)
                multiplicities = self.edgemultiplicities.get(v).copy()
                multiplicities[last] -= 1
                subtotal *= multinomial_coefficient(*multiplicities.values())
            subtotal *= multinomial_coefficient(
                *self.edgemultiplicities.get(self.last).values())
            total += subtotal
        return total

    def random_z_graph(self):
        while(True):
            ret = dict()
            for start in self.nonlast_vertices:
                end = self.rng.choice(self.edgelists.get(start))
                ret[start] = end
            if self.z_graphs.check(ret):
                return ret


class DPPermutations:
    """
    Doublet preserving permutations of a sequence.

    The cardinality of the alphabet must be <= 4 as is the case in DNA or RNA.

    Arguments:
    -- sequence: string representation of the sequence.
    """

    def __init__(self,
                 rng,
                 sequence,
                 sequence_type,
                 sequence_length,
                 di_features):
        self.rng = rng
        self.sequence = sequence
        self.length = sequence_length
        self.sequence_type = sequence_type
        if self.sequence_type == RNA:
            self.DINUCLEOTIDES = RNA_DINUCLEOTIDES
        elif self.sequence_type == DNA:
            self.DINUCLEOTIDES = DNA_DINUCLEOTIDES
        else:
            raise SequenceTypeError
        if self.length <= 3 or len(set(sequence)) <= 1:
            self.trivial = True
        else:
            self.trivial = False
            self.g = DoubletGraph(rng=self.rng, sequence=self.sequence)
            if di_features == None:
                self.dcounts = doublet_counts(self.sequence, self.DINUCLEOTIDES)
            else:
                self.dcounts = di_features

    def num_permutations(self):
        return 1 if self.trivial else self.g.num_permutations()
            
    def shuffle(self):
        if self.trivial:
            return self.sequence
        else:
            last_edges = self.g.random_z_graph()
            edgelists = dict((v, list(l)) for v,l in self.g.edgelists.items())
            for start, end in last_edges.items():
                edgelists.get(start).remove(end)
            for v in self.g.vertices:
                self.rng.shuffle(edgelists.get(v))
            for start, end in last_edges.items():
                edgelists.get(start).append(end)
            ret = self.g.first
            for _ in range(self.length - 1):
                ret += edgelists.get(ret[-1]).pop(0)
            if self.dcounts != doublet_counts(ret, self.DINUCLEOTIDES):
                raise RuntimeError(
                    'Something went very wrong! '
                    'Shuffled sequence is not doublet preserving.')
            return ret


################################################
#
#               RNAz/SISSIz helper functions
#
################################################


def shuffle_control(inpath, outpath):
    with open(outpath, "w+") as outf:
        cmd = "perl %s %s" % (os.path.join(RNAz_DIR, "rnazRandomizeAln.pl"), inpath)
        call(shlex.split(cmd), stdout=open(outpath, "w"))
    return 0


def generate_control_alignment(inpath, outpath, shuffle=False):
    if shuffle:
        shuffle_control(inpath, outpath)
        return 0
    with open(outpath, "w+") as outf:
        cmd = "SISSIz -n 1 -s --flanks 1750 %s" % inpath
        call(shlex.split(cmd), stdout=open(outpath, "w"))
    return 0


def select_seqs(filepath, identity, seqs):
    tmp_file = filepath+"_tmp"
    script_path = os.path.join(RNAz_DIR, ('rnazSelectSeqs.pl'))
    cmd = "perl %s --max-id=%s --num-seqs=%s %s" % (script_path, 
                                                    identity, 
                                                    seqs, 
                                                    filepath) 
    call(shlex.split(cmd), stdout=open(tmp_file, "w"))
    
    if is_alignment(tmp_file):
        os.remove(filepath)
        os.rename(tmp_file, filepath)
    else:    
        print("Careful: Sequence filtering failed for some reason. We will proceed with the full alignment.")


############################# Slice Alignments ###########################
"""
For {2..12} sequences create 10 alignments with random average pairwise 
identity in interval d = [50..95].
Note: Max. deviation in seq. length should be 65%, even though that still 
seems pretty excessive / previous screening if necessary. --TODO

"""
def alignment_windows(filepath, options):
    windows_directory = filepath+"_windows"
    
    if not os.path.isdir(windows_directory):
        os.mkdir(windows_directory)
    
    script_path = os.path.join(RNAz_DIR, ('rnazWindow.pl'))
    outfiles = []
    
    for n in range(2, 12+1):
        for i in range(0, options.n_windows):
            avg_identity = np.random.randint(low=50, high=95)
            align_id = "align_%s_%s" % (n, i+1)
            outpath = os.path.join(windows_directory, align_id)
            cmd =  'perl %s --min-id=50 --opt-id=%s --window=%s --slide=%s --no-reference --min-seqs=%s --max-seqs=%s --num-samples=1 %s' % (
                script_path, 
                avg_identity,
                options.window_length,
                options.slide,
                n,
                n,
                filepath
                ) 
            
            call(shlex.split(cmd), stdout=open(outpath, "w"))
            if check_file(outpath):
                outfiles.append(outpath)
        
    return outfiles


############# Shuffle alignment with structure conservation ############

def shuffle_alignment(filepath):
    outpath = filepath+"_shuffled"
    script_path = os.path.join(RNAz_DIR, 'rnazRandomizeAln.pl')
    cmd = "perl %s --level=0 %s" % (script_path, filepath)
    call(shlex.split(cmd), stdout=open(outpath, "w"))
    return outpath


##############################calculate parameters######################


def calculate_ShannonEntropy(seqs):
    '''
    Entropy is approximated by observing the total entropy of all individual columns using
    the relative frequency of nucleotides per column. 
    '''
    N = 0.0
    N += len(seqs[0])
    svalues = []
    for i in range(0, len(str(seqs[0]))):
        nA, nU, nG, nC, gap, none = 0, 0, 0, 0, 0, 0
        nucs_dict = {
            "A": nA,
            "U": nU,
            "T": nU,
            "G": nG,
            "C": nC,
            "-": gap,
            "\n": none,
            " ": none
        }
        for a in range(0, len(seqs)):
            if i < len(seqs[a]):
                literal = nucs_dict.get(seqs[a][i])
                if literal is not None:
                    literal_count = literal + 1
                    nucs_dict[seqs[a][i]] = literal_count
                else:
                    pass
        sA = float(nucs_dict.get('A')/N)*np.log2(float(nucs_dict.get('A')/N))
        sU = float(nucs_dict.get('U')/N)*np.log2(float(nucs_dict.get('U')/N))
        sG = float(nucs_dict.get('G')/N)*np.log2(float(nucs_dict.get('G')/N))
        sC = float(nucs_dict.get('C')/N)*np.log2(float(nucs_dict.get('C')/N))
        svalues.append(sum([sA, sG, sC, sU]))
        
    return -float(1/N)* sum(svalues)


############################ Check if a produced file is empty ###########

def check_file(filename):
    if not os.path.isfile(filename):
        return False
    if not os.path.getsize(filename) > 0:
        os.remove(filename)
        return False
    
    return True


################################################
#   
#               Tree edit distance handling
#
################################################

def ungap_sequence(seq):
    return re.sub(r'-', "", seq)


def shuffle_alignment_columns(seqs):
    indices = list(range(0, min([len(seq) for seq in seqs])))
    shuffle(indices)
    
    for i in range(0, len(seqs)):
        seqs[i] = str().join([seqs[i][n] for n in indices])
        
    return seqs


def fold_rna(seq):
    fc = vienna.fold_compound(seq)
    mfe_struct, mfe = fc.mfe()
    seq_ex = vienna.expand_Full(mfe_struct)
    return (seq_ex, mfe, mfe_struct)

    
def calculate_tree_edit_dist(seq1, seq2):
    seq_struc1, seq_struc2 = fold_rna(seq1)[0], fold_rna(seq2)[0]
    seq_tree1, seq_tree2 = vienna.make_tree(seq_struc1), vienna.make_tree(seq_struc2)
    return vienna.tree_edit_distance(seq_tree1, seq_tree2)


def calculate_mean_distance(seq_list):
    '''
    In case there are empty alignments created by RNAzWindow, which happens
    depending on set parameters for that run, 0 is returned. Zero-values are
    later removed from distance vector.
    Note: "Real" zero-alignments i.e. all sequences are equal can only be
    created if all input sequences in total are equal, so that should not be an
    issue.
    '''
    if len(seq_list) < 2:
        return 0
        
    distance_vector =[]
    for i in range(0, len(seq_list)-1):
        for a in range(i+1, len(seq_list)):    
            seq_struc1, seq_struc2 = fold_rna(seq_list[i])[0], fold_rna(seq_list[a])[0]
            seq_tree1, seq_tree2 = vienna.make_tree(seq_struc1), vienna.make_tree(seq_struc2)
            distance_vector.append(vienna.tree_edit_distance(seq_tree1, seq_tree2))
            
    return np.mean(a=distance_vector)




################################################
#
#               Plotting functions
#
################################################


limit_dict = {
    "SCI": (-0.1, 1.1),
    "z-score of MFE": (-10, 1),
    "Shannon-entropy": (-2, 2), 
    }

def plot_tree_edit_distribution(vector, edit_distances, out_path,):
    name = os.path.join(out_path, "tree_edit_distances")
    data = {
        "tree edit distance": vector + edit_distances,
        "class": ["control"]*len(vector) + ["input"]*len(edit_distances)
        }
    if len(data.get("tree edit distance")) < 5:
        return None
    print("%s.pdf" % name)
    print(data)
    sns.set_style("dark")
    sns.displot(data=data,x="tree edit distance", hue="class", kde=True,)
    plt.xlabel("tree edit distance")
    plt.ylabel("count")
    plt.savefig("%s.pdf" % name, dpi=300, bbox_inches="tight")
    plt.savefig("%s.svg" % name, dpi=300, bbox_inches="tight")
    plt.clf()


def feature_boxplot(df, out_path):
    # TODO: There is an easier way to "extend" the dataframe to a long format somewhere...
    values = [x for x in pd.concat([df["SCI"], df["z-score of MFE"], df["Shannon-entropy"], ])]
    classes = list(df["Class"])*3
    features = ["SCI"]*len(df["SCI"]) + ["z-score of MFE"]*len(df["z-score of MFE"]) + ["Shannon-entropy"]*len(df["Shannon-entropy"]) 
    long_df = pd.DataFrame(data={
        "value": values,
        "Class": classes,
        "feature": features,
        })
    sns.boxplot(data=long_df, x="feature", y="value", hue="Class")
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.savefig(out_path +"/features_boxplot.svg", dpi=300, bbox_inches="tight")
    plt.savefig(out_path +"/features_boxplot.pdf", dpi=300, bbox_inches="tight")
    plt.clf()

    
def plot_feature_distribution(df, out_path):
    # df["Class"] = [class_dict.get(a) for a in df["Class"]]
    df.index = list(range(0, len(df)))
    
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    sns.set_theme(style="ticks")
    sns.jointplot(data=df, x="SCI", y="z-score of MFE", hue="Class")  
    plt.savefig(out_path + "/2D_features.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_path + "/2D_features.svg", dpi=300, bbox_inches="tight")
    plt.clf()
    for ft in ["SCI", "z-score of MFE", "Shannon-entropy"]:
        g = sns.kdeplot(data=df, x=ft, hue="Class")
        plt.xlim(limit_dict.get(ft))
        plt.savefig(out_path + "/%s_distribution.svg" % ft.replace(' ','_').replace('-',''), dpi=300, bbox_inches="tight")
        plt.savefig(out_path + "/%s_distribution.pdf" % ft.replace(' ','_').replace('-',''), dpi=300, bbox_inches="tight")
        plt.clf()
        
    feature_boxplot(df, out_path)

    # pd.plotting.scatter_matrix(df[["SCI", "z-score of MFE", "Shannon-entropy"]], alpha=0.5)
    sns.pairplot(df[["SCI", "z-score of MFE", "Shannon-entropy", "Class"]], hue="Class")
    plt.yticks(rotation=90)
    plt.savefig(out_path +"/scatter_matrix.svg", dpi=300, bbox_inches="tight")
    plt.savefig(out_path +"/scatter_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.clf()

    corr = df[["SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"]].corr()
    corr.to_csv(out_path +"/pearson_correlation.txt")    
    print("Pearson correlation matrix of generated features: \n")
    print(corr)


################################################
#
#               Data generation
#
################################################


############################Core function###############################
"""
The hub through which all above functions are called. Core function is 
called by main() once input cmd line is parsed. For options please see
section 'main function call' below.
"""

class_dict = {
    1 : "OTHER",
    -1: "RNA",
    0 : "CDS",
    "OTHER": 1,
    "ncRNA": -1,
    "RNA": -1,
    "CDS": 0,
    }


def create_training_set(filepath, category, options):
    
    def read_windows(filename):
        alignments = AlignIO.parse(filename, "clustal")
        align_dict = {}
        alignment_count = 0
        for a in alignments:
            alignment_count += 1
            seqs = [str(record.seq) for record in a._records]
            align_dict[alignment_count] = seqs 
            
        return align_dict

    # Filter sequences by their identity
    select_seqs(filepath, options.max_id, options.n_seqs)
    
    # Slice alignments into overlapping alignment windows
    windows = alignment_windows(filepath, options)

    # Filter alignment windows by structural conservation, using a tree edit distance approximation
    # At first we need a baseline distribution of tree edit distances using shuffled alignments
    reference_distances = []
    edits = []
    
    if (category == -1) and options.structure_filter:
        for window in windows:
            align_dict = read_windows(window)
            for (k, seqs) in align_dict.items():
                shuffled_alignment = shuffle_alignment_columns(seqs)
                reference_distances.append(calculate_mean_distance([seq for seq in shuffled_alignment]))
                
        # We assume that tree edit distances follow an approximately gaussian distribution
        std = np.std(a=reference_distances)
        mu = np.mean(a=reference_distances)
        cutoff = mu - 1.645*std
    
        for window in windows:
            align_dict = read_windows(window)
            edit_distances = {}
            for (k, seqs) in align_dict.items():
                edit_distances[k] = calculate_mean_distance([ungap_sequence(seq) for seq in seqs])
            if category == -1:
                filter_windows(window, edit_distances, cutoff=cutoff)
            edits += list(edit_distances.values())
        
        # Plot the density of the control distribution
        if (category == -1):
            out_path = options.out_file + "_report"
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            plot_tree_edit_distribution(reference_distances, edits, out_path)
        
    # Calculate feature vectors
    scis = []
    zs = []
    entropies = []
    hexamers = []
    tree_edits_ = []
    codon_c = []
    
    for i in range(0, len(windows)):
        w = windows[i]
        print("Calculating features:", i, "/", len(windows))
        sci, zscore, entropy, hex_score, codon_conservation = get_feature_set(w, hexamer_model=options.hexamer_model, tree_path=options.tree_path)

        scis += sci
        zs += zscore
        entropies += entropy
        hexamers += hex_score
        codon_c += codon_conservation
    
    df = pd.DataFrame(data={
        "SCI": scis,
        "z-score of MFE": zs,
        "Shannon-entropy": entropies,
        "Hexamer Score": hexamers,
        "Codon conservation": codon_c,
        "Class": [category]*len(scis)
        })

    return df
   
    
def create_report(df, options):
    out_path = options.out_file + "_report"
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    
    plot_feature_distribution(df, out_path)
    df.to_csv(out_path +"/feature_vectors.csv", sep="\t")

    
######################### main function call for 2 classes #################


def build_data(options, filename):
    # Should negative sets be auto-generated for all instances? 
    if not options.negative and options.generate_control: 
        print("A negative training set will be auto-generated.")
        
    # Core function is called with set options:
    if not os.path.isdir(filename):
        filelist = [filename]
    else:
        filelist = [os.path.join(filename, f) for f in os.listdir(filename) if os.path.isfile(os.path.join(filename, f))]
        
    worktree = {}
    pos_label = class_dict.get(options.pos_label)
    if pos_label not in [0, -1]:
        print("Provided an invalid positive label. Choose from CDS, ncRNA (Default: ncRNA).")
        sys.exit()
    
    # Positive training instance will need to be supplied. Alignment happens now.
    for entry in filelist:
        inpath = entry
        if entry.endswith(".aln"):
            outpath = inpath
        else:
            outpath = entry.replace(".fasta", ".aln").replace(".fa", ".aln")
            print("Now aligning: %s" % entry)
            cline = ClustalwCommandline("clustalw2", infile=inpath, outfile=outpath)
            cline()
        worktree[outpath] = pos_label
    print('End of directory reached. Positive instance data sets found: ' + str(len(worktree)))
    # If a negative set is supplied it will be scanned now
    if options.negative:
        if not os.path.isdir(options.negative):
            filelist = [options.negative]
        else:
            filelist = [f for f in os.listdir(options.negative) if os.path.isfile(os.path.join(options.negative, f))]
            for entry in filelist:
                outpath = os.path.join(options.negative, entry + ".aln")
                inpath = os.path.join(options.negative, entry)
                if entry.endswith(".aln"):
                    outpath = inpath
                else:
                    print("Now aligning: %s" % entry)
                    cline = ClustalwCommandline("clustalw2", infile=inpath, outfile=outpath)
                    cline()
                worktree[outpath] = 1

    # Otherwise, one will be auto-generated using SISSIz
    if options.generate_control:
        negativeset = filename+"_negative"
        dummies = []
        if not os.path.isdir(negativeset):
            os.mkdir(negativeset)
        for (f, c) in worktree.items():
            outpath = os.path.join(negativeset, os.path.basename(f))
            generate_control_alignment(inpath=f, outpath=outpath, shuffle=options.shuffle_control)
            if not is_alignment(outpath):
                print("Failed to create a valid control alignment for %s." % f)
                continue
            dummies.append(outpath)
        for f in dummies:
            worktree[f] = 1
            
    if len(worktree) == 0:
        print("You supplied a directory, but I could not find readable fasta-files in it. Could you double-check the format?")
        sys.exit()
    
    print("Worktree generated:")
    for (k, v) in worktree.items():
        print(k, "-->", class_dict.get(v))
    vectortable = pd.concat([create_training_set(f, c, options) for (f, c) in list(worktree.items())])
    print("Data generation process exited normally.")
    
    vectortable["Class"] = [class_dict.get(a) for a in vectortable["Class"]]
    create_report(df=vectortable, options=options)
    if options.out_file.endswith(".tsv"):
        vectortable.to_csv(options.out_file, sep="\t", index=False)
        return vectortable, options.out_file
        
    vectortable.to_csv(options.out_file+"_trainingdata.tsv", sep="\t", index=False)
    return vectortable, options.out_file+"_trainingdata.tsv"


######################### main function call for 3 classes #################


def build_three_way_data(options):
    # Core function is called with set options:
    if not os.path.isdir(options.ncrna) and not os.path.isdir(options.protein):
        print("Supplied input path is not a directory. --input should point to a directory with fasta files.")
        sys.exit()

    worktree = {}
    
    # ncRNA training instance will need to be supplied. Alignment happens now.
    filelist = [f for f in os.listdir(options.ncrna) if os.path.isfile(os.path.join(options.ncrna, f))]
    for entry in filelist:
        outpath = os.path.join(options.ncrna, entry + ".aln")
        inpath = os.path.join(options.ncrna, entry)
        print("Now aligning: %s" % entry)
        cline = ClustalwCommandline("clustalw2", infile=inpath, outfile=outpath)
        cline()
        worktree[outpath] = -1
    
    # protein training instance will need to be supplied. Alignment happens now.
    filelist = [f for f in os.listdir(options.protein) if os.path.isfile(os.path.join(options.protein, f))]
    for entry in filelist:
        outpath = os.path.join(options.protein, entry + ".aln")
        inpath = os.path.join(options.protein, entry)
        print("Now aligning: %s" % entry)
        cline = ClustalwCommandline("clustalw2", infile=inpath, outfile=outpath)
        cline()
        worktree[outpath] = 0
    
    # If a negative set is supplied it will be scanned now
    if options.negative:
        filelist = [f for f in os.listdir(options.negative) if os.path.isfile(os.path.join(options.negative, f))]
        for entry in filelist:
            outpath = os.path.join(options.negative, entry + ".aln")
            inpath = os.path.join(options.negative, entry)
            print("Now aligning: %s" % entry)
            cline = ClustalwCommandline("clustalw2", infile=inpath, outfile=outpath)
            cline()
            worktree[outpath] = 1

    # Otherwise, one will be auto-generated using SISSIz
    else:
        negativeset = os.path.split(os.path.abspath(options.ncrna))[0]+"/_negative"
        dummies = []
        if not os.path.isdir(negativeset):
            os.mkdir(negativeset)
        
        keys = np.random.choice(list(worktree.keys()), size=int(len(worktree)/2))
        
        for (f, c) in [(k, worktree.get(k)) for k in keys]:
            outpath = os.path.join(negativeset, os.path.basename(f))
            generate_control_alignment(inpath=f, outpath=outpath)
            if not is_alignment(outpath):
                print("Failed to create a valid control alignment for %s." % f)
                continue
            dummies.append(outpath)
        for f in dummies:
            worktree[f] = 1
            
    if len(worktree) == 0:
        print("You supplied a directory, but I could not find readable fasta-files in it. Could you double-check the format?")
        sys.exit()
        
    vectortable = pd.concat([create_training_set(f, c, options) for (f, c) in list(worktree.items())])
    print("Data generation process exited normally.")
    
    vectortable.to_csv(options.out_file+"_trainingdata.csv", sep="\t")
    create_report(df=pd.read_csv(options.out_file+"_trainingdata.csv", sep="\t"), options=options)
    
    return vectortable, options.out_file+"_trainingdata.csv"



################################################
#
#               Combination
#
################################################

def combine(options):
    
    def has_files(xs):
        if len(xs) == 0:
            print("No files to combine. Did you add a wrong prefix? Aborting.")
            sys.exit()

    if os.path.isdir(options.in_file):
        xs = [os.path.join(options.in_file, x) for x in os.listdir(options.in_file) if x.endswith(".tsv") and x.startswith(options.prefix)]
        has_files(xs)
        df = pd.concat([pd.read_csv(x, sep="\t") for x in xs])

        if options.out_file.endswith(".tsv"):
            name = options.out_file
        else:
            name = options.out_file + ".tsv"
        
        df.sort_values(by="Class")
        df.to_csv(name, sep="\t", index=False)
        
    else:
        xs = [x for x in os.listdir() if x.endswith(".tsv") and x.startswith(options.prefix)]
        has_files(xs)
        df = pd.concat([pd.read_csv(x, sep="\t") for x in xs])

        if options.out_file.endswith(".tsv"):
            name = options.out_file
        else:
            name = options.out_file + ".tsv"
        
        df.sort_values(by="Class")
        df.to_csv(name, sep="\t", index=False)


################################################
#
#               Feature calculation
#
################################################


######################### Misc. ####################################

def ungap_sequence(seq):
    return re.sub(r'-', "", seq)


def get_GC_content(seq):
    Gs = seq.count("G")
    Cs = seq.count("C")
    
    return (Gs + Cs) / len(seq)


def average_GC_content(seqs):
    return np.mean([get_GC_content(s) for s in seqs])


def normalize_seq_length(length, min_len, max_len):
    return (length - min_len) / (max_len - min_len)


def calculate_sequence_features(seq, GC_content, min_len, max_len):
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
    return mono_features + di_features + [normalized_length]


######################### Shannon entropy ##########################

def log_2(x):
    if x == 0:
        return 0
    return np.log2(x)


def column_entropy(dinucleotide, column):
    col_string = str().join(column)
    l = len(column)
    f = col_string.count(dinucleotide) / l
    return f *log_2(f)


def shannon_entropy(seqs):
    if len(seqs) < 2:
        return 0
    columns = np.asarray([list(seq.strip("\n")) for seq in seqs]).transpose()
    entropy = 0.0
    
    if len(columns) < 4:
        return 0
    
    for column in columns:
        entropy += sum([column_entropy(n, column) for n in nucleotides])
    
    return round(entropy *(-1/len(columns)), 4)


########################### Structural conservation index ############

def fold_single(seq):
    fc = vienna.fold_compound(seq)
    mfe_structure, mfe = fc.mfe()
    return mfe_structure, mfe


def call_alifold(seqs):
    consensus_structure, mfe = vienna.alifold(seqs)
    return consensus_structure, mfe


def structural_conservation_index(seqs):
    if len(seqs) < 2:
        return 0
    e_consensus = call_alifold(seqs)[1]
    seqs_ungapped = [ungap_sequence(seq) for seq in seqs]
    single_energies = sum([fold_single(s)[1] for s in seqs_ungapped])
    if single_energies == 0:
        return 0
    return round(e_consensus / ( (1/len(seqs))*single_energies), 4)


########################### z-score of MFE ###########################

def get_bin(GC):
    low, high = 0, 0
    for i in range(1, len(bins)):
        lower_bound = bins[i]
        if GC <= lower_bound:
            low = int(bins[i-1]*100)
            high = int(lower_bound*100)
            break

    return "%s-%s" % (low, high)


def predict_std(seq, GC_bin, feature_vector):
    m = model_std_dict.get(GC_bin)
    return m.predict([feature_vector])[0]


def predict_mean(seq, GC_bin, feature_vector):
    m = model_mean_dict.get(GC_bin)
    return m.predict([feature_vector])[0]


def get_distribution(mfes):
    std = np.std(mfes)
    mu = np.mean(mfes)
    return mu, std


def z_(x, std, mu):
    return (x - mu) / std


def shuffle_me(seq):
    generator = DPPermutations(sequence=seq, rng=rng, 
                sequence_type="RNA", sequence_length=len(seq), di_features=None)

    references = []
    for i in range(0, 100):
        references.append(generator.shuffle())
    
    return references


def z_score_of_mfe(seqs):
    if len(seqs) < 2:
        return 0
    seqs = [ungap_sequence(seq) for seq in seqs]
    zs = []
    distributions = []
    
    lengths = [len(seq) for seq in seqs]
    
    for seq in seqs:
        GC_content = get_GC_content(seq)
        
        if (GC_content > 0.8) or (GC_content < 0.2):
            print("GC content outside of training range. Shuffling sequence instead...")
            references = shuffle_me(seq)
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
    
    return round(np.mean(zs), 4)


########################### Hexamer Score ###########################


def get_background_model(filename):
    coding, noncoding = {}, {}
    
    with open(filename, 'r') as f:
        for line in f.readlines():
            hexamer, c, nc = line.split("\t")
            coding[hexamer] = float(c)
            noncoding[hexamer] = float(nc)
    
    return coding, noncoding


def word_generator(seq,word_size,step_size,frame=0):
    for i in range(frame,len(seq),step_size):
        word =  seq[i:i+word_size]
        if len(word) == word_size:
            yield word


def kmer_ratio(seq,word_size,step_size, hexamer_model):
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
    
    
def hexamer_score(seqs, hexamer_model):
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
    return score


########################## Codon conservation #######################


stop_score = 0.0
framegap_score = -12.0
blosum_62 = dict(blosum.BLOSUM(62))
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


def generate_kmers():
    k = 3
    nucleotides = ['A', 'C', 'G', 'U']
    k_comb = product(nucleotides, repeat=k)
    return list(k_comb)

codons = generate_kmers()


def maxSubArraySum(arr):
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


def empirical_substitution_rates(ref_seq, seq):
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


def get_stationary_frequencies(seq):
    s = seq.replace("-", "")
    length = len(s)+1
    frequencies = {
        "A": s.count("A") / length,
        "U": s.count("U") / length,
        "G": s.count("G") / length,
        "C": s.count("C") / length,
        }
    return frequencies


def hamming_distance(codon1, codon2):
    d = 3
    for i in range(0, 3):
        if codon1[i] == codon2[i]:
            d -= 1
    return d


def statistical_penalty(codon1, codon2, a_a, frequencies, distance, ref_seq, seq, transitions):
    hamming = hamming_distance(codon1, codon2)
    s = 0
    
    for codon in codons:
        if hamming_distance(codon1, codon) != hamming:
            continue
            
        a1_b1 = transitions.get(codon1[0] + "_" + codon[0], 0)
        a2_b2 = transitions.get(codon1[1] + "_" + codon[1], 0)
        a3_b3 = transitions.get(codon1[2] + "_" + codon[2], 0)
        
        s += blosum_62.get(a_a)*(a1_b1/(distance+1) ) *(a2_b2/(1+distance)) *(a3_b3/(1+distance)) 
    return s


def gap_score(gap):
    if gap == 3:
        return 0.0
    else:
        return -2.0
    

def calculate_offset(seq, index):
    nucleotides = []
    old_index = index
    while len(nucleotides) < 3 and index < len(seq):
        if seq[index] != "-":
            nucleotides.append(seq[index])
        index += 1
    return index - old_index 


def sanitize_alignment(aln):
    for record in aln:
        record.id = str(record.id).split(".")[0]
    set_id = set()
    aln_ = []
    for record in aln:
        if record.id not in set_id:
            aln_.append(record)
        set_id.add(record.id)
    return MultipleSeqAlignment(records=aln_)


def codon_conservation(alignment, tree_path=None):
    seqs = [str(record.seq) for record in alignment._records]
    names = [str(record.id).split(".")[0] for record in alignment._records]
    
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
                    a_a = codon1.translate() + codon2.translate()
                    if a_a[0] == "*":
                        local_score = stop_score
                        break
                    else:
                        local_score = blosum_62.get(a_a) - statistical_penalty(codon1, codon2, a_a, frequencies, distance, ref_seq, seq, transitions)
                    
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


########################### Feature calculation ##################### 

def get_genome_coordinates(filename):
    try:
        alignments = AlignIO.parse(filename, "maf")
    except Exception:
        return [], [], [], []
    starts = []
    ends = []
    strand = []
    chrs = []
    
    for align in alignments:
        reference = align[0]
        starts.append(reference.annotations["start"])
        ends.append(reference.annotations["start"] + reference.annotations["size"])
        strand.append(reference.annotations["strand"])
        chrs.append(reference.id.split(".")[1])
        
    return starts, ends, strand, chrs


def get_sequence_id(id_, id_set):
    if id_ in id_set:
        if len(id_.split(".")) == 1:
            return id_.split(".")[0] + ".1"
        count = int(id_.split(".")[1])
        return id_.split(".")[0] + "." + str(count +1)
    return id_


def parse_windows(filename):
    try:
        alignments = AlignIO.parse(filename, "clustal")
        align_dict = {}
        alignment_count = 0
        for a in alignments:
            id_set = set()
            seq_count = 0
            for record in a:
                seq_count += 1
                id_ = record.id.split(".")[0]
                id_set.add(id_)
                id_updated = get_sequence_id(id_, id_set)
                id_set.add(id_updated)
                record.id = id_updated
            alignment_count += 1
            align_dict[alignment_count] = a
        return align_dict
    
    except Exception:
        try:
            alignments = AlignIO.parse(filename, "maf")
            align_dict = {}
            alignment_count = 0
            for a in alignments:
                id_set = set()
                seq_count = 0
                for record in a:
                    seq_count += 1
                    id_ = record.id.split(".")[0]
                    id_set.add(id_)
                    id_updated = get_sequence_id(id_, id_set)
                    id_set.add(id_updated)
                    record.id = id_updated
                alignment_count += 1
                align_dict[alignment_count] = a
            print(align_dict)
            return align_dict
        except Exception as e:
            print("Unknown alignment format found. Neither Clustal nor MAF. Aborting.")
            raise e


def get_reverse_complement(alignment):
    for record in alignment:
        record.seq = record.seq.reverse_complement()
    return alignment


def only_codon_conservation(window):
    alignment_dict = parse_windows(window)
    for (k, a) in alignment_dict.items():
        print(a)
        print(codon_conservation(a))
        

def get_feature_set(window, hexamer_model, reverse=False, tree_path=None):
    alignment_dict = parse_windows(window)
    scis, zs, entropies, hexamers, codon_cons = [], [], [], [], []
    count = 0
    length = len(alignment_dict)
    
    for (k, a) in alignment_dict.items():
        a_ = a
        if reverse:
            a_ = get_reverse_complement(a)
        count += 1
        print(count, "/", length)
        s = [str(record.seq) for record in a_._records]
        s = [seq.replace("T", "U") for seq in s]
        scis.append(structural_conservation_index(s))
        entropies.append(shannon_entropy(s))
        codon_cons.append(codon_conservation(a_, tree_path))
        seqs = [ungap_sequence(seq) for seq in s]
        zs.append(z_score_of_mfe(seqs))
        hexamers.append(hexamer_score(seqs, hexamer_model))
    
    return scis, zs, entropies, hexamers, codon_cons


################################################
#
#               Model building
#
################################################


################### Core #################################################

def unpack_features(df, structure):
    print(structure)
    
    if structure:
        y = df["Class"].ravel()
        x = np.asarray(df[["SCI", "z-score of MFE", "Shannon-entropy",]])
        return x, y
    
    y = df["Class"].ravel()
    x = np.asarray(df[["SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"]])
    return x, y


def build_model(options, filename):
    if os.path.isdir(filename):
        df = pd.read_csv(filename+"/feature_vectors.csv", sep="\t")
    else:
        df = pd.read_csv(filename, sep="\t")
        
    if not os.path.isdir(options.out_file):
        os.mkdir(options.out_file)
    
    df["Class"] = [class_dict.get(a) for a in df["Class"]]
    x, y = unpack_features(df, options.structure)
    
    if options.ml == "SVM":
        parameters = get_svm_hyperparameters(x, y, options)
        create_sklearn_svm_model(x, y, parameters, filepath=options.out_file)

    elif options.ml == "RF":
        parameters = get_rf_hyperparameters(x, y, options)
        create_sklearn_rf_model(x, y, parameters, filepath=options.out_file)
        
    elif options.ml == "LR":
        parameters = get_lr_hyperparameters(x, y, options)
        create_sklearn_lr_model(x, y, parameters, filepath=options.out_file)
    
    

###################### Plotting of Eval ####################################

def draw_roc(fpr, tpr, auc, color="royalblue", name=""):
    data = {
        "fpr": fpr,
        "tpr": tpr,
        }

    sns.set_style("ticks")
    textstr = " Area under curve: \n %s" % round(auc, 2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    g = sns.lineplot(data=data, x="fpr", y="tpr", color=color)
    g.text(0.6, 0.2, textstr, transform=g.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)
    
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim((-0.1, 1.0))
    plt.ylim((0.0, 1.1))
    plt.savefig(name + ".pdf", dpi=300, bbox_inches="tight")
    plt.savefig(name + ".svg", dpi=300, bbox_inches="tight")
    plt.show()


def draw_CV_rocs(x, y, m, outpath):
    if len(set(y)) > 2:
        print("No ROC curves for a 3-class problem. Skipping...")
        return None
    
    fs, ts, iis = [], [], []
    
    for i in range(1, 5+1):
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        fpr, tpr, _ = roc_curve(y_pred, y_test)
        fs += list(fpr)
        ts += list(tpr)
        iis += [i]*len(fpr)
    
    data = {
        "fpr": fs,
        "tpr": ts,
        "iteration": iis,
        }
    
    sns.lineplot(data=data, x="fpr", y="tpr", hue="iteration")
    plt.title("Cross-validation of generated model")
    plt.savefig(outpath+"/crossvalidation_rocs.pdf", dpi=220, bbox_inches="tight")
    plt.savefig(outpath+"/crossvalidation_rocs.svg", dpi=220, bbox_inches="tight")


###################### Parameter Optimization ###############################


def get_rf_hyperparameters(x, y, options):
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
            
    return parameters


def get_svm_hyperparameters(x, y, options):
    parameters = {
            "cost": 2.0,
            "gamma": 1.0,
            }
    if options.optimize == True:
        cv_results = svm_hyperparameter_search(x, y, options)
        cv_results.sort_values(by="rank_test_score", ascending=True, inplace=True)
        parameters["cost"] = cv_results["param_C"].iloc[0]
        parameters["gamma"] = cv_results["param_gamma"].iloc[0]
                
    return parameters


def get_lr_hyperparameters(x, y, options):
    parameters = {
            "cost": 2.0,
            "tolerance": 1.0,
            }
    if options.optimize == True:
        cv_results = lr_hyperparameter_search(x, y, options)
        cv_results.sort_values(by="rank_test_score", ascending=True, inplace=True)
        parameters["cost"] = cv_results["param_C"].iloc[0]
        parameters["tolerance"] = cv_results["param_tol"].iloc[0]

    return parameters


def rf_hyperparameter_search(x, y, options):
    m = RandomForestClassifier()
    
    stepsize_estimators = int(round( (options.high_estimators - options.low_estimators) / options.grid_steps , 0))
    stepsize_split = int(round( (options.high_split - options.low_split) / options.grid_steps, 0))
    stepsize_leaf = int(round( (options.high_leaf - options.low_leaf) / options.grid_steps, 0))
    
    parameter_grid = {
            "n_estimators": range(options.low_estimators, options.high_estimators, stepsize_estimators),
            "min_samples_split": range(options.low_split, options.high_split, stepsize_split),
            "min_samples_leaf": range(options.low_leaf, options.high_leaf, stepsize_leaf),
            }
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
                                      n_iter=50)
        searcher.fit(x, y)
    
    return pd.DataFrame(data=searcher.cv_results_)


def svm_hyperparameter_search(x, y, options):
    m = svm.SVC(verbose=1)
    
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


def lr_hyperparameter_search(x, y, options):
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

def scale_features(x):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X=x)
    return X, scaler


def create_sklearn_svm_model(X, Y, parameters, filepath):
    X, scaler = scale_features(X)
    
    m = svm.SVC(C=parameters.get("cost"), gamma=parameters.get("gamma"), verbose=1)
    m.fit(X, Y)
    pickle.dump(m, open(filepath + "/svm_classifier.model", "wb"))
    pickle.dump(scaler, open(filepath + "/svm_classifier.model_scaler", "wb"))


def create_sklearn_rf_model(X, Y, parameters, filepath):
    X, scaler = scale_features(X)
    
    m = RandomForestClassifier(**parameters)
    m.fit(X, Y)
    pickle.dump(m, open(filepath + "/rf_classifier.model", "wb"))
    pickle.dump(scaler, open(filepath + "/rf_classifier.model_scaler", "wb"))


def create_sklearn_lr_model(X, Y, parameters, filepath):
    X, scaler = scale_features(X)
    
    m = LogisticRegression(C=parameters.get("cost"), tol=parameters.get("tolerance"))
    m.fit(X, Y)
    pickle.dump(m, open(filepath + "/lr_classifier.model", "wb"))
    pickle.dump(scaler, open(filepath + "/lr_classifier.model_scaler", "wb"))

    

########### Model evaluation ##########################################


def test_model(options):
    if options.out_file: 
        out_folder = options.out_file
    else:
        out_folder = os.path.basename(options.model_path).replace(".model", "") + "_evaluation_out"
    
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    filename = options.in_file
    df = pd.read_csv(filename, sep="\t")
    x, y = unpack_features(df)
    y_pred, y_real = evaluate_classifier(x, y, options.model_path, out_folder)
    
    df["Assigned class label"] = [class_dict.get(a) for a in y_pred]
    df["True class label"] = [class_dict.get(a) for a in y_real]
    df.to_csv("%s/class_labels.csv" % out_folder, sep="\t")


def predict_with_sklearn_model(x, y, modelpath):
    model = pickle.load(open(modelpath, 'rb'))
    scaler = pickle.load(open(modelpath + "_scaler", 'rb'))
    x = scaler.transform(x)
    y_pred = model.predict(x)
    
    if type(y_pred[0]) != int and type(y_pred[0]) != float: 
        y_pred = [class_dict.get(y_pred[i]) for i in range(0, len(y_pred))]
    
    return y_pred


def evaluate_classifier(x, y, modelpath, outpath):
    try:
        y_pred = predict_with_sklearn_model(x, y, modelpath)
        color = "green"
    except Exception as e:
        print("This appears to be an invalid model file. Aborted.")
        raise e
        
    y_pred = [class_dict.get(x) for x in y_pred]
    y = [class_dict.get(x) for x in y]
    pos_label = min(y)
    
    tpr, fpr, _ = roc_curve(y_pred, y, pos_label=pos_label)
    auc = roc_auc_score(y_pred, y)
    
    f1 = f1_score(y, y_pred, pos_label=pos_label)
    acc = accuracy_score(y, y_pred,)
    recall = recall_score(y, y_pred, pos_label=pos_label)
    prec = precision_score(y, y_pred, pos_label=pos_label)
    
    result_data = {
            "Accuracy": [acc],
            "F1-Score": [f1],
            "Precision": [prec],
            "Recall": [recall],
             "AUC": [auc],
        }
    df = pd.DataFrame(result_data)
    df.to_json(outpath + "/statistical_evaluation.json")
    print(df)
    draw_roc(fpr, tpr, auc, color, outpath +"/roc_curve")
    
    return y_pred, y


################################################
#
#               Prediction
#
################################################


def load_target_model(filename):
    try:
        model = pickle.load(open(filename, 'rb'))
    except Exception as e:
        print("Could not load model file. Correct path?")
        sys.exit()
    return model


def svhip_prediction(options):
    model = load_target_model(options.model_path)
    df = pd.read_csv(options.in_file, sep="\t")
    
    if options.ncrna:
        target = df[["SCI", "z-score of MFE", "Shannon-entropy"]]
    else:
        target = df[["SCI", "z-score of MFE", "Shannon-entropy", "Hexamer Score", "Codon conservation"]]
    
    y = model.predict(target)
    y = [class_dict.get(y_) for y_ in y]
    predict_column = "Prediction"
    if options.prediction_label:
        predict_column = options.prediction_label
    df[predict_column] = y
    filename = options.in_file + ".svhip"
    if options.out_file:
        filename = options.out_file
    df.to_csv(filename, sep="\t", index=False)
            

def print_results(names, alignment_number, classes):
    if type(classes[0]) != str:
        classes = [class_dict.get(int(c)) for c in classes]
    
    print("\t \t   Alignment \t   Predicted")
    print("Filename \t | Number \t | Class")
    print("----------------------------------")
        
    for i in range(0, len(names)):
        print("%s \t | %s \t | %s" % (names[i], alignment_number[i], classes[i]))



###################################################
#
# Testing
#
###################################################

def svhip_check():
    tmp = tempfile.TemporaryFile(mode='w+b', buffering=- 1,)
    cmd = shlex.split("python3 svhip.py data --help")
    returncode = subprocess.call(cmd, stdout=tmp)
    
    if returncode == 0:
        print("TEST 1 / 5: SUCCESS")
    else:
        print("FAILED AT: TEST 1 / 5")
    
    
    ###################################################
    # data generation
    ###################################################
    
    def check_tsv_output(df):
        if (len(df) >= 20) and ("RNA" in list(df["Class"])) and ("OTHER" in list(df["Class"])) :
            return True
        return False
    
    
    cmd = shlex.split("python3 svhip.py data -i Example/tRNA_test.fasta -o temp.tsv -w 1 -g True ")
    returncode = subprocess.call(cmd, stdout=tmp)
    df = pd.read_csv("temp.tsv", sep="\t")
    shutil.rmtree("temp.tsv_report")
    shutil.rmtree("Example/tRNA_test.aln_windows")
    shutil.rmtree("Example/tRNA_test.fasta_negative")
    
    if returncode == 0 and check_tsv_output(df):
        print("TEST 2 / 5: SUCCESS")
    else:
        print("FAILED AT: TEST 2 / 5")
    
    
    ###################################################
    # model training
    ###################################################
    
    cmd = shlex.split("python3 svhip.py training -i temp.tsv -o tempmodel")
    returncode = subprocess.call(cmd, stdout=tmp)
    
    if returncode == 0 and os.path.exists("tempmodel/svm_classifier.model"):
        print("TEST 3 / 5: SUCCESS")
    else:
        print("FAILED AT: TEST 3 / 5")
        
    
    ###################################################
    # feature calculation
    ###################################################
    
    def check_tsv_output_2(df):
        if (len(df) == 8) :
            return True
        return False
    
    
    cmd = shlex.split("python3 svhip.py features -i Example/Arabidopsis_1.maf -o temp2.tsv -R True ")
    returncode = subprocess.call(cmd, stdout=tmp)
    df = pd.read_csv("temp2.tsv", sep="\t")
    
    if returncode == 0 and check_tsv_output_2(df):
        print("TEST 4 / 5: SUCCESS")
    else:
        print("FAILED AT: TEST 4 / 5")
    
    
    ###################################################
    # Prediction
    ###################################################
    
    def check_prediction_output(df):
        if "Prediction" in df.columns:
            return True
        return False
    
    cmd = shlex.split("python3 svhip.py predict -i temp2.tsv -M tempmodel/svm_classifier.model -o temp2.tsv --column-label Prediction")
    returncode = subprocess.call(cmd, stdout=tmp)
    df = pd.read_csv("temp2.tsv", sep="\t")
    
    if returncode == 0 and check_prediction_output(df):
        print("TEST 5 / 5: SUCCESS")
    else:
        print("FAILED AT: TEST 5 / 5")
    
    
    ###################################################
    # Cleanup
    ###################################################
    
    os.remove("temp.tsv")
    os.remove("temp2.tsv")
    shutil.rmtree("tempmodel")
    os.remove("Example/tRNA_test.aln")
    os.remove("Example/tRNA_test.dnd")



################################################
#
#               Main function
#
################################################


def has_input_options(parser, options):
    for entry in [options.in_file]:
        if not entry:
            parser.print_help()
            print("Input path is missing. Supply it with -i, --input. \n")
            sys.exit()


def has_output_options(parser, options):
    if not options.out_file:
        parser.print_help()
        print("Output path is missing. Supply it with -o, --output. \n")
        sys.exit()


def data(parser):
    parser.add_option("-N", "--negative",action="store",type="string",dest="negative",help="Should a specific negative data set be supplied for data generation? If this field is EMPTY it will be auto-generated based on the data at hand (This will be the desired option for most uses).")
    parser.add_option("-d", "--max-id",action="store",type="int",dest="max_id",default=95,help="During data preprocessing, sequences above identity threshold (in percent) will be removed. Default: 95.")
    parser.add_option("-n", "--num-sequences",action="store",type="int",dest="n_seqs",default=100,help="Number of sequences input alignments will be optimized towards. Default: 100.")
    parser.add_option("-l", "--window-length",action="store",type="int",dest="window_length",default=120,help="Length of overlapping windows that alignments will be sliced into. Default: 120.")
    parser.add_option("-s", "--slide",action="store",type="int",dest="slide",default=40,help="Controls the step size during alignment slicing and thereby the overlap of each window.")
    parser.add_option("-w", "--windows",action="store",type="int",dest="n_windows",default=10,help="The number of times the alignment should be fully sliced in windows - for variation.")
    parser.add_option("-g", "--generate-control", action="store",default=False,dest="generate_control",help="Flag to determine if a negative set should be auto-generated (Default: False).")
    parser.add_option("-c", "--shuffle-control", action="store", default=True,dest="shuffle_control",help="Use the column-based shuffling approach provided by the RNAz framework instead of SISSIz (Default: False).")
    parser.add_option("-p", "--positive-label", action="store", default="ncRNA",dest="pos_label",help="The label that should be assigned to the feature vectors generated from the (non-control) input data. Can be CDS (for protein coding sequences) or ncRNA. (Default: ncRNA).")
    parser.add_option("-H", "--hexamer-model", action="store",dest="hexamer_model",default=os.path.join(os.path.join(THIS_DIR, "hexamer_models"), "Human_hexamer.tsv"),help="The Location of the statistical Hexamer model to use. An example file is included with the download as Human_hexamer.tsv, which will be used as a fallback.")
    parser.add_option("-S", "--no-structural-filter", action="store", default=False,dest="structure_filter",help="Set this flag to True if no filtering of alignment windows for statistical significance of structure should occur (Default: False).")
    parser.add_option("-T", "--tree", action="store", default=None, dest="tree_path", help="If an evolutionary tree of species in the alignment is available in Newick format, you can pass it here. Names have to be identical. If None is passed, one will be estimated based on sequences at hand." )
    options, args = parser.parse_args()
    
    has_input_options(parser, options)
    has_output_options(parser, options)

    filename = options.in_file
    build_data(options, filename)
    
    
def svhip_combine(parser):
    parser.add_option("-p", "--prefix",action="store",type="string",dest="prefix",default="",help="Prefix for selection of files to combine. For example, if set to TEST, only valid feature vector containing files with the prefix TEST will be added.")
    options, args = parser.parse_args()
    
    if not options.in_file:
        options.in_file = ""
    
    has_output_options(parser, options)
    combine(options)
    

def training(parser):
    # Model training specific options
    parser.add_option("-S", "--structure",action="store",dest="structure",default=False,help="Flag determining if only secondary structure conservation features should be considered. If True, protein coding features will be included (Default: False).")
    parser.add_option("-M", "--model",action="store",type="string",dest="ml",default="SVM",help="The model type to be trained. You can choose LR (Logistic regression), SVM (Support vector machine) or RF (Random Forest). (Default: SVM)")
    parser.add_option("--optimize-hyperparameters",action="store",dest="optimize",default=True,help="Select if a parameter optimization should be performed for the ML model. Default is on.")
    parser.add_option("--optimizer",action="store",type="string",dest="optimizer",default="gridsearch",help="Select the optimizer for hyperparameter search. Search will be conducted with 5-fold crossvalidation and either of 'gridsearch' (default, more precise) or 'randomwalk' (faster).")
    
    parser.add_option("--low-c",action="store",type="int",dest="low_c",default=1,help="SVM hyperparameter search: Lowest value of the cost (C) parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--high-c",action="store",type="int",dest="high_c",default=1000,help="SVM hyperparameter search: Highest value of the cost (C) parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--low-gamma",action="store",type="int",dest="low_g",default=1,help="SVM hyperparameter search: Lowest value of the gamma parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--high-gamma",action="store",type="int",dest="high_g",default=1000,help="SVM hyperparameter search: Highest value of the gamma parameter to optimize. Does nothing if no SVM classifier is used.")
    parser.add_option("--hyperparameter-steps",action="store",type="int",dest="grid_steps",default=10,help="Number of values to try out for EACH hyperparameter. Values will be evenly spaced. Default: 10")
    parser.add_option("--logscale",action="store",dest="logscale",default=False,help="Flag that decides if a logarithmic scale should be used for the hyperparameter grid. If set, a log base can be set with --logbase.")
    parser.add_option("--logbase",action="store",type="int",dest="logbase",default=2,help="The logarithmic base if a log scale is used in hyperparameter search. Default: 10.")
    
    # Random Forest specific options 
    parser.add_option("--min-trees",action="store",type="int",dest="low_estimators",default=1,help="Random Forest hyperparameter search: Minimum number of trees before optimization. Does nothing if no RF classifier is used.")
    parser.add_option("--max-trees",action="store",type="int",dest="high_estimators",default=1000,help="Random hyperparameter search: Maximum number of trees before optimization. Does nothing if no RF classifier is used.")
    parser.add_option("--min-samples-split",action="store",type="int",dest="low_split",default=2,help="Random Forest hyperparameter search: Minimum number of samples for splitting an internal node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--max-samples-split",action="store",type="int",dest="high_split",default=16,help="Random hyperparameter search:  Maximum number of samples for splitting an internal node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--min-samples-leaf",action="store",type="int",dest="low_leaf",default=1,help="Random Forest hyperparameter search: Minimum number of samples for splitting a leaf node in the forest. Does nothing if no RF classifier is used.")
    parser.add_option("--max-samples-leaf",action="store",type="int",dest="high_leaf",default=16,help="Random hyperparameter search: Maximum number of samples for splitting a leaf node in the forest. Does nothing if no RF classifier is used.")
    options, args = parser.parse_args()
    
    has_input_options(parser, options)
    has_output_options(parser, options)
    
    filename = options.in_file
    build_model(options, filename)


def data_3(parser):
    parser.add_option("-a","--ncRNA",action="store",type="string", dest="ncrna",help="The input directory or file containing ncRNA training examples (Required).")
    parser.add_option("-b","--protein",action="store",type="string", dest="protein",help="The input directory or file containing protein encoding training examples (Required).")
    parser.add_option("-N", "--negative",action="store",type="string",dest="negative",help="Should a specific negative data set be supplied for data generation? If this field is EMPTY it will be auto-generated based on the data at hand (This will be the desired option for most uses).  ")
    parser.add_option("--log-file",action="store",type="string", dest="log_file",help="Name of log file", default="logs.info")   
    parser.add_option("-d", "--max-id",action="store",type="int",dest="max_id",default=95,help="During data preprocessing, sequences above identity threshold (in percent) will be removed. Default: 95.")
    parser.add_option("-n", "--num-sequences",action="store",type="int",dest="n_seqs",default=100,help="Number of sequences input alignments will be optimized towards. Default: 100.")
    parser.add_option("-l", "--window-length",action="store",type="int",dest="window_length",default=120,help="Length of overlapping windows that alignments will be sliced into. Default: 120.")
    parser.add_option("-s", "--slide",action="store",type="int",dest="slide",default=40,help="Controls the step size during alignment slicing and thereby the overlap of each window.")
    parser.add_option("-w", "--windows",action="store",type="int",dest="n_windows",default=10,help="The number of times the alignment should be fully sliced in windows - for variation.")
    parser.add_option("-h", "--hexamer-model", action="store",dest="pos_label",help="The Location of the statistical Hexamer model to use. An example file is included with the download as Human_hexamer.tsv.")
    options, args = parser.parse_args()

    has_output_options(parser, options)

    if not options.ncrna:
        parser.print_help()
        print("No ncRNA data sample. Supply it with -a, --ncRNA. \n")
        sys.exit()
    
    if not options.protein:
        parser.print_help()
        print("No protein data sample. Supply it with -b, --protein. \n")
        sys.exit()
        
    if not options.out_file:
        parser.print_help()
        print("You need to supply a name for the output directory with -o, --outfile.")
        sys.exit()

    build_three_way_data(options)


def evaluate(parser):
    # Test / prediction specific options
    parser.add_option("--model-path",action="store",type="string",dest="model_path",default="",help="If running a model test (--task test) or prediction (--task predict), this is the path of the model to evaluate. The data set to use should be handed over with -i, --input. ")
    options, args = parser.parse_args()
    has_input_options(parser, options)
    
    test_model(options)


def predict(parser):
    # Test / prediction specific options
    parser.add_option("-M", "--model-path",action="store",type="string",dest="model_path",default="",help="If running a model test (--task test) or prediction (--task predict), this is the path of the model to evaluate. The data set to use should be handed over with -i, --input. ")
    parser.add_option("--column-label",action="store",type="string",dest="prediction_label",default="Prediction",help="Column name for the prediction in the output.")
    parser.add_option("--structure", action="store", dest="ncrna", default=False, help="Set to True if only features for conservation of secondary structure should be used. Depends on type of model.")
    options, args = parser.parse_args()
    has_input_options(parser, options)
    
    svhip_prediction(options)


def features(parser):
    
    def flatten_me(forward, reverse):
        return [x for xs in zip(forward, reverse) for x in xs]
    
    parser.add_option("-R","--reverse",action="store", dest="reverse", default=False, help="Also scan the reverse complement when calculating features.")
    parser.add_option("-H", "--hexamer-model", action="store",dest="hexamer_model",default=hexamer_backup,help="The Location of the statistical Hexamer model to use. An example file is included with the download as Human_hexamer.tsv, which will be used as a fallback.")
    parser.add_option("-T", "--tree", action="store", default=None, dest="tree_path", help="If an evolutionary tree of species in the alignment is available in Newick format, you can pass it here. Names have to be identical. If None is passed, one will be estimated based on sequences at hand." )
    
    options, args = parser.parse_args()
    has_input_options(parser, options)
    starts, ends, directions, chromosomes = get_genome_coordinates(options.in_file)
    scis, zs, entropies, hexamers, codon_cons = get_feature_set(options.in_file, hexamer_model=options.hexamer_model, tree_path=options.tree_path)
    if options.reverse:
        scis_reverse, zs_reverse, entropies_reverse, hexamers_reverse, codons_reverse = get_feature_set(options.in_file, hexamer_model=options.hexamer_model, reverse=True, tree_path=options.tree_path)
        df = pd.DataFrame(data={
            "SCI": flatten_me(scis, scis_reverse),
            "z-score of MFE": flatten_me(zs, zs_reverse),
            "Shannon-entropy": flatten_me(entropies, entropies_reverse),
            "Hexamer Score": flatten_me(hexamers, hexamers_reverse),
            "Codon conservation": flatten_me(codon_cons, codons_reverse),
            })
        if len(starts)*2 == len(df):
            df["start"] = flatten_me(starts, starts)
            df["end"] = flatten_me(ends, ends)
            df["direction"] = flatten_me(["forward"]*len(scis), ["reverse"]*len(scis))
            df["chromosome"] = flatten_me(chromosomes, chromosomes)
        
    else:
        df = pd.DataFrame(data={
            "SCI": scis,
            "z-score of MFE": zs,
            "Shannon-entropy": entropies,
            "Hexamer Score": hexamers,
            "Codon conservation": codon_cons,
            })
        if len(starts) == len(df):
            df["start"] = starts
            df["end"] = ends
            df["direction"] = ["forward"]*len(df)
            df["chromosome"] = chromosomes
    print(df)
    if options.out_file:
        fname = options.out_file
        if not fname.endswith(".tsv"):
            fname = fname + ".tsv"
        df.to_csv(fname, sep="\t")


def codon_conservation_score(parser):
    options, args = parser.parse_args()
    has_input_options(parser, options)
    only_codon_conservation(options.in_file)
    


def main():
    args = sys.argv
    
    # Setup a command line parser, replacing the outdated "currysoup" module
    usage = "\n%prog  [options]"
    parser = OptionParser(usage,version="%prog " + __version__)
    parser.add_option("-i","--input",action="store",type="string", dest="in_file",help="The input directory or file (Required).")
    parser.add_option("-o","--outfile",action="store",type="string", dest="out_file",help="Name for the output directory (Required).")
    t = time.time()
    
    if len(args) < 2:
        print("Usage: svhip [Task] [Options] with Task being one of 'data', 'training', 'data_3', 'evaluate', 'predict', 'features'.")
        sys.exit()
    if args[1] == "data":
        data(parser)
    elif args[1] == "training":
        training(parser)
    elif args[1] == "data3":
        data_3(parser)
    elif args[1] == "evaluate":
        evaluate(parser)
    elif args[1] == "predict":
        predict(parser)
    elif args[1] == "features":
        features(parser)
    elif args[1] == "codon_conservation":
        codon_conservation_score(parser)
    elif args[1] == "check":
        svhip_check()
    elif args[1] == "combine":
        svhip_combine(parser)
    else:
        print("Usage: svhip [Task] [Options] with Task being one of 'data', 'training', 'data_3', 'evaluate', 'predict', 'features'.")
        sys.exit()
    print("Program ran for %s seconds." % round(time.time() - t, 2))
    

if __name__=='__main__':
    main()

