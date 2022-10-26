# Svhip
Retrainable machine learning pipeline for the detection of secondary structure conservation on a genome-level.

1 Introduction
Svhip is a software developed in Python 3.8 for analysis of multiple genome
alignments in MAF format for the identification of conserved functional gene
sites. It provides options for the search for both protein coding sequences
(CDS) as well as the identification of evolutionary conserved secondary struc-
tures, hinting at functional non-coding sequences. A core feature of Svhip is
the possibility to freely retrain the classifier to account for different genomic
contexts, usually done by providing preselected training examples in the form
of ClustalW-alignments. Some of it’s features directly build on the RNAz
framework (https://www.tbi.univie.ac.at/software/RNAz/#download) for the
identification of secondary structure sites of high conservation, with the core
difference being the unchangeability of the underlying RNAz model and it’s
lack of support for the identification of coding sequences.

2 Installation
In terms of external requirements, Svhip will require a working perl instal-
lation and the installation of the software ClustalW2. All needed python
libraries are contained in the included conda environment and we suggest
using it for the installation of these dependencies. We suggest installation
using conda and a new environment:

$ conda create --name svhip_env python=3.9

which will generate a new conda environment using python version 3.9.
Switch to the new environment:

$ conda activate svhip_env

Then we install Svhip from the bioconda channel using:

$ conda install -c bioconda svhip

This should download and install all required files.
