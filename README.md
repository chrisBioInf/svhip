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

3 Installation in case the above does not work
On certain systems, installation via bioconda, while generally preferred, may be impossible. 
A likely culprit is the local version of libgcc, which is system-relevant but may
not be compatible with the rather strict requirements of the ViennaRNA package, which is
in turn required to run Svhip. You can read more about the issue in the following link,
under the section "Troubleshooting": https://pypi.org/project/ViennaRNA/.

For this specific case, we provide a workaround via manual installation, which will
(hopefully) no longer be necessary in future versions. It uses the Mamba to 
provide the required libgcc version within an enclosed conda environment. 

Before we begin, move your working directory into the /svhip folder as downloaded 
from Git.
Like before, we start by creating the base environment that will contain Svhip:

$ conda create --name svhip_env python=3.9
$ conda activate svhip_env

Now, we install Mamba, install specific requirements and finally get ViennaRNA:

$ conda install -c conda-forge mamba
$ mamba install libgcc libgcc-ng libstdcxx-ng
$ mamba install -c biopython viennarna

You can check if the installation was succesful by opening a python console,
import the ViennaRNA package and predict the structure of some random RNA sequence, i.e.:

$ python
$ import RNA
$ RNA.fold("GGAAAGGTTTGGG")

Following this, we now have to install all the remaining requirements. For this, we provide 
the environment.yaml with the download. So leave the environment and update it 
with the missing packages:

$ conda deactivate
$ conda env update -n svhip_env --file environment.yaml
$ conda activate svhip_env

Now all the requirements should be in place. You can now run a manual installation of 
Svhip for this environment by simply executing 

$ bash install_svhip.sh

Congrats! By following these steps, you should have a functional installation of Svhip
without needing the bioconda recipe. You can still check the integrity of the installation
with:

$ svhip check


4  Do not forget to read the manual (included in this repository!).



