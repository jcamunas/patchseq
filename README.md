## Repository containing scripts, notebooks and preprocessed datasets for:

============================================================================

# Patch-seq of endocrine cell exocytosis links physiologic dysfunction in diabetes to single-cell transcriptomic phenotypes
## by Camunas-Soler J, Dai X, et al., https://doi.org/10.1101/555110

![alt text](readme/fig.png "scheme")


 #### Description of Repository

 * **data folder**: contains preprocessed datasets (gene count tables) for patch-seq dataset, FACS dataset, as well as electrophysiology and cell metadata. This folder also contains scRNAseq datasets from previous publications used in QC figures, as well as siRNA knockdown results.
    - Large files are compressed. Before runnning notebooks unzip tar files using `tar -xvzf filename.tar.gz`
    - Download data count matrix from Segerstolpe. et al dataset and unzip in **data folder**. Data can be found in [Array Express Segerstolpe et al.](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-5061/), file named *E-MTAB-5061.processed.1.zip*. After unzipping make sure *Sandberg_pancreas_refseq_rpkms_counts_3514sc.txt* is found in **data folder**.

 * **analysis folder**: contains several folders with analysis results used to produce final figures. Data is saved in csv / excel files or pickles for integration with notebooks. Includes:
    - Cell typing file for each dataset.
    - tSNE coordinates for plots used in manuscript (patch-seq, patched vs non-patched, cryopreserved cells, alpha cells)
    - Machine Learning model for cell type classification based on Electrophysiology.
    - Correlations of gene expression to total exocytosis for beta-cells (nondiabetic, and T2D donors) and pathway analysis (compressed folder that needs to be unzipped).
    - Gene Set Enrichment Analysis (GSEA) for genes correlated to each functional group (i.e. Exocytosis, Sodium currents)
    - Correlations using a subset of beta-cells (train/test split) to determine Predictive Set of genes and perform predictions of electrophysiology.

* **notebooks**: Python and R notebooks to generate figures from manuscript and supplementary notebooks to generate results in analysis folders.

* **functions**: Helper functions to run notebooks.

* **figures**: Figures produced in notebooks.

* **resources**: databases and resources used in notebooks. Before runnning notebooks unzip compressed data files using command:```tar -xvzf filename.tar.gz``` .
