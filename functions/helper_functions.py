# vim: fdm=indent
# author:     Joan Camunas
# date:       16/08/17
# content:    Dataset functions to correlate gene expression and phenotypes
# Modules
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

####
#BASIC QC FUNCTIONS
####
def get_number_genes_observed(Dataset, min_counts=4):
    '''Get number of genes observed by sample'''
    ds = Dataset
    n = (ds.counts.exclude_features(spikeins=True, other=True, inplace=False) >= min_counts).sum(axis=0)

    n.name = 'number of genes'
    return n


def ensembletable_curate(df_ensemble_gene, remove_none=True, append_dups=True):
    '''takes a dataframe containing index (ensembleID) and column with genenames (GeneName)
    and returns a new dataframe where all nones in genename are replaced by EnsembleID and repeated
    genes are made unique by appending EnsemblID'''
    df = df_ensemble_gene.copy()
    if remove_none is True:
        df_nones =df.copy()
        df_nones = df_nones[df['GeneName'] == 'none']
        df_nones['index'] = df_nones.index
        df_nones.drop('GeneName',inplace=True,axis=1)
        df_nones.rename(columns={'index': 'GeneName'}, inplace=True)
        #we update the values on the original table
        df.update(df_nones)

    if append_dups is True:
    #Duplicated genes are called with genname and EnsembleID appended (e.g. SNORA16A_ENSG00000280498)
    #Another way would be to just leave the ensemblID name
        df_dup = df.copy()
        df_dup = df_dup[df_dup['GeneName'].duplicated()]
        df_dup['GeneName'] = df_dup['GeneName'] + '_' + df_dup.index
        df.update(df_dup)

    return df

#standard df function
def get_qc_table_df(df, translation_dict=None):
    '''df: a count table
    mithocondrial: whether to count mithocondrial genes True/False
    translation_dict: Dictionary were keys are ensemble IDs and values are gene names
    store in column GeneName. If None, translation is not done
    '''
    if translation_dict:
        counts = df.rename(index=translation_dict,inplace=False)
    else:
        counts = df.copy()

    total_counts = counts.sum(axis=0)
    #count mapped
    searchfor = ['^__', 'NIST_ConsensusVector']
    mapped_counts = counts[~counts.index.str.contains('|'.join(searchfor))].sum(axis=0)
    #count ERCC
    ERCC_counts = counts[counts.index.str.contains('^ERCC-')].sum(axis=0)
    #n_genes_thres
    searchfor = ['^__', 'NIST_ConsensusVector','^ERCC-']
    human_counts = counts[~counts.index.str.contains('|'.join(searchfor))]
    n_genes_3 = (human_counts > 3).sum(axis=0)
    n_genes_1 = (human_counts > 1).sum(axis=0)
    #mithocondrial counts
    mito_counts = counts[counts.index.str.contains('^MT-')].sum(axis=0)

    c_qc = pd.concat([counts.sum(axis=0),
        100 * mapped_counts / total_counts,
        100 * ERCC_counts / total_counts,
        100 * counts.loc['__alignment_not_unique'] / total_counts,
        100 * counts.loc['__not_aligned'] / total_counts,
        100 * mito_counts / total_counts,
                     n_genes_1,
                     n_genes_3],
        axis=1)
    c_qc.rename(columns={0: 'all_counts', 1: 'percent_mapped', 2: 'percent_ercc',
        3: 'percent_multimapped', 4: 'percent_unmapped',5:'percent_mito' , 6:'n_genes_1',7:'n_genes_3'}, inplace=True)

    return c_qc


#qc function
def get_qc_table(Dataset, min_counts_genes=4, translation_dict=None):
    '''Dataset: A singlet dataset
    mithocondrial: whether to count mithocondrial genes True/False
    translation_dict: Dictionary were keys are ensemble IDs and values are gene names
    store in column GeneName. If None, translation is not done
    '''
    if translation_dict:
        counts = Dataset.counts.rename(index=translation_dict,inplace=False)
    else:
        counts = Dataset.counts

    total_counts = counts.sum(axis=0)

    c_qc = pd.concat([np.log10(counts.sum(axis=0)+0.01),
        100 * counts.exclude_features(spikeins=False).sum(axis=0) / total_counts,
        100 * counts.get_spikeins().sum(axis=0) / total_counts,
        100 * (counts.exclude_features(spikeins=False).sum(axis=0) - counts.get_spikeins().sum(axis=0)) / total_counts,
        100 * counts.loc[counts.index.str.contains('^mt-', case=False)].sum(axis=0) / total_counts,
        100 * counts.loc['__alignment_not_unique'] / total_counts,
        100 * counts.loc['__not_aligned'] / total_counts,
        get_number_genes_observed(Dataset=Dataset,min_counts=min_counts_genes)],
        axis=1)
    c_qc.rename(columns={0: 'log_all_reads', 1: 'percent_mapped', 2: 'percent_ercc', 3: 'percent_human', 4: 'percent_mithocondrial',
        5: 'percent_multimapped', 6: 'percent_unmapped', 7: 'genes_observed'}, inplace=True)

    return c_qc


def plot_qc_basic(qc_table, title,size_swarm=2):

    #First figures plots percentages
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.violinplot(data=qc_table.loc[:,qc_table.columns.str.contains('percent')], scale='width',inner=None)
    color_palette = sns.color_palette(['#FFF3AA'])
    sns.swarmplot(data=qc_table.loc[:,qc_table.columns.str.contains('percent')],
            palette=color_palette,
            size=size_swarm,
            ax=ax)
    ax.set_xticklabels(['mapped','ERCC','human','mithocondrial', 'multimapped', 'unmapped'])
    ax.set_ylabel('percentage')
    ax.set_title(title)
    #second figure plots number of genes
    fig2, ax2 = plt.subplots(figsize=(4, 4))
    ax2 = sns.violinplot(x=None, y='number of genes', data=qc_table,inner=None)
    sns.swarmplot(x=None, y='number of genes', data=qc_table, palette=color_palette, size=size_swarm, ax=ax2)
    ax2.set_xticklabels([title])
    #third figure plots genes vs several qc metrics
    fig3, axs3 = plt.subplots(1,6, figsize=(25, 4))
    axs3.flatten()
    plotted_columns = ['log_all_reads','percent_human','percent_ercc','percent_mithocondrial','percent_multimapped','percent_unmapped']
    renamed_columns = ['sequencing depth (log10)','human reads (%)','ERCC reads (%)','Mithocondrial reads (%)','Multimapped (%)','Unmapped (%)']
    for i,col in enumerate(plotted_columns):
        axs3[i] = sns.regplot(x=col, y="number of genes", data=qc_table,fit_reg=False,ax=axs3[i])
        axs3[i].set_xlabel(renamed_columns[i])
    fig3.tight_layout()
    plt.show()
    return fig, fig2, fig3


####
# Add features or filter datasets
####

def categorical_palette_feature(samplesheet, column, palette=sns.husl_palette(10), return_cmap=False):
    '''samplesheet: table samplesheet of dataset, typically containing features in columns and samples in rows
    column: feature to make palette from
    palette: a color palette to be used
    return_cmap: if true returns colormap and dictionary with unique keys to make legend
    returns: Series in which index is samplename and column is color'''
    categories = samplesheet[column].unique()
    colormap = dict(zip(map(str, categories), palette))
    cmap = pd.Series(samplesheet[column]).map(colormap)
    if return_cmap is True:
        return cmap, colormap
    else:
        return cmap


def samplesheet_add_feature(ds, name_feat=None, query_dict_counts=None, query_dict_metadata=None, local_dict=None):
    '''Take a single dataset and add a new column to the samplesheet metadata.
    Column is initialized as nan and values updated based on queries in query_dict
    ds: Singlet dataset
    name_feat: string with name containing new column to be added to samplesheet
    query_dict: Dictionary where key is a query language condition to select cells from
    localdict: local_dict (dict): A dictionary of local variables, useful if you are using
     @var assignments in your expression. By far the  most common usage of this argument is
      to set local_dict=locals()
    the counts_table based on count expression values and the value is the name of the
    cell type (value to be added to new samplesheet column.
    Example For alpha and beta cells: name_feat ='cell_type_ge'
    query_dict= {'INS > 3 & GCG < 3': 'beta', 'INS < 3 & GCG > 3':'alpha'}
    '''
    #initialize new column with NaNs
    ds.samplesheet[name_feat] = np.nan

    if query_dict_counts:
        for condition, ct in query_dict_counts.items():
            cells_names = ds.query_samples_by_counts(condition, inplace=False, local_dict=local_dict).samplenames
            ds.samplesheet.loc[cells_names,name_feat] = ct

    elif query_dict_metadata:
        for condition, ct in query_dict_metadata.items():
            cells_names = ds.query_samples_by_metadata(condition, inplace=False, local_dict=local_dict).samplenames
            ds.samplesheet.loc[cells_names,name_feat] = ct

    return ds


def filter_samplesheet(dataset=None, filter_dict={}):
    '''
    Takes a dataset and filters samplesheet based on a dictionary where keys are columns in samplesheet and items
    is a list of values in the column (e.g. filter_dict={'cell_type': ['alpha', 'beta']})'''
    s = dataset.copy()
    for key, items in filter_dict.items():
        s.samplesheet = s.samplesheet[s.samplesheet[key].isin(items)]
    return s


def add_column_based_gene_expression(dataset, gene, threshold=0, new_col_name=None):
    '''adds column new_col_name in samplesheet based in gene expression of gene gene
    ds: dataset
    gene: gene name
    threshold: threshold to separate
    new_col_name: name of new column'''
    ds1 = dataset.copy()
    if new_col_name is None:
        new_col_name = 'gene_group_'+gene

    name_above = gene + '+'
    name_below = gene + '-'

    ds1.samplesheet.loc[(ds1.counts.loc[gene,:] > threshold),new_col_name] = name_above
    ds1.samplesheet.loc[(ds1.counts.loc[gene,:] < threshold), new_col_name] = name_below
    return ds1


def add_column_based_phenotype(dataset, phenotype, threshold=0, new_col_name=None,):
    '''adds column new_col_name in samplesheet based in gene expression of gene gene
    ds: dataset
    gene: gene name
    threshold: threshold to separate
    new_col_name: name of new column'''
    ds1 = dataset.copy()
    if new_col_name is None:
        new_col_name = 'pheno_group_'+phenotype

    name_above = phenotype + '1'
    name_below = phenotype + '2'

    ds1.samplesheet.loc[(ds1.samplesheet.loc[:,phenotype] > threshold),new_col_name] = name_above
    ds1.samplesheet.loc[(ds1.samplesheet.loc[:,phenotype] < threshold), new_col_name] = name_below
    return ds1

####
#Functions to send data to voom and do DE
#######

#function to export stats of groups when splitting dataset for DE
def export_filter_conditions(df_mdata,filter_dict1,filter_dict2,filename_out='/Users/joan/Desktop/last_DEcompare.xlsx'):
    writer = pd.ExcelWriter(path=filename_out)
    filter_dict1 = pd.DataFrame.from_dict(filter_dict1, orient='index')
    header1 = [''] * filter_dict1.shape[1]
    header1[0] = 'Group 1 filtering'
    filter_dict1.to_excel(writer, sheet_name='stats',startcol=0,header=header1)
    #move savin position in excel
    n_cols = filter_dict1.shape[1] + 2
    filter_dict2 = pd.DataFrame.from_dict(filter_dict2, orient='index')
    header2 = [''] * filter_dict2.shape[1]
    header2[0] = 'Group 2 filtering'
    filter_dict2.to_excel(writer, sheet_name='stats', startcol=n_cols,header=header2)
    n_cols += filter_dict2.shape[1] + 2
    donor_stats = df_mdata.groupby('group')['Sex'].value_counts().unstack().fillna(0).T
    donor_stats.to_excel(writer, sheet_name='stats',startcol=n_cols, startrow=0)
    n_cols += donor_stats.shape[0] + 2
    donor_stats = df_mdata.groupby('group')['DonorID'].value_counts().unstack().fillna(0)
    donor_stats['Total'] = donor_stats.sum(axis=1)
    donor_stats = donor_stats.T
    donor_stats.to_excel(writer, sheet_name='stats', startcol=n_cols, startrow=0)
    writer.save()



########
# DE functions or percent
########

def DE_presplited(ds1, ds2, ids=['g1','g2'], min_cells=10,
    min_mean_counts=0,
    column=None,
    method='mann-whitney',
    min_log2FC=0.25,
    min_pct_diff = 0.1,
    min_pct_one_group= 0.1,
    min_pct_ratio=0.1,
    logbase=2,
    pseudocount=1):
    '''Does DE in dataset base in one column samplesheet, only works for two classes in column'''
    from statsmodels.sandbox.stats.multicomp import multipletests

    ids = ids

    #make two groups
    g1 =  ds1.copy()
    g2 =  ds2.copy()

    #calculate log fold change
    logFC1 =np.log2(pseudocount+g1.counts.unlog(base=logbase).mean(axis=1))
    logFC2 = np.log2(pseudocount+g2.counts.unlog(base=logbase).mean(axis=1))
    logFC = logFC1 - logFC2
    logFC.rename('log2_FC', inplace=True)

    #calculate percent expression
    pct1 = ( g1.counts>0 ).sum(axis=1) / g1.counts.shape[1]
    pct1.rename('pct1', inplace=True)
    pct2 =  ( g2.counts>0 ).sum(axis=1) / g2.counts.shape[1]
    pct2.rename('pct2', inplace=True)
    percent_diff = pct1 - pct2
    percent_diff.rename('pct_diff', inplace=True)
    #percent ratio: (p1-p2) / max(p1,p2) : to test for bimodal expression
    pct_max = np.max([pct1,pct2],axis=0)
    pct_rat = np.abs(percent_diff) / pct_max
    #merge stats
    names = ['log2_FC', 'pct_diff', str('pct_'+ids[0]),str('pct_'+ids[1]),'pct_ratio',str('log2_mean_'+ids[0]),str('log2_mean_'+ids[1])]
    results = pd.concat([logFC,percent_diff,pct1,pct2,pct_rat,logFC1,logFC2], keys=names, axis=1)
    #filter genes that are unlikely to be DE before testing
    cond1 = np.abs(results['log2_FC']) > min_log2FC
    cond2 = np.abs(results['pct_diff']) > min_pct_diff
    cond3 = ((np.abs(results['pct_'+ids[0]]) > min_pct_one_group) | (np.abs(results['pct_'+ids[1]]) > min_pct_one_group))
    cond4 = np.abs(results['pct_ratio']) > min_pct_ratio
    genes_test = results[cond1 & cond2 & cond3 & cond4].index.values

    g1.counts = g1.counts.loc[genes_test,:]
    g2.counts = g2.counts.loc[genes_test,:]


    #Calculate p value
    DE_genes = g1.compare(g2, method=method)
    DE_genes['pval_adj'] = multipletests(DE_genes['P-value'], method='fdr_bh')[1]
    DE_genes.sort_values('pval_adj', inplace=True)


    DE_genes = DE_genes.join(results)

    return DE_genes



def DE_metadata(ds,
    column=None,
                min_cells=10,
    min_mean_counts=0,
    method='mann-whitney',
    min_log2FC=0.25,
    min_pct_diff = 0.1,
    min_pct_one_group= 0.1,
    min_pct_ratio=0.1,
    logbase=2,
    pseudocount=1):
    '''Does DE in dataset base in one column samplesheet, only works for two classes in column'''
    from statsmodels.sandbox.stats.multicomp import multipletests
    s1 = ds.copy()
    s1.counts = s1.counts[(s1.counts > 0).sum(axis=1) > min_cells]
    s1.counts = s1.counts[s1.counts.mean(axis=1) > min_mean_counts]

    #split dataset between two datasets
    s1_split = s1.split(column)
    ids = s1.samplesheet[column].unique()

    #make two groups
    g1 =  s1_split[ids[0]]
    g2 =  s1_split[ids[1]]

    #calculate log fold change
    logFC1 =np.log2(pseudocount+g1.counts.unlog(base=logbase).mean(axis=1))
    logFC2 = np.log2(pseudocount+g2.counts.unlog(base=logbase).mean(axis=1))
    logFC = logFC1 - logFC2
    logFC.rename('log2_FC', inplace=True)

    #calculate percent expression
    pct1 = ( g1.counts>0 ).sum(axis=1) / g1.counts.shape[1]
    pct1.rename('pct1', inplace=True)
    pct2 =  ( g2.counts>0 ).sum(axis=1) / g2.counts.shape[1]
    pct2.rename('pct2', inplace=True)
    percent_diff = pct1 - pct2
    percent_diff.rename('pct_diff', inplace=True)
    #percent ratio: (p1-p2) / max(p1,p2) : to test for bimodal expression
    pct_max = np.max([pct1,pct2],axis=0)
    pct_rat = np.abs(percent_diff) / pct_max
    #merge stats
    names = ['log2_FC', 'pct_diff', str('pct_'+ids[0]),str('pct_'+ids[1]),'pct_ratio',str('log2_mean_'+ids[0]),str('log2_mean_'+ids[1])]
    results = pd.concat([logFC,percent_diff,pct1,pct2,pct_rat,logFC1,logFC2], keys=names, axis=1)
    #filter genes that are unlikely to be DE before testing
    cond1 = np.abs(results['log2_FC']) > min_log2FC
    cond2 = np.abs(results['pct_diff']) > min_pct_diff
    cond3 = ((np.abs(results['pct_'+ids[0]]) > min_pct_one_group) | (np.abs(results['pct_'+ids[1]]) > min_pct_one_group))
    cond4 = np.abs(results['pct_ratio']) > min_pct_ratio
    genes_test = results[cond1 & cond2 & cond3 & cond4].index.values

    g1.counts = g1.counts.loc[genes_test,:]
    g2.counts = g2.counts.loc[genes_test,:]


    #Calculate p value
    DE_genes = g1.compare(g2, method=method)
    DE_genes['pval_adj'] = multipletests(DE_genes['P-value'], method='fdr_bh')[1]
    DE_genes.sort_values('pval_adj', inplace=True)


    DE_genes = DE_genes.join(results)

    return DE_genes

def log_FC(ds1, ds2, logbase=2, pseudocount=1):
    ''' calculate logFC of dataset '''
    g1 = ds1.copy()
    g2 = ds2.copy()

    #calculate log fold change
    logFC1 =np.log2(pseudocount+g1.counts.unlog(base=logbase).mean(axis=1))
    logFC2 = np.log2(pseudocount+g2.counts.unlog(base=logbase).mean(axis=1))
    logFC = logFC1 - logFC2
    logFC.rename('log2_FC', inplace=True)

    return logFC


def fisher_exact_median(group1,group2):
    '''Group1 and group2 are two pandas series containing values that
    will be tested againts a fisher exact test of the mean of the whole dataset '''
    from scipy.stats import fisher_exact
    x_mean = pd.concat([group1,group2]).median()
    #x_mean = t_test.mean()
    t_table = np.empty((2,2))
    t_table[0][0] =(group1.dropna()> x_mean).sum()
    t_table[0][1] = (group1.dropna()< x_mean).sum()
    t_table[1][0]=(group2.dropna()> x_mean).sum()
    t_table[1][1] = (group2.dropna()< x_mean).sum()
    return fisher_exact(t_table)[1]


def percent_genes_observed(dataset, genelist=None , threshold=0, sample_label='no label', reset_index=True):
    '''Creates dataframe with percentage of cells expressing a gene above threshold
    dataset: datataset of interest
    genelist: columns in dataset.count that will be used. If not use all.
    threshold: threshold ni gene expression
    sample_label: method adds column to output dataframe with specified label (e.g. sample_label='Ins+')
    reset_index: If True puts index as new column called 'gene' and resets index to numbers, for easy concatenation'''
    ds =dataset.copy()
    if genelist:
        ds.counts = ds.counts.loc[genelist,:]

    res = (ds.counts > threshold).sum(axis=1) / (ds.counts).count(axis=1)
    res = pd.DataFrame(res,columns=['percent'])
    res['errauthoror'] = np.sqrt((ds.counts >0).sum(axis=1)) / (ds.counts).count(axis=1)
    res['sample'] = sample_label
    if reset_index is True:
        res['gene'] = res.index
        res.reset_index(drop=True,inplace=True)
    return res


########
# Correlations functions
########

def correlate_noNaN_phenotype(Dataset,phenotype_list):
    '''
    Takes Dataset and computes correlation between a value in the samplesheet and gene counts
    based on function singlet correlate_features_phenotypes.
    It removes cells that have NaN for a given samplesheet feature
    Dataset: singlet dataset
    phenotype list: list with columns in samplesheet
    '''
    dict_corr = {}
    for col in phenotype_list:
        dummyds = Dataset.copy()
        dummyds.samplesheet = dummyds.samplesheet.dropna(subset=[col], inplace=False)
        dict_corr[col]= dummyds.correlation.correlate_features_phenotypes(phenotypes=col,features='all',method='spearman', fillna=None)
    df_corr = pd.DataFrame.from_dict(dict_corr)
    return df_corr


def correlate(x, y, method):
    from scipy.stats import rankdata
    if method == 'pearson':
        xw = x
        yw = y
    elif method == 'spearman':
        xw = np.zeros_like(x, float)
        for ii, xi in enumerate(x):
            xw[ii] = rankdata(xi, method='average')
        yw = np.zeros_like(y, float)
        for ii, yi in enumerate(y):
            yw[ii] = rankdata(yi, method='average')
    else:
        raise ValueError('correlation method not understood')

    xw = ((xw.T - xw.mean(axis=1)) / xw.std(axis=1)).T
    yw = ((yw.T - yw.mean(axis=1)) / yw.std(axis=1)).T
    n = xw.shape[1]
    r = np.dot(xw, yw.T) / n
    return r


def pval_correlate(dataset, genes, phenotype, method='spearman', p_onesided=False, max_perm=1e6):
    '''Takes dataset, genename and phenotype and returns correlation, pvalue and standard error'''
    ds = dataset.copy()
    #read dataset counts for gene
    out = {}
    for gene in genes:

        #read gene correlations and array with phenotype
        x = np.array([ds.counts.loc[gene].values])
        y = ds.samplesheet.loc[:, [phenotype]]
        y = y.values.T

        #Calculate correlation for non-permuted dataset
        corr = correlate(x=x, y=y, method=method)
        #FIXME, picks element of array when doing correlation one gene at a time
        corr = corr[0][0]

        #if gene is nan output standard values and jump to next gene
        if np.isnan(corr):
            out[gene] = { 'correlation': corr , 'pval': 1, 'pval_err':  1, 'permutations': 0 }

        else:

            #calculate matrix x with repeats of gene expression and shuffle cells
            n_perm = 100
            p_val = 0
            se = 0

            while(n_perm <= max_perm):

                xp = np.tile(x, (n_perm,1))
                for row in xp:
                    np.random.shuffle(row)

                #calculate correlations for permutated dataset
                res = correlate(xp, y, method)

                #calculate p value for rare case where we want one-sided distribution
                if p_onesided:
                    if corr > 0:
                        p_val = (np.sum(res >= corr)) / n_perm
                    else:
                        p_val = (np.sum(res <= corr)) / n_perm
                #calculate p value for usual case two-sided distribution
                else:
                    p_val = np.sum(np.abs(res) >= abs(corr)) / n_perm

                #calculate standard deviation
                se= np.sqrt(p_val*(1-p_val)/n_perm)

                #if p_val is determined with good error stop else update value for next round
                if(se < 0.25 * p_val):
                    break
                else:
                    n_perm = n_perm * 10

            #save correlations and pvalues
            out[gene] = { 'correlation': corr , 'pval': p_val, 'pval_err':  se, 'permutations': n_perm }

    #make final pandas df and return
    dfout = pd.DataFrame.from_dict(out,orient='index')
    return(dfout)


#####
# Correlation plots in 1D
#####


def correlate_plots_sorted_collapsed(df,
    figsize=(20,10),
    xlim=(-1000,38000),
    legend_cols=3,
    loc=3,
    bbox_to_anchor=(0., 1.02, 1., .102)):
    '''
    Plots ordered correlations for all phenotypes after doing correlate
    Takes Dataframe and for each column sorts values and makes
    xlim: range of genes to plt and order
    ncols: number of columns in legensd

    '''

    dfsorted = df.apply(lambda x: x.sort_values().values)
    dfsorted = dfsorted.reset_index(drop=True)
    ax = dfsorted.plot(figsize=figsize,xlim=xlim)
    ax.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=legend_cols, mode="expand", borderaxespad=0.)
    return ax


def correlate_plots_sorted_subplots(df,
    figsize=(40,20),
    subplots_rows=4,
    subplots_cols=6,
    xlim=(-1000,38000),
    remove_plots_list=[21,22,23]):
    '''
    Plots all correlations for all phenotypes after doing correlate
    Takes Dataframe and for each column sorts values and makes one plot for each
    correlate in multiplot
    xlim: range of genes to plt and order
    remove_plots_list: takes list with axes index and removes them (e.g. [21,22,23]
    unless None
    '''

    fig,axs = plt.subplots(nrows=subplots_rows,ncols=subplots_cols,figsize=figsize,sharex=True)
    axs = axs.flatten()
    for i,col in enumerate(df.columns):
        s =  df[col].sort_values(ascending=True).dropna().reset_index(drop=True)
        if not s.empty:
            s.plot(xlim=xlim, title=col, ax=axs[i])
        axs[i].set_ylabel(r'spearman $\rho$')
        axs[i].set_xlabel(r'gene$')
    if remove_plots_list:
        for i in remove_plots_list:
            fig.delaxes(axs[i])

    return fig, axs


def correlate_plots_scatter_subplots(Dataset,
    dfcorr,
    pval=True,
    phenotype=None,
    color_gene_name=None,
    color_phenotype=None,
    figsize=(16,10),
    subplots_rows=4,
    subplots_cols=5,
    sharex=True,
    cmap='viridis',
    alpha=0.7,
    dotsize=20,
    remove_plots_list=None):
    '''
    Plots phenotype (x axis) vs gene expression (y axis) using a singlet dataset
    using a pandas series where index=gene value=correlation value
    Dataset: singlet dataset containing counts table and samplesheet with metadata
    Phenotype: string matching phenotype (column in samplesheet) to use in plot x-axis
    dfcorr= Pandas dataframe containing genename in index (matching counts table) and correlation value
    and additionally pvalue column.
    color_gene_name: Optional. string with an additional gene to plot its colored gene expression (e.g. INS)
    dfcorr correlations for all phenotypes after doing correlate
    color_phenotype_cat or color_phenotype_cont: Colors cells based on a categorycal phenotype (set2 recommended) or
    a continouous phenotype, max-min value range (viridis recommended)
    Takes Dataframe and for each column sorts values and makes one plot for each
    correlate in multiplot
    xlim: range of genes to plt and order
    figsize, subplots_rows, subplots-cols, sharex: plot parameters
    cmap: If None, gray dots, else matplotlib cmap to plot superimposed gene expression of color_gene_name
    alpha: set dot transparency
    remove_plots_list: takes list with axes index and removes them (e.g. [21,22,23] unless None
    '''
    import matplotlib as mpl
    from matplotlib import cm

    #set multiplot
    fig,axs = plt.subplots(nrows=subplots_rows,
        ncols=subplots_cols,
        figsize=figsize,
        squeeze=False,
        sharex=sharex)

    axs = axs.flatten()

    ds = Dataset.copy()
    #remove all na in phenotype variable. otherwise fills with zero!!
    ds.samplesheet = ds.samplesheet.loc[ds.samplesheet[phenotype].dropna().index]
    #make plot for each gene in dataframe dfcorr and plot correlation value
    for i,gene in enumerate(dfcorr.index):
        s1 = ds.samplesheet[phenotype]
        s2 = ds.counts.loc[gene]
        vec = pd.concat([s1, s2], axis=1, join='inner')
        #if gene expression had na we should find another option
        #removed the na and moved it upwards as otherwise color code of plot is wrong
        #vec.dropna(axis=0, how='any',inplace=True)

        if color_gene_name:
            axs[i] = ds.plot.scatter_reduced_samples(vectors_reduced=vec,
                ax=axs[i], color_by=color_gene_name,
                s=dotsize,
                alpha=alpha,
                cmap=cmap)
        else:
            axs[i] = ds.plot.scatter_reduced_samples(vectors_reduced=vec,
                ax=axs[i], color_by=color_phenotype,
                s=dotsize,
                cmap=cmap,
                alpha=alpha)
        #remove labels as they are many times long and mess up subplots, leave as title
        axs[i].set_xlabel('')

         #write value of correlation
        if pval == True:
            axs[i].text(0.67, 0.2, 'r= {:1.2f}'.format(dfcorr.loc[gene,'correlation']), transform=axs[i].transAxes)
            axs[i].text(0.65, 0.1, r'$p_{{val}}$= {:.1g}'.format(dfcorr.loc[gene,'pval']), transform=axs[i].transAxes)
        else:
            axs[i].text(0.67, 0.2, 'r= {:1.2f}'.format(dfcorr.loc[gene]), transform=axs[i].transAxes)

    ##prepare colorbar outside
    if color_gene_name:
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)

        #set normalization range for ColorBar masking potential NaN Values
        color_data = ds.counts.loc[color_gene_name]
        if np.isnan(color_data.values).any():
            unmask = ~np.isnan(color_data.values)
        else:
            unmask = np.ones(len(color_data), bool)

        cd_min = color_data.values[unmask].min()
        cd_max = color_data.values[unmask].max()

        norm = mpl.colors.Normalize(vmin=cd_min, vmax=cd_max)

    elif color_phenotype:
        if isinstance(cmap, str):
            cmap = cm.get_cmap(cmap)
        #prepare list of colours for colorbar
        color_data = ds.samplesheet.loc[:, color_phenotype]

        if hasattr(color_data, 'cat'):
            is_numeric = False
        else:
            is_numeric = np.issubdtype(color_data.dtype, np.number)
        #just to verify we print the result, so we can compare with ds result of plot
        print('PHENOTYPE IS OF TYPE NUMERIC (False means categorycal):' + str(is_numeric))

        #categorycal type receives a list of colors
        if not is_numeric:
            cd_unique = list(np.unique(color_data.values))
            c_unique = cmap(np.linspace(0, 1, len(cd_unique)))
        #numerical type is done as in gene expression
        else:
            if np.isnan(color_data.values).any():
                unmask = ~np.isnan(color_data.values)
            else:
                unmask = np.ones(len(color_data), bool)

            cd_min = color_data.values[unmask].min()
            cd_max = color_data.values[unmask].max()

            norm = mpl.colors.Normalize(vmin=cd_min, vmax=cd_max)

        #norm = mpl.colors.Normalize(vmin=cd_min, vmax=cd_max)

    #make colorbar
    if color_gene_name:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            label_colorbar = color_gene_name+", $\log_{10}\mathrm{cpm}$"
            plt.colorbar(sm,
                cax=fig.add_axes((1, 0.6,0.01,0.2)),
                label=label_colorbar,
                ticks=[0,1,2,3,4,5,6])
    #add colorbar if phenotype is numeric (not categorical):
    if color_phenotype:
        if is_numeric:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm._A = []
            label_colorbar = color_phenotype
            plt.colorbar(sm,
                cax=fig.add_axes((1, 0.6,0.01,0.2)),
                label=label_colorbar)
                #ticks=[0,1,2,3,4,5,6])
            #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])

    #remove unused subplots if needed
    if remove_plots_list:
        for i in remove_plots_list:
            fig.delaxes(axs[i])

    #add title to plot and common x label
    fig.suptitle(x=0.5, y=1.0,t=phenotype)
    fig.text(0.5, 0.01, phenotype, ha='center')

    return fig,axs

### make binning

def make_colormap_from_palette(palette_name='Set2', n=2,
                              bins=2):
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    colors = sns.color_palette(palette=palette_name,n_colors=n)
    return LinearSegmentedColormap.from_list('my_cmap', colors, N=bins)


def make_colormap_from_colorlist(list_colors=['#deebf7', '#08306b'],
                              bins=100):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('my_cmap', list_colors, N=bins)


def make_binning_phenotype(df, column,
    log=False, bins= 5, vmin=None,
    vmax=None,
    decimals=1,
    labels=None,
    nan_value=None):
    '''returns s, bins(new column with binned data, bins in data)
    bins : number of bins or list with intervals to set up bins (in that case provide labels)
    labels: labels for each bin if provided (if not provided picks the average value for each bin)
    vmin : set minimum value of bins (and collapse lower data to min)vmax : set maximum value of bins (and collapse higher data to max)
    nan_value : value or string to assign to nans if needed'''

    if column not in df.columns:
        ValueError("Phenotype not found in Table: ", column)

    ny = df[column]

    #constrain data to range of interest
    if vmin:
        ny = np.maximum(ny, vmin)

    if vmax:
        ny = np.minimum(ny, vmax)

    if log:
        vmin = np.maximum(vmin,1e-10)
        ny=np.log10(np.maximum(vmin,ny))
        #sets bins for logscale to decades
        vmax = np.amax(ny)
        vmax = round(vmax +0.5)

    s, bins =pd.cut(ny, bins=bins,
        include_lowest=True, retbins=True, labels= labels)

    if labels:
        if nan_value:
            s = s.cat.add_categories([nan_value])
            s = s.fillna(nan_value)

    if not labels:
        mid = [(a + b) /2 for a,b in zip(bins[:-1], bins[1:])]

        s = s.cat.rename_categories(mid)
        bins = np.around(mid, decimals=decimals)
        if nan_value:
            s = s.cat.add_categories([nan_value])
            s = s.fillna(nan_value).astype('float').round(decimals=decimals)
            bins = np.insert(bins, 0, nan_value, axis=0)

    return s, bins

#make tags
def make_group_tags(df, dict_groups):
    '''df: pandas Dataframe with columns that are found as elements in the dictionart dict_groups (genes)
    dict_groups: dictionary where keys are group (metabolism) and elements are lists of elements (genes)'''
    s = df.copy()
    genes = []
    for key, par in dict_groups.items():
        genes = genes + par

    ds_genetag = pd.DataFrame(index=s.index)
    for group, genes in dict_groups.items():
        for gene in genes:
            if gene in ds_genetag.index:
                ds_genetag.loc[gene,'group'] = group
    ds_genetag = ds_genetag[ds_genetag['group'].notnull()]

    n_groups =ds_genetag['group'].unique().shape[0]
    palette_col = sns.husl_palette(n_groups,s=0.7)
    lut_gene = dict(zip(ds_genetag['group'].unique(), palette_col))
    return lut_gene


###colormap utils 2
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 1.0.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.0 and 1.0.
    '''
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import AxesGrid
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap



### Plot gene counts with colorbar next to it

def plots_scatter_tsne_genes(Dataset,
    vectors_reduced,
    gene_list= [] ,
    phenotype_list = [],
    figsize=(10,10),
    subplots_rows=4,
    subplots_cols=5,
    sharex=True,
    sharey=True,
    cmap='viridis',
    alpha=0.7,
    dotsize=20,
    remove_plots_list=None):

    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from textwrap import wrap
    import re

    #set multiplot
    fig,axs = plt.subplots(nrows=subplots_rows,
        ncols=subplots_cols,
        figsize=figsize,
        squeeze=False,
        sharex=sharex,
        sharey=sharey)

    axs = axs.flatten()

    ds = Dataset.copy()
    vec = vectors_reduced.copy()
    #counter to add new figures from ephysiology below
    n=0
    #make plot for each gene in dataframe dfcorr and plot correlation value
    for i,gene in enumerate(gene_list):

        axs[i] = ds.plot.scatter_reduced_samples(vectors_reduced=vec,
            ax=axs[i], color_by=gene,
            s=dotsize,
            alpha=alpha,
            cmap=cmap)

        #remove labels as they are many times long and mess up subplots, leave as title
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(gene)


        divider = make_axes_locatable(axs[i])
        axs[i] = divider.append_axes('right', size='10%', pad=0.05)


        color_data = ds.counts.loc[gene]
        if np.isnan(color_data.values).any():
            unmask = ~np.isnan(color_data.values)
        else:
            unmask = np.ones(len(color_data), bool)


        cd_min = color_data.values[unmask].min()
        cd_max = color_data.values[unmask].max()


        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=cd_min, vmax=cd_max)
        mpl.colorbar.ColorbarBase(axs[i], cmap=cmap, norm=norm, orientation='vertical',label='log10(cpm)')
        n=i+1
        #phenotypes now

            #make plot for each gene in dataframe dfcorr and plot correlation value
    for i,gene in enumerate(phenotype_list):
        i=n+i
        axs[i] = ds.plot.scatter_reduced_samples(vectors_reduced=vec,
            ax=axs[i], color_by=gene,
            s=dotsize,
            alpha=alpha,
            cmap=cmap)

        #remove labels as they are many times long and mess up subplots, leave as title
        axs[i].set_ylabel('')
        axs[i].set_xlabel('')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        phenotype_title = re.sub(r"(\w)([A-Z])", r"\1 \2", gene)
        axs[i].set_title("\n".join(wrap(phenotype_title,20)))

        divider = make_axes_locatable(axs[i])
        axs[i] = divider.append_axes('right', size='10%', pad=0.05)


        color_data = ds.samplesheet.loc[:,gene]
        if np.isnan(color_data.values).any():
            unmask = ~np.isnan(color_data.values)
        else:
            unmask = np.ones(len(color_data), bool)


        cd_min = color_data.values[unmask].min()
        cd_max = color_data.values[unmask].max()


        cmap = mpl.cm.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=cd_min, vmax=cd_max)
        mpl.colorbar.ColorbarBase(axs[i], cmap=cmap, norm=norm, orientation='vertical')


    #remove unused subplots if needed
    if remove_plots_list:
        for i in remove_plots_list:
            fig.delaxes(axs[i])

    #add title to plot and common x label
    #fig.suptitle(x=0.5, y=1.0,t=phenotype)
    #fig.text(0.5, 0.01, phenotype, ha='center')

    return fig,axs




######
#correlation plots on 2D with highlighted features
#####

def get_informative_correlated_genes(df, cell_types=None, top_genes=None, threshold=0.55):
    '''
    Calculate euclidean distance of correlation of each gene to multiple phenotypes to pick features that
    are outside a hypersphere of a given threshold and that might be correlated to the features. Otherwise
    just pick genes furthest apart

    df: contains correlation of genes (rows) to phenotype for one or multiple subcategories (i.e. cell types)
    selected using splitcellsby in function get_correlation_features_phenotype()

    cell_types: List of columns used to calculate distance, if None columns all columns are used

    top_genes: Integer value of top genes to highlight instead of constant threshold distance.

    threshold: If top_genes is None,use this distance threshold to 0,0 to highlight all genes above.



    '''

    if cell_types:
        df = df[[cell_types]]

    #calculate euclidean distance of correlation matrix
    df_dist = np.sqrt(np.power(df, 2).sum(axis=1))

    if top_genes:
        return df_dist.nlargest(top_genes)

    #else return dataframe with genes with a correlation above the threshold
    return df_dist[df_dist>threshold]


def plot_informative_correlated_genes(df, cell_types = None, top_genes=40, threshold = 0.5,
    additional_genes = None, figsize=(12, 12), kind='scatter', fontsize='12',dot_size=120, title=None,
    fontsize_axis= 20, fontsize_labels=13, alpha=0.1):
    '''
    Takes a df containing two or multiple correlations for a gene list (rows)
    and plots scatter plot for two selected columns (cell_types) of correlations.
    cell_types: two columns of dataframe (can be cell types or phenotypes...)
    Highlights top 5 correlated and anticorrelated genes for each cell_type in red.
    threshold: Highlights genes that have distance greater than threshold from 0,0 correlation in blue
    and adds name.
    additional_genes: list with additional genes to highlight
    '''
    from adjustText import adjust_text

    mpl.rcParams['xtick.labelsize'] = fontsize_axis
    mpl.rcParams['ytick.labelsize'] = fontsize_axis
    mpl.rcParams['axes.titlesize'] = fontsize_axis
    mpl.rcParams['axes.labelsize'] = fontsize_axis

    #function only plots 2D correlation of features
    if (len(df.columns) is not 2) and (len(cell_types) is not 2):
        raise ValueError('Plot requires 2 columns of feature correlations to plot 2D plot')

    elif cell_types:

        df = df[cell_types]

    #pick genes that have a highest distance to zero (either top_genes or use threshold in radius)
    df_features = get_informative_correlated_genes(df, top_genes=top_genes, threshold=threshold)
    list_feat_rad = df_features.index.tolist()
    df_feat_rad = df.loc[df_features.index,:]


    #pick top correlated and anticorrelated genes in each axis
    list_feat_xy = []
    for cell_type in cell_types:
        list_feat_xy.extend(df.nlargest(5,columns=cell_type).index.tolist())
        list_feat_xy.extend(df.nsmallest(5,columns=cell_type).index.tolist())
    #select unique genes
    list_feat_xy = np.unique(list_feat_xy)
    #select features to plot as most correlated and anticorrelated to each axis
    df_feat_xy = df.loc[list_feat_xy,:]


    #finally add list of manually highlighted genes

    if additional_genes:
        list_additional = list(set(additional_genes).intersection(df.index.tolist()))
        df_feat_add = df.loc[list_additional,:]
    else:
        list_additional = []


    #make list of labels containing both top genes per axis and radial genes
    labels_list = np.append(list_feat_xy, list_feat_rad)
    labels_list = np.append(labels_list, list_additional)
    labels_list = np.unique(labels_list)
    df_labels = df.loc[labels_list,:]

    fig, ax = plt.subplots(figsize=figsize)

    #Plot all correlations
    ax = df.plot(df.columns[0], df.columns[1], kind=kind, ax=ax, s=dot_size, linewidth=0,color='gray', alpha=alpha)
    #ax = sns.kdeplot(df.columns[0], df.columns[1], ax=ax)
    #Plot all correlations that pass threshold in blue
    ax = df_feat_rad.plot(df_feat_rad.columns[0], df_feat_rad.columns[1], kind=kind, ax=ax, s=dot_size, linewidth=0,color='blue')
    #Plot most correlated and anticorrelated genes for each axis in red
    ax = df_feat_xy.plot(df_feat_xy.columns[0], df_feat_xy.columns[1], kind=kind, ax=ax, s=dot_size, linewidth=0,color='red')
    #Plot additional genes in blue too
    if additional_genes:
        ax = df_feat_add.plot(df_feat_add.columns[0], df_feat_add.columns[1], kind=kind, ax=ax, s=dot_size, linewidth=0,color='blue')


    ax.set_xlabel("Correlation, " + df.columns[0])
    ax.set_ylabel("Correlation, " + df.columns[1])
    ax.set_ylim(-1.,1.)
    ax.set_xlim(-1.,1.)

    texts = []
    #where k is the index (gene name) and v are the columns (correlation values for each axis to plot)
    for  k, v in df_labels.iterrows():
        texts.append(plt.text(v[0], v[1], k, size=fontsize_labels))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5), force_text=0.5)

    if title:
        ax.set_title(title)

    return {'fig': fig, 'ax': ax}

########

def plot_scatter_highlight(df, xcol = None, ycol= None, thres_col=None, threshold_y = None,
    threshold_x_symmetric = None, thres_threscol = None, figsize=(12, 12), kind='scatter', fontsize='12',dot_size=10, title=None,
    fontsize_axis=20, fontsize_labels=13, alpha=0.5):
    '''
    Takes a df containing mutiple columns and plots scatter plot
    top40: if True does not apply the threshold and just plots the 40 top genes in terms of x-y distance
    '''
    from adjustText import adjust_text

    mpl.rcParams['xtick.labelsize'] = fontsize_axis
    mpl.rcParams['ytick.labelsize'] = fontsize_axis
    mpl.rcParams['axes.titlesize'] = fontsize_axis
    mpl.rcParams['axes.labelsize'] = fontsize_axis

    df_features = df.copy()

    #pick genes that are at more extreme values than thresholds
    if threshold_x_symmetric:
        df_features = df_features[df_features[xcol].abs() > threshold_x_symmetric]
        list_feat = df_features.index.tolist()

    if threshold_y:
        df_features = df_features[df_features[ycol] > threshold_y]
        list_feat = df_features.index.tolist()

    if thres_col:
        df_features = df_features[df_features[thres_col] > thres_threscol]
        list_feat = df_features.index.tolist()


    #make list of labels containing both top genes per axis and radial genes
    labels_list = np.unique(list_feat)
    df_labels = df.loc[labels_list,:]

    fig, ax = plt.subplots(figsize=figsize)

    #Plot all correlations
    ax = df.plot(xcol, ycol, kind=kind, ax=ax, s=dot_size, linewidth=0,color='#fee0d2', alpha=alpha)#orange:#8da0cb
    #Plot all correlations that pass threshold in blue
    #Plot most correlated and anticorrelated genes for each axis in red
    ax = df_features.plot(xcol, ycol, kind=kind, ax=ax, s=dot_size, linewidth=0,color='#de2d26')#orange:fc8d62


    texts = []
    #where k is the index (gene name) and v are the columns (correlation values for each axis to plot)
    for  k, v in df_labels.iterrows():
        texts.append(plt.text(v[xcol], v[ycol], k, size=fontsize_labels))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='#de2d26', lw=0.5), force_text=0.5)

    if title:
        ax.set_title(title)

    return {'fig': fig, 'ax': ax}



################
# New correlations and electrophysiology pvalues
###############



def filter_dataset_ephys(ds, col, min_val=-1e6, max_val=1e6):
    '''
    removes NaNs by default too
    s: dataset
    col:column in electrophysiology to filter
    minimim and maximum thresholds to use'''
    s =ds.copy()
    #remove columns where phenotype is NaN
    s.samplesheet = s.samplesheet.dropna(subset=[col], inplace=False)
    #threshold data of parameter
    s.samplesheet = s.samplesheet[s.samplesheet[col] > min_val]
    s.samplesheet = s.samplesheet[s.samplesheet[col] < max_val]
    return s


def filter_genes_pp(df, min_cells=5, min_counts=10):
    '''Filter a dataframe to remove genes with less total counts than min_counts
    and genes not seen in at least min_cells'''
    s = df.copy()
    #filter genes not seen in cells
    f1 = (s>0).sum(axis=1)
    test = s.loc[f1>min_cells,:]
    #filter genes with not enough counts
    f2 = s.sum(axis=1)
    s = s.loc[f2>min_cells,:]
    return s


def filter_genes(df, min_cells = 0, dropout_val = 0, min_fraction=0, min_expression=0):
    '''filter genes in df (genes:rows, cells:columns)
    dropout_val: threshold value for genes to consider it observed
    min_cells: minimum number of cells above threshold
    min_fraction: minimum fraction of cells above threshold'''

    #remove thresholded genes
    s = df.copy()
    print('initial:')
    print(s.shape)
    #remove genes below a minimum value of average expression
    s = s[s.mean(axis=1)> min_expression]
    print('filtered mean:')
    print(s.shape)
    #Make nan any value below dropout_val
    #remove genes with a minimal count of cells or percentage of cells below thresholds
    s = s.clip(lower=dropout_val, inplace=False)
    s = s.replace(dropout_val, np.NaN, inplace=False)
    #remove genes observed in less than n cells
    print(s.shape)
    if min_cells >0:
        s = s.loc[s.dropna(thresh=min_cells, axis=0).index,:]
    #remove genes observed in less than n cells
    print('filtered n cells:')
    print(s.shape)
    if (min_fraction >=0) & (min_fraction <=1):
        print('aa')
        s = s[s.isnull().mean(axis=1) < 1-min_fraction]
        print('filtered fraction:')
        print(s.shape)
    else:
        printf('Fraction is not between 0 and 1')
    return df.loc[s.index,:]


def remove_dropout_genecount(df, dropout_val = 0):
    '''replace drop_out_val by nan'''
    return df.replace(dropout_val, np.NaN, inplace=False)

def correlation_pval(df_genes, series_ephys, method='ties', dropout_value=0):
    '''df_genes: table with gene counts
    series_ephys: series of one ephysiology value matched with df_genes
    method: 'ties' applies tie correction but considers dropouts into correlation /
    'omitdroput' filters dropouts and calculates correlation only in expressed cells without tie correction
    'pearsonr':calculates perason corr 'kendall' calculates kendall correlation '''

    from scipy.stats.mstats import spearmanr
    from scipy.stats import spearmanr as spearmanr2
    from scipy.stats import kendalltau
    from scipy.stats import pearsonr
    from statsmodels.sandbox.stats.multicomp import multipletests

    a = df_genes.T.copy()
    b = series_ephys.copy()

    if method is 'omitdropout':
        #remove dropouts from genecounts
        a = remove_dropout_genecount(df_genes, dropout_val = dropout_value).T.copy()

        res = a.apply(lambda col: spearmanr2(col, b, nan_policy='omit'), axis=0)
        c = pd.DataFrame(res, columns=[b.name])[b.name].apply(pd.Series)
        c.rename(columns={0:'correlation', 1:"pval"}, inplace=True)
        c['pval_adj'] = multipletests(c['pval'], method='fdr_bh')[1]
        c.sort_values(by='pval',inplace=True)
        c['pval'] = c['pval'].astype(float)

    if method is 'ties':

        res = a.apply(lambda col: spearmanr(col, b, use_ties=True), axis=0)
        c = pd.DataFrame(res, columns=[b.name])[b.name].apply(pd.Series)
        c.rename(columns={0:'correlation', 1:"pval"}, inplace=True)
        c['pval_adj'] = multipletests(c['pval'], method='fdr_bh')[1]
        c.sort_values(by='pval',inplace=True)
        c['pval'] = c['pval'].astype(float)

    if method is 'kendall':

        res = a.apply(lambda col: kendalltau(col, b), axis=0)
        c = pd.DataFrame(res, columns=[b.name])[b.name].apply(pd.Series)
        c.rename(columns={0:'correlation', 1:"pval"}, inplace=True)
        c['pval_adj'] = multipletests(c['pval'], method='fdr_bh')[1]
        c.sort_values(by='pval',inplace=True)
        c['pval'] = c['pval'].astype(float)

    if method is 'pearsonr':

        res = a.apply(lambda col: pearsonr(col, b), axis=0)
        c = pd.DataFrame(res, columns=[b.name])[b.name].apply(pd.Series)
        c.rename(columns={0:'correlation', 1:"pval"}, inplace=True)
        c['pval_adj'] = multipletests(c['pval'], method='fdr_bh')[1]
        c.sort_values(by='pval',inplace=True)
        c['pval'] = c['pval'].astype(float)

    return c




def correlate_plots_scatter_subplots_cat(Dataset,
    dfcorr,
    pval=True,
    phenotype= None,
    color_phenotype=None,
    color_data_order=None,
    subplots_rows=5,
    subplots_cols=4,
    figsize=3,
    aspect=1,
    y_jitter=0.1,
    alpha=0.7,
    spoint=20):
    '''Same functioning as correlate_plots_scatter_subplots but uses seaborn to do plot separating by categorical variable'''
    #make copy of datasrt to work and genelist
    s = Dataset.copy()
    genelist = dfcorr.index.tolist()


    #make vector of genecounts and phenotype to plt
    s1 = s.samplesheet[phenotype]
    s2 = s.counts.loc[genelist]
    vec = pd.concat([s1, s2.T], axis=1, join='inner')
    vec.dropna(axis=0, how='any',inplace=True)

    #add phenotype information for color
    vec2= pd.concat([vec, s.samplesheet[color_phenotype]],axis=1,join='inner')
    melted = vec2.melt(id_vars=[phenotype, color_phenotype], value_vars=genelist)
    melted.rename(columns={'value': 'log10(cpm)'},inplace=True)
    melted.rename(columns={phenotype: 'inactiv'},inplace=True)
    sns.set(font_scale=1.5)

    g = sns.lmplot(x='inactiv', y='log10(cpm)', data=melted, hue=color_phenotype,
                   col='variable', palette='husl', col_wrap=subplots_cols, size=figsize, aspect=aspect, hue_order=color_data_order,
                   sharex=True, sharey=True, legend=True, legend_out=True, x_estimator=None,
                   fit_reg=False, y_jitter=y_jitter, scatter_kws={'alpha': alpha, 's': spoint})

    for ax in g.axes.flat:
        #remove leading text from title
        ax.set_xlabel(' ')
        gene_name = ax.get_title().replace('variable = ', '')
        ax.set_title(gene_name)
        ax.text(x=0.7,y=0.9,s='r= {:1.2f}'.format(dfcorr.loc[gene_name, 'correlation']), fontsize=14, transform=ax.transAxes)
        ax.text(x=0.65,y=0.8,s=r'$p_{{val}}$= {:.1g}'.format(dfcorr.loc[gene_name, 'pval']), fontsize=14, transform=ax.transAxes)
    g.fig.text(0.4,0.01, phenotype,fontsize=14)

    return g

def filter_quantile_values(df, columns=None, qlow=0.1,qhigh=0.9, include_quantiles=True):
    '''takes dataframe df and for all the columns adds Nan to values above or below extreme quantiles
    df: dataframe to filter
    qlow: minimum quantilie
    qhigh: maximum quantile
    columns: do filtering in subset of columns
    include_quantiles: whether qlow and qhigh are included in the filtered dataset or removed
    (important for skewed distributions, with lots of zweros for instance)'''

    df1 = df.copy()
    s =df1
    if columns:
        s = df1[columns]

    if include_quantiles is True:
        s = s[s<=s.quantile(qhigh)]
        s = s[s>=s.quantile(qlow)]
    else:
        s = s[s<s.quantile(qhigh)]
        s = s[s>s.quantile(qlow)]

    df1[columns] = s

    return df1

def clip_quantile_values(df, columns=None, qlow=0.1,qhigh=0.9):
    '''takes dataframe df and for all the columns adds Nan to values above or below extreme quantiles
    df: dataframe to filter
    qlow: minimum quantilie
    qhigh: maximum quantile
    columns: do filtering in subset of columns
    include_quantiles: whether qlow and qhigh are included in the filtered dataset or removed
    (important for skewed distributions, with lots of zweros for instance)'''

    df1 = df.copy()
    s =df1
    if columns:
        s = df1[columns]

    s = s.clip(lower=s.quantile(qlow),upper=s.quantile(qhigh),axis=1)


   # df1[columns] = s

    return s



def filter_zeros(df, column):
    filter_zeros = df[column ]==0
    s = df[~filter_zeros]
    return s


#function to standarize columns of electrophysiologu ((x-mean)/sd)
def df_standarize(df,columns,norm=True):
    ts = df[columns]
    if norm:
        ts = (ts - ts.mean()) / ts.std()

    return ts


def get_zscores(df, num_bin=20):

    myMean = np.mean(df, axis=1)
    myVar = np.var(df, axis=1)
    bins = np.linspace(min(myMean), max(myMean), num_bin)

    df["mean"] = myMean
    df["var"] = myVar
    df["mean_bin"] = pd.cut(myMean, bins, right=True, labels=list(range(1,len(bins))), include_lowest=True)

    for _, group in df.groupby("mean_bin"):

        myDispersion = np.log10(group["var"] / group["mean"])
        myDispersionStd = np.std(myDispersion)

        if myDispersionStd == 0: z_scores = np.zeros(len(group))

        z_scores = (myDispersion - np.mean(myDispersion)) / myDispersionStd
        df.loc[group.index, "z_score"] = z_scores

    mean = df["mean"]
    z_score = df["z_score"]
    df.drop(["mean", "var", "mean_bin", "z_score"], axis=1, inplace=True) # clean up

    return mean, z_score, df


######
# PCA and tSNE plots on tables of electrophysiology
#####


def tsne(df, n_dims=2, perplexity=30, metric='correlation',
    rand_seed=0, early_exaggeration= 12.0, scale=True, **kwargs):
    '''t-SNE algorithm. Args:
    scale: Normalize to unit variance and mean 0 on columns of df (features are columns, cells are rows)
    This is the opposite than for counts tables!

    n_dims (int): Number of dimensions to use.
    perplexity (float): Perplexity of the algorithm.
    early_exaggeration: standard 12.0, should not change much things
    rand_seed (int): Random seed. -1 randomizes each run.
    **kwargs: Named arguments passed to the t-SNE algorithm.'''

    #from bhtsne import tsne
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    n = df.shape[0]
    if(n - 1 < 3 * perplexity):
        raise ValueError('Perplexity too high, reduce to <= {:}'.format((n - 1.)/3))

    data = df.copy()
    X = data.values

    if scale:
        X = StandardScaler().fit_transform(X)

    Y = TSNE(n_components=2,
             perplexity=perplexity,
             early_exaggeration=early_exaggeration,
             learning_rate=200.0, n_iter=1000,
             n_iter_without_progress=300,
             min_grad_norm=1e-07,
             metric=metric,
             init='random',
             verbose=0,
             random_state=rand_seed,
             method='barnes_hut',
             angle=0.5).fit_transform(X, y=None)

    #Y = tsne(data=X,
    #         dimensions=n_dims,
    #         perplexity=perplexity,
    #         theta=theta,
    #         rand_seed=rand_seed,
    #         **kwargs)

    vs = pd.DataFrame(Y, index=data.T.columns, columns=['dimension '+str(i+1) for i in range(n_dims)])
    return vs


def pca(df, n_dims=2, robust=True, random_state=None):
    '''Principal component analysis

    Args:
        n_dims (int): Number of dimensions (2+).
        transform (string or None): Whether to preprocess the data.
        robust (bool): Whether to use Principal Component Pursuit to \
                exclude outliers.

    Returns:
        dict of the left eigenvectors (vs), right eigenvectors (us) \
                of the singular value decomposition, eigenvalues \
                (lambdas), the transform, and the whiten function (for \
                plotting).
    '''
    from sklearn.decomposition import PCA

    X = df.copy()
    whiten = lambda x: ((x - X.mean(axis=0)) / X.std(axis=0, ddof=0))
    Xnorm = whiten(X)
    # NaN (e.g. features that do not vary i.e. dropout)
    Xnorm[np.isnan(Xnorm)] = 0

    if robust:
        #from numpy.linalg import matrix_rank
        #rank = matrix_rank(Xnorm.values)

        # Principal Component Pursuit (PSP)
        rpca = _RPCA(Xnorm.values)
        # L is low-rank, S is sparse (outliers)
        L, S = rpca.fit(max_iter=1000, iter_print=None)
        L = pd.DataFrame(L, index=X.index, columns=X.columns)
        whiten = lambda x: ((x - L.mean(axis=0)) / L.std(axis=0))
        Xnorm = whiten(L)
        Xnorm[np.isnan(Xnorm)] = 0
        #print('rPCA: original rank:', rank,
        #      'reduced rank:', matrix_rank(L),
        #      'sparse rank:', matrix_rank(S))

    pca = PCA(n_components=n_dims, random_state=random_state)
    vs = pd.DataFrame(
            pca.fit_transform(Xnorm.values),
            columns=['PC'+str(i+1) for i in range(pca.n_components)],
            index=X.index)
    us = pd.DataFrame(
            pca.components_,
            index=vs.columns,
            columns=X.columns).T

    return {
            'vs': vs,
            'us': us,
            'eigenvalues': pca.explained_variance_ * Xnorm.shape[0],
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'transform': pca.transform,
            'whiten': whiten,
            }


### Functions to plot phenotypes in groupped fashion


def plot_phenotypes_cluster(df, col_plot='all' , group_col='group_tsne',
                            groups_id=None,ncols=3,nrows=5,figsize=(13,10), plot_type='violin'):
    ''''Plots columns in dataframe based in one column that contains group information
    groups_plot is a list with columns to plot'''
    df_plot = df.copy()
    if col_plot is 'all':
        columns = df_plot.columns.drop(group_col)
    else:
        columns = col_plot

    if groups_id:
        df_plot = df_plot.loc[df[group_col].isin(groups_id),columns]#.join(res_tsne['group'])
    ncols=ncols
    nrows = nrows

    fig,axs= plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axs=axs.flatten()

    for i,col in enumerate(columns):
        if plot_type is 'violin':
            axs[i] = sns.violinplot(x=group_col, y=col, ax=axs[i],
                                    data=df_plot,palette='husl', inner='sticks',color=0.8, split=True)
        elif plot_type =='boxplot':
            axs[i] = sns.boxplot(x=group_col, y=col, data=df_plot, color=".8", ax=axs[i])
        #, color=".8", inner='sticks', ax=axs[i], bw=.2, cut=1, linewidth=1)
        elif plot_type =='strip':
            axs[i] = sns.stripplot(x=group_col, y=col, data=df_plot, jitter=True, palette='husl',dodge=True, alpha=0.6, ax=axs[i])
        elif plot_type =='boxstrip':
            axs[i] = sns.boxplot(x=group_col, y=col, data=df_plot, color=".8", ax=axs[i])
            axs[i] = sns.stripplot(x=group_col, y=col, data=df_plot, jitter=True, palette='husl',dodge=True, alpha=0.6, ax=axs[i])


    return fig, axs
    #plt.savefig(figures_output_folder + 'ephys_cluster_parameters.png')


#######
### Functions to find clusters
########


def find_clusters_hdbscan(df , tsne_columns=['dimension 1', 'dimension 2'], min_cluster_size = 5, min_samples=None, metric='euclidean'):
    '''df: is a dataframe (e.g. samplesheet with columns containing tsne coordinates)'''
    from sklearn.preprocessing import StandardScaler
    from hdbscan import HDBSCAN

    X = df[tsne_columns].values
    X =StandardScaler().fit_transform(X)
    db = HDBSCAN(algorithm='best', metric=metric, min_cluster_size = min_cluster_size, min_samples= min_samples).fit(X)

    labels = db.labels_

    df['group_tsne'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters)
    return df


def find_clusters_dbscan(df , tsne_columns=['dimension 1', 'dimension 2'], eps=0.3, min_samples=5, metric='euclidean'):
    '''df: is a dataframe (e.g. samplesheet with columns containing tsne coordinates)'''
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN

    X = df[tsne_columns].values
    X =StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
#db = SpectralClustering(n_clusters=4).fit_predict(X) #when dbscan doesnt work this does great

    labels = db

    df['group_tsne'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters)
    return df



def find_clusters_spectral(df , tsne_columns=['dimension 1', 'dimension 2'], n_clusters=2):
    '''df: is a dataframe (e.g. samplesheet with columns containing tsne coordinates)'''
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import SpectralClustering

    X = df[tsne_columns].values
    X =StandardScaler().fit_transform(X)
    db = SpectralClustering(n_clusters=n_clusters).fit_predict(X)
#db = SpectralClustering(n_clusters=4).fit_predict(X) #when dbscan doesnt work this does great

    labels = db

    df['group_tsne'] = labels
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters)
    return df


#####
#Functions to do percent expression with bootstrap and melt dataset with a samplesheet column
#####

def melt_datataset_gene_ephys(ds=None, gene_list=[], samplesheet_columns=[], groupby_col=None, var_name='gene'):
    '''Returns a melted pandas dataframe containing genes of interest and samplesheet columns
    ds: dataset
    gene_list: genes to keep
    samplesheet_columns:  samplesheet columns to keep
    groupby_col: If provided the dataset is melted onto this column
    '''
    s = ds.copy()
    a = s.counts.loc[gene_list,:].T
    if groupby_col:
        b = s.samplesheet[samplesheet_columns + [groupby_col]]
    else:
        b = s.samplesheet[samplesheet_columns]
    c = a.join(b)
    if groupby_col:
        c = c.melt(id_vars=groupby_col, value_vars=gene_list + samplesheet_columns,var_name=var_name)
    return c

def percent_expression_bootstrap_rows(df, n_iterations=100, thres=0):
    '''Function takes a dataframe with float values and returns bootstrapped mean and sd columnwise.
    Used to bootstrap percentage of cells showing expression at a certain level with error bar.
    df: dataframe where columns are genes and rows are cells
    thres: threshold of value to consider expressed'''
    from sklearn.utils import resample
    n_size = df.shape[0]
    output = {}
    for i in range(n_iterations):
        s = resample(df, n_samples=n_size)#pd.DataFrame(df.values[np.random.randint(n_size, size=n_size)])

        output[i] = s.apply(lambda x: (x > thres).sum(axis=0)) / s.count(axis=0)
        #output[i] = s.apply(lambda x: (x > thres).sum(axis=0).mean(axis=0))

    s = pd.DataFrame.from_dict(output,orient='index')
    mean = s.mean(axis=0)
    sd = s.std(axis=0)
    return mean, sd

def percent_difference_expression_bootstrap_rows(df, n_iterations=100, thres=0,
                                                 column_class='DiabetesStatus',categories=['healthy','T2D']):
    '''Function takes a dataframe with float values of expression and calculates percent difference
    in expression using bootstrap. Used to bootstrap percent difference of cells.
    df: dataframe where columns are genes and rows are cells
    thres: threshold of value to consider expressed'''
    from sklearn.utils import resample
    from statsmodels.sandbox.stats.multicomp import multipletests
    n_size = df.shape[0]
    output = {}
    for i in range(n_iterations):
        #column_class='DiabetesStatus'
        #categories=['healthy','T2D']
        #from sklearn.utils import resample
        s = resample(df, n_samples=df.shape[0])
        r1 = s[s[column_class] == categories[0]].drop(column_class, axis=1)
        x1 = r1.apply(lambda x: (x > thres).sum(axis=0)) / r1.count(axis=0)

        r2 = s[s[column_class] == categories[1]].drop(column_class, axis=1)
        x2 =r2.apply(lambda x: (x > thres).sum(axis=0)) / r2.count(axis=0)

        output[i] = x1-x2

    s = pd.DataFrame.from_dict(output,orient='index')
    mean = s.mean(axis=0)
    sd = s.std(axis=0)
    #compute pvalue from number of iterations in opposite direction of percent difference
    p_val = s[s>0].count(axis=0) /s.shape[0]
    p_val.map(lambda x: 2*(1-x) if (x>0.5) else 2*x)
    #merge
    res = pd.DataFrame([mean, sd, p_val], index=['mean','sd','pval']).T
    res['FDR'] = multipletests(res['pval'], method='fdr_bh')[1]
    #compute p_value from zscore  (mean and sd and assume gaussian distribution)
    from scipy.stats import norm
    res['p_zscore'] = np.abs(res['mean']/res['sd'])
    res['p_zscore']= res['p_zscore'].apply(lambda x: 2* norm.sf(abs(x)))
    return res






# Supplementary classes
class _RPCA:
    '''from: https://github.com/dganguli/robust-pca'''

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=None):
        ite = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and ite < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            ite += 1
            if iter_print is None:
                continue
            if (ite % iter_print) == 0 or ite == 1 or ite > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(ite, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):
        import matplotlib.pyplot as plt
        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')


##########
#### functions in
#### Correlations folder
###########

# read all folders where we keep picke with results
def find_parameter_name(filename, substring_pattern='beta_healthy_(.+?)_10'):
    import re
    m = re.search(substring_pattern,filename)
    try:
        if m:
            return m.group(1)
    except:
        print("Name not found. Check substring pattern and filename")

def find_files_in_folder(folder, pattern='*.pickle'):
    import sys
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(root+"/"+filename) # full path of match
    return matches

def import_correlations_folder(dir_name,substring_pattern):
    '''reads a folder containing standard excel files with correlation results and imports again the original dictionary
    example dir_name='/Users/joan/Desktop/Stanford/pancreas_singlecell/notebooks/patchclamp/pclamp_3_ephys_diff/beta_cell_healthy_corr_results/'
    substring_pattern='beta_healthy_(.+?)_10' '''
    import pickle
    # helper functions
    #get the full filename for files mathchin a name with a substring
    def find_parameter_name(filename, substring_pattern=substring_pattern):
        import re
        m = re.search(substring_pattern,filename)
        try:
            if m:
                return m.group(1)
        except:
            print("Name not found. Check substring pattern and filename")
    #find pickled files within folder
    def find_files_in_folder(folder, pattern='*.pickle'):
        import sys
        import os
        import fnmatch
        matches = []
        for root, dirnames, filenames in os.walk(folder):
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(root+"/"+filename) # full path of match
        return matches


    dir_name = dir_name#'/Users/joan/Desktop/Stanford/pancreas_singlecell/notebooks/patchclamp/pclamp_3_ephys_diff/beta_cell_healthy_corr_results/'# directory containing STAR log output files (Log.final.out)
    pattern_files = '*.pickle'
    # Find all pickled files and save
    matches = find_files_in_folder(dir_name, pattern='*.pickle')
    #recover pickled dataframes into dictionary where name is each parameter
    results_corr = {}
    for file in matches:
        name = find_parameter_name(file, substring_pattern=substring_pattern)#'beta_healthy_(.+?)_10'
        with open(file, 'rb') as handle:
            results_corr[name] = pickle.load(handle)
    #return dictionary with all correlations
    return results_corr

def get_hub_genes(correlations_dict, column_use='zscore', thres_col=2, signed_zscore='positive', n_groups=1):
    '''get all genes that correlate to more than one block of ephys parameters.
    correlations_dict: output of import_correlations_folder function. dictionaty of dataframes containing all correlations and zscores
    column_use: column used to select genes that pass threshold
    thres_col: threshold used in the column to decide tha one gene is significant for an ephys value
    signed_zscore: positive, negative, both, wether to select genes positively correlated, negatively or both
    n_groups: threshold of groups of parameters above which gene is kept, 0 keeps all genes correlated to at least one block,'''

    cols_order_group = {'Cell size': ['Cell size'],
                        'Exocytosis':['Total Exocitosis', 'Late exocytosis', 'Early exocytosis', 'Exocytosis norm Ca2+'],
                        'Calcium': ['Ca2+ entry','Early Ca2+ current', 'Late Ca2+ current', 'Late Ca2+ Conductance'],
                        'Sodium': ['Peak Na+ current','Na+ conductance']}

    col = column_use
    thres = thres_col
    ephysblocks_grouped = {}
    for i,group in enumerate(cols_order_group):
            #select all genes that positively correlate to at least one parameter withing group above threshold
            #subset_par = cols_order_group[col]
            #x = test_fin[subset_par]
            #print(x)
            x = pd.concat([correlations_dict[par][col] for par in cols_order_group[group]], axis=1)
            #count how many parameters within block pass threshold
            if signed_zscore=='positive':
                y = x[x>thres].count(axis=1)
            elif signed_zscore=='negative':
                y = x[x<-thres].count(axis=1)
            elif signed_zscore=='both':
                y = x[np.abs(x)>thres].count(axis=1)
            else:
                raise ValueError('signed_zsvore not understood: positive, negative or both')
            #seen in at least one parameter
            z = y[y>0]
            ephysblocks_grouped[group] = z
    df_groupped_corr = pd.concat(ephysblocks_grouped, axis=1).fillna(0)

    #pick genes that are seen in at least n groups of parameters
    df_hub_genes = df_groupped_corr[(df_groupped_corr> n_groups).sum(axis=1) > 0]

    return df_hub_genes

def filter_correlation_table(dict_df, z_th = 2, zscore_raw=False, pct_th = 0., raw_corr_th=0., p_boot_th=1, n_feat=1, n_blocks=1, signed='both',
                            cols_order_group={'Cell size': ['Cell size'],
                'Exocytosis':['Total Exocitosis', 'Late exocytosis', 'Early exocytosis', 'Exocytosis norm Ca2+'],
                'Calcium': ['Ca2+ entry','Early Ca2+ current', 'Late Ca2+ current', 'Late Ca2+ Conductance'],
                'Sodium': ['Peak Na+ current','Na+ conductance']}):
    def get_nodes_from_table(df, n_blocks = 0, cols_order_group=cols_order_group):

        x = df.copy()
        counts_nodes = pd.DataFrame([0 for i in range(len(x.index))],index=x.index, columns=['groups'])

        for i,group in enumerate(cols_order_group):
            pars = cols_order_group[group]
            #correlated parameters in block
            sel = np.abs(x[pars])>0
            #check than more than one gene in block
            update_x = x[(sel).sum(axis=1)>0].index
            #add +1 to genes that pass threshold for this block
            counts_nodes.loc[update_x,:] += 1
        #select genes that pop up in more than n blocks
        new_index = counts_nodes[counts_nodes['groups']>=n_blocks].index
        return x.reindex(new_index)

    if zscore_raw:
        zscore_col = 'zscore_raw_mean'
    else:
        zscore_col = 'zscore'

    final_table = []
    if p_boot_th < 1:
        for i,name in enumerate(dict_df):
            if signed=='positive':
                cond = (dict_df[name][zscore_col] > z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (dict_df[name]['corr_raw_mean'] > raw_corr_th) &\
                (np.abs(dict_df[name]['pval_bootstrap']) < p_boot_th)
            elif signed=='negative':
                cond = (dict_df[name][zscore_col] < -z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (dict_df[name]['corr_raw_mean'] < -raw_corr_th) &\
                (np.abs(dict_df[name]['pval_bootstrap']) < p_boot_th)
            elif signed=='both':
                cond = (np.abs(dict_df[name][zscore_col]) > z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (np.abs(dict_df[name]['corr_raw_mean']) > raw_corr_th) &\
                (np.abs(dict_df[name]['pval_bootstrap']) < p_boot_th)
            else:
                raise ValueError('signed must be positive, negative, or both.')

            final_table.append(pd.DataFrame(dict_df[name].loc[cond, 'zscore'].rename(name)))

    else:
        for i,name in enumerate(dict_df):
            if signed=='positive':
                cond = (dict_df[name][zscore_col] > z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (dict_df[name]['corr_raw_mean'] > raw_corr_th)
            elif signed=='negative':
                cond = (dict_df[name][zscore_col] < -z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (dict_df[name]['corr_raw_mean'] < -raw_corr_th)
            elif signed=='both':
                cond = (np.abs(dict_df[name][zscore_col]) > z_th) &\
                (np.abs(dict_df[name]['pct cells']) > pct_th) &\
                (np.abs(dict_df[name]['corr_raw_mean']) > raw_corr_th)
            else:
                raise ValueError('signed must be positive, negative, or both.')

            final_table.append(pd.DataFrame(dict_df[name].loc[cond, zscore_col].rename(name)))

    #table with all correlations passing threshold
    x = pd.concat(final_table, axis=1).fillna(0)

    #filter for genes correlated to a number of parameters
    cond_corr = np.abs(x)>0
    x = x[x[cond_corr].count(axis=1) >= n_feat]

    x = get_nodes_from_table(x, n_blocks)

    return x


def percent_exp(x):
    return (x>0).sum(axis=1) / x.count(axis=1)*100


def logmean(df, base=2, pseudocount=1):
    return np.log2(pseudocount+(base**df-pseudocount).mean(axis=1))


def get_all_zscores(dict_df,indexes):
    cols_order = ['Cell size', 'Total Exocitosis', 'Late exocytosis', 'Early exocytosis', 'Exocytosis norm Ca2+',  'Ca2+ entry','Early Ca2+ current', 'Late Ca2+ current', 'Late Ca2+ Conductance', 'Peak Na+ current','Na+ conductance']
    x_all ={}
    for col in cols_order:
        x_all[col] = dict_df[col].loc[indexes,'zscore']
    x_all = pd.DataFrame(x_all)
    return x_all


def df_get_mu_pvals(df, pars, column_class='DiabetesStatus',categories=['healthy','T2D']):
    from scipy.stats import mannwhitneyu
    from scipy.stats import ks_2samp
    from statsmodels.sandbox.stats.multicomp import multipletests
    pvals ={}
    for par in pars:
        cond1= df[column_class]==categories[0]
        group1 = df[cond1][par].dropna()
        cond2= df[column_class]==categories[1]
        group2 = df[cond2][par].dropna()
        pvals[par] = mannwhitneyu(group1, group2, use_continuity=True, alternative=None)[1]

        pvals =pd.Series(pvals)
    pvals =pd.DataFrame(pvals,columns=['pval'])
    pvals['FDR']= multipletests(pvals['pval'], method='fdr_bh')[1]
    return pvals

### make bulk samples

def make_bulk_raw_sc(ds,columns_split=['DonorID','cell_type','DiabetesStatus'],n_min_cells=10):
    dict_diab = ds.split(columns_split)
    sum_cells ={}
    for key,ds in dict_diab.items():
        if ds.n_samples>n_min_cells:
            sum_cells[key]=  ds.counts.sum(axis=1)
    #concetenate dictionary to make df
    df = pd.concat(sum_cells, axis=1)
    df.columns = ['%s%s%s' % (a, '|%s' % b, '|%s' % c) for a, b, c in df.columns]
    df.fillna(0,inplace=True)
    df = df *1e6 /df.sum(axis=0)
    df = np.log2(df+1)
    return df

def make_bulk_notnotnormalized_sc(ds,columns_split=['DonorID','cell_type','DiabetesStatus'],n_min_cells=10):
    dict_diab = ds.split(columns_split)
    sum_cells ={}
    for key,ds in dict_diab.items():
        if ds.n_samples>n_min_cells:
            sum_cells[key]=  ds.counts.sum(axis=1)
    #concetenate dictionary to make df
    df = pd.concat(sum_cells, axis=1)
    df.columns = ['%s%s%s' % (a, '|%s' % b, '|%s' % c) for a, b, c in df.columns]
    df.fillna(0,inplace=True)
    return df
