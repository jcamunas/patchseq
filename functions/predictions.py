# vim: fdm=indent
# author:     Joan Camunas
# date:       01/07/18
# content:    Dataset functions to correlate gene expression and phenotypes
# Modules

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def predict_ephys(counts1, counts2, genelist, ephys2, k=10, method='spearman', weight_method='linear', use_top_cell=True):
    '''counts1: count table of cells without electrophysiology
    counts2: count table of cells with electrophysiology
    ephys2:metadata table containing electrophysiology data for counts2
    method: metric to compute distance, euclidean or spearman
    weight_method: method to weight the cells when inferring electrophysiology, flat, linar or exp'''

    #functions
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

    #start here
    #select genes of interest for each dataset
    counts1 = counts1.loc[genelist,:]
    counts2 = counts2.loc[genelist,:]

    #calculate euclidean or correlation based distance betweeen each cell of both datasets
    if method=='euclidean':
        distance = []
        for cell_pc, exp_pc in counts2.T.iterrows():
            d =  np.sqrt(((counts1.values.T - exp_pc.values)**2).sum(axis=1))
            distance.append(d)
        distance = pd.DataFrame(data=np.vstack(distance), index=counts2.columns, columns=counts1.columns).T
    elif method=='spearman':
        distance = 1-correlate(counts1.values.T,counts2.values.T,method='spearman')
        distance = pd.DataFrame(data=distance.T, index=counts2.columns, columns=counts1.columns).T
    elif method=='pearson':
        distance = 1-correlate(counts1.values.T,counts2.values.T,method='pearson')
        distance = pd.DataFrame(data=distance.T, index=counts2.columns, columns=counts1.columns).T
    else:
        raise ValueError('method not understood')

    #pick the top k correlates of the ephys dataset for each cell of the non-ephys
    ranks = []
    ranks_dist = []
    for FACS_cell, row in distance.iterrows():
        if use_top_cell:
            ranks.append(distance.columns[np.argsort(row.values)[0:k]])
            ranks_dist.append(np.sort(row.values)[0:k])
        else:
            ranks.append(distance.columns[np.argsort(row.values)[1:k+1]])
            ranks_dist.append(np.sort(row.values)[1:k+1])

    ranks = pd.DataFrame(data=np.vstack(ranks), index=distance.index, columns=np.arange(1,k+1))
    ranks_dist = pd.DataFrame(data=np.vstack(ranks_dist), index=distance.index, columns=np.arange(1,k+1))

    #calculate the inferred electrophysiology for the cell without electrophysiology
    #3 methods: plain average, linear weight (from 1-0), or exponential weight
    for col in ephys2.columns:
        data = []
        for i in range(k):
            data.append(ephys2.loc[ranks[i+1], col].values)
        data = np.vstack(data)
        n = k - np.isnan(data).sum(axis=0)
        #use masked array of nans to do arithmetics correctly
        data = np.ma.masked_invalid(data)
        if weight_method == 'flat':
            weights = np.ones(k)
        elif weight_method == 'linear':
            weights = 1.0 - np.linspace(0, 1, k+1)[:-1]
        elif weight_method == 'exp':
            weights = np.exp(-np.arange(k))
        else:
            raise ValueError('weight_method is bogus!')
        data_mean = (data.T @ weights) / weights.sum()
        data_median= np.ma.median(data, axis=0)
        data_std = data.std(axis=0)
        ranks[col+' mean'] = data_mean
        #ranks[col+' mean'] = data_median
        ranks[col+' std'] = data_std
        ranks[col+' n'] = n
    #output table with mean and std of electrophysiology for each parameter (and nnumber of cells used, to avoid na)
    #mean is only mean for flat method, and std is only valid for that
    return ranks#, ranks_dist

def plot_correlation_prediction(df_pred,
                                phenotypes=['Cell size',
                                'Total Exocitosis',
                                'Early exocytosis',
                                'Late exocytosis',
                                'Ca2+ entry',
                                'Early Ca2+ current',
                                'Late Ca2+ current',
                                'Peak Na+ current',
                                'Na+ conductance'],
                                plot_error=False, stat='spearmanr'):
    sns.set(font_scale=1.3)
    sns.set_style('white')
    fig,axs=plt.subplots(3,4, figsize=(15,10))
    axs=axs.flatten()
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    for i,par in enumerate(phenotypes):
        if plot_error:
            axs[i].errorbar(x=df_pred[par],
                     y=df_pred[par + ' mean'],
                     yerr=df_pred[par + ' std'],
                     fmt='o')
        else:
            axs[i].scatter(x=df_pred[par],
                     y=df_pred[par + ' mean'],
                          c='#2b8cbe',alpha=0.9)
        axs[i].set_title(par,fontsize=18)
        if i in [0,4,8]:
            axs[i].set_ylabel('Predicted',fontsize=14)
        if i in [5,6,7,8]:
            axs[i].set_xlabel('Measured',fontsize=14)
        if stat is 'spearmanr':
            r = spearmanr(df_pred[par],df_pred[par + ' mean'],nan_policy='omit')[0]
        elif stat is 'pearsonr':
            r = pearsonr(np.ma.masked_invalid(df_pred[par]),np.ma.masked_invalid(df_pred[par + ' mean']))[0]
        axs[i].text(0.02, 0.9, 'r = {:01.2f}'.format(r), fontsize=14, transform=axs[i].transAxes,weight='bold',color='red')
    axs[9].remove()
    axs[10].remove()
    axs[11].remove()
    sns.despine()
    fig.tight_layout()
    return fig, axs

def filter_dataset_ephys(df,pars=['Cell size', 'Total Exocitosis','Early exocytosis','Late exocytosis',
                  'Ca2+ entry','Exocytosis norm Ca2+', 'Early Ca2+ current','Late Ca2+ current',
                      'Late Ca2+ Conductance','Peak Na+ current','Na+ conductance'], qlow=0.03,qhigh=0.97,include_quantiles=True,clipzero=True,removeneg=False,clip_quant=True, clip_qlow=0,clip_qhigh=1):

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

        return s

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

    x = df.copy()

    x = filter_quantile_values(x, columns=pars, qlow=qlow,qhigh=qhigh, include_quantiles=include_quantiles)
    for par in pars:
        if x[par].mean() < 0:
            x[par] = -1*(x[par])
        #clip value of electrophysiology to zero, no negative values
        if clipzero is True:
            x[par] = x[par].clip(lower=0)

    if clip_quant is True:
        x = clip_quantile_values(x, columns=pars, qlow=clip_qlow, qhigh=clip_qhigh)
        #do log plus 0.5
        #ts.samplesheet[par] = np.log(ts.samplesheet[par].astype(float)+floor_value)

    #mask negativezero values
    if removeneg is True:
        x[pars] = x[pars].mask(x[pars]<0)

    return x



def plot_correlation_duo_prediction(df_pred_tr, df_pred_ts,
                                phenotypes=['Cell size','Total Exocitosis','Early exocytosis',
                                            'Late exocytosis', 'Ca2+ entry', 'Early Ca2+ current',
                                            'Late Ca2+ current','Late Ca2+ Conductance', 'Peak Na+ current', 'Na+ conductance'], plot_error=False, stat='spearmanr'):
    sns.set(font_scale=1.3)
    sns.set_style('white')
    fig,axs=plt.subplots(3,4, figsize=(15,10))
    axs=axs.flatten()
    from scipy.stats import spearmanr
    from scipy.stats import pearsonr
    for i,par in enumerate(phenotypes):
        if plot_error:
            axs[i].errorbar(x=df_pred_tr[par],
                     y=df_pred_tr[par + ' mean'],
                     yerr=df_pred_tr[par + ' std'],
                     fmt='*')
            axs[i].errorbar(x=df_pred_ts[par],
                 y=df_pred_ts[par + ' mean'],
                 yerr=df_pred_ts[par + ' std'],
                 fmt='o')
        else:
            axs[i].scatter(x=df_pred_tr[par],
                     y=df_pred_tr[par + ' mean'], c='#2b8cbe',alpha=0.7)
            axs[i].scatter(x=df_pred_ts[par],
                 y=df_pred_ts[par + ' mean'], c='#e34a33',alpha=0.7)

        axs[i].set_title(par,fontsize=18)
        if i in [0,4,8]:
            axs[i].set_ylabel('Predicted',fontsize=14)
        if i in [5,6,7,8]:
            axs[i].set_xlabel('Measured',fontsize=14)
        if stat is 'spearmanr':
            r_tr = spearmanr(df_pred_tr[par],df_pred_tr[par + ' mean'],nan_policy='omit')[0]
            r_ts = spearmanr(df_pred_ts[par],df_pred_ts[par + ' mean'],nan_policy='omit')[0]
        elif stat is 'pearsonr':
            r_tr = pearsonr(np.ma.masked_invalid(df_pred_tr[par]),np.ma.masked_invalid(df_pred_tr[par + ' mean']))[0]
            r_ts = pearsonr(np.ma.masked_invalid(df_pred_ts[par]),np.ma.masked_invalid(df_pred_ts[par + ' mean']))[0]

        axs[i].text(0.02, 0.9, 'r = {:01.2f}'.format(r_tr), fontsize=16, transform=axs[i].transAxes,weight='bold',color='black')
        axs[i].text(0.02, 0.8, 'r = {:01.2f}'.format(r_ts), fontsize=16, transform=axs[i].transAxes,weight='bold',color='red')
    #axs[9].remove()
    axs[10].remove()
    axs[11].remove()
    sns.despine()
    fig.tight_layout()
    return fig, axs


def make_predictions_random_genes(x, x_electro, y, y_electro, use_top=False, n_cells = 5, n_bootstrap = 1000, n_genes = 484, method = 'spearman', weight_method= 'flat'):
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error

    #Fix parameters to test and method
    phenotypes = ['Cell size',
    'Total Exocitosis',  'Early exocytosis','Late exocytosis',
    'Ca2+ entry', 'Early Ca2+ current', 'Late Ca2+ Conductance',
    'Na+ conductance', 'Peak Na+ current']
    #random genes from expressed
    expressed_genes = x[x.mean(axis=1)>1].index.tolist()
    corr_random = {}
    mae_random={}
    for i,par in enumerate(y_electro.columns):
        corr_random[par] = []
        mae_random[par] = []

    for i in np.arange(n_bootstrap):
        genelist = np.random.choice(expressed_genes,size= n_genes, replace=True)

        #predict
        res_pred = predict_ephys(y, x, genelist, x_electro, k=n_cells, method= method, weight_method= weight_method, use_top_cell=use_top)
        #clean initial columns with top cells
        for col in y_electro.columns:
            res_pred[col] = y_electro[col]
        res_clean = res_pred.drop(np.arange(n_cells)+1,axis=1)

        for i,par in enumerate(phenotypes):
            r = spearmanr(res_clean[par],res_clean[par + ' mean'],nan_policy='omit')[0]
            corr_random[par].append(r)

            mae_temp = res_clean[[par, par + ' mean']].dropna(axis=0)
            mae = mean_absolute_error(mae_temp[par],mae_temp[par + ' mean'])
            mae_random[par].append(mae)
    corr_random = pd.DataFrame(corr_random).T
    mae_random = pd.DataFrame(mae_random).T

    return corr_random, mae_random


def correlation_pval_random(x,df_rand):
    '''x: is series containing parameters measured
    df_rand: values for same paramerer from random set of genes'''
    corr_pvals = {}
    for col in df_rand.T.columns:
        #print(col)
        cond = df_rand.T[col] > x[col]
        n_perm = df_rand.T.shape[0]
        pval = df_rand.T.loc[cond, col].count() / n_perm
        if pval ==0:
            corr_pvals[col] = '<' + str("{:.0E}".format(1/n_perm))
        else:
            corr_pvals[col] = '=' + str(pval)
    return corr_pvals

def plot_correlation_vs_random(x_m, x_rand, pval_labels):
    '''
    x_m=r_corr.T['corr_train'], x_rand = corr_rand, pval_labels = pval_rand_train
    '''
    #plot distributuions and position of best gene set corrrelation
    fig,axs=plt.subplots(2,5, figsize=(20,5))
    axs=axs.flatten()

    for i,par in enumerate(x_rand.columns):
        sns.distplot(x_rand[par],ax=axs[i])
        arrow_length = axs[i].get_ylim()[1]/3
        axs[i].arrow(x=x_m[par], y=arrow_length, dx=0, dy=-arrow_length,
                     fc="b", ec="b", linewidth=2,head_width=0.1, head_length=0.2,length_includes_head=True)
        axs[i].set_title(par)
        axs[i].set_xlim(-1,1)
        text_pval = 'p$_{val}$' + str(pval_labels[par])
        axs[i].text(0.02, 0.9, text_pval, fontsize=12, transform=axs[i].transAxes,weight='bold',color='blue')
        axs[i].set_xlabel('')
        for i in [5,6,7,8,9]:
            axs[i].set_xlabel('correlation',fontsize=16)
        for i in [0,5]:
            axs[i].set_ylabel('density',fontsize=16)
    #manual legend using patches
    import matplotlib.patches as patches
    p= patches.Rectangle((-0.95,2.2), 0.2, 0.3, fc='#9ecae1', ec='#3182bd')
    axs[0].add_patch(p)
    axs[0].arrow(x=-0.95, y=2.8, dx=0.2, dy=0, fc="b", ec="b", linewidth=2,head_width=0.15, head_length=0.1,length_includes_head=True)
    axs[0].text(-0.7, 2.2, 'random genes', fontsize=12)
    axs[0].text(-0.7, 2.8, 'PS genes', fontsize=12)
    #remove unused plots
    sns.despine()
    fig.tight_layout()
    return fig, axs

def get_group_genes(correlations_dict, column_use='zscore'):
    '''Read correlation resut table and get average of one column'''

    cols_order_group = {'Cell size': ['Cell size'],
                        'Exocytosis':['Total Exocitosis', 'Late exocytosis', 'Early exocytosis'],
                        'Calcium': ['Ca2+ entry','Early Ca2+ current', 'Late Ca2+ current', 'Late Ca2+ Conductance'],
                        'Sodium': ['Peak Na+ current','Na+ conductance']}

    grouped_genes = {}
    for i,col in enumerate(cols_order_group):
        #print(cols_order_group[col])
        x = pd.concat([correlations_dict[par][column_use] for par in cols_order_group[col]],axis=1)
        y = x.mean(axis=1)


        #y = y[~y.index.str.contains('RPL|RPS|HLA-A|HLA-B|HLA-C|HLA-G|HLA-E')]
        grouped_genes[col] = y
        grouped_genes = pd.DataFrame(grouped_genes)

    return grouped_genes

def group_correlation_table(df, thres=2,signed_zscore='both', return_sign=False,
                            cols_order_group={'Cell size': ['Cell size'],
                'Exocytosis':['Total Exocitosis', 'Late exocytosis', 'Early exocytosis', 'Exocytosis norm Ca2+'],
                'Calcium': ['Ca2+ entry','Early Ca2+ current', 'Late Ca2+ current', 'Late Ca2+ Conductance'],
                'Sodium': ['Peak Na+ current','Na+ conductance']}):
    """Takes a correlation table with z-scores and zeros and summarizes the number of parameters per group above threshold"""

    ephysblocks_grouped = {}
    for i,group in enumerate(cols_order_group):

        #select all genes that positively correlate to at least one parameter withing group above threshold
        subset_par = cols_order_group[group]
        x = df[subset_par]
        #print(x)
        #count how many parameters within block pass threshold
        if signed_zscore=='positive':
            y = x[x>thres].count(axis=1)
        elif signed_zscore=='negative':
            y = x[x<-thres].count(axis=1)
        elif signed_zscore=='both':
            if return_sign is True:
                y = x[np.abs(x)>thres].count(axis=1)
                y_sign = x[np.abs(x)>thres].sum(axis=1).apply(lambda x: np.sign(x))
                y =y*y_sign
                #print(y_sign)
            else:
                y = x[np.abs(x)>thres].count(axis=1)
        else:
            raise ValueError('signed_zsvore not understood: positive, negative or both')
        #seen in at least one parameter
        z = y[np.abs(y)>0]
        ephysblocks_grouped[group] = z
    df_groupped_corr = pd.concat(ephysblocks_grouped, axis=1).fillna(0)
    return df_groupped_corr
