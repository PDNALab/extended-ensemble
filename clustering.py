import pyemma
import numpy as np
import pyemma.coordinates as coor
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.pyplot import cm
from collections import OrderedDict
import mdtraj as md
import itertools
import time
import indices
from indices.base import BaseComparisons as bc
from indices.faith import Faith as Fai
import scipy.cluster.hierarchy as shc
from matplotlib import gridspec
from matplotlib import pyplot
import mdtraj
import seaborn as sns
import pandas  as pd



def feature(traj_loc, pdb_loc, sieve_res=2, random=None, sieve_traj=10, threshold=0.6):
    '''
    Contact fingeprint calculation.
    ---
    Input:
    traj_loc, pdb_loc: trajectory and pdb file location.
    sieve_res: calculate contact fingerprint every "sieve_res" (e.g. every two residue),
               default=2
    sieve_traj: calculate fingerprints every "sieve_traj" sample.
               default=10
    threshold: criterion for every "sieve_res" contact, within threshold 1 and 0 otherwise.
               default=0.6
    
    Output:
    inp: contact fingerprints for selected samples
    '''
    traj = md.load_dcd(traj_loc,top=pdb_loc)
    topfile=traj.top
    feat = coor.featurizer(topfile)
    residues = np.arange(0,topfile.n_residues)
    pairs = []                                                                                 
    for i,r1 in enumerate(residues):
        for r2 in residues[i+1::2]:
            pairs.append([r1,r2])
    pairs = np.array(pairs)
    feature=feat.add_residue_mindist(pairs, scheme='closest-heavy',threshold=threshold,periodic=False)
    inp = pyemma.coordinates.load(traj_loc, features=feat)
    if random:
        random_index = np.random.randint(0,len(inp),random)
        inp = inp[random_index]
    else:
        inp = inp[::sieve_traj]
    return inp


def binary_simi_matrix(inp,simi_scale='no_scaled',batch_size=1000000):
    '''
    Binary similarity matrix calculation.
    ---
    Input:
    inp: sample contact fingerprints with size np.array((n,m)). n is number of samples 
         and m is the length of each fingerprint.
    simi_scale: select which scale index for similarity calculation.
                default: simply add all 1 and all 0 together
    batch_size: calculate simi_matrix in batches if number of samples are too large.
                default=1000000
    
    Output:
    simi_matrix: similarity matrix with size np.array((n,n)).
    '''
    all_start=time.time()
    all_input = list(itertools.combinations(inp, 2))
    batch_size=1000000
    inp_sliced=[all_input[i*batch_size:(i+1)*batch_size] for i in range(int(len(all_input)/batch_size))]
    if int(len(inp_sliced)) < len(all_input)/batch_size:
        inp_sliced.append(all_input[len(inp_sliced)*batch_size:])
    for i in range(len(inp_sliced)):
        temp_start = time.time()
        temp_c = np.zeros((int(len(inp_sliced[i])),3))
        temp_input = np.array(inp_sliced[i])
        product = temp_input.reshape(-1,2,temp_input.shape[-1]).sum(1)
        for row in range(3):
            temp_c[:,row] = np.sum(product==row,axis=1)
        if i == 0:
            all_c = temp_c
        else:
            all_c = np.concatenate((all_c,temp_c),axis=0)
    all_end = time.time() 
    all_time = all_end - all_start 
    ###calculate similarity
    if simi_scale == 'no_scaled':
        simi = all_c[:,0]+all_c[:,2]
    elif simi_scale == "Faith":
        all_simi = all_c[:,0]+0.5*all_c[:,2]
        denominate = all_c[:,0]+all_c[:,1]+all_c[:,2]
        simi = all_simi/denominate
    simi_matrix = np.zeros((len(inp),len(inp)))
    indices = np.triu_indices(len(inp),k=1)
    indices = (indices[1],indices[0])
    simi_matrix[indices] = simi
    return simi_matrix, all_time


def agglomerative(inp,simi_matrix):
    '''
    Perform agglomerative hierachical clustering.
    ---
    Input:
    inp: sample contact fingerprints with size np.array((n,m)). n is number of samples 
         and m is the length of each fingerprint.
    simi_matrix: binary similarity matrix with size np.array((n,n)).
    
    Output: 
    tree: clustering results for constructing dentrogram in scipy style.
    hie_tree: sample index in each cluster along clustering process.
    dic: record which two sets are grouped together.
    all_time: clustering time
    '''
    df = pd.DataFrame(simi_matrix,columns=pd.MultiIndex.from_tuples([('{}'.format(i),'{}'.format(i)) for i in range(1,len(inp)+1)],names=['cluster', 'frame']))
    inp_copy=inp
    dic={}
    hie_tree=[]
    dentrom=[]
    all_start=time.time()
    while df.shape[0] > 2:
        ###update df
        frame_column=[i[1] for i in df.columns.to_list()]
        hie_tree.append(frame_column)
        new_max=np.argmax(df, axis=None)     ###2.5s

        del_index = np.unravel_index(new_max, df.shape)
        max_value = df.to_numpy()[del_index[0]][del_index[1]] ###0.1s

        ###get temp inp
        delete_row=[int(i) for i in df.columns[del_index[0]][1].split(',') ]
        delete_cluster_row=[int(i) for i in df.columns[del_index[0]][0].split(',')]
        delete_column=[int(i) for i in df.columns[del_index[1]][1].split(',') ]
        delete_cluster_column=[int(i) for i in df.columns[del_index[1]][0].split(',')]
        dentrom.append([*delete_cluster_column,*delete_cluster_row])

        delete_all=delete_column+delete_row
        insert_index='{}'.format(delete_all)[1:-1]   ###0.1s
        insert_cluster_index='{}'.format(2*len(inp) - df.shape[0]+1)
        df.drop(columns=[('{}'.format(delete_cluster_column)[1:-1],'{}'.format(delete_column)[1:-1]),('{}'.format(delete_cluster_row)[1:-1],'{}'.format(delete_row)[1:-1])],axis=1,inplace=True)
        df.drop(index=[*del_index],axis=0,inplace=True)
        df.reset_index(drop=True,inplace=True)
        temp_w_sim=[]            ###1s

        ###get temp inp
        frame_column=[i[1] for i in df.columns.to_list()]
        columns=[[int(d) for d in [*i.split(',')]] for i in frame_column]
        temp_inp=[[inp_copy[d-1] for d in m] for m in columns]    ###0.1s

        ###perform comparison
        for i in temp_inp:
            compare=bc(np.concatenate(([inp_copy[d-1] for d in delete_all],i),axis=0))
            temp_w_sim.append(compare.total_w_sim)         ###1s

        temp_w_sim=[0]+temp_w_sim
        df.loc[-1] = [0]*df.shape[1] # add a row
        df.index = df.index + 1  # shift index
        df = df.sort_index()  # sort by index
        df.insert(loc=0, column=(insert_cluster_index,insert_index), value=temp_w_sim)
        dic[insert_index]=max_value
    all_end = time.time() 
    all_time = all_end - all_start 
    dentrogram = np.vstack(dentrom)-np.ones((1,2))
    values = np.array([*dic.values()],ndmin=2)
    num_frames = np.array([len(i) for i in [[int(d) for d in [*k.split(',')]] for k in list(dic.keys())]],ndmin=2)
    tree = np.hstack((dentrogram,values.T,num_frames.T))
    last_two = [int(i[0]) for i in df.columns.to_list()]
    tree = np.vstack((tree, [[last_two[0]-1,last_two[1]-1,df.to_numpy()[1][0],len(inp)]]))
    return tree, hie_tree, dic, all_time


def plot_simi_threshold(tree,p,min_simi,save=False):
    '''
    Plot scipy.dentrogram tree.
    ---
    Input: 
    tree: clustering results for constructing dentrogram in scipy style with size np.array((n,4)).
    p: show last p steps clustering results.
    mini_simi: minimum similarity for clustering results.
    save: check if saving figure.
          default: False
    
    Output:
    scipy.dentrogram tree figure.
    '''
    plt.figure(figsize=(20, 8))
    dend = shc.dendrogram(tree,p=p,truncate_mode='lastp')
    plt.axhline(y=70, color='r', linestyle='-')
    plt.xlabel('Num_samples')
    plt.ylabel('Simi_value')
    if save:
        plt.savefig('simi_threshold_tree.png')


def plot_test_result(true_rmsd,hie_tree,cluster_step,save=False):
    '''
    Plot 1d rmsd validation figure with sample ratio bar.
    ---
    Input:
    true_rmsd: rmsd wrt reference pdb with size np.array((1,n)).
    hie_tree: sample index in each cluster along clustering process.
    cluster_step: check clustering results for selected step.
    save: check if saving figure.
          default=False
          
    Output:
    1d rmsd validation figure with sample ratio bar.
    label_index: sample index in each cluster for selected cluster_step.
    '''
    colors = ['grey', 'purple', 'blue', 'green', 'orange', 'red',
             'black','brown','navy','indigo','cyan','teal','violet','royalblue','yellow']
    fig, ax = plt.subplots(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[15, 1]) 
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    label_index = [[int(i)-1 for i in hie_tree[cluster_step][d].split(',')] for d in range(len(hie_tree[cluster_step]))]
    label_index = sorted(label_index, key=lambda x:len(x), reverse=True)
    if len(label_index) > 10:
        label_index = label_index[:10]
    test_rmsd = [[true_rmsd[i] for i in label_index[d]] for d in range(len(label_index))]
    for index,i in enumerate(test_rmsd):
        ax0.scatter(label_index[index],i,marker='.',color=colors[index])
    percentage=[len(i)/len(true_rmsd) for i in test_rmsd]
    sum_percentage = [0.0]+[sum(percentage[:i]) for i in range(1,len(percentage))]+[1.0]
    ax0.set_xlabel('Sample')
    ax0.set_ylabel('RMSD')
    cmap = mpl.colors.ListedColormap(colors[:len(test_rmsd)])
    norm = mpl.colors.BoundaryNorm(sum_percentage, cmap.N)
    cb2 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
                                    norm=norm,
                                    boundaries=sum_percentage,
                                    ticks=sum_percentage+[1.0],
                                    spacing='proportional')
    cb2.set_label('Cluster samples ratio')
    if save:
        plt.savefig('1d_rmsd_validation.png')
    return label_index


def plot_2d_rmsd(traj_file,pdb_file,label_index,indices='backbone',sieve=1,save=False):
    '''
    Plot 2d rmsd validation for selected cluster_step.
    ---
    Input:
    traj_file, pdb_file: trajectory and pdb file location.
    label_index: sample index in each cluster for selected cluster_step.
    indices: select indices to perform rmsd calculation.
    sieve: calculate fingerprints every "sieve_traj" sample.
           default=1 
    save: check if saving figure.
          default=False
          
    Output:
    2d rmsd validation figure with rmsd side bar.
    '''
    label_concate = np.concatenate(label_index)
    traj = md.load_dcd(traj_file,top=pdb_file)
    topfile=traj.top
    pdb=md.load_pdb(pdb_file)
    if indices == 'backbone':
        all_CA=topfile.select("backbone==1")
    elif indices == 'all C':
        all_CA=topfile.select("type C")
    else:
        all_CA=None
    traj_cluster=traj[::sieve][label_concate]
    rmsd_2d=np.zeros((len(traj_cluster),len(traj_cluster)))
    upper_indices = np.triu_indices(len(traj_cluster),k=0)
    lower_indices = (upper_indices[1],upper_indices[0])
    r2d = []
    for i in range(len(traj_cluster)):
        r2d.append(md.rmsd(traj_cluster[i:],traj_cluster[i],atom_indices=all_CA))
    rmsd_2d[lower_indices] = np.concatenate(r2d)
    rmsd_2d[upper_indices] = np.concatenate(r2d)
    sns.heatmap(rmsd_2d*10,square=True,xticklabels=100,yticklabels=100,cmap='bwr',cbar_kws={'label':'RMSD'},vmin=0,vmax=10)
    plt.plot(range(len(traj_cluster)),range(len(traj_cluster)),'-.',color='k',linewidth=2)
    plt.xlabel("Frame #")
    plt.ylabel("Frame #")
    ax = plt.gca()
    ax.tick_params(direction='out')
    plt.tight_layout()
    if save:
        plt.savefig('2d_rmsd_validation.png')


def plot_clustering_accuracy(traj_loc, pdb_loc, dic, indices='backbone',sieve_traj=1, save=False):
    '''
    Plot accuracy for each clustering step.
    ---
    Input:
    traj_loc, pdb_loc: trajectory and pdb file location.
    sieve_traj: calculate fingerprints every "sieve_traj" sample.
                default=1
    dic: record which two sets are grouped together.
    '''
    traj = md.load_dcd(traj_loc,top=pdb_loc)
    topfile=traj.top
    traj = traj[::sieve_traj]
    all_clust=[[int(d) for d in [*i.split(',')]] for i in list(dic.keys())]
    ave_rmsd = []
    all_rmsd = []
    if indices == 'backbone':
        all_CA=topfile.select("backbone==1")
    elif indices == 'all C':
        all_CA=topfile.select("type C")
    else:
        all_CA=None
    for index,i in enumerate(all_clust):   
        traj_comp=traj[[np.array(i)-1]]
        rmsd=md.rmsd(traj_comp,pdb,atom_indices=all_CA)
        ave_rmsd.append(np.average(rmsd))
        all_rmsd.append(rmsd)
    accuracy=[]
    for i in range(len(all_rmsd)):
        if len(all_rmsd[i])>1:
            ratio = correct_ratio(all_rmsd[i])
            accuracy.append(np.max(ratio))
    num_frames = np.array([len(i) for i in [[int(d) for d in [*k.split(',')]] for k in list(dic.keys())]],ndmin=2)
    plt.scatter(range(len(accuracy)),accuracy,marker='.',color='b')
    plt.scatter(range(num_frames.shape[1]),*(num_frames/num_frames.shape[1]).tolist(),color='r')
    plt.xlabel('Cluster_step')
    plt.ylabel('Accuracy')
    if save:
        plt.savefig('cluster_accuracy.png')

        
def correct_ratio(x,cluster_range):
    all_ratio = []
    for i in range(len(cluster_range)-1):
        all_ratio.append(np.sum((x>cluster_range[i]) & (x<cluster_range[i+1]))/x.shape[0])
    all_ratio.append(sum(all_ratio))
    ratio = [round(i,4) for i in all_ratio]
    return ratio

