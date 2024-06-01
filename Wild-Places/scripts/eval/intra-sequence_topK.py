import os 
import argparse 
import pandas as pd 
import numpy as np 
import sys
from tqdm import tqdm 
import torch
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils import get_latent_vectors, load_from_pickle, cosine_dist, euclidean_dist, query_to_timestamp
from logg3d.logg3d_utils import get_latent_vectors_logg3d
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../LoGG3D-Net/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../LoGG3D-Net/'))
sys.path.append('cluster/work/riner/users/PLR-2024/haozhu1/LoGG3D-Net/')
# from LoGG3D-Net.models.pipeline_factory import get_pipeline
from models.pipeline_factory import get_pipeline

import time


def eval_singlesession(database, embeddings, args):
    test_topK = np.arange(0, 6)
    # test_topK = [6144, 5120, 4096, 2048, 1024]
    stats = pd.DataFrame(columns = ['F1max', 'Recall@1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Locations', 'Model Inference Time'])
    
    model = get_pipeline('LOGG3D', logg3d_dim=args.logg3d_outdim)
    checkpoint = torch.load(args.logg3d_cpt)  # ,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    model = model.cuda()
    
    org_database = database
    eval_seq = str(args.databases).split('/')[-1].split('.')[-2]
    
    # Get embeddings, timestamps,coords and start time 
    database = load_from_pickle(database)

    timestamps = [query_to_timestamp(database[k]['query']) for k in range(len(database.keys()))]
    coords = np.array([[database[k]['easting'], database[k]['northing']] for k in range(len(database.keys()))])
    start_time = timestamps[0]

    # Thresholds, other trackers
    thresholds = np.linspace(0, 1, 1000)
    num_thresholds = len(thresholds)


    # Get similarity function 
    if args.similarity_function == 'cosine':
        dist_func = cosine_dist
    elif args.similarity_function == 'euclidean':
        dist_func = euclidean_dist
    else:
        raise ValueError(f'No supported distance function for {args.similarity_function}')
    
    
    plt.figure()
    plt.title('F1 Score of Seq ' + str(eval_seq))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1.1])
    plt.xticks(np.arange(0, 1.01, step=0.1))
    plt.grid(True)
    plt.legend()
    
    for topK in test_topK:
        print('starting evaluate top ', topK)
        num_true_positive = np.zeros(num_thresholds)
        num_false_positive = np.zeros(num_thresholds)
        num_true_negative = np.zeros(num_thresholds)
        num_false_negative = np.zeros(num_thresholds)
        
        num_revisits = 0
        num_correct_loc = 0
        
        
        time_test_start = time.time()
        
        embeddings = get_latent_vectors_logg3d(model=model, dataset=org_database, dataset_folder=args.pcd_path, topK=topK, gd_dim=args.logg3d_outdim * args.logg3d_outdim)

        for query_idx in tqdm(range(len(database)), desc = 'Evaluating Embeddings'):
            q_embedding = embeddings[query_idx]
            q_timestamp = timestamps[query_idx]
            q_coord = coords[query_idx]

            # Exit if time elapsed since start is less than time threshold 
            if (q_timestamp - start_time - args.time_thresh) < 0:
                continue 

            # Build retrieval database 
            tt = next(x[0] for x in enumerate(timestamps) if x[1] > (q_timestamp - args.time_thresh))
            seen_embeddings = embeddings[:tt+1]
            seen_coords = coords[:tt+1]

            # Get distances in feature space and world 
            dist_seen_embedding = dist_func(q_embedding, seen_embeddings)
            dist_seen_world = euclidean_dist(q_coord, seen_coords)

            # Check if re-visit 
            if np.any(dist_seen_world < args.world_thresh):
                revisit = True 
                num_revisits += 1 
            else:
                revisit = False 

            # Get top-1 candidate and distances in real world, embedding space 
            top1_idx = np.argmin(dist_seen_embedding)
            top1_embed_dist = dist_seen_embedding[top1_idx]
            top1_world_dist = dist_seen_world[top1_idx]

            if top1_world_dist < args.world_thresh:
                num_correct_loc += 1 

            # Evaluate top-1 candidate 
            for thresh_idx in range(num_thresholds):
                threshold = thresholds[thresh_idx]

                if top1_embed_dist < threshold: # Positive Prediction
                    if top1_world_dist < args.world_thresh:
                        num_true_positive[thresh_idx] += 1
                    else:
                        num_false_positive[thresh_idx] += 1
                else: # Negative Prediction
                    if not revisit:
                        num_true_negative[thresh_idx] += 1
                    else:
                        num_false_negative[thresh_idx] += 1

        # Find F1Max and Recall@1 
        recall_1 = num_correct_loc / num_revisits

        F1max = 0.0 

        Precisions, Recalls = [], []

        for thresh_idx in range(num_thresholds):
            nTruePositive = num_true_positive[thresh_idx]
            nFalsePositive = num_false_positive[thresh_idx]
            nTrueNegative = num_true_negative[thresh_idx]
            nFalseNegative = num_false_negative[thresh_idx]

            Precision = 0.0
            Recall = 0.0
            F1 = 0.0

            if nTruePositive > 0.0:
                Precision = nTruePositive / (nTruePositive + nFalsePositive)
                Recall = nTruePositive / (nTruePositive + nFalseNegative)
                F1 = 2 * Precision * Recall * (1/(Precision + Recall))

            if F1 > F1max:
                F1max = F1


            Precisions.append(Precision)
            Recalls.append(Recall)
    
        ## Ploting PR curve
        plt.plot(Recalls, Precisions, marker='.', label='topK = {}'.format(topK))
        plt.legend()
        
        time_test_end = time.time()
        elapsed_time  = time_test_end - time_test_start
        
        ## Saving to csv file
        stats.loc[topK] = [F1max, recall_1, len(embeddings), num_revisits, num_correct_loc, elapsed_time]
        
        print({'topK': topK, 'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc, 'Model Inference Time': elapsed_time})
        
        del num_true_positive
        del num_false_positive
        del num_true_negative
        del num_false_negative
        del Precisions
        del Recalls
        del embeddings
        
    if args.save_dir:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        stats.to_csv(os.path.join(args.save_dir, eval_seq + '_intra-run_results_KN%.csv'), index = False)
        
        eval_cpt = str(args.logg3d_cpt).split('/')[-1].split('.')[-2]
        
        save_plot_dir = os.path.join(args.save_dir, 'pr_curves')
        if not os.path.exists(save_plot_dir):
            os.makedirs(save_plot_dir)
        eval_seq = str(eval_seq).split('/')[-1]
        plt.savefig(save_plot_dir + '/' + eval_seq + '_KN%_' + eval_cpt + '.png')
    

    # return {'F1max': F1max, 'Recall@1': recall_1, 'Sequence Length': len(embeddings), 'Num. Revisits': num_revisits, 'Num. Correct Locations': num_correct_loc, 'Model Inference Time': elapsed_time}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading and saving paths 
    parser.add_argument('--databases', required = True, type = str, nargs = '+', help = 'List of paths to pickles containing info about database sets')
    parser.add_argument('--database_features', default = None, type = str, nargs = '+', help = 'List of paths to pickles containing feature vectors for database sets')
    parser.add_argument('--pcd_path', default = None, type = str, help = 'paths to datasets')
    parser.add_argument('--logg3d_cpt', default = None, type = str, help = 'paths to checkpoint')
    parser.add_argument('--logg3d_outdim', default = 16, type = int, help = 'feature dimension of LOGG3D-Net')
    
    
    
    parser.add_argument('--run_names', type = str, nargs = '+', help = 'List of names of runs being evaluated')
    parser.add_argument('--save_dir', type = str, default = None, help = 'Save Directory for results csv')
    # Eval parameters
    parser.add_argument('--world_thresh', type = float, default = 3, help = 'Distance to be considered revisit in world')
    parser.add_argument('--time_thresh', type = float, default = 600, help = 'Time before a previous frame can be considered a valid revisit')
    parser.add_argument('--similarity_function', type = str, default = 'cosine', help = 'Distance function used to calculate similarity of embeddings')
    args = parser.parse_args()

    # stats = pd.DataFrame(columns = ['F1max', 'Recall@1', 'Sequence Length', 'Num. Revisits', 'Num. Correct Locations', 'Model Inference Time'])
    for database, embeddings, location in zip(args.databases, args.database_features, args.run_names):
        eval_singlesession(database, embeddings, args)


