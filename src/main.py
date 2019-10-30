# setting the hyper parameters
import argparse

import pandas as pd

from keras_dec import DeepEmbeddingClustering

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_file', default='../data/embedding.tsv')
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--input_dim', type=int)

    parser.add_argument('--finetune_iters', default=500, type=int)
    parser.add_argument('--layerwise_pretrain_iters', default=1000, type=int)
    parser.add_argument('--iter_max', default=1000, type=int)
    parser.add_argument('--tol', default=0.001, type=float)

    parser.add_argument('--output_file', default='../data/cluster.csv')

    args = parser.parse_args()

    df_X = pd.read_csv(args.input_file, sep='\t', header=None)
    X = df_X.values
    print(len(X))
    dec = DeepEmbeddingClustering(n_clusters=args.n_clusters, input_dim=len(df_X.columns))
    dec.initialize(X, finetune_iters=args.finetune_iters, layerwise_pretrain_iters=args.layerwise_pretrain_iters)
    pred_y = dec.cluster(X, y=None, tol=args.tol, iter_max=args.iter_max)

    print(pred_y)

    d = {
        'pred_y': pred_y,
    }

    df_pred_clusters = pd.DataFrame(d)
    df_pred_clusters.to_csv(args.output_file, index=False, header=False)

