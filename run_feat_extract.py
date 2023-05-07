import argparse
from celltrack.datamodules.preprocess_seq2graph_3D import create_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_images', type=str, default="./data/CTC/Training/Fluo-N3DH-CE")
    parser.add_argument('--input_masks', type=str, default="./data/CTC/Training/Fluo-N3DH-CE")
    parser.add_argument('--input_seg', type=str, default="./data/CTC/Training/Fluo-N3DH-CE")
    parser.add_argument('--input_model', type=str, default=None, 
                        help="only valid when extracting metric learning features for GNN tracking")
    parser.add_argument('--output_csv', type=str, default="./data/basic_features/")
    parser.add_argument('--basic', action='store_true', default=False)
    parser.add_argument('--sequences', nargs='+', default=['01', '02'])
    parser.add_argument('--seg_dir', type=str, default='_GT/TRA')
    args = parser.parse_args()

    create_csv(**vars(args))


