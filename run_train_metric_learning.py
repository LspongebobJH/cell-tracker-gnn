from pytorch_metric_learning import losses, miners, reducers, distances, trainers, testers
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import torch
import tqdm
import pprint
from datetime import datetime

from celltrack.datamodules.sampler import MPerClassSampler_weighted
from celltrack.datamodules.dataset_3D import ImgDataset
from celltrack.models.resnet_3d.resnet import set_model_architecture, MLP
import os

import argparse

def train(device,
         base_dir,
         model_name,
         embedding_dim,
         dataset_dict_keys,
         batch_size,
         num_epochs,
         num_workers,
         lr_trunk,
         lr_embedder,
         weight_decay,
         epsilon_miner,
         shorter,
         patience,
         m_samples=4,
         avg_of_avgs=True,
         k="max_bin_count",
         normalized_feat=False,
         **data_config
         ):
    model_folder = os.path.join(base_dir, "saved_models")
    logs_folder = os.path.join(base_dir, "logs")
    tensorboard_folder = os.path.join(base_dir, "tensorboard")

    record_keeper, _, _ = logging_presets.get_record_keeper(logs_folder, tensorboard_folder)
    hooks = logging_presets.get_hook_container(record_keeper)

    trunk = set_model_architecture(model_name)
    trunk_output_size = trunk.input_features_fc_layer
    trunk = trunk.to(device)
    embedder = MLP([trunk_output_size, embedding_dim], normalized_feat=normalized_feat).to(device)

    # Set optimizers
    trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=lr_trunk, weight_decay=weight_decay)
    embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=lr_embedder, weight_decay=weight_decay)

    train_dataset = ImgDataset(**data_config, type_data='train')
    val_dataset = ImgDataset(**data_config, type_data='valid')
    test_data = ImgDataset(**data_config, type_data='test')

    print(train_dataset.curr_roi)
    print(f"train_dataset length:{len(train_dataset)}")
    print(f"val_dataset length:{len(val_dataset)}")
    print(f"test_data length:{len(test_data)}")

    # Set the loss function AND the mining function
    print("use MultiSimilarityLoss")
    loss = losses.MultiSimilarityLoss(distance=CosineSimilarity(), reducer=MeanReducer())
    miner = miners.MultiSimilarityMiner(epsilon=epsilon_miner)

    # Set the dataloader sampler
    sampler = MPerClassSampler_weighted(train_dataset.targets, frames=train_dataset.frames_for_sampler, m=m_samples,
                                        length_before_new_iter=len(train_dataset))

    # Package the above stuff into dictionaries.
    models = {"trunk": trunk, "embedder": embedder}
    optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
    loss_funcs = {"metric_loss": loss}
    mining_funcs = {"tuple_miner": miner}
    dataset_dict = {}

    for key in dataset_dict_keys:
        if key == 'train':
            dataset_dict['train'] = train_dataset
        if key == 'val':
            dataset_dict['val'] = val_dataset
        if key == 'test':
            dataset_dict['test'] = test_data

    print(f"dataset_dict keys: {dataset_dict.keys()}")

    accuracy_calculator = AccuracyCalculator(avg_of_avgs=avg_of_avgs, k=k)

    # Create the tester

    def end_of_testing_hook(tester):
        for split, (embeddings, labels) in tester.embeddings_and_labels.items():
            dataset = common_functions.EmbeddingDataset(embeddings.cpu().numpy(), labels.squeeze(1).cpu().numpy())
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=1)
            total_loss = 0
            with torch.no_grad():
                print(f"getting loss for {split} set")
                for E, L in tqdm.tqdm(dataloader):
                    total_loss += loss(E, L)
            total_loss /= len(dataloader)
            tester.all_accuracies[split]["loss"] = total_loss
        hooks.end_of_testing_hook(tester)

    if shorter:
        end_of_testing_hook_for_tester = hooks.end_of_testing_hook
    else:
        end_of_testing_hook_for_tester = end_of_testing_hook

    tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook=end_of_testing_hook_for_tester,
                                                dataloader_num_workers=num_workers,
                                                accuracy_calculator=accuracy_calculator
                                                )

    end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                                dataset_dict,
                                                model_folder,
                                                patience=patience)

    trainer = trainers.MetricLossOnly(models=models,
                                      optimizers=optimizers,
                                      batch_size=batch_size,
                                      loss_funcs=loss_funcs,
                                      mining_funcs=mining_funcs,
                                      dataset=train_dataset,
                                      sampler=sampler,
                                      dataloader_num_workers=num_workers,
                                      end_of_iteration_hook=hooks.end_of_iteration_hook,
                                      end_of_epoch_hook=end_of_epoch_hook)

    trainer.train(num_epochs=num_epochs)
    # TODO: when the number of epochs is too small, there may be no best checkpoint. Can fix?

    log_dir = os.path.join(os.getcwd(), base_dir)

    save_model = os.path.join(log_dir, "saved_models")
    for file in os.listdir(save_model):
        if file.startswith('trunk_best'):
            trunk_ckpt_path = os.path.join(save_model, file)
        if file.startswith('embedder_best'):
            embedder_ckpt_path = os.path.join(save_model, file)

    print(f"best trunk_ckpt: {trunk_ckpt_path}")
    print(f"best embedder_ckpt: {embedder_ckpt_path}")
    trunk_ckpt = torch.load(trunk_ckpt_path)
    embedder_ckpt = torch.load(embedder_ckpt_path)

    dict_params = {}
    dict_params['min_cell'] = test_data.min_cell
    dict_params['max_cell'] = test_data.max_cell
    dict_params['pad_value'] = test_data.pad_value

    dict_params['roi'] = test_data.curr_roi

    # models params
    dict_params['model_name'] = model_name
    dict_params['mlp_dims'] = [trunk_output_size, embedding_dim]
    dict_params['mlp_normalized_features'] = normalized_feat

    # models state_dict
    dict_params['trunk_state_dict'] = trunk_ckpt
    dict_params['embedder_state_dict'] = embedder_ckpt

    save_path = os.path.join(log_dir, 'all_params.pth')
    torch.save(dict_params, save_path)
    print(f'save: {save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='3D_SIM_MultiSimilarityLoss')
    parser.add_argument('--model_name', type=str, default='resnet18_3d')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--normalized_feat', action='store_true', default=False)
    parser.add_argument('--dataset_dict_keys', nargs='+', type=str, default=['val'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr_trunk', type=float, default=0.00001)
    parser.add_argument('--lr_embedder', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--epsilon_miner', type=float, default=0.1)

    parser.add_argument('--shorter', action='store_true', default=False)
    parser.add_argument('--normalize_type', type=str, default='MinMaxCell')

    parser.add_argument('--avg_of_avgs', action='store_true', default=False)
    parser.add_argument('--k', type=str, default="max_bin_count")
    parser.add_argument('--patience', type=int, default=25)

    parser.add_argument('--pad_value', type=int, default=0)
    parser.add_argument('--norm_value', type=int, default=0)
    parser.add_argument('--train_val_test_split', nargs='+', default=[80, 20, 0])

    parser.add_argument('--data_dir_img', type=str, default="./data/CTC/Training/Fluo-N3DH-CE")
    parser.add_argument('--data_dir_mask', type=str, default="./data/CTC/Training/Fluo-N3DH-CE")
    parser.add_argument('--subdir_mask', type=str, default='GT/TRA')
    parser.add_argument('--dir_csv', type=str, default="./data/basic_features/Fluo-N3DH-CE")

    parser.add_argument('--curr_seq', type=int, default=1)
    parser.add_argument('--num_sequences', type=int, default=2)
    parser.add_argument('--deviation', type=str, default='no_overlap')
    args = parser.parse_args()

    datetime_object = str(datetime.now())
    datetime_object = datetime_object.split('.')[0].replace(':', '-').replace(' ', '/')
    print(f"start time: {datetime_object}")
    base_dir = "logs/logs_" + args.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Config dictionary")
    pprint.pprint(args, sort_dicts=False)

    config = vars(args)
    config.pop('exp_name')

    train(device,
          patch_based=False,
          base_dir=base_dir,
          **vars(args))









