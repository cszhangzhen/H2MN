import os
import time
import glob

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import DataLoader
from scipy.stats import spearmanr

from models import Model
from parser import parameter_parser
from utils import tab_printer, GraphRegressionDataset, prec_at_ks, calculate_ranking_correlation

args = parameter_parser()
dataset = GraphRegressionDataset(args)
args.num_features = dataset.number_features

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
tab_printer(args)

model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def train():
    print('\nModel training.\n')
    start = time.time()
    val_loss_values = []
    patience_cnt = 0
    best_epoch = 0
    min_loss = 1e10

    with torch.autograd.detect_anomaly():
        for epoch in range(args.epochs):
            model.train()
            batches = dataset.create_batches(dataset.training_set)
            main_index = 0
            loss_sum = 0
            for index, batch_pair in enumerate(batches):
                optimizer.zero_grad()
                data = dataset.transform(batch_pair)
                prediction = model(data)
                loss = F.mse_loss(prediction, data['target'], reduction='sum')
                loss.backward()
                optimizer.step()
                main_index = main_index + batch_pair[0].num_graphs
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index
            # start validate at 9000th iteration
            if epoch + 1 < 9000:
                end = time.time()
                print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'time: {:.6f}s'.format(end - start))
            else:
                val_loss = validate()
                end = time.time()
                print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'loss_val: {:.6f},'.format(val_loss), 'time: {:.6f}s'.format(end - start))
                val_loss_values.append(val_loss)
                torch.save(model.state_dict(), '{}.pth'.format(epoch))
                if val_loss_values[-1] < min_loss:
                    min_loss = val_loss_values[-1]
                    best_epoch = epoch
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    break

                files = glob.glob('*.pth')
                for f in files:
                    epoch_nb = int(f.split('.')[0])
                    if epoch_nb < best_epoch:
                        os.remove(f)

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb > best_epoch:
                os.remove(f)
        print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - start))

        return best_epoch


def validate():
    model.eval()
    batches = dataset.create_batches(dataset.val_set)
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        for index, batch_pair in enumerate(batches):
            data = dataset.transform(batch_pair)
            prediction = model(data)
            loss = F.mse_loss(prediction, data['target'], reduction='sum')
            main_index = main_index + batch_pair[0].num_graphs
            loss_sum = loss_sum + loss.item()
        loss = loss_sum / main_index

    return loss


def evaluate():
    print('\nModel evaluation.')
    model.eval()
    scores = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))
    ground_truth = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))
    prediction_mat = np.zeros((len(dataset.testing_graphs), len(dataset.training_graphs)))

    rho_list = []
    tau_list = []
    prec_at_10_list = []
    prec_at_20_list = []

    with torch.no_grad():
        for i, g in enumerate(dataset.testing_graphs):
            if len(dataset.training_graphs) <= args.batch_size:
                source_batch = Batch.from_data_list([g] * len(dataset.training_graphs))
                target_batch = Batch.from_data_list(dataset.training_graphs)

                data = dataset.transform((source_batch, target_batch))
                target = data['target']
                ground_truth[i] = target.cpu().numpy()
                prediction = model(data)
                prediction_mat[i] = prediction.detach().cpu().numpy()

                scores[i] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()

                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                tau_list.append(calculate_ranking_correlation(kendalltau, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))
                prec_at_20_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 20))
            else:
                # Avoid GPU OOM error
                batch_index = 0
                target_loader = DataLoader(dataset.training_graphs, batch_size=args.batch_size, shuffle=False)
                for index, target_batch in enumerate(target_loader):
                    source_batch = Batch.from_data_list([g] * target_batch.num_graphs)
                    data = dataset.transform((source_batch, target_batch))
                    target = data['target']
                    num_graphs = target_batch.num_graphs
                    ground_truth[i,batch_index: batch_index+num_graphs] = target.cpu().numpy()
                    prediction = model(data)
                    prediction_mat[i,batch_index: batch_index+num_graphs] = prediction.detach().cpu().numpy()
                    scores[i,batch_index: batch_index+num_graphs] = F.mse_loss(prediction, target, reduction='none').detach().cpu().numpy()
                    batch_index += num_graphs
                
                rho_list.append(calculate_ranking_correlation(spearmanr, prediction_mat[i], ground_truth[i]))
                prec_at_10_list.append(prec_at_ks(ground_truth[i], prediction_mat[i], 10))

    rho = np.mean(rho_list)
    prec_at_10 = np.mean(prec_at_10_list)
    model_error = np.mean(scores) * 0.5
    print_evaluation(model_error, rho, prec_at_10)


def print_evaluation(model_error, rho, prec_at_10):
    print("\nmse(10^-3): " + str(round(model_error * 1000, 5)))
    print("Spearman's rho: " + str(round(rho, 5)))
    print("p@10: " + str(round(prec_at_10, 5)))


if __name__ == "__main__":
    best_model = train()
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    evaluate()
