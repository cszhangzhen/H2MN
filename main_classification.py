import os
import time
import glob
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import Model
from parser import parameter_parser
from utils import tab_printer, GraphClassificationDataset

args = parameter_parser()
dataset = GraphClassificationDataset(args)
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
            main_index = 0
            loss_sum = 0
            batches = dataset.create_batches(dataset.training_funcs, dataset.collate)
            for index, batch_pair in enumerate(batches):
                optimizer.zero_grad()
                data = dataset.transform(batch_pair)
                prediction = model(data)
                loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')
                loss.backward()
                optimizer.step()
                main_index = main_index + len(batch_pair[2])
                loss_sum = loss_sum + loss.item()
            loss = loss_sum / main_index
            # start validate at 9000th iteration
            if epoch + 1 < 0:
                end = time.time()
                print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'time: {:.6f}s'.format(end - start))
            else:
                val_loss, aucscore = validate(dataset, dataset.validation_funcs)
                end = time.time()
                print('Epoch: {:05d},'.format(epoch + 1), 'loss_train: {:.6f},'.format(loss), 'loss_val: {:.6f},'.format(val_loss), 'AUC: {:.6f},'.format(aucscore), 'time: {:.6f}s'.format(end - start))
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


def validate(datasets, funcs):
    model.eval()
    main_index = 0
    loss_sum = 0
    with torch.no_grad():
        pred = []
        gt = []
        batches = datasets.create_batches(funcs, datasets.collate)
        for index, batch_pair in enumerate(batches):
            data = datasets.transform(batch_pair)
            prediction = model(data)
            loss = F.binary_cross_entropy(prediction, data['target'], reduction='sum')
            main_index = main_index + len(batch_pair[2])
            loss_sum = loss_sum + loss.item()
            
            batch_gt = batch_pair[2]
            batch_pred = list(prediction.detach().cpu().numpy())

            pred = pred + batch_pred
            gt = gt + batch_gt
        
        loss = loss_sum / main_index
        gt = np.array(gt, dtype=np.float32)
        pred = np.array(pred, dtype=np.float32)
        score = roc_auc_score(gt, pred)

        return loss, score


if __name__ == "__main__":
    best_model = train()
    model.load_state_dict(torch.load('{}.pth'.format(best_model)))
    print('\nModel evaluation.')
    test_loss, test_auc = validate(dataset, dataset.testing_funcs)
    print('Test set results, loss = {:.6f}, AUC = {:.6f}'.format(test_loss, test_auc))
