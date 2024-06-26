import argparse
import os
import pandas as pd
import tqdm
import numpy as np
import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import matplotlib as plt
from dataset import CSVDataset
from utils import plot_result, get_deterministic_coefficient, get_mean_squared_error, get_Nash_efficiency_coefficient, get_Kling_Gupta_efficiency,\
    init_results, print_results, print_args
from adversarial_domain_adaptation_utils import save_backbone_features
from model.Raindrop_encoder import Raindrop_encoder
from model.Joint_encoder import Joint_encoder

torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed_all(1)


def predict(model, test_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for index, (raindrop, runoff_history, runoff) in enumerate(test_loader):
            raindrop, runoff_history, runoff = \
                raindrop.cuda().float(), runoff_history.cuda().float(), runoff.cuda().float()
            prediction = model.inference(raindrop, runoff_history)

            predictions.extend(
                prediction.flatten().detach().cpu().numpy().tolist())
            targets.extend(runoff.flatten().detach().cpu().tolist())
    predictions, targets = np.array(predictions), np.array(targets)

    return predictions, targets


def train(model, train_loader, test_loader, writer, save_path):
    print_args(args)

    print('\tsaving checkpoints to:', save_path)
    log_file = save_path + '/TRAIN_LOG_{}.csv'.format(args.exp_description)
    print('\tsaving training log to:', save_path + log_file)

    print('\n--- starting training...')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-4, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.N_EPOCH)

    global_step = 0
    for epoch in range(args.N_EPOCH):
        model.train()
        predictions = []
        targets = []

        for step, (raindrop, runoff_history, runoff) in enumerate(train_loader):
            raindrop, runoff_history, runoff = \
                raindrop.cuda().float(), runoff_history.cuda().float(), runoff.cuda().float()

            optimizer.zero_grad()
            train_loss, info = model(raindrop, runoff_history, runoff)
            train_loss.backward()
            optimizer.step()

            predictions.extend(info['predictions'])
            targets.extend(info['targets'])

            for key in info['WRITER_KEYS']:
                writer.add_scalar('step_log/' + key, info[key], global_step)

            global_step += 1

        scheduler.step()

        predictions, targets = np.array(predictions), np.array(targets)

        TRAIN_MSE = get_mean_squared_error(predictions, targets)
        TRAIN_DC = get_deterministic_coefficient(predictions, targets)
        TRAIN_NSE = get_Nash_efficiency_coefficient(predictions, targets)
        TRAIN_KGE = get_Kling_Gupta_efficiency(predictions, targets)

        predictions, targets = predict(model, test_loader)
        TEST_MSE = get_mean_squared_error(predictions, targets)
        TEST_DC = get_deterministic_coefficient(predictions, targets)
        TEST_NSE = get_Nash_efficiency_coefficient(predictions, targets)
        TEST_KGE = get_Kling_Gupta_efficiency(predictions, targets)

        tqdm.tqdm.write('Epoch: [{}/{}], TRAIN_MSE: {:.2f}, TEST_MSE: {:.2f}, TRAIN_DC: {:.2f}%, TEST_DC: {:.2f}%, TRAIN_NSE: {:.3f}, TEST_NSE: {:.3f}, TRAIN_KGE: {:.3f}, TEST_KGE: {:.3f} '
                        .format(epoch + 1, args.N_EPOCH, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE))

        writer.add_scalars('epoch_log/mean_squared_error',
                           {'train': TRAIN_MSE, 'test': TEST_MSE}, epoch)
        writer.add_scalars('epoch_log/deterministic_coefficient',
                           {'train': TRAIN_DC, 'test': TEST_DC}, epoch)
        writer.add_scalars('epoch_log/Nash_efficiency_coefficient',
                           {'train': TRAIN_NSE, 'test': TEST_NSE}, epoch)
        writer.add_scalars('epoch_log/Kling_Gupta_efficiency',
                           {'train': TRAIN_KGE, 'test': TEST_KGE}, epoch)
        writer.add_scalar('epoch_log/learning rate', scheduler.get_last_lr()[-1], epoch)

        with open(log_file, 'a+') as f:
            if epoch == 0:
                print('exp_description,', args.exp_description, file=f)
                print(
                    'Epoch,TRAIN_MSE,TEST_MSE,TRAIN_DC,TEST_DC, TRAIN_NSE,TEST_NSE, TRAIN_KGE,TEST_KGE', file=f)
            print('{},{},{},{},{},{},{},{},{}'.format(epoch,
                                          round(TRAIN_MSE, 2), round(TEST_MSE, 2),
                                          round(TRAIN_DC, 2), round(TEST_DC, 2),
                                          round(TRAIN_NSE, 3), round(TEST_NSE, 3),
                                          round(TRAIN_KGE, 3), round(TEST_KGE, 3)), file=f)

        if epoch % 100 == 0 or epoch == args.N_EPOCH - 1:
            torch.save(model.state_dict(), save_path +
                       '/epoch_{}.pt'.format(epoch))
            torch.save(model.state_dict(), save_path +
                       '/last.pt'.format(epoch))
            img_scatter, img_line = plot_result(targets, predictions)
            writer.add_figure("prediction/scatter", img_scatter, epoch)
            writer.add_figure("prediction/line", img_line, epoch)

        if TRAIN_DC > 95 and args.few_shot_num is not None:
            break

    if args.dataset == 'WaterBench':
        save_backbone_features(model, train_loader)

    torch.save(model.state_dict(), save_path + '/last.pt'.format(epoch))

    return model, round(TRAIN_MSE, 2), round(TEST_MSE, 2), round(TRAIN_DC, 2), round(TEST_DC, 2), round(TRAIN_NSE, 3), round(TEST_NSE, 3), round(TRAIN_KGE, 3), round(TEST_KGE, 3)

def main(args,dataset_file):
    writer = SummaryWriter(comment=args.exp_description)
    save_path = writer.get_logdir()
    dataset_file = args.dataset+'/'+dataset_file
    train_set = CSVDataset(forecast_range=args.forecast_range, dataset=dataset_file, mode='train',
                           train_test_split_ratio=args.train_test_split_ratio,
                           sample_length=args.sample_length,
                           training_set_scale=args.training_set_scale,
                           training_set_start=args.training_set_start,
                           few_shot_num=args.few_shot_num)

    test_set = CSVDataset(forecast_range=args.forecast_range, dataset=dataset_file, mode='test',
                          train_test_split_ratio=args.train_test_split_ratio, sample_length=args.sample_length)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    input_size = train_set.get_input_size()

    if args.structure == 'joint':
        model = Joint_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers, args.dropout,
                              input_size + 1, args.head, args.head_hidden_size, args.head_num_layers).cuda()
    elif args.structure == 'direct':
        model = Raindrop_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers, args.dropout,
                                 input_size, args.head, args.head_hidden_size, args.head_num_layers,
                                 residual=False).cuda()
    elif args.structure == 'residual':
        model = Raindrop_encoder(args.backbone, args.backbone_hidden_size, args.backbone_num_layers, args.dropout,
                                 input_size, args.head, args.head_hidden_size, args.head_num_layers,
                                 residual=True).cuda()
    else:
        raise RuntimeError('model {} not defined!'.format(args.structure))

    for name, param in model.named_parameters():
        if 'weight' in name:
            init.xavier_uniform_(param)

    return train(model, train_loader, test_loader, writer, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flood Forecasting')

    parser.add_argument('--exp_description', default='stage1', type=str)
    parser.add_argument('--dataset', default='WaterBench', help='data path')
    parser.add_argument('--N_EPOCH', default=1000, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--train_test_split_ratio', default=0.7, type=float)
    parser.add_argument('--training_set_scale', default=1.0, type=float)
    parser.add_argument('--training_set_start', default=0, type=float)
    parser.add_argument('--sample_length', default=72, type=int)
    parser.add_argument('--forecast_range', default=6, type=int)
    parser.add_argument('--weight_decay', default=0.008, type=bool)

    parser.add_argument('--structure', default='residual', type=str)  # joint, direct, residual
    # RNN, LSTM, GRU, ANN, STGCN, TCN
    parser.add_argument('--backbone', default='STGCN', type=str)
    parser.add_argument('--backbone_hidden_size', default=36, type=int)
    parser.add_argument('--backbone_num_layers', default=3, type=int)
    parser.add_argument('--head', default='conv1d', type=str)  # conv1d, linear
    parser.add_argument('--head_hidden_size', default=36, type=int)
    parser.add_argument('--head_num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)

    parser.add_argument('--few_shot_num', default=None, type=int)
    args = parser.parse_args()


    if args.few_shot_num is not None:

        for scale in np.arange(start=0.075, stop=0.08, step=0.005):
            for start in np.arange(start=0, stop=args.few_shot_num, step=1):
                args.training_set_scale = round(scale, 3)
                args.training_set_start = round(start, 3)

                l = os.listdir('dataset/' + args.dataset + '/')
                sensors = list(l)
                for dataset_file in sensors:
                    if os.path.splitext(dataset_file)[1] == '.csv':
                        results = init_results(args, stage=1)
                        model, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE = main(args,dataset_file)

                        results['TRAIN_MSE'].append(TRAIN_MSE)
                        results['TEST_MSE'].append(TEST_MSE)
                        results['TRAIN_DC'].append(TRAIN_DC)
                        results['TEST_DC'].append(TEST_DC)
                        results['TRAIN_NSE'].append(TRAIN_NSE)
                        results['TEST_NSE'].append(TEST_NSE)
                        results['TRAIN_KGE'].append(TRAIN_KGE)
                        results['TEST_KGE'].append(TEST_KGE)

                        print_results(results)


    else:
        l = os.listdir('dataset/'+args.dataset+'/')
        sensors = list(l)
        for dataset_file in sensors:
            if os.path.splitext(dataset_file)[1] == '.csv':
                results = init_results(args, stage=1)
                model, TRAIN_MSE, TEST_MSE, TRAIN_DC, TEST_DC, TRAIN_NSE, TEST_NSE, TRAIN_KGE, TEST_KGE = main(args,dataset_file)

                results['TRAIN_MSE'].append(TRAIN_MSE)
                results['TEST_MSE'].append(TEST_MSE)
                results['TRAIN_DC'].append(TRAIN_DC)
                results['TEST_DC'].append(TEST_DC)
                results['TRAIN_NSE'].append(TRAIN_NSE)
                results['TEST_NSE'].append(TEST_NSE)
                results['TRAIN_KGE'].append(TRAIN_KGE)
                results['TEST_KGE'].append(TEST_KGE)

                print_results(results)
