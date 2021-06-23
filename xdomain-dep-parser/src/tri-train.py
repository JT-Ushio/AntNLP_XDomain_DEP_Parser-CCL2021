import argparse
import os, time
import random
import logging
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import neptune.new as neptune
import numpy as np
from antu.io import Vocabulary, glove_reader
from antu.io.configurators import IniConfigurator
from antu.utils.dual_channel_logger import dual_channel_logger

import warnings
warnings.filterwarnings("ignore")
from module.exp_scheduler import ExponentialLRwithWarmUp
from parser import Parser
from eval.PTB_evaluator import ptb_evaluation
from utils.conllu_reader import PTBReader
from utils.conllu_dataset import CoNLLUDataset, conllu_fn

def main():
    # Configuration file processing
    parser = argparse.ArgumentParser(description="Usage for DEP Parsing.")
    parser.add_argument('--CFG', type=str, help="Path to config file.")
    parser.add_argument('--DEBUG', action='store_true', help="DEBUG mode.")
    args, extra_args = parser.parse_known_args()
    cfg = IniConfigurator(args.CFG, extra_args)
    if not os.path.exists(cfg.ckpt_dir):
        os.makedirs(cfg.ckpt_dir)

    # Set seeds
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # Logger setting
    logger = dual_channel_logger(
        __name__,
        file_path=cfg.LOG,
        file_model='w',
        formatter="%(asctime)s - %(levelname)s - %(message)s",
        time_formatter="%m-%d %H:%M")

    # Setup neptune mode="debug" run="CCL-9",
    run = neptune.init(project='ushio/CCL2021', name=cfg.exp_name,
                       tags=['parallel homework', ], mode='debug')

    #                   tags=[cfg.DOMAIN, str(cfg.MIN_PROB), 'SNorm', 'giga100', ], mode='debug')
    run['parameters'] = vars(cfg)

    # Build data reader
    field_list=['word', 'tag', 'head', 'rel', 'prob']
    data_reader = PTBReader(
        field_list=field_list, spacer=r'[\t]', min_prob=cfg.MIN_PROB,
        root='0\t**root**\t_\t**rcpos**\t**rpos**\t_\t0\t**rrel**\t_\t0',)

    # Build vocabulary with pretrained glove
    vocabulary = Vocabulary()
    g_word, _ = glove_reader(cfg.GLOVE)
    vocabulary.extend_from_pretrained_vocab({'glove': g_word,})
    counters = {'word': Counter(), 'tag': Counter(), 'rel': Counter(), 'char': Counter()}

    # Build the dataset
    DEBUG = cfg.data_dir+'/train.debug'
    TRAIN, DEV, TEST = (cfg.TRAIN, cfg.DEV, cfg.TEST) if not args.DEBUG else (DEBUG, DEBUG, DEBUG)
    train_set = CoNLLUDataset(
        TRAIN, data_reader, vocabulary, counters, {'word': cfg.MIN_COUNT},
        no_pad_namespace={'rel'}, no_unk_namespace={'rel'})
    dev_set  = CoNLLUDataset(DEV,  data_reader, vocabulary)
    # print(counters['char'])
    # test_set = CoNLLUDataset(TEST, data_reader, vocabulary)

    # Build the data-loader
    train = DataLoader(train_set, cfg.N_BATCH, True,  None, None, cfg.N_WORKER, conllu_fn, cfg.N_WORKER>0)
    dev   = DataLoader(dev_set,   cfg.N_BATCH, False, None, None, cfg.N_WORKER, conllu_fn, cfg.N_WORKER>0)
    # test  = DataLoader(test_set,  cfg.N_BATCH, False, None, None, cfg.N_WORKER, conllu_fn, cfg.N_WORKER>0)
    train_ = DataLoader(train_set, cfg.N_BATCH, False, None, None, cfg.N_WORKER, conllu_fn, cfg.N_WORKER>0)

    # Build parser model
    parser1 = Parser(vocabulary, cfg)
    parser2 = Parser(vocabulary, cfg)
    parser3 = Parser(vocabulary, cfg)

    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')
        cuda2 = torch.device('cuda:2')
        parser1 = parser1.to(cuda0)
        parser2 = parser2.to(cuda1)
        parser3 = parser3.to(cuda2)


    CELoss = nn.CrossEntropyLoss()
    # build optimizers
    optim1 = AdamW(parser.parameters(), cfg.LR, cfg.BETAS, cfg.EPS)
    sched1 = ExponentialLRwithWarmUp(
        optim, cfg.LR_DECAY, cfg.LR_ANNEAL, cfg.LR_DOUBLE, cfg.LR_WARM)
    optim2 = AdamW(parser.parameters(), cfg.LR, cfg.BETAS, cfg.EPS)
    sched2 = ExponentialLRwithWarmUp(
        optim, cfg.LR_DECAY, cfg.LR_ANNEAL, cfg.LR_DOUBLE, cfg.LR_WARM)
    optim3 = AdamW(parser.parameters(), cfg.LR, cfg.BETAS, cfg.EPS)
    sched3 = ExponentialLRwithWarmUp(
        optim, cfg.LR_DECAY, cfg.LR_ANNEAL, cfg.LR_DOUBLE, cfg.LR_WARM)

    # load checkpoint if wanted
    start_epoch = best_uas = best_las = best_epoch = 0
    def load_ckpt(ckpt_path: str):
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch']+1
        best_uas, best_las, best_epoch = ckpt['best']
        parser1.load_state_dict(ckpt['parser1'])
        optim1.load_state_dict(ckpt['optim1'])
        sched1.load_state_dict(ckpt['sched1'])
        parser2.load_state_dict(ckpt['parser2'])
        optim2.load_state_dict(ckpt['optim2'])
        sched2.load_state_dict(ckpt['sched2'])
        parser3.load_state_dict(ckpt['parser3'])
        optim3.load_state_dict(ckpt['optim3'])
        sched3.load_state_dict(ckpt['sched3'])
        return start_epoch, best_uas, best_las, best_epoch

    if cfg.IS_RESUME:
        start_epoch, best_uas, best_las, best_epoch = load_ckpt(cfg.LAST)

    @torch.no_grad()
    def validation(parser, data_loader: DataLoader, pred_path: str, gold_path: str):
        pred = {'arcs': [], 'rels': []}
        arc_losses, rel_losses = [], []
        for data in data_loader:
            if cfg.N_WORKER:
                for x in data.keys(): data[x] = data[x].cuda()
            arc_loss, rel_loss, arcs, rels = parser(data)
            arc_losses.append(arc_loss.item())
            rel_losses.append(rel_loss.item())
            pred['arcs'].extend(arcs)
            pred['rels'].extend(rels)
        uas, las = ptb_evaluation(vocabulary, pred, pred_path, gold_path)
        return float(np.mean(arc_losses)), float(np.mean(rel_losses)), uas, las

    # Train model
    tot_dataio = 0
    tot_calc = 0
    for epoch in range(start_epoch, cfg.N_EPOCH):
        parser1.train()
        parser2.train()
        parser3.train()
        arc_losses, rel_losses = [], []
        for n_iter, data in enumerate(train):
            if cfg.N_WORKER:
                for x in data.keys():
                    data0[x] = data[x].to(cuda0)
                    data1[x] = data[x].to(cuda1)
                    data2[x] = data[x].to(cuda2)

            arc_loss, rel_loss = parser1(data)
            arc_losses.append(arc_loss.item())
            rel_losses.append(rel_loss.item())
            ((arc_loss+rel_loss)/cfg.STEP_UPDATE).backward()

            arc_loss, rel_loss = parser2(data)
            arc_losses.append(arc_loss.item())
            rel_losses.append(rel_loss.item())
            ((arc_loss+rel_loss)/cfg.STEP_UPDATE).backward()

            arc_loss, rel_loss = parser3(data)
            arc_losses.append(arc_loss.item())
            rel_losses.append(rel_loss.item())
            ((arc_loss+rel_loss)/cfg.STEP_UPDATE).backward()

            # Actual update
            if n_iter % cfg.STEP_UPDATE == cfg.STEP_UPDATE-1:
                nn.utils.clip_grad_norm_(parser1.parameters(), cfg.CLIP)
                nn.utils.clip_grad_norm_(parser2.parameters(), cfg.CLIP)
                nn.utils.clip_grad_norm_(parser3.parameters(), cfg.CLIP)
                optim1.step()
                optim1.zero_grad()
                sched1.step()
                optim2.step()
                optim2.zero_grad()
                sched2.step()
                optim3.step()
                optim3.zero_grad()
                sched3.step()
            end2 = time.perf_counter()
            tot_calc += end2 - beg2
        end = time.perf_counter()
        tot_dataio += end - beg
        if epoch%cfg.STEP_VALID != cfg.STEP_VALID-1: continue

        # save current parser
        torch.save({
            'epoch': epoch,
            'best': (best_uas, best_las, best_epoch),
            'parser1': parser1.state_dict(),
            'optim1': optim1.state_dict(),
            'sched1': sched1.state_dict(),
            'parser2': parser2.state_dict(),
            'optim2': optim2.state_dict(),
            'sched2': sched2.state_dict(),
            'parser3': parser3.state_dict(),
            'optim3': optim3.state_dict(),
            'sched3': sched3.state_dict(),
        }, cfg.LAST)

        # validate parer on dev set
        parser1.eval()
        arc_loss, rel_loss, uas, las = validation(parser1, dev, cfg.PRED_DEV, DEV)
        if uas > best_uas and las > best_las or uas+las > best_uas+best_las:
            best_uas, best_las, best_epoch = uas, las, epoch
            os.popen(f'cp {cfg.LAST} {cfg.BEST}')
        run["valid1/uas"].log(uas)
        run["valid1/las"].log(las)
        run["valid1/arc_loss"].log(arc_loss)
        run["valid1/rel_loss"].log(rel_loss)
        logger.info(
            f'|{epoch:5}| Arc({float(np.mean(arc_losses)):.2f}) '
            f'Rel({float(np.mean(rel_losses)):.2f}) Best({best_epoch})')
        run["train/train_arc_loss"].log(float(np.mean(arc_losses)))
        run["train/train_rel_loss"].log(float(np.mean(rel_losses)))
        logger.info(f'|  Dev| UAS:{uas:6.2f}, LAS:{las:6.2f}, '
                    f'Arc({arc_loss:.2f}), Rel({rel_loss:.2f})')

    run["time"] = tot_dataio-tot_calc
    run["tot_time"] = tot_dataio
    logger.info(f'*Best Dev Result* UAS:{best_uas:6.2f}, LAS:{best_las:6.2f}, Epoch({best_epoch})')
    run["best"] = (best_uas, best_las, best_epoch)

if __name__ == '__main__':
    main()
