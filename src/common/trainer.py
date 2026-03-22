# coding: utf-8
# @email: enoche.chow@gmail.com

r"""
################################
"""

import os
import itertools
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
import matplotlib.pyplot as plt

from time import time
from logging import getLogger

from utils.utils import get_local_time, early_stopping, dict2str
from utils.topk_evaluator import TopKEvaluator


class AbstractTrainer(object):
    r"""Trainer Class is used to manage the training and evaluation processes of recommender system models.
    AbstractTrainer is an abstract class in which the fit() and evaluate() method should be implemented according
    to different training and evaluation strategies.
    """

    def __init__(self, config, model):
        self.config = config
        self.model = model

    def fit(self, train_data):
        r"""Train the model based on the train data.

        """
        raise NotImplementedError('Method [next] should be implemented.')

    def evaluate(self, eval_data):
        r"""Evaluate the model based on the eval data.

        """

        raise NotImplementedError('Method [next] should be implemented.')


class Trainer(AbstractTrainer):
    r"""The basic Trainer for basic training and evaluation strategies in recommender systems. This class defines common
    functions for training and evaluation processes of most recommender system models, including fit(), evaluate(),
   and some other features helpful for model training and evaluation.

    Generally speaking, this class can serve most recommender system models, If the training process of the model is to
    simply optimize a single loss without involving any complex training strategies, such as adversarial learning,
    pre-training and so on.

    Initializing the Trainer needs two parameters: `config` and `model`. `config` records the parameters information
    for controlling training and evaluation, such as `learning_rate`, `epochs`, `eval_step` and so on.
    More information can be found in [placeholder]. `model` is the instantiated object of a Model Class.

    """

    def __init__(self, config, model):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric'].lower()
        self.valid_metric_bigger = config['valid_metric_bigger']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd

        self.req_training = config['req_training']

        self.start_epoch = 0
        self.cur_step = 0

        tmp_dd = {}
        for j, k in list(itertools.product(config['metrics'], config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_score = -1
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd
        self.train_loss_dict = dict()
        self.optimizer = self._build_optimizer()

        #fac = lambda epoch: 0.96 ** (epoch / 50)
        lr_scheduler = config['learning_rate_scheduler']        # check zero?
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)
        self.lr_scheduler = scheduler

        self.eval_type = config['eval_type']
        self.evaluator = TopKEvaluator(config)

        self.item_tensor = None
        self.tot_item_num = None

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx, loss_func=None):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): The train data.
            epoch_idx (int): The current epoch id.
            loss_func (function): The loss function of :attr:`model`. If it is ``None``, the loss function will be
                :attr:`self.model.calculate_loss`. Defaults to ``None``.

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        if not self.req_training:
            return 0.0, []
        self.model.train()
        loss_func = loss_func or self.model.calculate_loss
        total_loss = None
        loss_batches = []
        for batch_idx, interaction in enumerate(train_data):
            self.optimizer.zero_grad()
            losses = loss_func(interaction)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()
            if self._check_nan(loss):
                self.logger.info('Loss is nan at epoch: {}, batch index: {}. Exiting.'.format(epoch_idx, batch_idx))
                return loss, torch.tensor(0.0)
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()
            loss_batches.append(loss.detach())
            # for test
            #if batch_idx == 0:
            #    break
        return total_loss, loss_batches

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        valid_result = self.evaluate(valid_data)
        valid_score = valid_result[self.valid_metric] if self.valid_metric else valid_result['NDCG@20']
        return valid_score, valid_result

    def _check_nan(self, loss):
        if torch.isnan(loss):
            #raise ValueError('Training loss is nan')
            return True

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            test_data (DataLoader, optional): None
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                # get nan loss
                break
            #for param_group in self.optimizer.param_groups:
            #    print('======lr: ', param_group['lr'])
            self.lr_scheduler.step()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            # eval: To ensure the test result is the best model under validation data, set self.eval_step == 1
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                # test
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                    #self.model.save_best()

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break
        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid


    @torch.no_grad()
    def evaluate(self, eval_data, is_test=False, idx=0):
        r"""Evaluate the model based on the eval data.
        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        self.model.eval()

        # batch full users
        batch_matrix_list = []
        embedding = self.model.pre_epoch_processing
        for batch_idx, batched_data in enumerate(eval_data):
            # predict: interaction without item ids
            scores = self.model.full_sort_predict(batched_data)
            masked_items = batched_data[1]
            # mask out pos items
            scores[masked_items[0], masked_items[1]] = -1e10
            # rank and get top-k
            _, topk_index = torch.topk(scores, max(self.config['topk']), dim=-1)  # nusers x topk
            batch_matrix_list.append(topk_index)
        return self.evaluator.evaluate(batch_matrix_list, eval_data, is_test=is_test, idx=idx)

    def plot_train_loss(self, show=True, save_path=None):
        r"""Plot the train loss in each epoch

        Args:
            show (bool, optional): whether to show this figure, default: True
            save_path (str, optional): the data path to save the figure, default: None.
                                       If it's None, it will not be saved.
        """
        epochs = list(self.train_loss_dict.keys())
        epochs.sort()
        values = [float(self.train_loss_dict[epoch]) for epoch in epochs]
        plt.plot(epochs, values)
        plt.xticks(epochs)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path)


class DanceTrainer(Trainer):
    r"""Trainer for the DANCE model with 3-phase training:
    Phase 1: Pre-train behavior simulator on factual graph
    Phase 2: Generate counterfactual graph using simulator embeddings
    Phase 3: Train final model with balanced (factual + counterfactual) graph
    """

    def __init__(self, config, model):
        super(DanceTrainer, self).__init__(config, model)
        self.pretrain_epochs = config['pretrain_epochs'] or 50
        self.verbose = config['verbose'] or 10

    def _generate_counterfactual_graph(self, batch_size=512):
        """
        Phase 2: Use pre-trained simulator to predict counterfactual scores
        for ALL user-item pairs.
        """
        self.logger.info('=' * 60)
        self.logger.info('PHASE 2: GENERATING COUNTERFACTUAL GRAPH')
        self.logger.info('=' * 60)

        self.model.eval()
        h_user, h_item = self.model.get_embeddings()

        n_users = self.model.n_users
        n_items = self.model.n_items

        R_counterfactual = torch.zeros(n_users, n_items, device=self.device)

        for u_start in range(0, n_users, batch_size):
            u_end = min(u_start + batch_size, n_users)
            batch_users = h_user[u_start:u_end]
            scores = torch.mm(batch_users, h_item.t())
            scores = torch.sigmoid(scores)
            R_counterfactual[u_start:u_end] = scores

        self.logger.info(
            'Counterfactual graph R* generated: %s, mean=%.4f' %
            (str(R_counterfactual.shape), R_counterfactual.mean().item())
        )
        return R_counterfactual

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        r"""DANCE 3-phase training.

        Phase 1: Pre-train on factual graph (use_counterfactual=False)
        Phase 2: Generate counterfactual graph
        Phase 3: Train with balanced graph (use_counterfactual=True)
        """
        # ============================================================
        # PHASE 1: PRE-TRAIN BEHAVIOR SIMULATOR
        # ============================================================
        self.logger.info('=' * 60)
        self.logger.info('PHASE 1: PRE-TRAINING BEHAVIOR SIMULATOR (%d epochs)' % self.pretrain_epochs)
        self.logger.info('=' * 60)

        self.model.disable_counterfactual()
        best_pretrain_score = -1

        for epoch_idx in range(self.pretrain_epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()
            training_end_time = time()

            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)

            if verbose:
                self.logger.info('[Simulator] ' + train_loss_output)

            if (epoch_idx + 1) % self.verbose == 0 and valid_data is not None:
                valid_score, valid_result = self._valid_epoch(valid_data)
                if verbose:
                    self.logger.info('[Simulator] epoch %d valid: %s' % (epoch_idx, dict2str(valid_result)))
                if valid_score > best_pretrain_score:
                    best_pretrain_score = valid_score
                    best_simulator_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if verbose:
                        self.logger.info('[Simulator] Best simulator updated (score=%.4f)' % valid_score)

        # Restore best simulator weights
        if best_pretrain_score > 0:
            self.model.load_state_dict(best_simulator_state)
            self.logger.info('Loaded best simulator (score=%.4f)' % best_pretrain_score)

        # ============================================================
        # PHASE 2: GENERATE COUNTERFACTUAL GRAPH
        # ============================================================
        R_counterfactual = self._generate_counterfactual_graph()

        # ============================================================
        # PHASE 3: BUILD BALANCED GRAPH AND TRAIN FINAL MODEL
        # ============================================================
        self.logger.info('=' * 60)
        self.logger.info('PHASE 3: TRAINING WITH BALANCED GRAPH')
        self.logger.info('=' * 60)

        self.model.set_balanced_graph(R_counterfactual)

        # Reset optimizer and scheduler for phase 3
        self.optimizer = self._build_optimizer()
        lr_scheduler = self.config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        # Reset early stopping state
        self.cur_step = 0
        self.best_valid_score = -1
        tmp_dd = {}
        for j, k in list(itertools.product(self.config['metrics'], self.config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd

        # Phase 3 training loop (reuse parent's logic)
        for epoch_idx in range(self.start_epoch, self.epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()

            self.train_loss_dict[self.pretrain_epochs + epoch_idx] = \
                sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    update_output = '██ DANCE--Best validation results updated!!!'
                    if verbose:
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result

                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid



class PrismTrainer(Trainer):
    r"""Trainer for PRISM: 3-phase training with calibrated counterfactual and progressive refinement."""

    def __init__(self, config, model):
        super(PrismTrainer, self).__init__(config, model)
        self.pretrain_epochs = config['pretrain_epochs'] or 50
        self.verbose = config['verbose'] or 10
        self.refine_step = config['refine_step'] or 50

    def _generate_calibrated_counterfactual(self, label='PHASE 2'):
        self.logger.info('=' * 60)
        self.logger.info('%s: GENERATING CALIBRATED COUNTERFACTUAL GRAPH' % label)
        self.logger.info('=' * 60)
        self.model.eval()
        h_user, h_item = self.model.get_embeddings()
        R_counterfactual = self.model.build_calibrated_counterfactual(h_user, h_item)
        mean_val = R_counterfactual.mean().item()
        std_val = R_counterfactual.std().item()
        self.logger.info('Calibrated R* generated: shape=%s, mean=%.4f, std=%.4f' %
            (str(R_counterfactual.shape), mean_val, std_val))
        return R_counterfactual

    def fit(self, train_data, valid_data=None, test_data=None, saved=False, verbose=True):
        # PHASE 1: PRE-TRAIN BEHAVIOR SIMULATOR
        self.logger.info('=' * 60)
        self.logger.info('PHASE 1: PRE-TRAINING BEHAVIOR SIMULATOR (%d epochs)' % self.pretrain_epochs)
        self.logger.info('=' * 60)
        self.model.disable_counterfactual()
        best_pretrain_score = -1

        for epoch_idx in range(self.pretrain_epochs):
            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()
            training_end_time = time()
            self.train_loss_dict[epoch_idx] = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            if verbose:
                self.logger.info('[Simulator] ' + train_loss_output)
            if (epoch_idx + 1) % self.verbose == 0 and valid_data is not None:
                valid_score, valid_result = self._valid_epoch(valid_data)
                if verbose:
                    self.logger.info('[Simulator] epoch %d valid: %s' % (epoch_idx, dict2str(valid_result)))
                if valid_score > best_pretrain_score:
                    best_pretrain_score = valid_score
                    best_simulator_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if verbose:
                        self.logger.info('[Simulator] Best simulator updated (score=%.4f)' % valid_score)

        if best_pretrain_score > 0:
            self.model.load_state_dict(best_simulator_state)
            self.logger.info('Loaded best simulator (score=%.4f)' % best_pretrain_score)

        # PHASE 2: GENERATE CALIBRATED COUNTERFACTUAL GRAPH
        R_counterfactual = self._generate_calibrated_counterfactual()

        # PHASE 3: TRAIN WITH BALANCED GRAPH + PROGRESSIVE REFINEMENT
        self.logger.info('=' * 60)
        self.logger.info('PHASE 3: TRAINING WITH BALANCED GRAPH + PROGRESSIVE REFINEMENT')
        self.logger.info('=' * 60)
        self.model.set_balanced_graph(R_counterfactual)

        self.optimizer = self._build_optimizer()
        lr_scheduler = self.config['learning_rate_scheduler']
        fac = lambda epoch: lr_scheduler[0] ** (epoch / lr_scheduler[1])
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=fac)

        self.cur_step = 0
        self.best_valid_score = -1
        tmp_dd = {}
        for j, k in list(itertools.product(self.config['metrics'], self.config['topk'])):
            tmp_dd[f'{j.lower()}@{k}'] = 0.0
        self.best_valid_result = tmp_dd
        self.best_test_upon_valid = tmp_dd

        for epoch_idx in range(self.start_epoch, self.epochs):

            training_start_time = time()
            self.model.pre_epoch_processing()
            train_loss, _ = self._train_epoch(train_data, epoch_idx)
            if torch.is_tensor(train_loss):
                break
            self.lr_scheduler.step()
            self.train_loss_dict[self.pretrain_epochs + epoch_idx] =                 sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss)
            post_info = self.model.post_epoch_processing()
            if verbose:
                self.logger.info(train_loss_output)
                if post_info is not None:
                    self.logger.info(post_info)

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                valid_score, valid_result = self._valid_epoch(valid_data)
                self.best_valid_score, self.cur_step, stop_flag, update_flag = early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger)
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" %                                      (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid result: \n' + dict2str(valid_result)
                _, test_result = self._valid_epoch(test_data)
                if verbose:
                    self.logger.info(valid_score_output)
                    self.logger.info(valid_result_output)
                    self.logger.info('test result: \n' + dict2str(test_result))
                if update_flag:
                    if verbose:
                        self.logger.info('PRISM--Best validation results updated!!!')
                    self.best_valid_result = valid_result
                    self.best_test_upon_valid = test_result
                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' %                                   (epoch_idx - self.cur_step * self.eval_step)
                    if verbose:
                        self.logger.info(stop_output)
                    break

        return self.best_valid_score, self.best_valid_result, self.best_test_upon_valid
