from abc import ABCMeta, abstractmethod
from graph_scorer import GraphScorer
import dgl
import torch as th
import copy
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim


class HyperGNNScorerGeneral(GraphScorer):
    """
    Implements methods used by all Hyper-GNN Scorers.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    def collate(self, samples):
        # The input `samples` is a list of pairs
        #  (graph, label, self_label).

        graphs, labels, labels_self = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        return batched_graph, th.tensor(labels), th.tensor(labels_self)

    def fit(self, train_dataloader, valid_dataloader, optimizer, loss_func_tune, loss_func_pre, n_epochs= 100, alpha=1, scheduler=None,
            train_mode=None, loss_func_node=None, pretrain_node=False, model_name=None, pretrain_he=False, joint=False):
        "Fits the deep neural network."
        def binary_accuracy(preds, y):
            """
            Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
            """
            num_correct = 0
            for i in range(preds.shape[0]):
                if th.all(th.eq(th.argmax(preds[i,:]), y[i])):
                    num_correct += 1
            acc = num_correct / preds.shape[0]
            return acc

        def train(model, iterator, optimizer, loss_func_tune, loss_func_pre, alpha, tuning=False, train_mode=train_mode,
                  pretrain_node=False, loss_func_node=loss_func_node, model_name=model_name):

            epoch_loss = 0
            epoch_acc = 0
            model.train()

            for iter, (bg, label_tune, label_self) in enumerate(iterator):
                label_tune = label_tune.float()
                label_self = label_self.float()

                optimizer.zero_grad()

                if not tuning:
                    prediction_tune, prediction_pre, prediction_node, label_pre_node = model(bg)
                    prediction_tune = prediction_tune.squeeze()
                    prediction_pre = prediction_pre.squeeze()
                    if pretrain_node:
                        if train_mode == 'pretrain':
                            loss = loss_func_node(prediction_node, label_pre_node)
                    else:
                        if train_mode == 'separate':
                            loss = loss_func_tune(prediction_tune, label_tune.long())
                        if train_mode == 'joint':
                            loss = loss_func_tune(prediction_tune, label_tune.long())
                            loss = loss + alpha * loss_func_pre(prediction_pre, label_self.long())
                        if train_mode == 'pretrain':
                            loss = loss_func_pre(prediction_pre, label_self.long())

                if tuning:
                    prediction_tune, _, _, _ = model(bg)
                    prediction_tune = prediction_tune.squeeze()
                    loss = loss_func_tune(prediction_tune, label_tune.long())

                loss.backward()
                optimizer.step()

                epoch_loss += loss.detach().item()

                acc = binary_accuracy(prediction_tune, label_tune)
                epoch_acc += acc

            return epoch_loss / len(iterator), epoch_acc / len(iterator)

        def evaluate(model, iterator, loss_func_tune, loss_func_pre, alpha, tuning=False, train_mode=train_mode,
                     pretrain_node=False, loss_func_node=loss_func_node, model_name=model_name):

            epoch_loss = 0
            epoch_acc = 0
            model.eval()

            with th.no_grad():
                for iter, (bg, label, label_self) in enumerate(iterator):
                    label = label.float()
                    label_self = label_self.float()

                    if not tuning:
                        prediction_tune, prediction_pre, prediction_node, label_pre_node = model(bg)
                        prediction_tune = prediction_tune.squeeze()
                        prediction_pre = prediction_pre.squeeze()
                        if pretrain_node:
                            if train_mode == 'pretrain':
                                loss = loss_func_node(prediction_node, label_pre_node)
                        else:
                            if train_mode == 'separate':
                                loss = loss_func_tune(prediction_tune, label.long())
                            if train_mode == 'joint':
                                loss = loss_func_tune(prediction_tune, label.long())
                                loss = loss + alpha * loss_func_pre(prediction_pre, label_self.long())
                            if train_mode == 'pretrain':
                                loss = loss_func_pre(prediction_pre, label_self.long())

                    if tuning:
                        prediction_tune, _, _, _ = model(bg)
                        prediction_tune = prediction_tune.squeeze()
                        loss = loss_func_tune(prediction_tune, label.long())

                    epoch_loss += loss.detach().item()

                    acc = binary_accuracy(prediction_tune, label)
                    epoch_acc += acc

            return epoch_loss / len(iterator), epoch_acc / len(iterator)

        training_loss_history = []
        validation_loss_history = []
        min_validation_loss = th.Tensor([float("Inf")])

        if pretrain_node:
            print("Begin node-level pretraining...")
            for epoch in range(n_epochs):
                train_loss, train_acc = train(self._model, train_dataloader, optimizer, loss_func_tune, loss_func_pre,
                                              alpha, pretrain_node=True)
                valid_loss, valid_acc = evaluate(self._model, valid_dataloader, loss_func_tune, loss_func_pre, alpha,
                                                 pretrain_node=True)

                training_loss_history.append(train_loss)
                validation_loss_history.append(valid_loss)

                if scheduler is not None:
                    scheduler.step(valid_loss)

                print(
                    f'| Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc * 100:.2f}')
                if valid_loss < min_validation_loss:
                    min_validation_loss = valid_loss
                    best_model = copy.deepcopy(self._model)
                    print("Updated model due to improved validation loss.")
            self._model = best_model
            th.save(self._model, './pre_trained.model')

        if pretrain_he or joint or train_mode == 'separate':
            print("Begin hyperedge-level pretraining/joint training/separate training...")
            training_loss_history = []
            validation_loss_history = []
            min_validation_loss = th.Tensor([float("Inf")])
            for epoch in range(n_epochs):
                optimizer2 = optim.Adam(self._model.parameters(), lr=0.001)
                scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=3, factor=0.2, verbose=True,
                                                                  min_lr=10 ** -5)
                train_loss, train_acc = train(self._model, train_dataloader, optimizer2, loss_func_tune, loss_func_pre, alpha,
                                              pretrain_node=False)
                valid_loss, valid_acc = evaluate(self._model, valid_dataloader, loss_func_tune, loss_func_pre, alpha,
                                                 pretrain_node=False)

                training_loss_history.append(train_loss)
                validation_loss_history.append(valid_loss)

                scheduler2.step(valid_loss)

                print(
                    f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}')
                if valid_loss < min_validation_loss:
                    min_validation_loss = valid_loss
                    best_model = copy.deepcopy(self._model)
                    print("Updated model due to improved validation loss.")

            self._model = best_model
            th.save(self._model, './pre_trained.model')

        if train_mode == 'pretrain':
            print("Beginning tuning...")
            pretrained_model = th.load('./pre_trained.model')

            optimizer = optim.Adam(pretrained_model.parameters(), lr=0.01)
            tuning_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2,
                                                                    verbose=True, min_lr=10 ** -5)
            loss_func = th.nn.CrossEntropyLoss()
            training_loss_history = []
            validation_loss_history = []
            validation_acc_history = []
            min_validation_loss = th.Tensor([float("Inf")])
            for epoch in range(n_epochs):
                train_loss, train_acc = train(pretrained_model, train_dataloader, optimizer, loss_func_tune, loss_func_pre,
                                              alpha, tuning=True)
                valid_loss, valid_acc = evaluate(pretrained_model, valid_dataloader, loss_func_tune, loss_func_pre,
                                                 alpha, tuning=True)

                training_loss_history.append(train_loss)
                validation_loss_history.append(valid_loss)
                validation_acc_history.append(valid_acc)

                tuning_scheduler.step(valid_loss)

                print(
                    f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}')
                if valid_loss < min_validation_loss:
                    min_validation_loss = valid_loss
                    best_model = copy.deepcopy(pretrained_model)
                    print("Updated model due to improved validation loss.")

            self._model = best_model
            return True

    def predict_on_ordered_set(self, model, data_loader):
        """
        Args:
            model: DGL model that makes a prediction over a batched graph.
            data_loader: Pytorch data_loader that uses the collate function from this class.

        Returns:
            list of predictions.
        """
        test_preds = []
        model.eval()
        with th.no_grad():
            for iter, (bg, label, label_pre) in enumerate(data_loader):
                prediction, _,_,_ = model(bg)
                prediction = prediction.squeeze()
                preds = prediction.data.cpu().numpy()
                test_preds.append(preds.tolist())

        return test_preds

    def train(self, g_list_train, labels_train_tune, labels_train_pre, g_list_validation, labels_validation_tune,
              labels_validation_pre, eval_metric, alpha=1, train_mode=None, pretrain_node=False, model_name=None,
              pretrain_he=False, joint=False):
        self._alpha = alpha

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        train_dataloader = DataLoader(list(zip(g_list_train, labels_train_tune, labels_train_pre)), batch_size=self._batch_size,
                                      shuffle=True, collate_fn=self.collate)
        valid_dataloader = DataLoader(list(zip(g_list_validation, labels_validation_tune, labels_validation_pre)),
                                      batch_size=self._batch_size, shuffle=False, collate_fn=self.collate)

        print(f'The model has {count_parameters(self._model):,} trainable parameters')

        self.fit(train_dataloader, valid_dataloader, self._optimizer, self._loss_func_tune, self._loss_func_pre,
                 n_epochs=self._n_epochs, alpha=self._alpha, scheduler=self._scheduler, train_mode=train_mode,
                 pretrain_node=pretrain_node, loss_func_node=self._loss_func_node, model_name=model_name,
                 pretrain_he=pretrain_he, joint=joint)

        return True


    def predict_proba(self, g_list):
        "Predicts probability of correct classification for each graph in g_list"

        dataloader = DataLoader(list(zip(g_list, np.ones_like(list(range(len(g_list)))), np.ones_like(list(range(len(g_list)))))),
                                batch_size=self._batch_size, shuffle=False, collate_fn=self.collate)
        predicted_proba = self.predict_on_ordered_set(self._model, dataloader)
        return np.concatenate(predicted_proba, axis=0)


    def predict_labels(self, g_list):
        """
        Predicts binary label for whether the graph is a correct classification.  Relies on predict_proba.
       """

        graph_predicted_proba = self.predict_proba(g_list)
        cols = np.argmax(graph_predicted_proba, axis=1)
        rows = np.array(list(range(graph_predicted_proba.shape[0])))
        predicted_labels = np.zeros_like(graph_predicted_proba)
        for i in range(rows.shape[0]):
            predicted_labels[rows[i]][cols[i]] = 1
        return predicted_labels