import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
from training import make_train_state, update_train_state, compute_accuracy
from dataset import generate_batches


class NewsClassifierTrainer:
    def __init__(self, classifier, dataset, loss_func, optimizer, scheduler, args):
        """
        Args:
            model (torch.nn.Module): The text classification model.
            loss_func: The loss function for training the model.
            optimizer: The optimizer used for updating the model parameters during training.
        """
        self.classifier = classifier
        self.dataset = dataset
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args

    def train(self):
        """
        Train the text classification model.
        
        Args:
            train_loader: The data loader providing training data.
            num_epochs (int): The number of training epochs.
        """
        self.dataset.class_weights = self.dataset.class_weights.to(self.args.device)

        self.train_state = make_train_state(self.args)

        epoch_bar = tqdm(desc='training routine', total=self.args.num_epochs, position=0)

        self.dataset.set_split('train')
        train_bar = tqdm(desc='split=train', total=self.dataset.get_num_batches(self.args.batch_size), position=1, leave=True)

        self.dataset.set_split('val')
        val_bar = tqdm(desc='split=val', total=self.dataset.get_num_batches(self.args.batch_size), position=1, leave=True)
        
        try:
            for epoch_index in range(self.args.num_epochs):
                self.train_state['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on

                self.dataset.set_split('train')
                batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size, device=self.args.device)
                running_loss = 0.0
                running_acc = 0.0
                self.classifier.train()

                for batch_index, batch_dict in enumerate(batch_generator):
                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    self.optimizer.zero_grad()

                    # step 2. compute the output
                    y_pred = self.classifier(batch_dict['x_data'])

                    # step 3. compute the loss
                    loss = self.loss_func(y_pred, batch_dict['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # step 4. use loss to produce gradients
                    loss.backward()

                    # step 5. use optimizer to take gradient step
                    self.optimizer.step()
                    # -----------------------------------------
                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)

                    # update bar
                    train_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                    train_bar.update()

                self.train_state['train_loss'].append(running_loss)
                self.train_state['train_acc'].append(running_acc)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                self.dataset.set_split('val')
                batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size, device=self.args.device)
                running_loss = 0.
                running_acc = 0.
                self.classifier.eval()

                for batch_index, batch_dict in enumerate(batch_generator):

                    # compute the output
                    y_pred =  self.classifier(batch_dict['x_data'])

                    # step 3. compute the loss
                    loss = self.loss_func(y_pred, batch_dict['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)

                    # compute the accuracy
                    acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)
                    val_bar.set_postfix(loss=running_loss, acc=running_acc, epoch=epoch_index)
                    val_bar.update()

                self.train_state['val_loss'].append(running_loss)
                self.train_state['val_acc'].append(running_acc)

                self.train_state = update_train_state(args=self.args, model=self.classifier, train_state=self.train_state)

                self.scheduler.step(self.train_state['val_loss'][-1])

                if self.train_state['stop_early']:
                    break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()
        except KeyboardInterrupt:
            print("Exiting loop")

    def evaluate(self):
        self.classifier.load_state_dict(torch.load(self.train_state['model_filename']))

        self.dataset.set_split('test')
        batch_generator = generate_batches(self.dataset, batch_size=self.args.batch_size, device=self.args.device)
        running_loss = 0.
        running_acc = 0.
        self.classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):
            # compute the output
            y_pred =  self.classifier(batch_dict['x_data'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # compute the accuracy
            acc_t = compute_accuracy(y_pred, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index + 1)

        self.train_state['test_loss'] = running_loss
        self.train_state['test_acc'] = running_acc

        print("Test loss: {};".format(self.train_state['test_loss']))
        print("Test Accuracy: {}".format(self.train_state['test_acc']))
    