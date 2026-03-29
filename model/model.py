import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class ImageCaptionModel(nn.Module):

    def __init__(self, decoder: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, encoded_images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        return self.decoder(encoded_images, captions)

    def compile(self, optimizer: torch.optim.Optimizer, loss: Callable, metrics: list[Callable]) -> None:
        """
        Stores optimizer and loss/metric functions on the model so that
        train_model() and test() can use them without extra arguments.
        """
        self.optimizer         = optimizer
        self.loss_function     = loss
        self.accuracy_function = metrics[0]

    def train_epoch(self, train_captions: torch.Tensor, train_image_features: torch.Tensor, padding_index: int, batch_size: int = 30) -> tuple[float, float, float]:
        """
        TODO: Runs through one epoch over all training examples.

        :param train_captions:       integer tensor [N x (WINDOW_SIZE+1)] – full
                                     caption sequences including <start> and <end>.
        :param train_image_features: float tensor   [N x 2048]
        :param padding_index:        int – token id of *PAD*; used for the mask.
        :param batch_size:           int
        :returns: (perplexity, loss, accuracy) on the training set
        """

        # Switch model to training mode
        super().train()

        # NOTE:
        # - The captions passed to the decoder should have the last token in the window removed:
        #     [<START> student working on homework <STOP>] --> [<START> student working on homework]
        #
        # - When computing loss, the decoder labels should have the first word removed:
        #     [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

        ## HINT: shuffle the training examples to make training smoother over multiple epochs.

        ## TODO: implement the training loop above
        
        num_examples = len(train_captions)
        indices = torch.randperm(num_examples)
        shuffled_captions = train_captions[indices]
        shuffled_images = train_image_features[indices]

        total_loss = 0.0
        total_seen = 0.0
        total_correct = 0.0

        for end in range(batch_size, num_examples + 1, batch_size):
            start = end - batch_size
            
            batch_captions = shuffled_captions[start:end]
            batch_images = shuffled_images[start:end]

            decoder_input = batch_captions[:, :-1]
            decoder_labels = batch_captions[:, 1:]

            mask = decoder_labels != padding_index

            probs = self(batch_images, decoder_input)

            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()

            num_predictions = mask.float().sum().item()
            total_loss += loss.item() * num_predictions
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

        avg_loss = float(total_loss / total_seen)
        avg_acc = float(total_correct / total_seen)
        avg_prp = float(np.exp(avg_loss))

        return avg_prp, avg_loss, avg_acc

    def test(self, test_captions: torch.Tensor, test_image_features: torch.Tensor, padding_index: int, batch_size: int = 30) -> tuple[float, float]:
        """
        DO NOT CHANGE

        Runs through one epoch over all testing examples.

        :param test_captions:        integer tensor [N x (WINDOW_SIZE+1)]
        :param test_image_features:  float tensor   [N x 2048]
        :param padding_index:        int
        :param batch_size:           int
        :returns: (perplexity, per-symbol-accuracy) on the test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0

        self.eval()
        with torch.no_grad():
            for index, end in enumerate(range(batch_size, len(test_captions) + 1, batch_size)):

                start = end - batch_size
                batch_image_features = test_image_features[start:end, :]
                decoder_input  = test_captions[start:end, :-1]
                decoder_labels = test_captions[start:end,  1:]

                probs = self(batch_image_features, decoder_input)
                mask  = decoder_labels != padding_index
                num_predictions = mask.float().sum()
                loss     = self.loss_function(probs, decoder_labels, mask)
                accuracy = self.accuracy_function(probs, decoder_labels, mask)

                total_loss    += loss.item() * num_predictions.item()
                total_seen    += num_predictions.item()
                total_correct += num_predictions.item() * accuracy

                avg_loss = float(total_loss / total_seen)
                avg_acc  = float(total_correct / total_seen)
                avg_prp  = np.exp(avg_loss)
                print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()
        return avg_prp, avg_acc


def accuracy_function(prbs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    """
    DO NOT CHANGE

    Computes the batch accuracy.

    :param prbs:   float tensor  [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]  (logits)
    :param labels: integer tensor [BATCH_SIZE x WINDOW_SIZE]
    :param mask:   bool tensor    [BATCH_SIZE x WINDOW_SIZE]
    :return: scalar float – accuracy between 0 and 1
    """
    predictions = torch.argmax(prbs, dim=-1)
    correct  = (predictions == labels) & mask
    accuracy = correct.float().sum() / mask.float().sum()
    return accuracy.item()


def loss_function(prbs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    DO NOT CHANGE

    Calculates the model cross-entropy loss.
    Uses reduce_sum (not reduce_mean) so that per-symbol accuracy
    can be computed correctly in the calling code.

    :param prbs:   float tensor  [batch_size x window_size x vocab_size]  (logits)
    :param labels: integer tensor [batch_size x window_size]
    :param mask:   bool tensor    [batch_size x window_size]
    :return: scalar loss tensor
    """
    B, T, V     = prbs.shape
    prbs_flat   = prbs.reshape(B * T, V)
    labels_flat = labels.reshape(B * T).long()
    mask_flat   = mask.reshape(B * T).float()

    loss_unreduced = F.cross_entropy(prbs_flat, labels_flat, reduction='none')
    loss = (loss_unreduced * mask_flat).sum() / mask_flat.sum()
    return loss
