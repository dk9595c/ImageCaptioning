import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerBlock, PositionalEncoding

################################################################################### :)

class RNNDecoder(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, dropout: float = 0.4) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO:# Define your layers
        self.image_proj = nn.Linear(2048, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, vocab_size)


    def forward(self, encoded_images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions:       tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits  of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]

        1. Embed both your images and captions
        2. You can use your images as the inital hidden state in the shape [1 x BATCH_SIZE x HIDDEN_SIZE]
        3. Feed the embedded captions and the initial hidden state into an RNN
        4. Feed the RNN output into a classification layer to get the LOGITS
        """

        # TODO:
        img_emb = self.image_proj(encoded_images)
        h0 = img_emb.unsqueeze(0)
        
        cap_emb = self.embedding(captions)
        cap_emb = self.dropout(cap_emb)
        
        rnn_out, _ = self.rnn(cap_emb, h0)
        
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)
        
        return logits

######################################################################################## :)

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int, window_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define your layers
        self.image_proj = nn.Linear(2048, hidden_size)
        self.pos_encoder = PositionalEncoding(vocab_size, hidden_size, window_size, dropout)
        self.transformer_block1 = TransformerBlock(hidden_size, dropout=dropout)
        self.transformer_block2 = TransformerBlock(hidden_size, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, vocab_size)


    def forward(self, encoded_images, captions):
        """
        :param encoded_images: tensor of shape [BATCH_SIZE x 2048]
        :param captions:       tensor of shape [BATCH_SIZE x WINDOW_SIZE]
        :return: batch logits  of shape [BATCH_SIZE x WINDOW_SIZE x VOCAB_SIZE]

        1. Project and reshape the image embedding
        2. Embed the input caption tokens and apply positional encoding
        3. Pass through your transformer block(s)
        4. Feed the output through your classifier to get the LOGITS
        """

        # TODO
        context = self.image_proj(encoded_images).unsqueeze(1)
        
        seq = self.pos_encoder(captions)
        
        seq = self.transformer_block1(seq, context)
        seq = self.transformer_block2(seq, context)
        
        logits = self.classifier(seq)
        
        return logits
