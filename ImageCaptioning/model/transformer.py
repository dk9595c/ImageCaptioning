import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMatrix(nn.Module):

    def __init__(self, use_mask: bool = False) -> None:
        super().__init__()
        # Mask is [batch_size x window_size_queries x window_size_keys]
        self.use_mask = use_mask

    def forward(self, K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        STUDENT MUST WRITE:

        Computes attention weights given key and query matrices.

        :param K: is [batch_size x window_size_keys x embedding_size]
        :param Q: is [batch_size x window_size_queries x embedding_size]
        :return: attention matrix [batch_size x window_size_queries x window_size_keys]
        """
        window_size_queries = Q.size(1)   # window size of queries
        window_size_keys    = K.size(1)   # window size of keys
        embedding_size_keys = K.size(2)

        # TODO:
        # 1) compute attention weights using queries and key matrices (if use_mask==True, then make sure to add the attention mask before softmax)
        # 2) return the attention matrix

        # Check lecture slides for how to compute self-attention
        # Remember:
        # - Q is [batch_size x window_size_queries x embedding_size]
        # - K is [batch_size x window_size_keys x embedding_size]
        # - Mask is [batch_size x window_size_queries x window_size_keys]

        # Here, queries are matmuled with the transpose of keys to produce for every query vector, weights per key vector.
        # This can be thought of as: for every query word, how much should I pay attention to the other words in this window?
        # Those weights are then used to create linear combinations of the corresponding values for each query.
        # Those queries will become the new embeddings.

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(embedding_size_keys)

        if self.use_mask:
            mask = torch.ones(window_size_queries, window_size_keys, device=Q.device)
            mask = torch.triu(mask, diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)

        return attention_weights


class AttentionHead(nn.Module):
    def __init__(self, input_size: int, output_size: int, is_self_attention: bool) -> None:
        super().__init__()
        self.use_mask = is_self_attention

        # TODO: Initlize your weights
        self.W_k = nn.Linear(input_size, output_size, bias=False)
        self.W_v = nn.Linear(input_size, output_size, bias=False)
        self.W_q = nn.Linear(input_size, output_size, bias=False)
        self.attention_matrix = AttentionMatrix(use_mask=self.use_mask)

    def forward(self, inputs_for_keys: torch.Tensor, inputs_for_values: torch.Tensor, inputs_for_queries: torch.Tensor) -> torch.Tensor:
        """
        Runs a single attention head.

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x input_size]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x input_size]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x output_size]
        """

        # TODO:
        # 1) Project inputs into K, V, Q using the linear layers.
        # 2) Compute attention weights via AttentionMatrix.
        # 3) Return the weighted combination of V.

        K = self.W_k(inputs_for_keys)
        V = self.W_v(inputs_for_values)
        Q = self.W_q(inputs_for_queries)

        weights = self.attention_matrix(K, Q)
        output = torch.bmm(weights, V)

        return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, emb_sz: int, use_mask: bool) -> None:
        super().__init__()

        # TODO:
        # Create AttentionHeads, each with output size emb_sz // 3.
        # After concatenating their outputs (giving a tensor of size emb_sz),
        # add a final nn.Linear(emb_sz, emb_sz) layer to combine them.
        
        head_size = emb_sz // 3
        
        self.head1 = AttentionHead(emb_sz, head_size, use_mask)
        self.head2 = AttentionHead(emb_sz, head_size, use_mask)
        self.head3 = AttentionHead(emb_sz, head_size, use_mask)
        
        self.final_linear = nn.Linear(head_size * 3, emb_sz)


    def forward(self, inputs_for_keys: torch.Tensor, inputs_for_values: torch.Tensor, inputs_for_queries: torch.Tensor) -> torch.Tensor:
        """
        Runs multiheaded attention.

        Requirements:
            - 3 attention heads, each of output size emb_sz // 3
            - Concatenate the three head outputs along the last dimension
            - Pass through a final linear layer

        :param inputs_for_keys:    tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_values:  tensor of [batch_size x KEY_WINDOW_SIZE   x emb_sz]
        :param inputs_for_queries: tensor of [batch_size x QUERY_WINDOW_SIZE x emb_sz]
        :return:                   tensor of [BATCH_SIZE x QUERY_WINDOW_SIZE x emb_sz]
        """

        out1 = self.head1(inputs_for_keys, inputs_for_values, inputs_for_queries)
        out2 = self.head2(inputs_for_keys, inputs_for_values, inputs_for_queries)
        out3 = self.head3(inputs_for_keys, inputs_for_values, inputs_for_queries)
        
        concat_out = torch.cat((out1, out2, out3), dim=-1)
        output = self.final_linear(concat_out)
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, emb_sz: int, multiheaded: bool = False, dropout: float = 0.2) -> None:
        super().__init__()

        # TODO:
        # 1) Create a masked self-attention layer and an unmasked cross-attention layer
        # 2) Create your layernorm layers
        # 3) Create your feed-forward sublayer
        # 4) Add dropout layers
        
        self.masked_self_attention = MultiHeadedAttention(emb_sz, use_mask=True)
        self.norm1 = nn.LayerNorm(emb_sz)
        self.dropout1 = nn.Dropout(dropout)
        
        self.cross_attention = MultiHeadedAttention(emb_sz, use_mask=False)
        self.norm2 = nn.LayerNorm(emb_sz)
        self.dropout2 = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(emb_sz, emb_sz * 4),
            nn.ReLU(),
            nn.Linear(emb_sz * 4, emb_sz)
        )
        self.norm3 = nn.LayerNorm(emb_sz)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, context_sequence: torch.Tensor) -> torch.Tensor:
        """
        Runs one Transformer decoder block.

        :param inputs:           tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]
        :param context_sequence: tensor of shape [BATCH_SIZE x CONTEXT_SEQ_LENGTH x EMBEDDING_SIZE]
        :return:                 tensor of shape [BATCH_SIZE x INPUT_SEQ_LENGTH   x EMBEDDING_SIZE]
        """

        attn_out = self.masked_self_attention(inputs, inputs, inputs)
        x = self.norm1(inputs + self.dropout1(attn_out))
        
        cross_out = self.cross_attention(context_sequence, context_sequence, x)
        x = self.norm2(x + self.dropout2(cross_out))
        
        ffn_out = self.ffn(x)
        out = self.norm3(x + self.dropout3(ffn_out))
        
        return out


def positional_encoding(length: int, depth: int) -> torch.Tensor:
    """
    Generates a sinusoidal positional encoding matrix using the Halved approach.
    """
    pe = torch.zeros(length, depth)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    
    # Calculate the frequency divisor
    div_term = torch.exp(torch.arange(0, depth, 2).float() * -(math.log(10000.0) / depth))
    
    # Apply sin to the first half of the dimensions
    pe[:, 0:depth//2] = torch.sin(position * div_term)
    
    # Apply cos to the second half of the dimensions
    pe[:, depth//2:] = torch.cos(position * div_term)
    
    return torch.FloatTensor(pe)


class PositionalEncoding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, window_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.embed_size = embed_size

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding which precomputed and stored as a buffer (not trainable)
        # HINT: call positional_encoding(length=window_size, depth=embed_size)
        pos_enc = positional_encoding(length=window_size, depth=embed_size)
        self.register_buffer('pos_encoding', pos_enc[:window_size, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        STUDENT MUST WRITE:

        :param x: integer tensor of token ids [BATCH_SIZE x WINDOW_SIZE]
        :return:  float tensor [BATCH_SIZE x WINDOW_SIZE x EMBED_SIZE]

        Steps:
          1. Embed x with self.embedding.
          2. Scale the embeddings by sqrt(embed_size)
          3. Add self.pos_encoding, broadcasted over the batch dimension
          4. Apply dropout
        """
        ## TODO:
        emb = self.embedding(x)
        emb = emb * math.sqrt(self.embed_size)
        
        seq_len = x.size(1)
        emb = emb + self.pos_encoding[:seq_len, :]
        
        out = self.dropout(emb)
        
        return out
