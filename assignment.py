import os
import argparse
import numpy as np
import pickle
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from types import SimpleNamespace
from typing import Optional
from model.model import ImageCaptionModel, accuracy_function, loss_function
from model.decoder import TransformerDecoder, RNNDecoder


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Perform command-line argument parsing (or parse arguments with defaults).
    To parse in an interactive context pass a list:

        parse_args(['--type', 'rnn', '--task', 'train', '--data', '../data/data.p'])
    """
    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--type',        required=True,             choices=['rnn', 'transformer'],    help='Type of model to train')
    parser.add_argument('--task',        required=True,             choices=['train', 'test', 'both'], help='Task to run')
    parser.add_argument('--data',        default='data/data.p',                                    help='File path to the assignment data file.')
    parser.add_argument('--epochs',      type=int,   default=3,                                       help='Number of epochs used in training.')
    parser.add_argument('--lr',          type=float, default=1e-3,                                    help="Model's learning rate")
    parser.add_argument('--optimizer',   type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help="Model's optimizer")
    parser.add_argument('--batch_size',  type=int,   default=100,                                     help="Model's batch size.")
    parser.add_argument('--hidden_size', type=int,   default=256,                                     help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size', type=int,   default=20,                                      help='Window size of text entries.')
    parser.add_argument('--chkpt_path', default='./checkpoints',                                      help='where the model checkpoint is saved/loaded')
    parser.add_argument('--no_save',    action='store_true',                                          help='if set, do not save model checkpoints')
    parser.add_argument('--check_valid', default=True, action='store_true',                           help='if training, also print validation after each epoch')
    parser.add_argument('--device',     type=str,
                        default='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cpu / cuda / mps)')

    if args is None:
        return parser.parse_args()       # called from command line
    return parser.parse_args(args)      # called from notebook / tests


def main(args: argparse.Namespace) -> None:

    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open(args.data, 'rb') as data_file:
        data_dict = pickle.load(data_file)

    feat_prep = lambda x: np.repeat(np.array(x).reshape(-1, 2048), 5, axis=0)

    train_captions  = torch.tensor(np.array(data_dict['train_captions']),  dtype=torch.long)
    test_captions   = torch.tensor(np.array(data_dict['test_captions']),   dtype=torch.long)
    train_img_feats = torch.tensor(feat_prep(data_dict['train_image_features']), dtype=torch.float32)
    test_img_feats  = torch.tensor(feat_prep(data_dict['test_image_features']),  dtype=torch.float32)
    word2idx        = data_dict['word2idx']

    train_captions  = train_captions.to(device)
    test_captions   = test_captions.to(device)
    train_img_feats = train_img_feats.to(device)
    test_img_feats  = test_img_feats.to(device)

    if args.task in ('train', 'both'):

        decoder_class = {
            'rnn'         : RNNDecoder,
            'transformer' : TransformerDecoder,
        }[args.type]

        decoder = decoder_class(
            vocab_size  = len(word2idx),
            hidden_size = args.hidden_size,
            window_size = args.window_size,
        )

        model = ImageCaptionModel(decoder).to(device)
        compile_model(model, args)

        train_model(
            model, train_captions, train_img_feats, word2idx['<pad>'], args,
            valid=(test_captions, test_img_feats)
        )

    if args.task in ('test', 'both'):
        if args.task != 'both':
            model = load_model(args, device)

        if not (args.task == 'both' and args.check_valid):
            test_model(model, test_captions, test_img_feats, word2idx['<pad>'], args)


def save_model(model: ImageCaptionModel, args: argparse.Namespace) -> None:
    save_dir = os.path.join(args.chkpt_path, args.type)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'model_state_dict' : model.state_dict(),
        'decoder_type'     : args.type,
        'vocab_size'       : model.decoder.vocab_size,
        'hidden_size'      : model.decoder.hidden_size,
        'window_size'      : model.decoder.window_size,
        'args'             : vars(args),
    }
    torch.save(checkpoint, os.path.join(save_dir, 'model.pt'))
    print(f"Model saved to {save_dir}/model.pt")


def load_model(args: argparse.Namespace, device: Optional[torch.device] = None) -> ImageCaptionModel:
    if device is None:
        device = torch.device(args.device if hasattr(args, 'device') else 'cpu')
    checkpoint = torch.load(os.path.join(args.chkpt_path, 'model.pt'), map_location=device)

    decoder_class = {
        'rnn'         : RNNDecoder,
        'transformer' : TransformerDecoder,
    }[checkpoint['decoder_type']]

    decoder = decoder_class(
        vocab_size  = checkpoint['vocab_size'],
        hidden_size = checkpoint['hidden_size'],
        window_size = checkpoint['window_size'],
    )
    model = ImageCaptionModel(decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    saved_args = SimpleNamespace(**checkpoint['args'])
    compile_model(model, saved_args)

    print(f"Model loaded from '{args.chkpt_path}/model.pt'")
    return model


def compile_model(model: ImageCaptionModel, args: argparse.Namespace) -> None:
    optimizer_map = {
        'adam'    : optim.Adam,
        'rmsprop' : optim.RMSprop,
        'sgd'     : optim.SGD,
    }
    optimizer = optimizer_map[args.optimizer](model.parameters(), lr=args.lr)
    model.compile(
        optimizer = optimizer,
        loss      = loss_function,
        metrics   = [accuracy_function],
    )


def plotter(history: dict, save_dir: str, model_type: str) -> None:
    epochs = range(1, len(history['train_perp']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'{model_type.upper()} Training Curves', fontsize=13)

    # Perplexity
    ax1.plot(epochs, history['train_perp'], label='Train', marker='o', markersize=4)
    if history['val_perp']:
        ax1.plot(epochs, history['val_perp'], label='Val (best ★)', marker='s', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Perplexity')
    ax1.set_title('Perplexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train', marker='o', markersize=4)
    if history['val_acc']:
        ax2.plot(epochs, history['val_acc'], label='Val', marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to {plot_path}")


def train_model(model: ImageCaptionModel, captions: torch.Tensor, img_feats: torch.Tensor, pad_idx: int, args: argparse.Namespace, valid: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> None:
    """
    Runs the full training loop for args.epochs epochs.
    Always saves the BEST model perf and training-curves plot on completion.
    """
    best_perp = float('inf')
    history = {
        'train_perp': [], 'train_loss': [], 'train_acc': [],
        'val_perp':   [], 'val_acc':    [],
    }

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        try:
            train_perp, train_loss, train_acc = model.train_epoch(
                captions, img_feats, pad_idx, batch_size=args.batch_size
            )
            history['train_perp'].append(float(train_perp))
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
        except KeyboardInterrupt:
            if epoch > 0:
                print("\nEarly stopping via keyboard interrupt.")
                break
            raise

        if args.check_valid and valid is not None:
            val_perp, val_acc = model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
            history['val_perp'].append(float(val_perp))
            history['val_acc'].append(float(val_acc))
            pbar.set_postfix(
                t_loss=float(train_loss), t_acc=float(train_acc), t_perp=float(train_perp),
                v_acc=float(val_acc), v_perp=float(val_perp)
            )
            if not args.no_save and val_perp < best_perp:
                best_perp = val_perp
                save_model(model, args)
                tqdm.write(f"  ** Best model saved (val_perp={val_perp:.3f}, val_acc={val_acc:.3f}) **")
        else:
            pbar.set_postfix(loss=float(train_loss), acc=float(train_acc), perp=float(train_perp))
            if not args.no_save and train_perp < best_perp:
                best_perp = train_perp
                save_model(model, args)
                tqdm.write(f"  ** Best model saved (train_perp={train_perp:.3f}) **")

    if not args.no_save and history['train_perp']:
        save_dir = os.path.join(args.chkpt_path, args.type)
        os.makedirs(save_dir, exist_ok=True)
        plotter(history, save_dir, args.type)


def test_model(model: ImageCaptionModel, captions: torch.Tensor, img_feats: torch.Tensor, pad_idx: int, args: argparse.Namespace) -> tuple[float, float]:
    return model.test(captions, img_feats, pad_idx, batch_size=args.batch_size)


if __name__ == '__main__':
    main(parse_args())
