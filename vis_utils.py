import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import torch
import uuid
from IPython.display import display, HTML, Javascript


def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        if len(layer_attention.shape) != 4:
            print(layer_attention.shape)
            raise ValueError(
                "The attention tensor does not have the correct number of dimensions. "
            )
        if isinstance(layer_attention, torch.Tensor):
            layer_attention = layer_attention.squeeze(0)
        else:
            layer_attention = np.squeeze(layer_attention, axis=0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)

    if isinstance(squeezed[0], torch.Tensor):
        return torch.stack(squeezed)
    return np.stack(squeezed)


def num_layers(attention):
    return len(attention)


def num_heads(attention):
    return attention[0][0].shape[0]


def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]


def plot_attention(image, titles, attention_plot, vgg_attention=False):
    temp_image = image.astype(float)
    fig = plt.figure(figsize=(30, 30))
    len_result = len(titles)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (14, 14))
        if vgg_attention:
            temp_att = np.kron(temp_att, np.ones((16, 16), dtype=int))
            temp_att = scipy.ndimage.gaussian_filter(temp_att, 10)
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(titles[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.tight_layout()
    plt.show()


def plot_encoder_attention(image, attention):
    image = image.astype(float)
    fig = plt.figure(figsize=(50, 50))
    num_attention = len(attention)
    for l in range(num_attention):
        temp_att = np.resize(attention[l], (14, 14))
        ax = fig.add_subplot(14, 14, l + 1)
        img = ax.imshow(image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())
    plt.tight_layout()
    plt.show()


def plot_decoder_text_attention(
        attention=None,
        tokens=None,
        sentence_b_start=None,
        prettify_tokens=True,
        layer=None,
        heads=None,
        encoder_attention=None,
        decoder_attention=None,
        cross_attention=None,
        encoder_tokens=None,
        decoder_tokens=None,
        include_layers=None,
        html_action='view'
):
    def _to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().numpy()
        return np.asarray(t)

    attn_data = []

    if attention is not None:
        if tokens is None:
            raise ValueError("'tokens' is required")
        if include_layers is None:
            include_layers = list(range(num_layers(attention)))
        attention = format_attention(attention, include_layers)

        if sentence_b_start is None:
            att_np = _to_numpy(attention)
            att    = att_np.reshape(1, 1, att_np.shape[-2], att_np.shape[-1])
            attn_data.append({
                'name'      : None,
                'attn'      : att.tolist(),
                'left_text' : tokens,
                'right_text': tokens,
            })
        else:
            slice_a = slice(0, sentence_b_start)
            slice_b = slice(sentence_b_start, len(tokens))
            att_np  = _to_numpy(attention)
            attn_data.append({'name': 'All',                      'attn': att_np.tolist(),                          'left_text': tokens,          'right_text': tokens})
            attn_data.append({'name': 'Sentence A -> Sentence A', 'attn': att_np[:, :, slice_a, slice_a].tolist(), 'left_text': tokens[slice_a], 'right_text': tokens[slice_a]})
            attn_data.append({'name': 'Sentence B -> Sentence B', 'attn': att_np[:, :, slice_b, slice_b].tolist(), 'left_text': tokens[slice_b], 'right_text': tokens[slice_b]})
            attn_data.append({'name': 'Sentence A -> Sentence B', 'attn': att_np[:, :, slice_a, slice_b].tolist(), 'left_text': tokens[slice_a], 'right_text': tokens[slice_b]})
            attn_data.append({'name': 'Sentence B -> Sentence A', 'attn': att_np[:, :, slice_b, slice_a].tolist(), 'left_text': tokens[slice_b], 'right_text': tokens[slice_a]})

    elif encoder_attention is not None or decoder_attention is not None or cross_attention is not None:
        if encoder_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'encoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(encoder_attention)))
            enc_att = _to_numpy(format_attention(encoder_attention, include_layers))
            attn_data.append({'name': 'Encoder', 'attn': enc_att.tolist(), 'left_text': encoder_tokens, 'right_text': encoder_tokens})
        if decoder_attention is not None:
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'decoder_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(decoder_attention)))
            dec_att = _to_numpy(format_attention(decoder_attention, include_layers))
            attn_data.append({'name': 'Decoder', 'attn': dec_att.tolist(), 'left_text': decoder_tokens, 'right_text': decoder_tokens})
        if cross_attention is not None:
            if encoder_tokens is None:
                raise ValueError("'encoder_tokens' required if 'cross_attention' is not None")
            if decoder_tokens is None:
                raise ValueError("'decoder_tokens' required if 'cross_attention' is not None")
            if include_layers is None:
                include_layers = list(range(num_layers(cross_attention)))
            crs_att = _to_numpy(format_attention(cross_attention, include_layers))
            attn_data.append({'name': 'Cross', 'attn': crs_att.tolist(), 'left_text': decoder_tokens, 'right_text': encoder_tokens})
    else:
        raise ValueError("You must specify at least one attention argument.")

    if layer is not None and layer not in include_layers:
        raise ValueError(f"Layer {layer} is not in include_layers: {include_layers}")

    vis_id = 'bertviz-%s' % (uuid.uuid4().hex)

    if len(attn_data) > 1:
        options = '\n'.join(
            f'<option value="{i}">{attn_data[i]["name"]}</option>'
            for i, d in enumerate(attn_data)
        )
        select_html = f'Attention: <select id="filter">{options}</select>'
    else:
        select_html = ""

    vis_html = f"""
        <div id="{vis_id}" style="font-family:'Helvetica Neue', Helvetica, Arial, sans-serif;">
            <span style="user-select:none">
                Layer: <select id="layer"></select>
                {select_html}
            </span>
            <div id='vis'></div>
        </div>
    """

    for d in attn_data:
        attn_seq_len_left = len(d['attn'][0][0])
        if attn_seq_len_left != len(d['left_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_left} positions, while number of tokens is "
                f"{len(d['left_text'])} for tokens: {' '.join(d['left_text'])}"
            )
        attn_seq_len_right = len(d['attn'][0][0][0])
        if attn_seq_len_right != len(d['right_text']):
            raise ValueError(
                f"Attention has {attn_seq_len_right} positions, while number of tokens is "
                f"{len(d['right_text'])} for tokens: {' '.join(d['right_text'])}"
            )
        if prettify_tokens:
            d['left_text']  = format_special_chars(d['left_text'])
            d['right_text'] = format_special_chars(d['right_text'])

    params = {
        'attention'     : attn_data,
        'default_filter': "0",
        'root_div_id'   : vis_id,
        'layer'         : layer,
        'heads'         : heads,
        'include_layers': include_layers,
    }

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    if html_action == 'view':
        display(HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>'))
        display(HTML(vis_html))
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        display(Javascript(vis_js))

    elif html_action == 'return':
        html1  = HTML('<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"></script>')
        html2  = HTML(vis_html)
        vis_js = open(os.path.join(__location__, 'head_view.js')).read().replace("PYTHON_PARAMS", json.dumps(params))
        html3  = Javascript(vis_js)
        script = '\n<script type="text/javascript">\n' + html3.data + '\n</script>\n'
        return HTML(html1.data + html2.data + script)

    else:
        raise ValueError("'html_action' parameter must be 'view' or 'return'")
