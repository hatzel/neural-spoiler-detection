import random
import torch
import numpy as np
import sys
from io import StringIO


def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def seed_for_testing():
    seed(2)


def print_confusion_matrix(confusion):
    print(f"{'':<18}{'Predicted Spoiler':>20}{'Predicted Non-Spoiler':>28}")
    print(f"{'Actual Spoiler':>18}{confusion['tp']:>20}{confusion['fn']:>28}")
    print(f"{'Actual Non-Spoiler':<18}{confusion['fp']:>20}{confusion['tn']:>28}")


def print_colored(token_attention, out=sys.stdout):
    """
    Prints a sequence of token, attention pairs with tokens being colorized
    according to their attention.
    """
    # Set text to white
    _set_foreground_color(255, 255, 255, out=out)
    for token, attention in token_attention:
        if token == "[PAD]":
            break
        else:
            if token.startswith("##"):
                _set_background_color(0, 0, int(255 * attention), out=out)
                out.write(token[2:])
            else:
                _set_background_color(0, 0, 0, out=out)
                out.write(" ")
                _set_background_color(0, 0, int(255 * attention), out=out)
                out.write(token)
    out.write("\u001b[0m\n")
    out.flush()


def _set_background_color(r, g, b, out=sys.stdout):
    out.write(f"\x1b[48;2;{r};{g};{b}m")


def _set_foreground_color(r, g, b, out=sys.stdout):
    out.write(f"\x1b[38;2;{r};{g};{b}m")


def latex_colored(token_attention):
    out = ""
    for token, attention in token_attention:
        if token.startswith("##"):
            token = token[2:]
        else:
            out += " "
        out += ("\\colorbox{blue!"
                + str(int(attention * 0.3 * 255))
                + "}{\strut " + token + "}")
    return out


def spoiler_header(actual, predicted, boundary=0.5):
    out = StringIO("")
    if actual != (predicted.item() > boundary):
        _set_foreground_color(255, 0, 0, out=out)
    out.write(f"Predicted: {predicted}\tActual: {actual}")
    out.write("\u001b[0m\n")
    return out.getvalue()


def merge_compound_tokens(tokens):
    out = []
    out_count = []
    for token in tokens:
        if token.startswith("##") and len(out) > 0:
            out[-1] += token[2:]
            out_count[-1] += 1
        else:
            out.append(token)
            out_count.append(1)
    return out, out_count
