import random
import torch
import numpy as np
import sys


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
    _set_foreground_color(255, 255, 255)
    for token, attention in token_attention:
        if token == "[PAD]":
            break
        else:
            if token.startswith("##"):
                _set_background_color(0, 0, int(255 * attention))
                out.write(token[2:])
            else:
                _set_background_color(0, 0, 0)
                out.write(" ")
                _set_background_color(0, 0, int(255 * attention))
                out.write(token)
    out.write("\u001b[0m\n")
    out.flush()


def _set_background_color(r, g, b):
    sys.stdout.write(f"\x1b[48;2;{r};{g};{b}m")


def _set_foreground_color(r, g, b):
    sys.stdout.write(f"\x1b[38;2;{r};{g};{b}m")


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
