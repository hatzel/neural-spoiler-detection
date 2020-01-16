import random
import torch
import numpy as np
import sys
from io import StringIO
from itertools import zip_longest


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


def print_colored(token_attention, spoiler_labels=(), predictions=[],
                  out=sys.stdout):
    """
    Prints a sequence of token, attention pairs with tokens being colorized
    according to their attention.
    """
    bold_state = False
    # Set text to white
    _set_foreground_color(255, 255, 255, out=out)
    if len(token_attention) != len(spoiler_labels):
        spoiler_labels = spoiler_labels[:len(token_attention)]
        predictions = predictions[:len(token_attention)]
    for (token, attention), label, prediction\
            in zip_longest(token_attention, spoiler_labels, predictions):
        if token == "[PAD]":
            break
        else:
            if not token.startswith("##"):
                _set_background_color(0, 0, 0, out=out)
                out.write(" ")
            if prediction and prediction > 0.5:
                out.write("\x1b[4:3m")
            elif prediction and prediction < 0.5 or prediction is None:
                out.write("\x1b[59m")
                out.write("\x1b[4:1m")
            if prediction:
                pred_color = 255 * clamp(scaled_prediction(prediction))
                out.write(f"\x1b[58;2;{int(pred_color)};0;0m")
            if label and not bold_state:
                out.write("\x1b[1m")
                bold_state = True
            elif not label and bold_state:
                bold_state = False
                out.write("\x1b[1:0m")
            if token.startswith("##"):
                _set_background_color(0, 0, int(255 * attention), out=out)
                out.write(token[2:])
            else:
                _set_background_color(0, 0, int(255 * attention), out=out)
                out.write(token)
    out.write("\u001b[0m\n")
    out.flush()


def _set_background_color(r, g, b, out=sys.stdout):
    out.write(f"\x1b[48;2;{int(r)};{int(g)};{int(b)}m")


def _set_foreground_color(r, g, b, out=sys.stdout):
    out.write(f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m")


def _reset_foreground_color(out=sys.stdout):
    out.write("\x1b[0m\n")


def scaled_prediction(prediction):
    return ((prediction / 8) + 0.5)


def scale_tensors_to_max(in_tensor):
    return in_tensor / (in_tensor.max(0).values
         if len(in_tensor) > 0
         else torch.tensor(1)
     )


def clamp(value, lower=0, upper=1):
    return max(lower, min(upper, value))


def latex_colored(token_attention, spoiler_labels=(), predictions=torch.tensor([]),
                  out=sys.stdout):
    out = ""
    if len(token_attention) != len(spoiler_labels):
        spoiler_labels = spoiler_labels[:len(token_attention)]
        predictions = predictions[:len(token_attention)]
    predicted_spoiler = (t > 0.5 for t in predictions)
    predictions = scale_tensors_to_max(predictions)
    for (token, attention), label, prediction, predicted_spoiler\
            in zip_longest(token_attention, spoiler_labels,
                           predictions, predicted_spoiler):
        if token.startswith("##"):
            token = token[2:]
        else:
            out += " "
        if label is not None and label:
            out += "\\textbf{"
        if prediction:
            prediction_color = int(100 * clamp(scaled_prediction(prediction)))
            thickness = "{2}" if predicted_spoiler else "{1}"
            if prediction_color != 0:
                out += f"\\cul[red!{prediction_color}]" + thickness + "{"
        token = token.replace("&gt;", "\\textgreater")
        token = token.replace("&lt;", "\\textless")
        out += ("\\colorbox{blue!"
                + str(int(attention * 0.5 * 100))
                + "}{\strut " + token + "}")
        if prediction and prediction_color != 0:
            out += "}"
        if label is not None and label:
            out += "}"
    return out


def spoiler_token_header(correct, length, out=sys.stdout):
    try:
        accuracy = correct / length
        _set_foreground_color(255, 255 * accuracy, 255 * accuracy, out=out)
        print(f"\nToken accuracy: {accuracy}", file=out, end="")
        _reset_foreground_color(out=out)
        out.flush()
    except ZeroDivisionError:
        print("WARNING: length zero, no token accuracy to calculate")


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
