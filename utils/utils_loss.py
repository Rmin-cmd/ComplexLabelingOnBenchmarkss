import torch
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loss_function(loss_func, preds, label_en, targets, temperature=1, beta=0.5):

    angs = preds.angle()
    # angs = preds.real
    label_ang = label_en.angle()
    # label_ang = label_en.real

    abs_lbl = preds.abs()
    # abs_lbl = preds.imag
    label_abs = label_en.abs()
    # label_abs = label_en.imag

    # label_oh = F.one_hot(labels.long(), num_classes=9).float()
    #
    # mse_loss = nn.MSELoss()

    distances_ang = torch.abs(angs - label_ang[:, None].to(device))
    distances_abs = torch.abs(abs_lbl - label_abs[:, None].to(device))
    # over_phase.append(len(distances[distances > 2 * torch.pi]))
    distances_ang = torch.min(distances_ang, 2 * np.pi - distances_ang)

    # Apply a negative exponential to convert distances to similarities
    similarities_ang = -distances_ang / temperature  # Control sharpness with temperature
    similarities_abs = -distances_abs / temperature  # Control sharpness with temperature

    # soft_similarity_ang = torch.softmax(similarities_ang, dim=1)
    # soft_similarity_abs = torch.softmax(similarities_abs, dim=1)

    logits = beta * similarities_ang + (1 - beta) * similarities_abs

    loss = loss_func(logits, targets.to(torch.int64))

    # loss_ang = loss_func(similarities_ang, targets.to(torch.int64))
    # loss_abs = loss_func(similarities_abs, targets.to(torch.int64))
    # loss = beta * loss_ang + (1 - beta) * loss_abs
    # loss = mse_loss(soft_similarity, label_oh)
    # loss = mse_loss(angs, label_ang[:, None].repeat(1, 9))

    probabilities = torch.softmax(logits, dim=1)

    predicted_labels = torch.argmax(probabilities, dim=1)

    # plot_on_unit_circle(preds, label_en, labels, predicted_labels)

    return loss, predicted_labels, targets
