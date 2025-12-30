def logit_accuracy(logits, truth, class_dim=1):
    pred = logits.argmax(dim=class_dim)
    correct = (pred == truth)
    return correct.float().mean()
