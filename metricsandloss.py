import tensorflow.keras.backend as K


# Tversky loss


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)

def focal_tversky_loss(y_true, y_pred, gamma=2):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


# Dice coefficient - F1-score

def dice_coef(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (2. * true_pos + smooth) /(2.*true_pos + false_neg + false_pos + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Jaccard index - IoU


def jacard_coef(y_true, y_pred, smooth=1): # TP/(FN+FP+TP)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def iou_class(y_true,y_pred,n_classes):
    EPS = 1e-12
    class_wise = K.zeros(n_classes)
    for cl in range(n_classes):
        intersection = K.sum((gt == cl)*(pr == cl))
        union = K.sum(K.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou
    return class_wise

# OTHER METRICS

def precision(y_true, y_pred, smooth=1): # TP/(TP+FP)
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos+smooth)/(true_pos+false_pos+smooth)

def recall(y_true, y_pred,smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    return (true_pos+smooth)/(true_pos+false_neg+smooth)


    




