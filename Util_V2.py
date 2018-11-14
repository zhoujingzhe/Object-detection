import keras.backend as K
from keras.losses import categorical_crossentropy
import keras
import numpy as np
import tensorflow as tf
#hyperparameter
from settings import setting
weight_Classification_loss = setting["weight_Classification_loss"]
weight_Object_loss = setting["weight_Object_loss"]
weight_Localization_loss = setting["weight_Localization_loss"]
_batch_size = setting['batch_size']
_epoch = 0
initial_lr = 0.01
Hdecay = setting["decay"]
lr_minimum_rate = 60.0

#########################
print(weight_Classification_loss)
print(weight_Object_loss)
print(weight_Localization_loss)
print(Hdecay)
print(_batch_size)
#########################
_epsilon = K.epsilon()
_epsilon = K.cast(_epsilon, 'float32')

def generating_consequences(results):
    global _batch_size
    _batch_size = K.shape(results)[0]
    Pc = results[:, :, :, 0]
    Pc = K.reshape(x = Pc, shape=(-1, 19, 14, 1))
    x ,y ,w ,h = return_coordinates(y_pred=results)
    x = K.reshape(x=x, shape=(-1, 19, 14, 1))
    y = K.reshape(x=y, shape=(-1, 19, 14, 1))
    w = K.reshape(x=w, shape=(-1, 19, 14, 1))
    h = K.reshape(x=h, shape=(-1, 19, 14, 1))
    Boxes = K.concatenate([x, y, w, h], axis=-1)
    Class = results[:, :, :, 5:]
    Box_scores = Pc * Class
    Box_classes = K.argmax(Box_scores, axis=-1)
    Box_class_scores = K.max(Box_scores, axis=-1)
    Box_classes = K.reshape(x=Box_classes, shape=(_batch_size, -1))
    Box_class_scores = K.reshape(x=Box_class_scores, shape=(_batch_size, -1))
    Boxes = K.reshape(x=Boxes, shape=(_batch_size, -1, 4))
    TOPK = tf.nn.top_k(input=Box_class_scores, k=5)
    indices = TOPK.indices
    temp = K.zeros(K.shape(x=indices), dtype='int32')
    tmp = K.arange(0, _batch_size, 1, dtype='int32')
    tmp = K.reshape(x=tmp, shape=(_batch_size, -1))
    temp = temp + tmp
    indices = tf.stack([temp, indices], axis= 2)
    scores = tf.gather_nd(params=Box_class_scores, indices=indices)
    boxes = tf.gather_nd(params=Boxes, indices=indices)
    classes = tf.gather_nd(params=Box_classes, indices=indices)
    return boxes, classes, scores

def transform_to_coordinate(x, y, w, h):
    x1 = x - K.cast(w / 2, 'float32')
    y1 = y - K.cast(h / 2, 'float32')
    x2 = x + K.cast(w / 2, 'float32')
    y2 = y + K.cast(h / 2, 'float32')
    return [x1, y1, x2, y2]

def Checking_if_object(x1_window, y1_window, x2_window, y2_window, x_max_true, x_min_true, y_max_true, y_min_true):
    x_middle_true = (x_max_true + x_min_true)/2.0
    y_middle_true = (y_max_true + y_min_true)/2.0
    matching_x = tf.logical_and(K.greater_equal(x=x_middle_true, y=x1_window), K.greater_equal(x=x2_window, y= x_middle_true))
    matching_y = tf.logical_and(K.greater_equal(x=y_middle_true, y=y1_window), K.greater_equal(x=y2_window, y= y_middle_true))
    matching = tf.logical_and(matching_x, matching_y)
    return matching

def return_coordinates(y_pred):
    global _epsilon
    xpred = y_pred[:, :, :, 1]
    xpred = tf.clip_by_value(t=xpred, clip_value_min = 0 + _epsilon, clip_value_max = 1 - _epsilon)
    xpred = K.cast(xpred, 'float32')
    xpred = xpred * 64 + np.arange(0, 608, 32).reshape(19, 1)
    ypred = y_pred[:, :, :, 2]
    ypred = tf.clip_by_value(t=ypred, clip_value_min = 0 + _epsilon, clip_value_max = 1 - _epsilon)
    ypred = K.cast(ypred, 'float32')
    ypred = ypred * 64 + np.arange(0, 417, 32).reshape(1, 14)
    wpred = y_pred[:, :, :, 3] * 64 * 10
    wpred = K.clip(x=wpred, max_value=640, min_value=50)
    hpred = y_pred[:, :, :, 4] * 64 * 10
    hpred = K.clip(x=hpred, max_value=480, min_value=50)
    return [xpred, ypred, wpred, hpred]

def Loss_v2(y_true, y_pred):
    Pc_pred = y_pred[:, :, :, 0]
    Pc_pred = K.cast(Pc_pred, 'float32')
    global _epsilon
    Pc_pred = tf.clip_by_value(t=Pc_pred, clip_value_min = _epsilon, clip_value_max = 1 - _epsilon)
    xpred, ypred, wpred, hpred = return_coordinates(y_pred)
    x1_pred, y1_pred, x2_pred, y2_pred = transform_to_coordinate(xpred, ypred, wpred, hpred)

    C_Class_Array = y_pred[:, :, :, 5:]
    x_max_true = y_true[:, :, :, 0]
    x_min_true = y_true[:, :, :, 1]
    y_max_true = y_true[:, :, :, 2]
    y_min_true = y_true[:, :, :, 3]
    C_index_true = y_true[:, :, :, 7:]
    C_index_true = K.cast(C_index_true, dtype='float32')

    X_matrix = np.ndarray((19, 14, 2), dtype='float32')

    X_matrix[:, :, 0] = np.arange(0, 608, 32).reshape(19, 1)
    X_matrix[:, :, 1] = np.arange(0, 417, 32).reshape(1, 14)

    x1_window = X_matrix[:, :, 0]
    y1_window = X_matrix[:, :, 1]
    x2_window = X_matrix[:, :, 0] + 64
    y2_window = X_matrix[:, :, 1] + 64

    matching = Checking_if_object(x1_window, y1_window, x2_window, y2_window, x_max_true, x_min_true,
                                  y_max_true, y_min_true)
    mat = K.cast(matching, 'float32')

    C_Class_Array = tf.clip_by_value(t=C_Class_Array, clip_value_min = _epsilon, clip_value_max = 1 - _epsilon)
    Classification_loss = categorical_crossentropy(y_true= C_index_true, y_pred= C_Class_Array)

    Classification_loss = K.reshape(x=Classification_loss, shape=(-1, 19, 14, 1))

    Classification_loss  = (1 - C_Class_Array) * C_index_true * Classification_loss * weight_Classification_loss

    Localization_loss = weight_Localization_loss * mat * (K.square(x1_pred - x_min_true) + K.square(x2_pred - x_max_true) + K.square(
        y1_pred - y_min_true) + K.square(y2_pred - y_max_true))

    Object_loss = -(1 - Pc_pred) * K.log(Pc_pred) * mat - (1 - mat) * K.log(1-Pc_pred) * Pc_pred
    Object_loss = Object_loss * weight_Object_loss

    Total_loss = K.mean(axis=-1, x= K.mean(axis=-1, x=Classification_loss)) + K.mean(axis=-1, x=Localization_loss) + K.mean(axis=-1, x=Object_loss)
    Totalloss = K.mean(x=Total_loss, axis=-1)

    return Totalloss


def Loss_v3(y_true, y_pred):
    Pc_pred = y_pred[:, :, :, 0]
    Pc_pred = K.cast(Pc_pred, 'float32')

    xpred, ypred, wpred, hpred = return_coordinates(y_pred)
    x1_pred, y1_pred, x2_pred, y2_pred = transform_to_coordinate(xpred, ypred, wpred, hpred)

    C_Class_Array = y_pred[:, :, :, 5:]
    x_max_true = y_true[:, :, :, 0]
    x_min_true = y_true[:, :, :, 1]
    y_max_true = y_true[:, :, :, 2]
    y_min_true = y_true[:, :, :, 3]
    C_index_true = y_true[:, :, :, 7:]
    C_index_true = K.cast(C_index_true, dtype='float32')
    
    X_matrix = np.ndarray((19, 14, 2), dtype='float32')

    X_matrix[:, :, 0] = np.arange(0, 608, 32).reshape(19, 1)
    X_matrix[:, :, 1] = np.arange(0, 417, 32).reshape(1, 14)

    x1_window = X_matrix[:, :, 0]
    y1_window = X_matrix[:, :, 1]
    x2_window = X_matrix[:, :, 0] + 64
    y2_window = X_matrix[:, :, 1] + 64

    matching = Checking_if_object(x1_window, y1_window, x2_window, y2_window, x_max_true, x_min_true,
                                  y_max_true, y_min_true)
    mat = K.cast(matching, 'float32')
   
    global _epsilon

    C_Class_Array = tf.clip_by_value(t=C_Class_Array, clip_value_min = _epsilon, clip_value_max = 1 - _epsilon)
    
    Classification_loss = categorical_crossentropy(y_true= C_index_true, y_pred= C_Class_Array) * weight_Classification_loss

    Localization_loss = weight_Localization_loss * mat * (K.square(x1_pred - x_min_true) + K.square(x2_pred - x_max_true) + K.square(
        y1_pred - y_min_true) + K.square(y2_pred - y_max_true))

    Object_loss = weight_Object_loss * (mat * K.square(1 - Pc_pred) + (1 - mat) * K.square(Pc_pred))

    Total_loss = K.mean(axis=-1, x=Classification_loss) + K.mean(axis=-1, x=Localization_loss) + K.mean(axis=-1, x=Object_loss)
    Totalloss = K.mean(x=Total_loss, axis=-1)

    return Totalloss

class DecayByEpoch(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log=[]):
        global Hdecay
        new_lr = initial_lr * 1.0 / (1.0 + Hdecay * epoch)
        if initial_lr / new_lr > lr_minimum_rate:
            lr = self.model.optimizer.lr
        else:
            K.set_value(self.model.optimizer.lr, new_lr)
            lr = self.model.optimizer.lr
        print(K.eval(lr))

class lr_minimum(keras.callbacks.Callback):
    def on_batch_end(self, batch, log=[]):
        global Hdecay, _epoch, _batch_size
        iterations = batch + _epoch * 1000.0 / _batch_size
        print('iterations:', iterations)

        new_lr = initial_lr * 1.0 / (1.0 + Hdecay * iterations)
        print('The New_lr:', new_lr)
        if initial_lr / new_lr > lr_minimum_rate:
            K.set_value(self.model.optimizer.lr, initial_lr/lr_minimum_rate)
        else:
            K.set_value(self.model.optimizer.lr, new_lr)
    def on_epoch_end(self, epoch, log=[]):
        lr = self.model.optimizer.lr
        global _epoch
        _epoch += 1
        print('Each epoch, the lr is', K.eval(lr))