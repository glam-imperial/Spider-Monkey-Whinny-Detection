import tensorflow as tf


def safe_log(x):
    x_ok = tf.not_equal(x, 0.)
    f = lambda x: tf.math.log(x)
    safe_f = lambda x: tf.ones_like(x) * (-16.118095651)
    safe_x = tf.where(x_ok, x, tf.ones_like(x) * 1e-7)
    return tf.where(x_ok, f(safe_x), safe_f(x))


def get_loss(loss_argument_dict,
             model_configuration):
    global_pooling = model_configuration["global_pooling"]

    # TODO: Would be best if this were to be put in the YAML files.
    # TODO: Also, currently I bring predicted logits/probabilities for 2 classes (POS/NEG). For binary cross entropy loss, I should use only the POS class (axis=1), otherwise this positive weighting is null. Currently, the loss works as if there is no explicit positive weight.
    positive_weight = 4288 / 1375

    if global_pooling == "Prediction":
        # The prediction-pooling methods bring label prediction probabilities, so we cannot use the TF and Keras
        # builtin loss functions that accept logits and are NaN-safe.
        # We need to ensure NaN-safety in the loss function then.

        # The below is what keras implements in the binary cross entropy function for NaN safety.
        # The clip_by_value, is, of course, a bad choice: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b
        # output = tf.clip_by_value(loss_argument_dict["pred_whinny_single"], 1e-5, 1. - 1e-5)
        #
        # # Compute cross entropy from probabilities.
        # bce = loss_argument_dict["true_whinny_single"] * tf.math.log(output + 1e-5) * (4288 / 1375)
        # bce += (1 - loss_argument_dict["true_whinny_single"]) * tf.math.log(1 - output + 1e-5)
        #
        # bce = tf.reduce_sum(bce, axis=1)
        # loss = tf.reduce_mean(bce)

        # The below safe_log function is inspired from: https://stackoverflow.com/a/42497444
        # This may be fine in terms of protecting from NaNs, but I think that it stops gradient backprop in the unsafe region.
        # Thus, the main trick I use: both for NaN-safety, and normal-ish backprop, is found in model.py,
        # in the way I output prediction probabilities from the prediction-pooling function.
        # Using a couple of trials only, this is *better* for pooling-prediction methods.
        loss = loss_argument_dict["true_whinny_single"] * -safe_log(loss_argument_dict["pred_whinny_single"]) * (positive_weight) +\
               (1. - loss_argument_dict["true_whinny_single"]) * -safe_log(1. - loss_argument_dict["pred_whinny_single"])

        loss = tf.reduce_sum(loss, axis=1)
        loss = tf.reduce_mean(loss)
    else:
        # The embedding pooling methods would output logits -- I use the builtit function.
        # TODO: Worth using a custom loss here as well?
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=loss_argument_dict["true_whinny_single"],
                                                        logits=loss_argument_dict["pred_whinny_single"],
                                                        pos_weight=positive_weight)

        loss = tf.reduce_mean(loss)

    return loss


def get_loss_argument_dict(pred,
                           y_tf_placeholder_dict):
    pred_whinny_single = pred["whinny_single"]  # [batch_size, 2]
    true_whinny_single = y_tf_placeholder_dict["whinny_single"]  # [batch_size, 2]

    loss_argument_dict = dict()
    loss_argument_dict["pred_whinny_single"] = pred_whinny_single
    loss_argument_dict["true_whinny_single"] = true_whinny_single

    return loss_argument_dict


def flatten_data(data, flattened_size):
    flattened_data = tf.reshape(data[:, :],
                                (-1,))
    flattened_data = tf.reshape(flattened_data,
                                (flattened_size, 1, 1, 1))
    return flattened_data
