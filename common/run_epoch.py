import numpy as np

import common.specaugment as specaugment
import common.mixup as mixup

# TODO: Calculate the fancy metrics in tensorflow and store/update?
# TODO: Make moving stuff to numpy optional -- much faster.
# TODO: Move the nan in numpy or the batch generator.
# TODO: Add the preprocessing for the CNN models.


class RunEpoch:
    def __init__(self,
                 sess,
                 partition,
                 is_training,
                 steps_per_epoch,
                 are_test_labels_available,
                 init_op,
                 next_element,
                 batch_size,
                 name_to_metadata,
                 input_gaussian_noise,
                 optimizer,
                 loss,
                 pred,
                 y_names,
                 input_feed_dict,
                 config_dict):
        self.sess = sess
        self.partition = partition
        self.is_training = is_training
        self.steps_per_epoch = steps_per_epoch
        self.are_test_labels_available = are_test_labels_available
        self.init_op = init_op
        self.next_element = next_element
        self.batch_size = batch_size
        self.name_to_metadata = name_to_metadata
        self.optimizer = optimizer
        self.loss = loss
        self.pred = pred
        self.input_gaussian_noise = input_gaussian_noise
        self.y_names = y_names
        self.input_feed_dict = input_feed_dict
        self.config_dict = config_dict

    def run_epoch(self):
        # Initialize an iterator over the dataset split.
        self.sess.run(self.init_op)

        # Store variable sequence.
        stored_variables = dict()

        for y_name in self.y_names:
            stored_variables[y_name] = dict()
            stored_variables[y_name]["pred"] = list()

            if not ((self.partition == "test") and (not self.are_test_labels_available)):
                stored_variables[y_name]["true"] = list()

        stored_variables["support"] = list()

        stored_variables["loss"] = list()

        # Run epoch.
        for step in range(self.steps_per_epoch):
            if step % 20 == 0:
                print(step, self.steps_per_epoch)

            # batch_tuple = self.next_element
            batch_tuple = self.sess.run(self.next_element)

            # TODO: Get the ids here as well if needed.
            name_to_array = dict()
            for name in self.name_to_metadata.keys():
                name_to_array[name] = np.nan_to_num(batch_tuple[name])

            # Augment data.
            if self.is_training:
                # if self.config_dict["mixup"]:
                #     mu = mixup.Mixup(self.config_dict["train_batch_size"], 0.2)
                for name in name_to_array.keys():
                    attribute_metadata = self.name_to_metadata[name]
                    variable_type = attribute_metadata["variable_type"]

                    if variable_type == "x":
                        # if self.config_dict["input_channels_aug"] > 1:
                        #     array = np.zeros(list(name_to_array[name].shape) + [self.config_dict["input_channels_aug"], ], dtype=np.float32)
                        #     for c in range(self.config_dict["input_channels_aug"]):
                        #         array[:, :, :, c] = name_to_array[name]
                        #     name_to_array[name] = array

                        if (self.config_dict["specaug"]) and (name == "logmel_spectrogram"):
                            sa = specaugment.SpecAugment(name_to_array[name])
                            _ = sa.time_mask()
                            name_to_array[name] = sa.freq_mask()

                        # if self.config_dict["mixup"]:
                        #     name_to_array[name] = mu.mixup_data(name_to_array[name])

                        if self.input_gaussian_noise > 0.0:
                            jitter = np.random.normal(scale=self.input_gaussian_noise,
                                                      size=name_to_array[name].shape)
                            array_plus_jitter = name_to_array[name] + jitter
                            name_to_array[name] = array_plus_jitter
                    elif variable_type == "y":
                        pass
                        # if self.config_dict["mixup"]:
                        #     name_to_array[name] = mu.mixup_data(name_to_array[name])
                    else:
                        pass
            else:
                pass
                # for name in name_to_array.keys():
                #     attribute_metadata = self.name_to_metadata[name]
                #     variable_type = attribute_metadata["variable_type"]

                    # if variable_type == "x":
                    #     if self.config_dict["input_channels_aug"] > 1:
                    #         array = np.zeros(
                    #             list(name_to_array[name].shape) + [self.config_dict["input_channels_aug"], ],
                    #             dtype=np.float32)
                    #         for c in range(self.config_dict["input_channels_aug"]):
                    #             array[:, :, :, c] = name_to_array[name]
                    #         name_to_array[name] = array

            feed_dict = dict()
            for name in self.name_to_metadata.keys():
                if name in name_to_array.keys():
                    tensor = self.input_feed_dict[name]
                    feed_dict[tensor] = name_to_array[name]

            if self.is_training:
                out_tf = list()
                # TODO: Alter this.
                out_tf.append(self.pred["whinny_single"])
                out_tf.append(self.pred["whinny_continuous"])
                out_tf.append(self.pred["attention_weights"])
                optimizer_index = None
                loss_index = None
                if self.optimizer is not None:
                    out_tf.append(self.optimizer)
                    optimizer_index = len(out_tf) - 1
                if self.loss is not None:
                    out_tf.append(self.loss)
                    loss_index = len(out_tf) - 1

                return_np = self.sess.run(out_tf,
                                          feed_dict=feed_dict)

                out_np = return_np[0:3]
                y_np = [name_to_array["whinny_single"],]

            else:
                out_tf = list()
                # TODO: Alter this.
                out_tf.append(self.pred["whinny_single"])
                out_tf.append(self.pred["whinny_continuous"])
                out_tf.append(self.pred["attention_weights"])

                return_np = self.sess.run(out_tf,
                                          feed_dict=feed_dict)

                out_np = return_np[0:3]
                y_np = [name_to_array["whinny_single"], ]

            if self.is_training and (self.config_dict["mixup"]):
                pass
            else:
                # TODO: Fix this.
                for y_i, name in enumerate(["whinny_single", ]):
                    attribute_metadata = self.name_to_metadata[name]
                    stored_variables[name]["pred"].append(out_np[y_i])
                    if not ((self.partition == "test") and (not self.are_test_labels_available)):
                        stored_variables[name]["true"].append(y_np[y_i])

        if self.is_training and (self.config_dict["mixup"]):
            pass
        else:
            for y_i, y_name in enumerate(["whinny_single", ]):
                if not ((self.partition == "test") and (not self.are_test_labels_available)):
                    stored_variables[y_name]["true"] = np.vstack(stored_variables[y_name]["true"])
                stored_variables[y_name]["pred"] = np.vstack(stored_variables[y_name]["pred"])
                # print(stored_variables[y_name]["pred"].shape)
                # print(stored_variables[y_name]["true"].shape)
                # print(stored_variables[y_name]["pred"])
                # print(stored_variables[y_name]["true"])

        subject_to_id = None
        return stored_variables, subject_to_id
