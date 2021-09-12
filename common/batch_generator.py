from pathlib import Path

import tensorflow as tf


class BatchGenerator:
    def __init__(self,
                 tf_records_folder,
                 is_training,
                 partition,
                 are_test_labels_available,
                 name_to_metadata,
                 batch_size,
                 buffer_size,
                 path_list=None,
                 use_autopad=False):
        self.tf_records_folder = tf_records_folder
        self.is_training = is_training
        self.partition = partition
        self.are_test_labels_available = are_test_labels_available
        self.name_to_metadata = name_to_metadata
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.path_list = path_list
        self.use_autopad = use_autopad

        if (self.path_list is None) or (len(self.path_list) == 0):
            root_path = Path(self.tf_records_folder)
            self.path_list = [str(x) for x in root_path.glob('*.tfrecords')]

        print("Number of files:", len(self.path_list))

    def get_tf_dataset(self):
        dataset = tf.data.TFRecordDataset(self.path_list,
                                          num_parallel_reads=8)

        features_dict = dict()
        for attribute_name, attribute_metadata in self.name_to_metadata.items():
            dtype = attribute_metadata["tfrecords_type"]
            variable_type = attribute_metadata["variable_type"]
            if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
                features_dict[attribute_name] = tf.FixedLenFeature([], dtype)

        dataset = dataset.map(lambda x: tf.parse_single_example(x,
                                                                features=features_dict
                                                                ))

        def map_func(attribute):
            for attribute_name, attribute_value in attribute.items():
                attribute_metadata = self.name_to_metadata[attribute_name]
                variable_type = attribute_metadata["variable_type"]
                shape = self.name_to_metadata[attribute_name]["shape"]
                dtype = self.name_to_metadata[attribute_name]["tf_dtype"]

                if variable_type == "id":
                    attribute[attribute_name] = tf.cast(tf.reshape(attribute[attribute_name],
                                                                   shape),
                                                        dtype)
                elif variable_type in ["x", "y", "support"]:
                    attribute[attribute_name] = tf.reshape(tf.decode_raw(attribute[attribute_name],
                                                                         dtype),
                                                           shape)
                else:
                    raise ValueError

            return attribute

        dataset = dataset.map(map_func)

        # dataset = dataset.repeat()
        if self.is_training:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)

        padded_shapes = dict()
        for attribute_name, attribute_metadata in self.name_to_metadata.items():
            if self.use_autopad:
                padded_shape = attribute_metadata["padded_shape"]
            else:
                padded_shape = attribute_metadata["numpy_shape"]
            variable_type = attribute_metadata["variable_type"]
            if not ((self.partition == "test") and (not self.are_test_labels_available) and (variable_type == "y")):
                padded_shapes[attribute_name] = padded_shape

        dataset = dataset.padded_batch(self.batch_size, padded_shapes=padded_shapes)

        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)

        next_element = iterator.get_next()

        init_op = iterator.make_initializer(dataset)

        return dataset, iterator, next_element, init_op
