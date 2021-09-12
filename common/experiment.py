import os

import tensorflow as tf

from common.batch_generator import BatchGenerator
from common.run_epoch import RunEpoch
import model
import losses
from evaluation import PerformanceMonitor, CustomSaver


# TODO: The GPU stuff could be even more general.
# TODO: BatchGenerator at every epoch?
# TODO: Use tensorflow Estimator: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/3_NeuralNetworks/neural_network.py


def experiment_run(config_dict):
    tfrecords_folder = config_dict["tfrecords_folder"]
    output_folder = config_dict["output_folder"]
    gpu = config_dict["gpu"]
    are_test_labels_available = config_dict["are_test_labels_available"]
    path_list_dict = config_dict["path_list_dict"]
    train_size = config_dict["train_size"]
    devel_size = config_dict["devel_size"]
    test_size = config_dict["test_size"]
    train_batch_size = config_dict["train_batch_size"]
    devel_batch_size = config_dict["devel_batch_size"]
    test_batch_size = config_dict["test_batch_size"]
    name_to_metadata = config_dict["model_configuration"]["name_to_metadata"]
    method_string = config_dict["method_string"]
    model_configuration = config_dict["model_configuration"]
    initial_learning_rate = config_dict["initial_learning_rate"]
    number_of_epochs = config_dict["number_of_epochs"]
    input_gaussian_noise = config_dict["input_gaussian_noise"]
    val_every_n_epoch = config_dict["val_every_n_epoch"]
    patience = config_dict["patience"]
    target_to_measures = config_dict["target_to_measures"]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = repr(gpu)

    train_steps_per_epoch = train_size // train_batch_size
    if train_size % train_batch_size > 0:
        train_steps_per_epoch += 1
    devel_steps_per_epoch = devel_size // devel_batch_size
    if devel_size % devel_batch_size > 0:
        devel_steps_per_epoch += 1
    test_steps_per_epoch = test_size // test_batch_size
    if test_size % test_batch_size > 0:
        test_steps_per_epoch += 1

    y_names = list()
    x_names = list()
    support_names = list()
    for attribute_name, attribute_metadata in name_to_metadata.items():
        variable_type = attribute_metadata["variable_type"]
        if variable_type == "y":
            y_names.append(attribute_name)
        elif variable_type == "x":
            x_names.append(attribute_name)
        elif variable_type == "support":
            support_names.append(attribute_name)
        elif variable_type == "id":
            # TODO: Do something for ids.
            pass
        else:
            print(variable_type)
            raise ValueError

    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            dataset_train, \
            iterator_train, \
            next_element_train, \
            init_op_train = BatchGenerator(tf_records_folder=tfrecords_folder,
                                           is_training=True,
                                           partition="train",
                                           are_test_labels_available=are_test_labels_available,
                                           name_to_metadata=name_to_metadata,
                                           batch_size=train_batch_size,
                                           buffer_size=(train_steps_per_epoch + 1) // 4,
                                           path_list=path_list_dict["train"]).get_tf_dataset()
            dataset_devel, \
            iterator_devel, \
            next_element_devel, \
            init_op_devel = BatchGenerator(tf_records_folder=tfrecords_folder,
                                           is_training=False,
                                           partition="devel",
                                           are_test_labels_available=are_test_labels_available,
                                           name_to_metadata=name_to_metadata,
                                           batch_size=devel_batch_size,
                                           buffer_size=(devel_steps_per_epoch + 1) // 4,
                                           path_list=path_list_dict["devel"]).get_tf_dataset()
            dataset_test, \
            iterator_test, \
            next_element_test, \
            init_op_test = BatchGenerator(tf_records_folder=tfrecords_folder,
                                          is_training=False,
                                          partition="test",
                                          are_test_labels_available=are_test_labels_available,
                                          name_to_metadata=name_to_metadata,
                                          batch_size=test_batch_size,
                                          buffer_size=(test_steps_per_epoch + 1) // 4,
                                          path_list=path_list_dict["test"]).get_tf_dataset()

            y_tf_placeholder_train_dict = dict()
            x_tf_placeholder_train_dict = dict()
            support_tf_placeholder_train_dict = dict()
            id_tf_placeholder_train_dict = dict()

            for attribute_name, attribute_metadata in name_to_metadata.items():
                placeholder_shape = attribute_metadata["placeholder_shape"]
                dtype = name_to_metadata[attribute_name]["tf_dtype"]
                variable_type = attribute_metadata["variable_type"]

                if variable_type == "y":
                    y_tf_placeholder_train_dict[attribute_name] = tf.placeholder(dtype, placeholder_shape)
                elif variable_type == "x":
                    if config_dict["input_channels_aug"] > 1:
                        x_tf_placeholder_train_dict[attribute_name] = tf.placeholder(dtype, list(placeholder_shape) + [config_dict["input_channels_aug"],])
                    else:
                        x_tf_placeholder_train_dict[attribute_name] = tf.placeholder(dtype, placeholder_shape)
                elif variable_type == "support":
                    support_tf_placeholder_train_dict[attribute_name] = tf.placeholder(dtype, placeholder_shape)
                elif variable_type == "id":
                    id_tf_placeholder_train_dict[attribute_name] = tf.placeholder(dtype, placeholder_shape)
                else:
                    raise ValueError

            with tf.variable_scope("Model"):
                model_configuration_effective = {k: v for k, v in model_configuration.items()}
                # TODO: Make more general by removing hardcoding to whinny_single.
                model_configuration_effective["number_of_outputs"] = name_to_metadata["whinny_single"]["number_of_outputs"]
                model_configuration_effective["is_training"] = True

                pred_train,\
                pred_test, \
                keras_model_train,\
                keras_model_test = model.get_model(x_tf_placeholder_dict=x_tf_placeholder_train_dict,
                                                   support_tf_placeholder_dict=support_tf_placeholder_train_dict,
                                                   model_configuration=model_configuration_effective)

            loss_argument_dict = losses.get_loss_argument_dict(pred=pred_train,
                                                               y_tf_placeholder_dict=y_tf_placeholder_train_dict)

            loss = losses.get_loss(loss_argument_dict,
                                   model_configuration)

            optimizer = tf.compat.v1.train.AdamOptimizer(initial_learning_rate)

            optimizer = optimizer.minimize(loss)

            custom_saver = CustomSaver(output_folder,
                                       method_string,
                                       target_to_measures,
                                       keras_model_test)

            sess.run(tf.compat.v1.global_variables_initializer())

            current_patience = 0

            performance_monitor = PerformanceMonitor(output_folder,
                                                     method_string,
                                                     custom_saver,
                                                     target_to_measures,
                                                     are_test_labels_available,
                                                     model_configuration)
            # best_performance_dict = evaluation.initialise_best_performance_dict()

            print("Start training base model.")
            print("Fresh base model.")
            for ee, epoch in enumerate(range(number_of_epochs)):
                print("EPOCH:", epoch + 1)

                input_feed_dict = dict()

                for name, tensor in y_tf_placeholder_train_dict.items():
                    input_feed_dict[name] = tensor
                for name, tensor in x_tf_placeholder_train_dict.items():
                    input_feed_dict[name] = tensor
                for name, tensor in support_tf_placeholder_train_dict.items():
                    input_feed_dict[name] = tensor
                for name, tensor in id_tf_placeholder_train_dict.items():
                    input_feed_dict[name] = tensor

                run_epoch = RunEpoch(sess=sess,
                                     partition="train",
                                     is_training=True,
                                     steps_per_epoch=train_steps_per_epoch,
                                     are_test_labels_available=are_test_labels_available,
                                     init_op=init_op_train,
                                     next_element=next_element_train,
                                     batch_size=train_batch_size,
                                     name_to_metadata=name_to_metadata,
                                     input_gaussian_noise=input_gaussian_noise,
                                     optimizer=optimizer,
                                     loss=loss,
                                     pred=pred_train,
                                     y_names=y_names,
                                     input_feed_dict=input_feed_dict,
                                     config_dict=config_dict)

                train_items, train_subject_to_id = run_epoch.run_epoch()

                # train_measures = evaluation.get_measures(train_items,
                #                                          model_configuration)
                performance_monitor.get_measures(items=train_items,
                                                 partition="train")

                print("TRAIN:")
                # evaluation.report_measures(train_measures)
                performance_monitor.report_measures(partition="train")

                if (ee + 1) % val_every_n_epoch == 0:
                    input_feed_dict = dict()
                    for name, tensor in y_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in x_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in support_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in id_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor

                    run_epoch = RunEpoch(sess=sess,
                                         partition="devel",
                                         is_training=False,
                                         steps_per_epoch=devel_steps_per_epoch,
                                         are_test_labels_available=are_test_labels_available,
                                         init_op=init_op_devel,
                                         next_element=next_element_devel,
                                         batch_size=devel_batch_size,
                                         name_to_metadata=name_to_metadata,
                                         input_gaussian_noise=input_gaussian_noise,
                                         optimizer=None,
                                         loss=None,
                                         pred=pred_test,
                                         y_names=y_names,
                                         input_feed_dict=input_feed_dict,
                                         config_dict=config_dict)

                    devel_items, devel_subject_to_id = run_epoch.run_epoch()

                    # devel_measures = evaluation.get_measures(devel_items,
                    #                                          model_configuration)
                    performance_monitor.get_measures(items=devel_items,
                                                     partition="devel")
                    print("DEVEL:")
                    # evaluation.report_measures(devel_measures)
                    performance_monitor.report_measures(partition="devel")

                    # noticed_improvement = evaluation.monitor_improvement(best_performance_dict,
                    #                                                      devel_measures,
                    #                                                      saver_paths,
                    #                                                      saver_dict,
                    #                                                      sess)
                    noticed_improvement = performance_monitor.monitor_improvement()

                    if noticed_improvement:
                        current_patience = 0
                    else:
                        current_patience += 1
                        if current_patience > patience:
                            break

                else:
                    pass

            for target in target_to_measures.keys():
                for measure in target_to_measures[target]:
                    custom_saver.load_model(target=target,
                                            measure=measure)
                    input_feed_dict = dict()
                    for name, tensor in y_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in x_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in support_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor
                    for name, tensor in id_tf_placeholder_train_dict.items():
                        input_feed_dict[name] = tensor

                    run_epoch = RunEpoch(sess=sess,
                                         partition="test",
                                         is_training=False,
                                         steps_per_epoch=test_steps_per_epoch,
                                         are_test_labels_available=are_test_labels_available,
                                         init_op=init_op_test,
                                         next_element=next_element_test,
                                         batch_size=test_batch_size,
                                         name_to_metadata=name_to_metadata,
                                         input_gaussian_noise=input_gaussian_noise,
                                         optimizer=None,
                                         loss=None,
                                         pred=pred_test,
                                         y_names=y_names,
                                         input_feed_dict=input_feed_dict,
                                         config_dict=config_dict)

                    test_items, test_subject_to_id = run_epoch.run_epoch()

                    performance_monitor.get_test_measures(test_items=test_items,
                                                          target=target,
                                                          measure=measure)

            # evaluation.report_best_performance_measures(best_performance_dict,
            #                                             are_test_labels_available,
            #                                             test_measures_dict)
            performance_monitor.report_best_performance_measures()

            # results_summary = evaluation.get_results_summary(output_folder,
            #                                                  method_string,
            #                                                  best_performance_dict,
            #                                                  test_measures_dict,
            #                                                  test_items_dict,
            #                                                  are_test_labels_available)

            results_summary = performance_monitor.get_results_summary()

            return results_summary
