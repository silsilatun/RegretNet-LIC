import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import logging
import json
import os
from data_generator import (
    generate_item_values_uniform_01,
    generate_item_values_1x2_asymmetric,
)


def generate_model(
    num_bidders,
    num_items,
    num_alloc_cells,
    num_pay_cells,
    num_hidden_layers,
    utility_type,
):
    # input layer
    input_layer = keras.layers.Input(
        shape=(
            num_bidders,
            num_items,
        )
    )
    flatten_layer = keras.layers.Flatten()(input_layer)

    # allocation network
    alloc_hidden_layer = keras.layers.Dense(
        num_alloc_cells, activation="tanh", kernel_initializer="glorot_uniform"
    )(flatten_layer)
    for _ in range(num_hidden_layers - 2):
        alloc_hidden_layer = keras.layers.Dense(
            num_alloc_cells, activation="tanh", kernel_initializer="glorot_uniform"
        )(alloc_hidden_layer)

    if utility_type == "additive":
        alloc_output = keras.layers.Dense(
            (num_bidders + 1) * (num_items),
            activation="linear",
            kernel_initializer="glorot_uniform",
        )(
            alloc_hidden_layer
        )  # num_bidders + 1 : bidders + nobody gets the item

        alloc_output_reshape = keras.layers.Reshape((num_bidders + 1, num_items))(
            alloc_output
        )
        alloc_output_1 = keras.layers.Lambda(
            lambda s: tf.nn.softmax(
                s,
                axis=1,
            ),  # prevent the same item from being allocated to multiple bidders
            output_shape=(num_bidders + 1, num_items),
        )(alloc_output_reshape)
        alloc_output_2 = keras.layers.Lambda(
            lambda t: tf.slice(
                t, [0, 0, 0], [-1, num_bidders, num_items]
            ),  # drop the last column (nobody gets the item)
            output_shape=(num_bidders, num_items),
        )(alloc_output_1)
    elif utility_type == "unit-demand":
        alloc_output = keras.layers.Dense(
            num_bidders * num_items,
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(alloc_hidden_layer)
        alloc_output = keras.layers.Dense(
            (num_bidders + 1) * (num_items + 1),
            activation="linear",
            kernel_initializer="glorot_uniform",
        )(alloc_hidden_layer)
        alloc_output_reshape = keras.layers.Reshape((num_bidders + 1, num_items + 1))(
            alloc_output
        )
        alloc_output_item = keras.layers.Lambda(
            lambda s: tf.nn.softmax(
                s,
                axis=1,
            ),  # prevent the same item from being allocated to multiple bidders
            output_shape=(num_bidders + 1, num_items + 1),
        )(alloc_output_reshape)
        alloc_output_bidder = keras.layers.Lambda(
            lambda s: tf.nn.softmax(
                s,
                axis=2,
            ),  # unit-demand constraint
            output_shape=(num_bidders + 1, num_items + 1),
        )(alloc_output_reshape)
        alloc_output_1 = keras.layers.Lambda(
            lambda t: tf.math.reduce_min(t, axis=0),
            output_shape=(num_bidders + 1, num_items + 1),
        )([alloc_output_item, alloc_output_bidder])

        alloc_output_2 = keras.layers.Lambda(
            lambda t: tf.slice(
                t, [0, 0, 0], [-1, num_bidders, num_items]
            ),  # drop the last column (nobody gets the item)
            output_shape=(num_bidders, num_items),
        )(alloc_output_1)

    # payment network
    pay_hidden_layer = keras.layers.Dense(
        num_pay_cells, activation="tanh", kernel_initializer="glorot_uniform"
    )(flatten_layer)
    for _ in range(num_hidden_layers - 2):
        pay_hidden_layer = keras.layers.Dense(
            num_pay_cells, activation="tanh", kernel_initializer="glorot_uniform"
        )(pay_hidden_layer)
    pay_output = keras.layers.Dense(
        num_bidders, activation="sigmoid", kernel_initializer="glorot_uniform"
    )(pay_hidden_layer)

    # caluculate payment
    alloc_bid = keras.layers.Multiply()([alloc_output_2, input_layer])  # allocation*bid
    alloc_bid_sum = keras.layers.Lambda(
        lambda s: tf.reduce_sum(s, axis=-1), output_shape=(num_bidders,)
    )(alloc_bid)
    pay = keras.layers.Multiply()([alloc_bid_sum, pay_output])

    model = keras.Model(inputs=input_layer, outputs=[alloc_output_2, pay])
    model.summary()

    return model


class Trainer(object):
    def __init__(self, model, num_bidders, num_items):
        self.model = model
        self.num_bidders = num_bidders
        self.num_items = num_items

    # calculate utility
    @tf.function
    def _compute_util(self, allocation, payment, val):
        utils = tf.reduce_sum(tf.multiply(allocation, val), axis=-1) - payment
        return tf.reduce_mean(tf.reduce_sum(utils, axis=-1))

    # calculate revenue
    @tf.function
    def _compute_rev(self, payment):
        return tf.reduce_mean(tf.reduce_sum(payment, axis=-1))

    # calculate regret
    # get misreport profiles
    @tf.function
    def _get_misreports(self, val, mis):
        true_val = tf.tile(val, [self.num_bidders, 1, 1])
        true_val_r = tf.reshape(
            true_val,
            [self.num_bidders, self.batch_size, self.num_bidders, self.num_items],
        )
        mis_r = tf.tile(tf.expand_dims(mis, 0), [self.num_bidders, 1, 1, 1])
        misrep = tf.reshape(
            true_val_r * (1 - self.adv_mask) + mis_r * self.adv_mask,
            [-1, self.num_bidders, self.num_items],
        )
        return true_val, misrep

    # calculate utility from misreporting
    @tf.function
    def _compute_mis_util(self, val, mis):
        true_val, misrep = self._get_misreports(val, mis)
        m_alloc, m_pay = self.model(misrep)
        m_util = tf.reduce_sum(tf.multiply(m_alloc, true_val), axis=-1) - m_pay
        m_util_r = (
            tf.reshape(m_util, [self.num_bidders, self.batch_size, self.num_bidders])
            * self.u_mask
        )
        return m_util_r

    # calculate regret
    @tf.function
    def _compute_reg(self, val, mis):
        true_val, misrep = self._get_misreports(val, mis)

        m_alloc, m_pay = self.model(misrep)
        m_util = tf.reduce_sum(tf.multiply(m_alloc, true_val), axis=-1) - m_pay
        m_util_r = (
            tf.reshape(m_util, [self.num_bidders, self.batch_size, self.num_bidders])
            * self.u_mask
        )

        t_alloc, t_pay = self.model(true_val)
        t_util = tf.reduce_sum(tf.multiply(t_alloc, true_val), axis=-1) - t_pay
        t_util_r = tf.reshape(
            t_util, [self.num_bidders, self.batch_size, self.num_bidders]
        )

        excess_from_utility = tf.nn.relu(
            tf.reshape(
                m_util_r - t_util_r,
                [self.num_bidders, self.batch_size, self.num_bidders],
            )
            * self.u_mask
        )
        regret = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(2,)), axis=(1))
        return regret

    # Training
    @tf.function
    def _main_train_step(self, batch, batch_misreports):
        with tf.GradientTape() as tape:
            # Compute loss
            allocation, payment = self.model(batch)
            rev = self._compute_rev(payment)  # scalar mean revenue

            regret_per_bidder = self._compute_reg(
                batch, batch_misreports
            )  # tensor [num_bidders]
            current_rgt = tf.reduce_sum(regret_per_bidder)

            # Loss matching trainer.py: -revenue + w_rgt * rgt
            # lagrange_multiplier is scalar.
            loss = -rev + self.lagrange_multiplier * current_rgt

        grads_main = tape.gradient(loss, self.model.trainable_weights)
        self.main_opt.apply_gradients(zip(grads_main, self.model.trainable_weights))
        return rev, current_rgt, regret_per_bidder, loss

    @tf.function
    def _mis_train_step(self, optimizer, batch, batch_misreports):
        with tf.GradientTape() as tape:
            tape.watch(batch_misreports)
            mis_util = -tf.reduce_sum(self._compute_mis_util(batch, batch_misreports))
        grads_mis = tape.gradient(mis_util, batch_misreports)
        optimizer.apply_gradients(zip([grads_mis], [batch_misreports]))

    def _train_batch(self, batch, batch_misreports, batch_id):
        # reset misreport optimizer
        for var in self.mis_opt.variables:
            var.assign(tf.zeros_like(var))
        # update misreports
        for _ in range(self.mis_iter):
            self._mis_train_step(self.mis_opt, batch, batch_misreports)
            if self.val_model == "uniform_01":
                batch_misreports.assign(tf.clip_by_value(batch_misreports, 0.0, 1.0))

        if self.misreport_data == "reuse":
            self.misreports[batch_id] = batch_misreports

        rev, current_rgt_for_update, regret_per_bidder, loss = self._main_train_step(
            batch, batch_misreports
        )

        rgt_val_for_log = current_rgt_for_update / (rev + 1e-8)
        log_term = tf.math.log(tf.maximum(rgt_val_for_log, 1e-9)) - tf.math.log(
            tf.maximum(self.rgt_target, 1e-9)
        )

        new_lagr_val = self.lagrange_multiplier + self.rgt_lagrangian_lr * log_term
        self.lagrange_multiplier.assign(tf.nn.relu(new_lagr_val))

        self.rgt_target.assign(
            tf.maximum(self.rgt_target * self.rgt_target_mult, self.rgt_target_end_val)
        )

        return rev, current_rgt_for_update, loss

    def train(
        self,
        model_name,
        sample_size,
        batch_size,
        val_model,
        main_learning_rate,
        mis_learning_rate,
        rgt_lagrangian_lr,
        max_epoch,
        mis_iter,
        lagrange_multiplier_init,
        print_metrics_iter,
        dataset,
        init_misreports,
        misreport_data,
        rgt_target_start,
        rgt_target_end,
    ):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        log_dir = "logs/" + model_name
        train_summary_writer = tf.summary.create_file_writer(log_dir)
        train_rev = tf.keras.metrics.Mean("train_rev", dtype=tf.float32)
        train_reg = tf.keras.metrics.Mean("train_reg", dtype=tf.float32)
        train_lag = tf.keras.metrics.Mean("train_lag", dtype=tf.float32)

        start = time.time()
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.num_batch = int(self.sample_size / self.batch_size)
        self.mis_learning_rate = mis_learning_rate
        self.mis_iter = mis_iter
        self.val_model = val_model
        self.dataset = [
            tf.Variable(batch)
            for batch in tf.unstack(
                tf.split(dataset, num_or_size_splits=self.num_batch, axis=0)
            )
        ]
        self.misreports = [
            tf.Variable(mis)
            for mis in tf.unstack(
                tf.split(init_misreports, num_or_size_splits=self.num_batch, axis=0)
            )
        ]
        self.lagrange_multiplier = tf.Variable(
            lagrange_multiplier_init, dtype=tf.float32
        )
        self.rgt_lagrangian_lr = tf.constant(rgt_lagrangian_lr, dtype=tf.float32)

        self.misreport_data = misreport_data

        self.rgt_target_start_val = tf.constant(rgt_target_start, dtype=tf.float32)
        self.rgt_target_end_val = tf.constant(rgt_target_end, dtype=tf.float32)
        self.rgt_target = tf.Variable(self.rgt_target_start_val, dtype=tf.float32)

        total_iterations = max_epoch * self.num_batch
        self.rgt_target_mult = (
            self.rgt_target_end_val / self.rgt_target_start_val
        ) ** (1.5 / total_iterations)
        logger.info("total_iterations:" + str(total_iterations))
        logger.info("rgt_target_mult:" + str(self.rgt_target_mult.numpy()))

        self.mis_opt = tf.optimizers.Adam(learning_rate=self.mis_learning_rate)
        self.main_opt = tf.optimizers.Adam(learning_rate=main_learning_rate)

        self.adv_mask = np.zeros(
            [self.num_bidders, self.batch_size, self.num_bidders, self.num_items]
        )
        self.adv_mask[
            np.arange(self.num_bidders), :, np.arange(self.num_bidders), :
        ] = 1.0
        self.u_mask = np.zeros([self.num_bidders, self.batch_size, self.num_bidders])
        self.u_mask[np.arange(self.num_bidders), :, np.arange(self.num_bidders)] = 1.0
        self.adv_mask = tf.constant(self.adv_mask, dtype=tf.float32)
        self.u_mask = tf.constant(self.u_mask, dtype=tf.float32)

        self.iter = 0

        for i in range(max_epoch):
            perm = np.random.permutation(self.num_batch)

            for j in perm:
                # Train
                rev_val, rgt_val, loss_val = self._train_batch(
                    self.dataset[j], self.misreports[j], j
                )
                train_rev(rev_val)
                train_reg(rgt_val)
                train_lag(self.lagrange_multiplier)
                if self.iter % print_metrics_iter == 0:
                    logger.info("iter:" + str(self.iter))
                    logger.info("time:" + str(time.time() - start))
                    logger.info(
                        "rev:"
                        + str(train_rev.result())
                        + " rgt:"
                        + str(train_reg.result())
                        + " lag_mult:"
                        + str(train_lag.result())
                        + " rgt_target:"
                        + str(self.rgt_target.numpy())
                    )
                self.iter += 1

            with train_summary_writer.as_default():
                tf.summary.scalar("revenue", train_rev.result(), step=i + 1)
                tf.summary.scalar("regret", train_reg.result(), step=i + 1)
                tf.summary.scalar("regret_multiplier", train_lag.result(), step=i + 1)
                tf.summary.scalar("time", float(time.time() - start), step=i + 1)

            train_rev.reset_state()
            train_reg.reset_state()
            train_lag.reset_state()

            logger.info("epoch" + str(i) + "end")
            logger.info("time:" + str(time.time() - start))

        time_elapsed = time.time() - start
        logger.info("time:" + str(time_elapsed))
        logger.info("Training completed.")

        return time_elapsed

    # Test

    @tf.function
    def _test_get_misreports(self, batch, batch_misreports):
        true_val = tf.tile(batch, [self.num_bidders * self.test_num_misreports, 1, 1])
        true_val_r = tf.reshape(
            true_val,
            [
                self.num_bidders,
                self.test_num_misreports,
                self.test_batch_size,
                self.num_bidders,
                self.num_items,
            ],
        )
        adv = tf.tile(
            tf.expand_dims(batch_misreports, 0), [self.num_bidders, 1, 1, 1, 1]
        )
        misrep = tf.reshape(
            true_val_r * (1 - self.test_adv_mask) + adv * self.test_adv_mask,
            [-1, self.num_bidders, self.num_items],
        )
        return true_val, misrep

    @tf.function
    def _test_compute_mis_util(self, batch, batch_misreports):
        true_val, misrep = self._test_get_misreports(batch, batch_misreports)
        m_alloc, m_pay = self.model(misrep)
        m_util = tf.reduce_sum(tf.multiply(m_alloc, true_val), axis=-1) - m_pay
        m_util_r = (
            tf.reshape(
                m_util,
                [
                    self.num_bidders,
                    self.test_num_misreports,
                    self.test_batch_size,
                    self.num_bidders,
                ],
            )
            * self.test_u_mask
        )
        return -tf.reduce_sum(m_util_r)

    @tf.function
    def _test_compute_reg(self, batch, batch_misreports):
        true_val, misrep = self._test_get_misreports(batch, batch_misreports)

        m_alloc, m_pay = self.model(misrep)
        m_util = tf.reduce_sum(tf.multiply(m_alloc, true_val), axis=-1) - m_pay
        m_util_r = (
            tf.reshape(
                m_util,
                [
                    self.num_bidders,
                    self.test_num_misreports,
                    self.test_batch_size,
                    self.num_bidders,
                ],
            )
            * self.test_u_mask
        )

        t_alloc, t_pay = self.model(true_val)
        t_util = tf.reduce_sum(tf.multiply(t_alloc, true_val), axis=-1) - t_pay
        t_util_r = tf.reshape(
            t_util,
            [
                self.num_bidders,
                self.test_num_misreports,
                self.test_batch_size,
                self.num_bidders,
            ],
        )

        excess_from_utility = tf.nn.relu(
            tf.reshape(
                m_util_r - t_util_r,
                [
                    self.num_bidders,
                    self.test_num_misreports,
                    self.test_batch_size,
                    self.num_bidders,
                ],
            )
            * self.test_u_mask
        )
        return tf.reduce_mean(
            tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(1, 3)), axis=1),
            axis=0,
        )

    @tf.function
    def _test_mis_step(self, optimizer, batch, batch_misreports):
        with tf.GradientTape() as tape:
            tape.watch(batch_misreports)
            mis_util = self._test_compute_mis_util(batch, batch_misreports)
        grads_mis = tape.gradient(mis_util, batch_misreports)
        optimizer.apply_gradients(zip([grads_mis], [batch_misreports]))

    def test(
        self,
        model_name,
        test_batch_size,
        test_sample_size,
        val_model,
        num_misreports,
        mis_learning_rate,
        test_mis_iter,
        dataset,
        init_misreports,
    ):
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        start = time.time()

        log_dir = "logs_test/" + model_name
        test_summary_writer = tf.summary.create_file_writer(log_dir)

        self.test_adv_mask = np.zeros(
            [
                self.num_bidders,
                num_misreports,
                test_batch_size,
                self.num_bidders,
                self.num_items,
            ]
        )
        self.test_adv_mask[
            np.arange(self.num_bidders), :, :, np.arange(self.num_bidders), :
        ] = 1.0
        self.test_u_mask = np.zeros(
            [self.num_bidders, num_misreports, test_batch_size, self.num_bidders]
        )
        self.test_u_mask[
            np.arange(self.num_bidders), :, :, np.arange(self.num_bidders)
        ] = 1.0
        self.test_adv_mask = tf.constant(self.test_adv_mask, dtype=tf.float32)
        self.test_u_mask = tf.constant(self.test_u_mask, dtype=tf.float32)

        self.test_num_misreports = num_misreports
        self.test_batch_size = test_batch_size

        self.val_model = val_model

        self.test_mis_opt = keras.optimizers.Adam(learning_rate=mis_learning_rate)

        self.test_num_batch = int(test_sample_size / test_batch_size)

        self.dataset = [
            tf.Variable(batch)
            for batch in tf.unstack(
                tf.split(dataset, num_or_size_splits=self.test_num_batch, axis=0)
            )
        ]
        self.misreports = [
            tf.Variable(mis)
            for mis in tf.unstack(
                tf.split(
                    init_misreports, num_or_size_splits=self.test_num_batch, axis=0
                )
            )
        ]

        total_util = 0
        total_pay = 0
        total_regret = 0
        for i in range(self.test_num_batch):
            batch = self.dataset[i]
            batch_misreports = tf.Variable(
                tf.reshape(
                    self.misreports[i],
                    [num_misreports, test_batch_size, self.num_bidders, self.num_items],
                )
            )

            # Revenue
            alloc, pay = self.model(batch)
            total_util += self._compute_util(alloc, pay, batch)
            test_rev = self._compute_rev(pay)
            total_pay += test_rev

            # Regret
            for var in self.test_mis_opt.variables:
                var.assign(tf.zeros_like(var))
            for _ in range(test_mis_iter):
                self._test_mis_step(self.test_mis_opt, batch, batch_misreports)
                if self.val_model == "uniform_01":
                    batch_misreports.assign(
                        tf.clip_by_value(batch_misreports, 0.0, 1.0)
                    )

            test_reg = self._test_compute_reg(batch, batch_misreports)
            total_regret += test_reg
            logger.info("test_iter:" + str(i + 1))

            with test_summary_writer.as_default():
                tf.summary.scalar("revenue", test_rev.numpy(), step=i + 1)
                tf.summary.scalar("regret", test_reg.numpy(), step=i + 1)
                tf.summary.scalar("time", float(time.time() - start), step=i + 1)

        time_elapsed = time.time() - start
        logger.info("time:" + str(time_elapsed))
        return (
            total_util / self.test_num_batch,
            total_pay / self.test_num_batch,
            total_regret / self.test_num_batch,
        )


def load_model(model_name):
    config_dir = "RegretNet_configs"
    config_file_path = os.path.join(config_dir, model_name + ".json")
    with open(config_file_path, "r") as file:
        config = json.load(file)
    num_bidders = config["num_bidders"]
    num_items = config["num_items"]
    num_alloc_cells = config["num_alloc_cells"]
    num_pay_cells = config["num_pay_cells"]
    num_hidden_layers = config["num_hidden_layers"]

    model_dir = "RegretNet_saved_models"
    model_file_path = os.path.join(model_dir, model_name + ".h5")
    model = generate_model(
        num_bidders, num_items, num_alloc_cells, num_pay_cells, num_hidden_layers
    )
    model.load_weights(model_file_path)
    return model


def train_model(model_name, seed=12345678):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    config_dir = "RegretNet_configs"
    config_file_path = os.path.join(config_dir, model_name + ".json")
    with open(config_file_path, "r") as file:
        config = json.load(file)

    num_bidders = config["num_bidders"]
    num_items = config["num_items"]
    num_alloc_cells = config["num_alloc_cells"]
    num_pay_cells = config["num_pay_cells"]
    num_hidden_layers = config["num_hidden_layers"]
    val_model = config["val_model"]
    utility_type = config["utility_type"]
    sample_size = config["sample_size"]
    batch_size = config["batch_size"]
    main_learning_rate = config["main_learning_rate"]
    mis_learning_rate = config["mis_learning_rate"]
    rgt_lagrangian_lr = config["rgt_lagrangian_lr"]
    max_epoch = config["max_epoch"]
    mis_iter = config["mis_iter"]
    lagrange_multiplier_init = config["lagrange_multiplier_init"]
    print_metrics_iter = config["print_metrics_iter"]
    misreport_data = config["misreport_data"]
    rgt_target_start = config["rgt_target_start"]
    rgt_target_end = config["rgt_target_end"]

    model = generate_model(
        num_bidders,
        num_items,
        num_alloc_cells,
        num_pay_cells,
        num_hidden_layers,
        utility_type,
    )
    trainer = Trainer(model, num_bidders, num_items)
    if val_model == "uniform_01":
        dataset = generate_item_values_uniform_01(sample_size, num_bidders, num_items)
        if misreport_data == "reuse":
            init_misreports = generate_item_values_uniform_01(
                sample_size, num_bidders, num_items
            )
        elif misreport_data == "LIC":
            init_misreports = dataset.copy()
        else:
            raise ValueError("Unsupported misreport data type: " + misreport_data)
    elif val_model == "1x2_asymmetric":
        dataset = generate_item_values_1x2_asymmetric(sample_size)
        if misreport_data == "reuse":
            init_misreports = generate_item_values_1x2_asymmetric(sample_size)
        elif misreport_data == "LIC":
            init_misreports = dataset.copy()
        else:
            raise ValueError("Unsupported misreport data type: " + misreport_data)
    else:
        raise ValueError("Unsupported validation model: " + val_model)

    time = trainer.train(
        model_name,
        sample_size,
        batch_size,
        val_model,
        main_learning_rate,
        mis_learning_rate,
        rgt_lagrangian_lr,
        max_epoch,
        mis_iter,
        lagrange_multiplier_init,
        print_metrics_iter,
        dataset,
        init_misreports,
        misreport_data=misreport_data,
        rgt_target_start=rgt_target_start,
        rgt_target_end=rgt_target_end,
    )

    model_dir = "RegretNet_saved_models"
    model_file_path = os.path.join(model_dir, model_name + ".h5")
    model.save(model_file_path, include_optimizer=False)

    result_dir = "RegretNet_training_times"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file_path = os.path.join(result_dir, model_name + "_training_time.json")
    with open(result_file_path, "w") as file:
        json.dump(
            {
                "training_time": time,
            },
            file,
        )
    print(f"Test results saved to {result_file_path}")


def test_model(model_name, seed=12345678):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    config_dir = "RegretNet_configs"
    config_file_path = os.path.join(config_dir, model_name + ".json")
    with open(config_file_path, "r") as file:
        config = json.load(file)
    num_bidders = config["num_bidders"]
    num_items = config["num_items"]
    num_alloc_cells = config["num_alloc_cells"]
    num_pay_cells = config["num_pay_cells"]
    num_hidden_layers = config["num_hidden_layers"]
    utility_type = config["utility_type"]
    val_model = config["val_model"]
    test_batch_size = config["test_batch_size"]
    test_sample_size = config["test_sample_size"]
    test_num_misreports = config["test_num_misreports"]
    mis_learning_rate = config["mis_learning_rate"]
    test_mis_iter = config["test_mis_iter"]

    model_dir = "RegretNet_saved_models"
    model_file_path = os.path.join(model_dir, model_name + ".h5")
    model = generate_model(
        num_bidders,
        num_items,
        num_alloc_cells,
        num_pay_cells,
        num_hidden_layers,
        utility_type,
    )
    model.load_weights(model_file_path)

    trainer = Trainer(model, num_bidders, num_items)

    if val_model == "uniform_01":
        dataset = generate_item_values_uniform_01(
            test_sample_size, num_bidders, num_items
        )
        init_misreports = generate_item_values_uniform_01(
            test_sample_size * test_num_misreports, num_bidders, num_items
        )
    elif val_model == "1x2_asymmetric":
        dataset = generate_item_values_1x2_asymmetric(test_sample_size)
        init_misreports = generate_item_values_1x2_asymmetric(
            test_sample_size * test_num_misreports
        )
    else:
        raise ValueError("Unsupported validation model: " + val_model)

    util, pay, regret = trainer.test(
        model_name,
        test_batch_size,
        test_sample_size,
        val_model,
        test_num_misreports,
        mis_learning_rate,
        test_mis_iter,
        dataset,
        init_misreports,
    )

    result_dir = "RegretNet_test_results"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file_path = os.path.join(result_dir, model_name + "_test_results.json")
    with open(result_file_path, "w") as file:
        json.dump(
            {
                "util": util.numpy().tolist(),
                "pay": pay.numpy().tolist(),
                "regret": regret.numpy().tolist(),
            },
            file,
        )
    print(f"Test results saved to {result_file_path}")

    return util, pay, regret


if __name__ == "__main__":
    np.random.seed(1234567879)
    tf.random.set_seed(1234567879)

    config_dir = "RegretNet_configs"
    l_models = [f.rstrip(".json") for f in os.listdir(config_dir)]
    for model in l_models:
        print(f"Training {model}...")
        train_model(model)
        print(f"Compreted training of {model} successfully.\n")
        print(f"Testing {model}...")
        print(test_model(model))
        print(f"Compreted test of {model} successfully.\n")
    print("All models trained and tested successfully.")
