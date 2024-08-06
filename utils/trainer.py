import tensorflow as tf



class Trainer:
    def __init__(self, model, learning_rate=1e-3, 
                 decay_steps=100000, decay_rate=0.96,
                 ema_decay=0.999, max_grad_norm=10.0):
        self.model = model
        self.ema_decay = ema_decay
        self.max_grad_norm = max_grad_norm

        self.learning_rate = tf.optimizers.schedules.ExponentialDecay(
                learning_rate, decay_steps, decay_rate)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
     
    def update_weights(self, loss, gradient_tape):
        grads = gradient_tape.gradient(loss, self.model.trainable_variables)

        global_norm = tf.linalg.global_norm(grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm, use_norm=global_norm)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    @tf.function
    def train_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
            mean_mae = tf.reduce_mean(mae)
            loss = mean_mae
        self.update_weights(loss, tape)

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def test_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)
        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae

        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return loss

    @tf.function
    def predict_on_batch(self, dataset_iter, metrics):
        inputs, targets = next(dataset_iter)
        preds = self.model(inputs, training=False)

        mae = tf.reduce_mean(tf.abs(targets - preds), axis=0)
        mean_mae = tf.reduce_mean(mae)
        loss = mean_mae
        nsamples = tf.shape(preds)[0]
        metrics.update_state(loss, mean_mae, mae, nsamples)

        return preds
