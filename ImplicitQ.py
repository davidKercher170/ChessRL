target_model = create_q_policy_model()  # create_model returns a model with the same architecture
target_model.set_weights(model.get_weights())
target_model.trainable = False

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.002,
    decay_steps=10000,
    decay_rate=0.97,
    staircase=True)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=5.0)
# optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer,
          loss={"QHead": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                "ValueHead": tf.keras.losses.Huber(reduction='sum_over_batch_size'),
                "PolicyHead": tf.keras.losses.CategoricalCrossentropy(from_logits=True)},

          loss_weights={
              'QHead': 1.0,
              'ValueHead': 1.0,
              'PolicyHead': 1.0 })

@tf.function
def update_target_network(tau=0.01):
    for main_var, target_var in zip(model.trainable_variables, target_model.trainable_variables):
        target_var.assign((1 - tau) * target_var + tau * main_var)

@tf.function
def train_step(batch):
    s, a, r, s_next, done = batch

    with tf.GradientTape() as tape:
        # Compute outputs for current state using main model
        q, v, pi = model(s, training=True)
        # Compute value for next state using target network
        v_next = target_model(s_next, training=False)[1]  # get v from target model
        v_next = v_next * -1  # Reflect opponent's perspective
        v_next = tf.stop_gradient(v_next)

        # Q-loss
        q_a = tf.gather(q, a, axis=1, batch_dims=1)
        y = tf.where(done == 1, r, r + gamma * v_next)
        L_Q = tf.reduce_mean(tf.square(q_a - y))

        # V-loss (expectile regression)
        adv = q_a - v
        L_V = tf.reduce_mean(tf.where(adv > 0, tau * adv**2, (1 - tau) * adv**2))

        # Policy loss (advantage-weighted regression)
        w = tf.exp(adv / beta)
        w = tf.minimum(w, 10.0)
        log_pi_a = tf.gather(tf.nn.log_softmax(pi, axis=-1), a, axis=1, batch_dims=1)
        L_pi = -tf.reduce_mean(w * log_pi_a)

        # Conservative Q-Learning penalty
        logsumexp_q = tf.reduce_logsumexp(q, axis=1)
        L_CQL = tf.reduce_mean(logsumexp_q - q_a)

        # Total loss
        loss = L_Q + L_V + L_pi + 0.5 * L_CQL

    # Compute and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 3.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

num_epochs = 50
for epoch in range(num_epochs):
    total_loss = 0.0
    num_batches = 0
    for batch in dataset:
        loss = train_step(batch)
        total_loss += loss.numpy()
        num_batches += 1
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
    # Soft update the target network
    update_target_network()
