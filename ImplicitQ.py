@tf.function
def train_step(batch):
    s, a, r, s_next, done = batch
    batch_size = tf.shape(s)[0]  # Dynamically get batch size

    # Combine current and next states for a single forward pass
    s_combined = tf.concat([s, s_next], axis=0)
    with tf.GradientTape() as tape:
        # Single forward pass
        q_combined, v_combined, pi_combined = model(s_combined, training=True)

        # Split outputs
        q, v, pi = q_combined[:batch_size], v_combined[:batch_size], pi_combined[:batch_size]
        v_next = v_combined[batch_size:]

        # Apply game-specific adjustment and stop gradients
        v_next = v_next * -1  # Reflects opponent's perspective
        v_next = tf.stop_gradient(v_next)  # Prevent gradients through v_next

        # Q-loss
        q_a = tf.gather(q, a, axis=1, batch_dims=1)  # Select Q(s, a) for action
        y = tf.where(done == 1, r, r + gamma * v_next)  # TD target
        L_Q = tf.reduce_mean(tf.square(q_a - y))

        # V-loss (expectile regression)
        adv = q_a - v
        L_V = tf.reduce_mean(tf.where(adv > 0, tau * adv**2, (1 - tau) * adv**2))

        # Policy loss (advantage-weighted regression)
        w = tf.exp(tf.clip_by_value(adv / beta, -5, 2))
        log_pi_a = tf.gather(tf.nn.log_softmax(pi, axis=-1), a, axis=1, batch_dims=1)
        L_pi = -tf.reduce_mean(w * log_pi_a)

        # Conservative Q-Learning penalty
        logsumexp_q = tf.reduce_logsumexp(q, axis=1)
        L_CQL = tf.reduce_mean(logsumexp_q - q_a)

        # Total loss
        loss = L_Q + L_V + L_pi + 0.05 * L_CQL

    # Compute and apply gradients
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 5.0)  # Clip total gradient norm to 5.0
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss
