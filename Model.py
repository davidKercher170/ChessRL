class PartialConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding_size=0, groups=1, **kwargs):
        super(PartialConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding_size = padding_size
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size, strides=strides, padding='valid', use_bias=False, groups=groups, kernel_initializer='he_normal'
        )
        self.bias = self.add_weight(shape=(filters,), initializer='zeros', trainable=True)

    def build(self, input_shape):
        padded_shape = list(input_shape)
        padded_shape[1] += 2 * self.padding_size
        padded_shape[2] += 2 * self.padding_size
        self.conv.build(padded_shape)

    def call(self, inputs, mask=None):
        paddings = [[0, 0], [self.padding_size, self.padding_size],
                    [self.padding_size, self.padding_size], [0, 0]]
        x_padded = tf.pad(inputs, paddings, mode='CONSTANT', constant_values=0)
        if mask is None:
            mask = tf.ones_like(inputs[..., :1])  # Single-channel mask
            mask_padded = tf.pad(mask, paddings, mode='CONSTANT', constant_values=0)
        else:
            mask_padded = tf.pad(mask, paddings, mode='CONSTANT', constant_values=0)

        conv_out = self.conv(x_padded)

        ones_kernel = tf.ones((self.kernel_size, self.kernel_size, 1, 1))
        mask_sum = tf.nn.conv2d(mask_padded, ones_kernel, strides=[1, self.strides, self.strides, 1],
                               padding='VALID')
        mask_sum = tf.maximum(mask_sum, 1e-10)

        normalized_out = conv_out / mask_sum
        output = tf.where(mask_sum > 0, normalized_out + self.bias, 0.0)

        return output

class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class ECABlock(tf.keras.layers.Layer):
    def __init__(self, k_size=3, **kwargs):
        super(ECABlock, self).__init__(**kwargs)
        self.k_size = k_size  # Kernel size for 1D conv

    def build(self, input_shape):
        self.filters = input_shape[-1]

        self.gate = self.add_weight(
            name='gate',
            shape=(1,),
            initializer=tf.constant_initializer(0.3),
            trainable=True,
            constraint=ClipConstraint(0.0, 2.0)
        )

        # Define layers
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.global_max_pool = tf.keras.layers.GlobalMaxPooling2D()

        self.reshape = tf.keras.layers.Reshape((self.filters, 1))

        # Separate 1D convolutions for avg and max pooled features
        self.conv_avg = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self.k_size, padding="same",
            use_bias=False, activation='sigmoid'
        )
        self.conv_max = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self.k_size, padding="same",
            use_bias=False, activation='sigmoid'
        )

        self.reshape_attention = tf.keras.layers.Reshape((1, 1, self.filters))
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # Global pooling
        gap_avg = self.global_avg_pool(inputs)  # (batch_size, filters)
        gap_max = self.global_max_pool(inputs)  # (batch_size, filters)

        # Reshape to (batch_size, filters, 1) for 1D conv
        gap_avg = self.reshape(gap_avg)
        gap_max = self.reshape(gap_max)

        # Apply separate 1D convolutions
        attn_avg = self.conv_avg(gap_avg)
        attn_max = self.conv_max(gap_max)

        # Merge attention weights (sum or max, sum is preferred)
        attention = attn_avg + attn_max  # Element-wise sum (batch_size, filters, 1)

        # Reshape back to (batch_size, 1, 1, filters)
        attention = self.reshape_attention(attention)

        # Apply gate and multiply with input
        return self.multiply([inputs, self.gate * attention])

class SpatialAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        # Row convolution to pool each row into a single value
        self.row_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 8),
            strides=(1, 8),
            padding='valid',
            use_bias=False
        )

        # Column convolution to pool each column into a single value
        self.col_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(8, 1),
            strides=(8, 1),
            padding='valid',
            use_bias=False
        )
        # Learnable gate parameter
        self.gate = self.add_weight(
            name='gate',
            shape=(1,),
            initializer=tf.constant_initializer(0.3),
            trainable=True,
            constraint=ClipConstraint(0.0, 2.0)
        )

        # New 3x3 convolution to mix features after adding row and col attentions
        self.fusion_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(3, 3),
            padding='same',
            use_bias=False  # Optional: align with previous layers, or set True if desired
        )

        # Multiplication layer
        self.multiply = tf.keras.layers.Multiply()

    def call(self, inputs):
        # Compute row attention: (batch_size, 8, 1, 1)
        row_attn = self.row_conv(inputs)
        # Compute column attention: (batch_size, 1, 8, 1)
        col_attn = self.col_conv(inputs)
        # Add with broadcasting: (batch_size, 8, 8, 1)
        spatial_attn = row_attn + col_attn
        spatial_attn = self.fusion_conv(spatial_attn)

        # Apply sigmoid to get attention weights
        spatial_attn = tf.sigmoid(spatial_attn)
        # Scale with gate and apply to input
        return self.multiply([inputs, self.gate * spatial_attn])

class PositionalBiasLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalBiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.pos_bias = self.add_weight(
            shape=(1, 8, 8, 1),
            initializer='zeros',
            trainable=True,
            name='pos_bias',
            constraint=ClipConstraint(0.0, 1.0)
        )

    def call(self, inputs):
      batch_size = tf.shape(inputs)[0]
      pos_bias_expanded = tf.tile(self.pos_bias, [batch_size, 1, 1, 1])  # (batch_size, 8, 8, 1)
      return layers.Concatenate(axis=-1)([inputs, pos_bias_expanded])
        
class ResidualBlock(tf.keras.layers.Layer):
    """
    A Keras Layer implementing a Residual Block with optional projection and SE.
    """
    def __init__(self,
                 filters,
                 kernel_size=3,
                 use_attention=False,
                 groups=1,
                 name=None):
        super(ResidualBlock, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_attention = use_attention
        self.padding_size = kernel_size // 2
        self.groups = groups

        # Define the layers for the 'shortcut' (projection) if needed
        self.shortcut_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=1,
            strides=1,
            padding='same'
        )
        self.shortcut_bn = tf.keras.layers.BatchNormalization()

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv1 = PartialConv2D(filters, kernel_size, padding_size=self.padding_size, groups=self.groups)
        self.conv2 = PartialConv2D(filters, kernel_size, padding_size=self.padding_size, groups=self.groups)
        self.bn2 = tf.keras.layers.BatchNormalization()

        if use_attention:
          self.eca_block = ECABlock()
          self.spatial_attention_block = SpatialAttentionBlock()

        # Activation
        self.relu = tf.keras.layers.ReLU()

    def call(self, input_tensor):


        shortcut = self.shortcut_conv(input_tensor)
        shortcut = self.shortcut_bn(shortcut)

        # Residual path: conv1 -> bn1 -> relu
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)

        # conv2 -> bn2
        x = self.conv2(x)
        x = self.bn2(x)

        # Optionally apply SE
        if self.use_attention:
          x = self.eca_block(x)
          x = self.spatial_attention_block(x)

        # Merge shortcut and residual
        x = tf.keras.layers.Add()([x, shortcut])
        x = self.relu(x)
        return x

def create_q_policy_model():
    inputs = layers.Input(shape=(8, 8, 18), dtype='float32', name="BitboardInput")

    x = PositionalBiasLayer()(inputs)

    for _ in range(8):
      x = ResidualBlock(filters=128, use_attention=True)(x)

    # Q-head: outputs Q(s, a) for all 1792 actions
    q_x = layers.Conv2D(filters=32, kernel_size=1, activation="swish")(x)
    q_x = SpatialAttentionBlock()(q_x)
    q_x = ECABlock()(q_x)
    q_x = layers.BatchNormalization()(q_x)
    q_x = layers.Flatten()(q_x)
    q_x = layers.LeakyReLU(alpha=0.01)(q_x)
    q = layers.Dense(1792, activation='linear', name="QHead")(q_x)

    v_x = layers.DepthwiseConv2D(kernel_size=8, activation="swish")(x)
    v_x = layers.BatchNormalization()(v_x)
    v_x = layers.Flatten()(v_x)
    v_x = layers.Dense(128, activation='relu')(v_x)
    v = layers.Dense(1, activation='tanh', name="ValueHead")(v_x)

    # Policy head: outputs π(a|s) logits for all 1792 actions
    pi_x = layers.Conv2D(filters=32, kernel_size=1, activation="swish")(x)
    pi_x = SpatialAttentionBlock()(pi_x)
    pi_x = ECABlock()(pi_x)
    pi_x = layers.BatchNormalization()(pi_x)
    pi_x = layers.Flatten()(pi_x)
    pi_x = layers.LeakyReLU(alpha=0.01)(pi_x)
    pi = layers.Dense(1792, activation='linear', name="PolicyHead")(pi_x)

    model = tf.keras.Model(inputs=inputs, outputs=[q, v, pi])
    return model
