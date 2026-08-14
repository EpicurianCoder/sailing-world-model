from tensorflow.keras import layers, models


def build_vision_model():
    vision_input = layers.Input(shape=(3, 256, 256), name="vision_grid_input")
    x = layers.Permute((2, 3, 1))(vision_input)

    # Extract details
    x = layers.Conv2D(16, (7, 7), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Extract Context
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Compress down to 32x32x32
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Extract Naviation
    x = layers.Conv2D(64, (3, 3), strides=(4, 4), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Flatten()(x)

    latent_output = layers.Dense(128, activation='relu', name="vision_latent_vector")(x)

    return models.Model(inputs=vision_input, outputs=latent_output, name="Vision_CNN")


def build_memory_model(telemetry_size=22, action_size=3):
    latent_input = layers.Input(shape=(128,), name="latent_input") # Reverted to 128
    telemetry_input = layers.Input(shape=(telemetry_size,), name="telemetry_input")
    prev_action_input = layers.Input(shape=(action_size,), name="prev_action_input")

    state_h_input = layers.Input(shape=(128,), name="state_h_input")
    state_c_input = layers.Input(shape=(128,), name="state_c_input")

    merged_input = layers.Concatenate()([latent_input, telemetry_input, prev_action_input])
    reshaped_input = layers.Reshape((1, 128 + telemetry_size + action_size))(merged_input)

    lstm_out, state_h, state_c = layers.LSTM(128, return_state=True, name="memory_lstm")(
        reshaped_input, initial_state=[state_h_input, state_c_input]
    )

    return models.Model(
        inputs=[latent_input, telemetry_input, prev_action_input, state_h_input, state_c_input],
        outputs=[lstm_out, state_h, state_c],
        name="Memory_LSTM"
    )


def build_ppo_controller_and_critic():
    # Inputs -> 128 + 128 = 256
    latent_input = layers.Input(shape=(128,), name="vision_latent_input")
    memory_input = layers.Input(shape=(128,), name="lstm_hidden_input")

    # Combine the CNN and LSTM output into single state
    combined_state = layers.Concatenate()([latent_input, memory_input])

    # Actor - The controller and the sailor themselves
    a = layers.Dense(128, activation='relu')(combined_state)
    a = layers.Dense(64, activation='relu')(a)
    action_output = layers.Dense(3, activation='tanh', name="actor_output")(a)

    # Critic - output the linear reward prediciton
    c = layers.Dense(128, activation='relu')(combined_state)
    c = layers.Dense(64, activation='relu')(c)
    value_output = layers.Dense(1, activation='linear', name="critic_value")(c)

    return models.Model(
        inputs=[latent_input, memory_input], 
        outputs=[action_output, value_output],
        name="Controller_ANN_Dense"
    )
