import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv3D, Flatten, Dense, MaxPooling3D, LSTM
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import gym
import scipy.signal
import time
import argparse, gym_unrealcv
import os

#only messages with severity ERROR or higher will be shown. Lower-severity messages (like WARN and INFO) will be suppressed. 
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

attnSize = 7
attention_shape = attnSize*attnSize
feature_size = 512
hidden_units = 512
statefull_lstm_flag = True
prediction_learning_rate = 1e-5 #original: 1e-12

#the get_batches function is a generator function that splits the given dataset into smaller batches of a specific size.
def get_batches(dataset, batch_size):
    """Yield successive batches from the dataset."""
    for i in range(0, dataset.shape[0], batch_size):
        #yield keyword is used in Python to define a generator function that can be paused and resumed, allowing it to generate a sequence of results over time, rather than computing them all at once and returning them in a list for instance. 
        yield dataset[i:i + batch_size]

#preprocess function converts it to grayscale, and make its value between -1 and 1 (normalization)
def preprocess(observation):
    # Add an extra dimension for batch size
    image = tf.expand_dims(observation, axis=0)

    # Use the VGG16 preprocess_input method
    image = preprocess_input(image)

    # Remove the batch size dimension
    image = tf.squeeze(image, axis=0)
    
    return image

def stack_frames(stacked_frames, frame, buffer_size):
    if stacked_frames is None:
        #the * operator can be used to "unpack" the list or tuple into separate arguments.
        #here, frame.shape returns a tuple representing the shape of the frame (for example, (height, width)). So, *frame.shape would unpack this tuple into separate arguments.
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx, _ in enumerate(stacked_frames):
            stacked_frames[idx,:] = frame
    else:
        stacked_frames[0:buffer_size-1,:] = stacked_frames[1:,:]
        stacked_frames[buffer_size-1, :] = frame

    return stacked_frames    

def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, *observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def get(self):
        # Get all data of the buffer and normalize the advantages
        #"cleaning" buffer here
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
#end of the class buffer

def create_actor_critic_network(attention_dimensions, num_actions):
    input_tensor = keras.Input(shape=attention_dimensions)
    #TODO: 5.1 Do i need to add an extra middle layer to project the attention matrix into a larger dimension and then project to target dimension??
    #TODO: 5.2 or, use weighted_loss which is (1, 49, 512) as input, and use a middle dense, and then flat, and then dense with output shape?
    # Output layer
    # 1DGloablAverafe -- (1, 512)
    x = Dense(num_actions, activation=None)(input_tensor)  ##note! we just need raw value; we don't need to softmax for actor at this. Critic will generate real values not percentage

    model = keras.models.Model(inputs=input_tensor, outputs=x)
    
    return model

#MARK: can only process a batch of 1 at a time
def create_cnn_predicted_attention(observation_dimensions):
    # Input is a 4D tensor with shape (frames, height, width, channels)
    input_tensor = keras.Input(shape=observation_dimensions)
    
    # Load VGG16 model
    # observation_dimensions[1:] is used to pass the spatial dimensions (height, width, channels) to the VGG16 model
    vgg16_model = VGG16(input_shape=observation_dimensions[1:], weights='imagenet', include_top=False)
    
    # We set the layers of VGG16 to be not trainable
    for layer in vgg16_model.layers:
        layer.trainable = False
    
    # We use the VGG16 model in a TimeDistributed manner
    x = keras.layers.TimeDistributed(vgg16_model)(input_tensor)

    print(x.shape)

    vgg16_Features = tf.reshape(x, (-1, attention_shape, 512))
    
    previous_input_sequence = tf.reshape(vgg16_Features[0,...], (1, attention_shape, 512))

    # create RNN Predictor
    # Initial hidden state
    # TODO: 1. when should we reset the hidden state?? -- before each new episode?? -- is it the summary of previous scenes??
    hidden_state = tf.Variable(tf.zeros((attention_shape, hidden_units)))
    hidden_state_with_time_axis = tf.expand_dims(hidden_state, 0)
    
    # Bahdanau Attention
    W1 = tf.keras.layers.Dense(hidden_units)
    W2 = tf.keras.layers.Dense(hidden_units)
    V = tf.keras.layers.Dense(1)
    
    # LSTM layers
    LSTM1 = tf.keras.layers.LSTM(hidden_units,
                                  batch_input_shape= (1, attention_shape, feature_size),
                                  time_major = True, 
                                  return_sequences=True,
                                  return_state=True,
                                  stateful= statefull_lstm_flag)

    LSTM2 = tf.keras.layers.LSTM(hidden_units,
                                  batch_input_shape= (1, attention_shape, feature_size),
                                  time_major = True, 
                                  return_sequences=True,
                                  return_state=True,
                                  stateful= statefull_lstm_flag)

    LSTM3 = tf.keras.layers.LSTM(hidden_units,
                                  batch_input_shape= (1, attention_shape, feature_size),
                                  time_major = True, 
                                  return_sequences=True,
                                  return_state=True,
                                  stateful= statefull_lstm_flag)
    
    score = tf.nn.tanh(W1(previous_input_sequence) + W2(hidden_state_with_time_axis))
    attention_weights = tf.nn.softmax(V(score), axis=1)
    context_vector = tf.multiply(previous_input_sequence, attention_weights)
    x_hat = context_vector
    encoder_output1, _, _ = LSTM1(x_hat)
    encoder_output2, _, _ = LSTM2(encoder_output1)
    prediction, next_hidden_state, cell_state = LSTM3(encoder_output2)
    #TODO: 1. may need to clean the LSTM here -- before each new episode??
    hidden_state.assign(next_hidden_state)
    
    #this is used vgg16_features of current obs and compare the loss
    #shape is (1, 49, 512)
    curr_input_sequence = tf.reshape(vgg16_Features[1,...], (1, attention_shape, 512))
	
    #TODO: 3. why?? prediction error, instead of (prediction minus previous) which is predicted suprise and then compare predicted suprise with actual suprise??
    pred_loss = tf.square(tf.subtract(curr_input_sequence, prediction))

    # if we were to just do zero order hold as prediction, this is the loss or error
    #this is actual suprise/move
    frame_diff = tf.square(tf.subtract(curr_input_sequence, previous_input_sequence)) 

    # MARK: prediction loss weighted by frame diffeence (zero-order hold difference), errors will be weighted high for blocks with motion/change -- which is our focus -- what we want to predict
    #TODO:2. output weighted_loss or lossGrid??
    weighted_loss = tf.multiply(frame_diff, pred_loss) 

    #tf.reduce_mean(weighted_loss): When not specify any axis, it computes the mean of all the elements in the weighted_loss tensor, regardless of its shape. The result will be a single scalar value.
    loss = tf.reduce_mean(weighted_loss)/(attnSize*attnSize*512)

    #it will compute the mean across the third dimension
    #final lossGrid tensor will also have the shape (1, 49)
    lossGrid = tf.nn.softmax(tf.reduce_mean(weighted_loss, 2), axis=1)
    #TODO: 4. this loss calculated is kind of mismatch with those formula in paper
    #      any paper about attention algorithm?
    ## return last features, predictions, current featrues, control logits
    model = keras.models.Model(inputs=input_tensor, outputs=[loss, lossGrid])

    return model

#used in update policy gradient where a represent the action
#this a relatively small function with just a couple of TensorFlow operations, so the overhead of turning it into a tf.function may not be worth the performance gains.    
def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability


# Sample action from actor
#@tf.function
def sample_action_may_train_prediction(observation, duplicateFrames):
    if duplicateFrames: #if the frames are just duplicate, don't train prediction
        loss, lossGrid = cnnAttentionPredictor(observation)
        logits = actor(lossGrid)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action, lossGrid
    else:
        with tf.GradientTape() as tape:
            loss, lossGrid = cnnAttentionPredictor(observation)
        prediction_grads = tape.gradient(loss, cnnAttentionPredictor.trainable_variables)
        prediction_optimizer.apply_gradients(zip(prediction_grads, cnnAttentionPredictor.trainable_variables))
    
        #The second argument, 1, specifies that one sample should be drawn. The tf.squeeze function is used to remove dimensions of size 1 from the shape of the resulting tensor. The axis=1 argument specifies that the second dimension should be removed.
        logits = actor(lossGrid)
        #tf.random.categorical function expects logits as input and applies softmax internally.
        #This line is performing the action sampling. tf.random.categorical takes logits as input and draws a sample from a categorical distribution (in this case, the distribution is over potential actions). 
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action, lossGrid



# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    lossGrid, action_buffer, logprobability_buffer, advantage_buffer
):
    ## MARK: since we don't calculate the gradient of cnnAttentionPredictor, we can move the batch_concatenation process out of tape or even this function
    # shape of lossGrid now is (batch, 47)
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        actor_logits = actor(lossGrid)
        ratio = tf.exp(
            logprobabilities(actor_logits, action_buffer)
            - logprobability_buffer
        )
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )

        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantage_buffer, min_advantage)
        )
    policy_grads = tape.gradient(policy_loss, actor.trainable_variables)
    policy_optimizer.apply_gradients(zip(policy_grads, actor.trainable_variables))
    
    new_actor_logits = actor(lossGrid)

    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(new_actor_logits, action_buffer)
    )
    kl = tf.reduce_sum(kl)
    return kl

# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(lossGrid, return_buffer):
    ## MARK: since we don't calculate the gradient of cnnAttentionPredictor, we can move the batch_concatenation process out of tape
    #shape of lossGrid now is (batch, 47)
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        critic_logits = critic(lossGrid)
        value_loss = tf.reduce_mean((return_buffer - critic_logits) ** 2)
    value_grads = tape.gradient(value_loss, critic.trainable_variables)
    value_optimizer.apply_gradients(zip(value_grads, critic.trainable_variables))

# Hyperparameters of the PPO algorithm
steps_per_epoch = 100 #store memories of 100 steps
batch_size = 1 ## MARK: RNN_Predictor can process one stacked frames at a time
epochs = 10000
gamma = 0.9
clip_ratio = 0.2
policy_learning_rate = 1e-4
value_function_learning_rate = 1e-4
train_policy_iterations = 80
train_value_iterations = 80
lam = 0.97
target_kl = 0.01

# True if you want to render the environment
render = False
#the frist agent is the tracker
agentIndex = 0
'''may continue training by making it ture'''
load_checkpoint = True
#inpSize = 336
inpSize = 224
stack_size = 2
channelNum = 3
parser = argparse.ArgumentParser(description=None)
parser.add_argument("-e", "--env_id", nargs='?', default='UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', #'UnrealArm-DiscreteRgbd-v0', #'RobotArm-Discrete-v0',
                    help='Select the environment to run')
parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
args = parser.parse_args()
env = gym.make(args.env_id, resolution=(inpSize,inpSize))
print("Env created")
### should i set here or, as the train.py code, set in the beginning of each episode before env.reset??
env.seed(1)

num_actions = env.action_space[agentIndex].n
# Initialize the buffer

observation_dimensions = (stack_size, inpSize, inpSize, channelNum)
# to define a tuple with a single element, you need to add a trailing comma.
attention_dimensions = (49,)
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
actor = create_actor_critic_network(attention_dimensions, num_actions)
critic = create_actor_critic_network(attention_dimensions, 1)
cnnAttentionPredictor = create_cnn_predicted_attention(observation_dimensions)
#cnnAttentionPredictor.build(observation_dimensions)

load_dir_path = 'tmpIntegratedModel/checkpoints'
dir_path = 'tmpIntegratedModel/checkpoints'
f = open('IntegratedModel.txt', 'a')
# Check if the directory exists
if not os.path.exists(load_dir_path):
    # If the directory does not exist, create it
    print("!!!Successfully created the load directory")
    os.makedirs(load_dir_path)
if not os.path.exists(dir_path):
    # If the directory does not exist, create it
    print("!!!Successfully created the save directory")
    os.makedirs(dir_path)

load_predictor_checkpoint_file = os.path.join(load_dir_path, 'predictor.ckpt')
load_actor_checkpoint_file = os.path.join(load_dir_path, 'actor.ckpt')
load_critic_checkpoint_file = os.path.join(load_dir_path, 'critic.ckpt')

predictor_checkpoint_file = os.path.join(dir_path, 'predictor.ckpt')
actor_checkpoint_file = os.path.join(dir_path, 'actor.ckpt')
critic_checkpoint_file = os.path.join(dir_path, 'critic.ckpt')

if load_checkpoint:
    print('...Loading Checkpoint...')
    cnnAttentionPredictor.load_weights(load_predictor_checkpoint_file)
    actor.load_weights(load_actor_checkpoint_file)
    critic.load_weights(load_critic_checkpoint_file)

# Initialize the policy and the value function optimizers
prediction_optimizer = keras.optimizers.Adam(learning_rate=prediction_learning_rate)
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)


'''Begin to train'''
# Initialize the observation, episode return and episode length
observations, episode_return, episode_length = env.reset(), 0, 0
curr_observation = preprocess(observations[agentIndex])
stacked_frames = None
stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
duplicateFrames = True
# ------ train ------
# Iterate over the number of epochs
for epoch in range(epochs):
    # Initialize the sum of the returns, lengths and number of episodes for each epoch
    sum_return = 0
    sum_length = 0
    num_episodes = 0

    # Iterate over the steps of each epoch
    for t in range(steps_per_epoch):
        if render:
            env.render()

        # reshape to have a batch dimension with size of 1 in order to be fed into neural network
        reshaped_stacked_frames = stacked_frames.reshape(1, *stacked_frames.shape)
        # Get the logits, action, and take one step in the environment
        logits, action, lossGrid = sample_action_may_train_prediction(reshaped_stacked_frames, duplicateFrames)

        # action[0] -- the action of first batch
        observations, rewards, done, _ = env.step(action[0].numpy())
        reward = rewards[agentIndex]
        #print(reward) #--- reward is from -1 to 1
        episode_return += reward
        episode_length += 1
        
        # Get the value and log-probability of the action
        value_t = critic(lossGrid)
        logprobability_t = logprobabilities(logits, action)

        # Store obs, act, rew, v_t, logp_pi_t
        buffer.store(stacked_frames, action, reward, value_t, logprobability_t)

        # Update the observation
        curr_observation = preprocess(observations[agentIndex])
        stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
        duplicateFrames = False

        # Finish trajectory if reached to a terminal state
        terminal = done
        if terminal or (t == steps_per_epoch - 1):
            #MARK: critic
            last_value = 0 #if done
            if not done:
                reshaped_stacked_frames = stacked_frames.reshape(1, *stacked_frames.shape)
                _, _, lossGrid = sample_action_may_train_prediction(reshaped_stacked_frames, duplicateFrames) 
                last_value = critic(lossGrid)
            buffer.finish_trajectory(last_value)
            sum_return += episode_return
            sum_length += episode_length
            num_episodes += 1
            observations, episode_return, episode_length = env.reset(), 0, 0
            curr_observation = preprocess(observations[agentIndex])
            stacked_frames = None
            stacked_frames = stack_frames(stacked_frames, curr_observation, stack_size)
            duplicateFrames = True
        
    print(f"Epoch: {epoch + 1}.")
    # Get values from the buffer
    #advantage_buffer is calculated based on the critic value buffer
    (
        observation_buffer,
        action_buffer,
        advantage_buffer,
        return_buffer,
        logprobability_buffer,
    ) = buffer.get()
   
    #observation_buffer.shape -- (100, 2, 224, 224, 3)
    #this whole batch cannot be directly passed to ConvLayer since the first CNN layer will generate a tensor of (350, 32, 4, 224, 224) where 32 is num of filters
    #hence we need use mini-batch to collect returns from the mini batch of the stacked images in buffer

    lossGrid = []
    for obs_batch in get_batches(observation_buffer, batch_size):
        _, lossGrid_batch = cnnAttentionPredictor(obs_batch)
        lossGrid.append(lossGrid_batch)
    #shape of lossGrid now is (batch, 47)
    lossGrid = tf.concat(lossGrid, axis=0)

    # Update the policy and implement early stopping using KL divergence
    for _ in range(train_policy_iterations):
        kl = train_policy(
            lossGrid, action_buffer, logprobability_buffer, advantage_buffer
        )
        if kl > 1.5 * target_kl:
            # Early Stopping
            break
    
    # Update the value function
    for _ in range(train_value_iterations):
        train_value_function(lossGrid, return_buffer)

    # Print mean return and length for each epoch
    print(
        f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
    )
    
    report_line = f"Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}\n"
    f.write(report_line)
    f.flush()

    print('...Saving Checkpoint...')

    cnnAttentionPredictor.save_weights(predictor_checkpoint_file)
    actor.save_weights(actor_checkpoint_file)
    critic.save_weights(critic_checkpoint_file)
    tf.keras.backend.clear_session()

f.close()
