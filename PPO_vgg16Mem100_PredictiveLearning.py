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
n_hidden1 = 512
n_hidden2 = 512
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


class RNN_Predictor (tf.keras.Model):
    #batch_input_shape -- 49; input_shape -- 512; 
    #TODO: need to set the stateful be true later
	def __init__(self, batch_input_shape, input_shape, hidden_units, statefull_lstm_flag = False):
	  
		super(RNN_Predictor, self).__init__()
	  
		# hidden state
		self.hidden_units = hidden_units
		
		self.hidden_state = tf.zeros((attention_shape, self.hidden_units))
		
        # Bahdanau Attention, the model learns to assign different weights to different parts of the input for each step of the output.
		self.W1 = tf.keras.layers.Dense(hidden_units) 
		
		# Bahdanau Attention layer to process hidden
		self.W2 = tf.keras.layers.Dense(hidden_units)

		# Bahdanau Attention layer to generate context vector from embedded input and hidden state
		self.V = tf.keras.layers.Dense(1)

		# Could not find how to choose the "teacher_force" option for the LSTM, I am assuming that is how it generates the hidden states over the sequence.
		# 
		# We also use a grid of 64 LSTMs to process the indiviual image blocks. The weights of the LSTMs are the same. 
        # We implement this by passing the 64 inputs as a "batch" into a single LSTM, so the batch size is not the default "BATCH_SIZE", which is 1
		# Note LSTM expects a 3D tensor (batch_size, seq_length, feature)
        # If True, the inputs and outputs will be in shape [timesteps, batch, feature], whereas in the False case, it will be [batch, timesteps, feature]. Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation. However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.
		self.LSTM1 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (1, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag) #Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
                                         #processing batch 2 has to wait the completion of processing batch 1
		self.LSTM2 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (1, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag) 
		
		self.LSTM3 = tf.keras.layers.LSTM(self.hidden_units,
										 batch_input_shape= (1, batch_input_shape, input_shape),
										 time_major = True, #(timesteps, batch, ...)
										 return_sequences=True,
										 return_state=True, # return hidden state
										 stateful= statefull_lstm_flag)	   
		
	def call(self, x): 
        # batch size, 1, becomes the seq_length; attention size becomes the batch size
		# Dimension of x is [seq_length(i.e., 1), attention_shape=49,  hidden_units]

		# print ('In RNN Decoder x=', x.shape, 'hidden=', hidden_state.shape)

		# Bahadanou attention -- note we have hooks for temporal attention too.
		if (len(self.hidden_state.shape) == 2):
		   hidden_with_time_axis = tf.expand_dims(self.hidden_state, 0)
		else :
		   print ('Error: hidden_state should be 2 dimensional. It is:', hidden_state.shape)
		   hidden_with_time_axis = self.hidden_state
		
		# score shape == (seq_length, attention_shape=49, hidden_units)
		# W1 and W2 are two dense layer with the same output shape -- hidden_units
		score = tf.nn.tanh(self.W1(x) + self.W2(hidden_with_time_axis))
		
		# attention_weights shape == (seq_length, attention_shape=49, 1)
		# you get 1 at the last axis because you are applying score to self.V
		attention_weights = tf.nn.softmax(self.V(score), axis=1)
		
		# context_vector shape == (seq_length, attention_shape=49, hidden_units) -- broadcasting automatically happens and attention_weights will be converted to (1, 49, 512) and then do the element-wise multiplication
		context_vector = tf.multiply(x, attention_weights)
		
		#print ('context_vector=', context_vector.shape, 'attention_weights=', attention_weights.shape)
		
		# option 1: concatenate context vector with input x - very large state vector
		# shape after concatenation == (attention_shape=64, 1, embedding_dim + hidden_units)	
		# x_hat = tf.concat([context_vector, x], axis=-1)
		
		x_hat = context_vector  # (seq_length, attention_shape=49, embedding_dim)

		# LSTM expects a 3D tensor (seq_length, batch_size,  feature) since time_major == True
		# we use batch_size = 49 -- the 49 blocks of inceptionV3 features

		#  output, next_hidden_state, cell_state = self.LSTM (x_hat)
	 
		encoder_output1, _ , _ = self.LSTM1 (x_hat)
		encoder_output2, _ , _ = self.LSTM2 (encoder_output1)
		output, next_hidden_state, cell_state = self.LSTM3 (encoder_output2)

		# model.add(LSTM(100, activation='relu', input_shape=(n_in,1)))
		# model.add(RepeatVector(n_in))
		# model.add(LSTM(100, activation='relu', return_sequences=True))
		# model.add(TimeDistributed(Dense(1)))
		# model.compile(optimizer='adam', loss='mse')

		self.hidden_state = next_hidden_state

		#print ('x =', x_hat.shape, 'output=', output.shape, 'h state=', next_hidden_state.shape, 'cell state=', cell_state.shape)			   
			   
		return output, self.hidden_state, attention_weights

	def reset_hidden_state(self):
		self.hidden_state = tf.zeros((attention_shape, self.hidden_units))
		# LSTM expects a 3D tensor (batch_size, seq_length, feature)
		# we use batch_size = 64 (attention_shape) -- the 64 blocks of inceptionV3 features
		# seq_length, and feature = 2048 inceptionV3 features
		# hidden state dimension is (batch_size=attention_shape, features)



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

    vgg16_Features = tf.reshape(x, (2, attention_shape, 512))
    
    predictModel = RNN_Predictor(batch_input_shape=attention_shape, input_shape=feature_size, hidden_units=n_hidden1)
    #Note: This is used the vgg16_features of last obs to make prediction of the current obs
	### cannot just use [0] --- since we want the shape be (1, attention_shape, 512) which includes the batch size

    previous_input_sequence = tf.reshape(vgg16_Features[0,...], (1, attention_shape, 512))
    prediction, hidden, attention_weights = predictModel(previous_input_sequence)

    #this is used vgg16_features of current obs and compare the loss
    #shape is (1, 49, 512)
    curr_input_sequence = tf.reshape(vgg16_Features[1,...], (1, attention_shape, 512))
	
    pred_loss = tf.square(tf.subtract(curr_input_sequence, prediction))

    # if we were to just do zero order hold as prediction, this is the loss or error
    frame_diff = tf.square(tf.subtract(curr_input_sequence, previous_input_sequence)) 

    # MARK: prediction loss weighted by frame diffeence (zero-order hold difference), errors will be weighted high for blocks with motion/change
    #TODO: output weighted_loss or lossGrid??
    weighted_loss = tf.multiply(frame_diff, pred_loss) 

    #tf.reduce_mean(weighted_loss): When not specify any axis, it computes the mean of all the elements in the weighted_loss tensor, regardless of its shape. The result will be a single scalar value.
    loss = tf.reduce_mean(weighted_loss)/(attnSize*attnSize*512)

    #it will compute the mean across the third dimension
    #final lossGrid tensor will also have the shape (1, 49)
    lossGrid = tf.nn.softmax(tf.reduce_mean(weighted_loss, 2), axis=1)

    ## return last features, predictions, current featrues, control logits
    model = keras.models.Model(inputs=input_tensor, outputs=[loss, lossGrid])

    return model

def create_actor_critic_network(attention_dimensions, num_actions):
    input_tensor = keras.Input(shape=attention_dimensions)

    # Output layer
    x = Dense(num_actions, activation=None)(input_tensor)  ##note! we just need raw value; we don't need to softmax for actor at this. Critic will generate real values not percentage

    model = keras.models.Model(inputs=input_tensor, outputs=x)
    
    return model

# @tf.function
# def train_prediction(observation):
#     with tf.GradientTape() as tape:
#         loss, _ = cnnAttentionPredictor(observation)
#     prediction_grads = tape.gradient(loss, cnnAttentionPredictor.trainable_variables)
#     prediction_optimizer.apply_gradients(zip(prediction_grads, cnnAttentionPredictor.trainable_variables))

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
# @tf.function
def sample_action_may_train_prediction(observation, duplicateFrames):
    if duplicateFrames: #if the frames are just duplicate, don't train prediction
        loss, lossGrid = cnnAttentionPredictor(observation)
        logits = actor(lossGrid)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action
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
        return logits, action



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
load_checkpoint = False
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
# TODO: be careful about this dimension # to define a tuple with a single element, you need to add a trailing comma.
attention_dimensions = (49,)
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
actor = create_actor_critic_network(attention_dimensions, num_actions)
critic = create_actor_critic_network(attention_dimensions, 1)
cnnAttentionPredictor = create_cnn_predicted_attention(observation_dimensions)

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
        logits, action = sample_action_may_train_prediction(reshaped_stacked_frames, duplicateFrames)
        print("pass")
        # action[0] -- the action of first batch
        observations, rewards, done, _ = env.step(action[0].numpy())
        reward = rewards[agentIndex]
        #print(reward) #--- reward is from -1 to 1
        episode_return += reward
        episode_length += 1
        
        # Get the value and log-probability of the action
        value_t = critic(reshaped_stacked_frames)
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
            last_value = 0 if done else critic(stacked_frames.reshape(1, *stacked_frames.shape))
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
