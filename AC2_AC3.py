import tensorflow as tf

import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class ActorCritic(object):
    
    def __init__(self, scope, sess, globalAC=None):
        self.sess = sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(0.0001, name="RMSPropA")
        self.critic_optimizer = tf.train.RMSPropOptimizer(0.0001, name="RMSPropC")
        
        # if the scope is the global network (global agent)
        if scope == global_net_scope:
            with tf.variable_scope(scope):
                # placeholder for the state
                self.state = tf.placeholder(tf.float32, [None, state_shape], "state")
                # global network (global agent) and get the actor and critic parameters
                self.actor_params, self.critic_params = self.build_network(scope)[-2:]
        
        # if the network is not the global networ then
        else:
            with tf.variable_scope(scope):
                # placeholder for the state
                self.state = tf.placeholder(tf.float32, [None, state_shape], "state")
                # Our env is continuous , so the actor network (policy network) returns the mean and the variance of 
                # the action and then we build the action distribution out of this mean and variance and select the action based
                # on this action distribution
                
                # placeholder for obtaining the action distribution
                self.action_dist = tf.placeholder(tf.float32, [None, action_shape], "action")
                
                # placeholder for the target value
                self.target_value = tf.placeholder(tf.float32, [None, 1], "Vtarget")
                
                # Build the worker network (worker agent) and get the mean and variance of the action,
                # the value of the state, and actor and critic network parameters:
                mean, variance, self.value, self.actor_params, self.critic_params = self.build_network(scope)
                
                # Compute the TD error which is the difference between the target value of the state 
                # and the predicted value of the state
                td_error = tf.subtract(self.target_value, self.value, name = "TD_error")
                
                # now, let's define the critic network loss
                with tf.name_scope("critic_loss"):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))
                
                with tf.name_scope("wrap_action"):
                    mean, variance = mean * action_bound[1], variance + 1e-4
                
                # create a normal distribution based on the mean and variance of the action
                normal_dist = tf.distributions.Normal(mean, variance)
                
                # now, let's define the actr network loss
                with tf.name_scope("actor_loss"):
                    
                    # compute the log probability of the action
                    log_prob = normal_dist.log_prob(self.action_dist)
                    
                    # define the entropy of the policy
                    entropy_pi = normal_dist.entropy()
                    
                    # compute the actor network loss
                    self.loss = log_prob * td_error + (beta * entropy_pi)
                    self.action_loss = tf.reduce_mean(-self.loss)
                    
                # select the action based on the normal distribution
                with tf.name_scope("select_action"):
                    self.action = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0),
                                                   action_bound[0], action_bound[1])
                
                # compute the gradients of the actor and critic network loss of the worker agent (local agent)
                with tf.name_scope("local_gad"):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)
                
            # now, let's perform the sync operation
            with tf.name_scope("sync"):
                
                # After computing the gradient of the loss of the actor and critic network, worker agent
                # sends (push) those gradients to the global agent
                
                with tf.name_scope("push"):
                    self.update_actor_params = self.actor_optimizer.apply_gradients(zip(self.actor_grads,
                                                                                        globalAC.actor_params))
                    self.update_critic_params = self.critic_optimizer.apply_gradients(zip(self.actor_grads,
                                                                                          globalAC.critic_params))
                    
                # gloabal agent updates their parameter with the gradients received from the worker agents 
                # (local agents). Then the worker agents, pull the updated parameter from the global agent
                with tf.name_scope("pull"):
                    self.pull_actor_params = [l_p.assign(g_p) for l_p, g_p in zip(self.actor_params,
                                                                                  globalAC.actor_params)]
                    self.pull_critic_params = [l_p.assign(g_p) for l_p, g_p in zip(self.critic_params,
                                                                                   globalAC.critic_params)]
        
    def build_network(self, scope):
        
        # initialize the weight:
        w_init = tf.random_normal_initializer(0., .1)
        
        # define the actor network which returns the mean and variance of the action
        with tf.variable_scope("actor"):
            l_a = tf.layers.dense(self.state, 200, tf.nn.relu, kernel_initializer=w_init, name="la")
            mean = tf.layers.dense(l_a, action_shape, tf.nn.tanh, kernel_initializer=w_init, name="mean")
            variance = tf.layers.dense(l_a, action_shape, tf.nn.softplus, kernel_initializer=w_init, name="variance")
        
        # define the critic network which returns the value of the state
        with tf.variable_scope("critic"):
            l_c = tf.layers.dense(self.state, 100, tf.nn.relu, kernel_initializer=w_init, name="lc")
            value = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="value")
        
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/actor")
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + "/critic")
        
        return mean, variance, value, actor_params, critic_params
    
    def update_global(self, feed_dict):
        self.sess.run([self.update_actor_params, self.update_critic_params], feed_dict)
        
    # we also define a function called pull_from_global for updateing the parameters 
    # the worker networks by pulling from the global network, that is, the pull operation
    def pull_from_global(self):
        self.sess.run([self.pull_actor_params, self.pull_critic_params])
        
    def select_action(self, state):
        state = state[np.newaxis, :]
        return self.sess.run(self.action, {self.state : state})[0]








class Worker:
    
    def __init__(self, env, name, globalAC, sess):
        self.env = env
        self.name = name
        self.AC = ActorCritic(name, sess, globalAC)
        self.sess = sess
        
    
    def work(self):
        global global_rewards, global_episodes
        
        # intitialize the time step
        total_step = 1
        
        # initialize a list for storing the states, actions and rewards
        batch_states, batch_actions, batch_rewards = [], [], []
        
        # when the global episodes are less than the number of episodes and coordinator is active
        while not coord.should_stop() and global_episodes < num_episodes:
            # initialize the state by resetting the environment
            state = self.env.reset()
            
            # initialize the return
            Return = 0
            
            # for each step in the environment
            for t in range(num_timesteps):
                
                # render the environment of only the worker 0:
                if self.name == "W_0":
                    self.env.render()
                    
                # select the action
                action = self.AC.select_action(state)
                
                # perform the selected action
                next_state, reward, done, _ = self.env.step(action)
                
                # set done t true if we reached the final step of the episode else set to false
                done = True if t == num_timesteps - 1 else False
                
                # update the return
                Return += reward
                
                # store the state, action, and reward into the lists
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append((reward+8)/8)
                
                # now, let's update the global network. If done is true then set the value of next state to 0 else 
                # the compute the value of the next state
                if total_step % update_global == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = self.sess.run(self.AC.value, {self.AC.state : next_state[np.newaxis, :]})[0, 0]
                    
                    batch_target_value = []
                    
                    # compute the target value which is sum of reward and discounted value of next state
                    for reward in batch_rewards[::-1]:
                        v_s_ = reward + gamma * v_s_
                        batch_target_value.append(v_s_)
                        
                    # reverse the target value
                    batch_target_value.reverse()
                    
                    # stack the state, action and target value
                    batch_states, batch_actions, batch_target_value = np.vstack(batch_states), np.vstack(batch_actions), np.vstack(batch_target_value)
                    
                    # define the feed dictionary
                    feed_dict = {
                        self.AC.state : batch_states,
                        self.AC.action_dist : batch_actions,
                        self.AC.target_value : batch_target_value
                    }
                    
                    # update the global network
                    self.AC.update_global(feed_dict)
                    
                    # empty the lists:
                    batch_states, batch_actions, batch_rewards = [], [], []
                    
                    # update the worker network by pulling the parameters from the global network
                    self.AC.pull_from_global()
                    
                # update global rewards:
                if done:
                    if len(global_rewards) < 5:
                        global_rewards.append(Return)
                    else:
                        global_rewards.append(Return)
                        global_rewards[-1] = (np.mean(global_rewards[-5:]))
                    
                    global_episodes += 1
                    break
                
                    
