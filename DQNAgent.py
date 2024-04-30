import tensorflow as tf
from model import Model
from chess import Chess
from memory import Memory
import random
import numpy as np
class DQNAgent:
    def __init__(self, gamma, tau, learning_rate, max_memory_size):
        self.q_learner = Model()
        self.target_q_learner = Model()
        self.learning_rate = learning_rate
        self.tau=tau
        self.memory=Memory(max_memory_size)
        self.q_learner_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.q_learner_criterion = tf.keras.losses.MeanSquaredError()
        self.update_target_frequency = 10
        self.steps = 0
        self.gamma = gamma
        self.target_q_learner.set_weights(self.q_learner.get_weights())
    def get_action(self, legal_moves):
        if np.random.rand() < self.tau:
            return random.choice(legal_moves)
        else:
            maxqval=-999999999
            bestmove=legal_moves[0]
            for move in legal_moves:
                Qval = self.q_learner(self.memory.preprocess_input(move))
                if Qval>maxqval:
                    maxqval=Qval
                    bestmove=move
            return bestmove
    def target_get_action(self, legal_moves):
        maxqval=-999999999
        bestmove=legal_moves[0]
        for move in legal_moves:
            
            Qval = self.q_learner_target(self.memory.preprocess_input(move))
            if Qval>maxqval:
                maxqval=Qval
                bestmove=move
        return bestmove
    def update(self, batch_size):
        states, rewards, next_states = self.memory.sample(batch_size)
        with tf.GradientTape() as tape:
            Qvals = self.q_learner(states)
            next_Q = self.q_learner_target(next_states)
            Qprime = rewards + self.gamma * next_Q
            loss_model = self.q_learner_criterion(Qvals, Qprime)
            self.loss = loss_model
            grads = tape.gradient(loss_model, self.q_learner.trainable_variables)
            self.q_learner_optimizer.apply_gradients(zip(grads, self.q_learner.trainable_variables))
    def save_model(self):
            self.q_learner.save('model.h5')
            self.target_q_learner.save('target_model.h5')
    def load_model(self):
        self.q_learner = tf.keras.models.load_model('model.h5')
        self.target_q_learner = tf.keras.models.load_model('target_model.h5')