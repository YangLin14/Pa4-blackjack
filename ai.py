import copy
import random

from game import Game, states

HIT = 0
STAND = 1
DISCOUNT = 0.95 #This is the gamma value for all value calculations

class Agent:
    def __init__(self):

        # For MC values
        self.MC_values = {} # Dictionary: Store the MC value of each state
        self.S_MC = {}      # Dictionary: Store the sum of returns in each state
        self.N_MC = {}      # Dictionary: Store the number of samples of each state
        # MC_values should be equal to S_MC divided by N_MC on each state (important for passing tests)

        # For TD values
        self.TD_values = {}  # Dictionary: Store the TD value of each state
        self.N_TD = {}       # Dictionary: Store the number of samples of each state

        # For Q-learning values
        self.Q_values = {}   # Dictionary: Store the Q-Learning value of each state and action
        self.N_Q = {}        # Dictionary: Store the number of samples of each state for each action

        # Initialization of the values
        for s in states:
            self.MC_values[s] = 0
            self.S_MC[s] = 0
            self.N_MC[s] = 0
            self.TD_values[s] = 0
            self.N_TD[s] = 0
            self.Q_values[s] = [0,0] # First element is the Q value of "Hit", second element is the Q value of "Stand"
            self.N_Q[s] = [0,0] # First element is the number of visits of "Hit" at state s, second element is the Q value of "Stand" at s

        # Game simulator
        # NOTE: see the comment of `init_cards()` method in `game.py` for description of the initial game states       
        self.simulator = Game()

    # NOTE: do not modify this function
    # This is the fixed policy given to you, for which you need to perform MC and TD policy evaluation. 
    @staticmethod
    def default_policy(state):
        user_sum = state[0]
        user_A_active = state[1]
        actual_user_sum = user_sum + user_A_active * 10
        if actual_user_sum < 14:
            return 0
        else:
            return 1

    # NOTE: do not modify this function
    # This is the fixed learning rate for TD and Q learning. 
    @staticmethod
    def alpha(n):
        return 10.0/(9 + n)
   
    #TODO: Take one step of transition in the game simulator
    #Hint: Take the given action, and return the next state given by the game engine. 
    #Hint: Useful functions: self.simulator.act_hit, self.simulator.act_stand, self.simulator.state 
    #Hint: If a state is terminal ("game_over"), i.e., taking any action from it doesn't lead to any next state, then you can return None
    #Hint: You need the act_hit and act_stand functions in game.py. Note that they are already generating random next cards. 
    #Hint: You can keep track the reward of states with this function as well, e.g., as one of the return values
    #Hint: After this function, you can also define another function that simulates one full trajectory, but it's optional
    def make_one_transition(self, action):
        if self.simulator.game_over():
            return None, 0
        
        if action == HIT:
            self.simulator.act_hit()
        elif action == STAND:
            self.simulator.act_stand()
        
        reward = self.simulator.check_reward()
        next_state = self.simulator.state
        
        if self.simulator.game_over():
            next_state = None
            
        return next_state, reward

    #TODO: Implement MC policy evaluation
    def MC_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "MC")
            self.simulator.reset()  # The simulator is already reset for you for each new trajectory

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Make sure to update self.MC_values, self.S_MC, self.N_MC for the autograder
            # Don't forget the DISCOUNT
            episode = []
            curr_state = self.simulator.state
            
            while curr_state is not None:
                action = self.default_policy(curr_state)
                
                next_state, reward = self.make_one_transition(action)
                
                episode.append((curr_state, action, reward))
                curr_state = next_state
            
            G = 0
            for t in reversed(range(len(episode))):
                state, action, reward = episode[t]
                
                G = reward + DISCOUNT * G
                
                self.N_MC[state] += 1
                self.S_MC[state] += G
                self.MC_values[state] = self.S_MC[state] / self.N_MC[state]
    
    #TODO: Implement TD policy evaluation
    def TD_run(self, num_simulation, tester=False):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "TD")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Make sure to update self.TD_values and self.N_TD for the autograder
            # Don't forget the DISCOUNT
            curr_state = self.simulator.state
            
            while curr_state is not None:
                action = self.default_policy(curr_state)
                
                next_state, reward = self.make_one_transition(action)
                
                next_val = self.TD_values[next_state] if next_state is not None else 0
                td_target = reward + DISCOUNT * next_val
                
                self.N_TD[curr_state] += 1
                alpha = self.alpha(self.N_TD[curr_state])
                self.TD_values[curr_state] += alpha * (td_target - self.TD_values[curr_state])
                
                curr_state = next_state
                
    #TODO: Implement Q-learning
    def Q_run(self, num_simulation, tester=False, epsilon=0.4):

        # Perform num_simulation rounds of simulations in each cycle of the overall game loop
        for simulation in range(num_simulation):

            # Do not modify the following three lines
            if tester:
                self.tester_print(simulation, num_simulation, "Q")
            self.simulator.reset()

            # TODO
            # Note: Do not reset the simulator again in the rest of this simulation
            # Hint: self.simulator.state gives you the current state of the trajectory
            # Hint: Use the "make_one_transition" function to take steps in the simulator, and keep track of the states
            # Hint: Go through game.py file and figure out which functions will be useful
            # Hint: The learning rate alpha is given by "self.alpha(...)"
            # Hint: Implement epsilon-greedy method in "self.pick_action(...)"
            # Important: When calling pick_action, use the given parameter epsilon=0.4 to match the autograder
            # Make sure to update self.Q_values, self.N_Q for the autograder
            # Don't forget the DISCOUNT
            curr_state = self.simulator.state
            
            while curr_state is not None:
                action = self.pick_action(curr_state, epsilon)
                
                next_state, reward = self.make_one_transition(action)
                
                if next_state is None:
                    target = reward
                else:
                    max_next_q = max(self.Q_values[next_state][HIT], self.Q_values[next_state][STAND])
                    target = reward + DISCOUNT * max_next_q
                
                self.N_Q[curr_state][action] += 1
                alpha = self.alpha(self.N_Q[curr_state][action])
                
                old_q = self.Q_values[curr_state][action]
                self.Q_values[curr_state][action] = old_q + alpha * (target - old_q)
                
                curr_state = next_state

    #TODO: Implement epsilon-greedy policy
    def pick_action(self, s, epsilon):
        # TODO: Replace the following random value with an action following the epsilon-greedy strategy
        if random.random() < epsilon:
            return random.randint(0, 1)

        if self.Q_values[s][HIT] >= self.Q_values[s][STAND]:
            return HIT
        else:
            return STAND

    ####Do not modify anything below this line####

    #Note: do not modify
    def autoplay_decision(self, state):
        hitQ, standQ = self.Q_values[state][HIT], self.Q_values[state][STAND]
        if hitQ > standQ:
            return HIT
        if standQ > hitQ:
            return STAND
        return HIT #Before Q-learning takes effect, just always HIT

    # NOTE: do not modify
    def save(self, filename):
        with open(filename, "w") as file:
            for table in [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q]:
                for key in table:
                    key_str = str(key).replace(" ", "")
                    entry_str = str(table[key]).replace(" ", "")
                    file.write(f"{key_str} {entry_str}\n")
                file.write("\n")

    # NOTE: do not modify
    def load(self, filename):
        with open(filename) as file:
            text = file.read()
            MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text, _  = text.split("\n\n")
            
            def extract_key(key_str):
                return tuple([int(x) for x in key_str[1:-1].split(",")])
            
            for table, text in zip(
                [self.MC_values, self.TD_values, self.Q_values, self.S_MC, self.N_MC, self.N_TD, self.N_Q], 
                [MC_values_text, TD_values_text, Q_values_text, S_MC_text, N_MC_text, NTD_text, NQ_text]
            ):
                for line in text.split("\n"):
                    key_str, entry_str = line.split(" ")
                    key = extract_key(key_str)
                    table[key] = eval(entry_str)

    # NOTE: do not modify
    @staticmethod
    def tester_print(i, n, name):
        print(f"\r  {name} {i + 1}/{n}", end="")
        if i == n - 1:
            print()
"""
I asked Gemini: How exactly do we implement Q-learning and TD(0) in code?

Gemini answered:
How to Code Q-Learning & TD
    # Initialize V arbitrarily
    V = {s: 0.0 for s in env.get_states()}

    for _ in range(episodes):
        state = env.reset() # Start new game
        
        while not env.is_terminal(state):
            # 1. Pick action based on the policy we are evaluating
            action = policy[state] 
            
            # 2. Take step (Observer R and Next State)
            next_state, reward = env.step(action)
            
            # 3. THE TD UPDATE FORMULA
            # Target = Reward + Discount * Guess_of_Next_State
            td_target = reward + gamma * V[next_state]
            
            # Update = Old + Alpha * (Target - Old)
            V[state] = V[state] + alpha * (td_target - V[state])
            
            state = next_state
            
    return V

Q-LearningUsed for Control (Finding the optimal policy). It is Off-Policy (it learns the optimal value even if it takes random exploration actions).$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$Pythonimport random

def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=1000):
    # Initialize Q-table: key=(state, action), value=0.0
    Q = {} 
    # (Helper to ensure we handle new states safely)
    def get_q(s, a): return Q.get((s, a), 0.0)

    for _ in range(episodes):
        state = env.reset()
        
        while not env.is_terminal(state):
            actions = env.get_actions(state)
            
            # 1. Epsilon-Greedy Strategy (Exploration vs Exploitation)
            if random.random() < epsilon:
                action = random.choice(actions) # Explore
            else:
                # Exploit: Choose action with highest current Q
                action = max(actions, key=lambda a: get_q(state, a))
            
            # 2. Take Step
            next_state, reward = env.step(action)
            
            # 3. Find max Q for the NEXT state (Off-Policy lookahead)
            if env.is_terminal(next_state):
                max_next_q = 0.0
            else:
                next_actions = env.get_actions(next_state)
                max_next_q = max([get_q(next_state, a) for a in next_actions])
            
            # 4. THE Q-LEARNING UPDATE FORMULA
            current_q = get_q(state, action)
            target = reward + gamma * max_next_q
            
            Q[(state, action)] = current_q + alpha * (target - current_q)
            
            state = next_state
            
    return Q
"""