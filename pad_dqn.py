import pygame
import pygame.locals
import numpy as np
import random
from copy import deepcopy
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

BOARD_WIDTH = 6
BOARD_HEIGHT = 5
ORB_SIZE = 110
DISPLAY_WIDTH = BOARD_WIDTH * ORB_SIZE
DISPLAY_HEIGHT = BOARD_HEIGHT * ORB_SIZE

class Board:
    
    def __init__(self, cursor, board_position):
        self.cursor = cursor
        self.moves = 0
        self.board_position = deepcopy(board_position)
        self.state = self.init_state()

    #sets state given by input
    def set_state(self, new_state):
        self.state = deepcopy(new_state)

    #loads orb sprites into a 2D list, orb_table
    def load_sprite_table(self, filename, width, height):
        image = pygame.image.load(filename).convert_alpha()
        image_width, image_height = image.get_size()
        sprite_table = []
        #fills sprite_table
        for sprite_y in range(int(image_width / width)):
            line = []
            sprite_table.append(line)
            for sprite_x in range(int(image_height / height)):
                rect = (sprite_y * width, sprite_x * height, width, height)
                line.append(image.subsurface(rect))
        return sprite_table

    #swaps two adjacent orbs
    def swap(self, action):
        #list of possible movements (up, down, left, right)
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        new_cursor = (self.cursor[0] + actions[action][0], self.cursor[1] + actions[action][1])
        #checks if orb coordinates are within board dimensions
        if (new_cursor[0] < 0 or new_cursor[1] < 0 or
            new_cursor[0] > BOARD_HEIGHT - 1 or new_cursor[1] > BOARD_WIDTH - 1):
            #increments moves by 2 to discourage invalid moves
            self.moves += 2
            return self.state
        #swaps and updates the board argument
        temp = self.board_position[self.cursor[0]][self.cursor[1]]
        self.board_position[self.cursor[0]][self.cursor[1]] = self.board_position[new_cursor[0]][new_cursor[1]]
        self.board_position[new_cursor[0]][new_cursor[1]] = temp
        #updates state, see the get_state() definition for how the state is interpreted
        grid1 = -1
        grid2 = -1
        for i in range(0, 6):
            if (self.state[self.cursor[0]][self.cursor[1]][i] == 1):
                grid1 = i
            if (self.state[new_cursor[0]][new_cursor[1]][i] == 1):
                grid2 = i
            if grid1 != -1 and grid2 != -1 and grid1 != grid2:
                self.state[self.cursor[0]][self.cursor[1]][grid1] = 0
                self.state[new_cursor[0]][new_cursor[0]][grid1] = 1
                self.state[self.cursor[0]][self.cursor[1]][grid2] = 1
                self.state[new_cursor[0]][new_cursor[0]][grid2] = 0
                break
        self.cursor = new_cursor
        self.moves += 1
        return self.state
    
    #draws background orb sprites
    #Fr = Fire, Wt = Water, Wd = Wood, Lt = Light, Dk = Dark, Ht = Heart
    def draw_board(self, background, orbs):
        for x in range(BOARD_HEIGHT):
            for y in range(BOARD_WIDTH):
                screen.blit(background[(x + y) % 2][0], (y * ORB_SIZE, x * ORB_SIZE))
                if self.board_position[x][y] == 'Fr':
                    screen.blit(orbs[0][0], (y * ORB_SIZE, x * ORB_SIZE))
                elif self.board_position[x][y] == 'Wt':
                    screen.blit(orbs[1][0], (y * ORB_SIZE, x * ORB_SIZE))
                elif self.board_position[x][y] == 'Wd':
                    screen.blit(orbs[2][0], (y * ORB_SIZE, x * ORB_SIZE))
                elif self.board_position[x][y] == 'Lt':
                    screen.blit(orbs[3][0], (y * ORB_SIZE, x * ORB_SIZE))
                elif self.board_position[x][y] == 'Dk':
                    screen.blit(orbs[4][0], (y * ORB_SIZE, x * ORB_SIZE))
                elif self.board_position[x][y] == 'Ht':
                    screen.blit(orbs[5][0], (y * ORB_SIZE, x * ORB_SIZE))
                if self.cursor[0] == x and self.cursor[1] == y:
                    screen.blit(background[2][0], (y * ORB_SIZE, x * ORB_SIZE))
        return
    
    #calculates the number of combos
    def calculate_combos(self):
        #a copy of board_position is created so the original board isn't altered
        board_copy = deepcopy(self.board_position)
        total_combos = 0
        #removed_orbs is 2D list of orbs that need to be cleared
        removed_orbs = self.find_matches(board_copy)
        #iterates until all skyfall orb combos are resolved
        while len(removed_orbs) > 0:
            total_combos += len(removed_orbs)
            for combo in removed_orbs:
                for orb_coord in combo:
                    #replaces board position with a "null" character
                    board_copy[orb_coord[0]][orb_coord[1]] = 'XX'
            #drops orbs down if orbs are erased
            for y in range(BOARD_WIDTH):
                for x in range(BOARD_HEIGHT - 1, 0, -1):
                    if board_copy[x][y] == 'XX':
                        for check_above in range(x - 1, -1, -1):
                            if board_copy[check_above][y] != 'XX':
                                board_copy[x][y] = board_copy[check_above][y]
                                board_copy[check_above][y] = 'XX'
                                break
            removed_orbs = self.find_matches(board_copy)
        return total_combos

    #creates a 2D list of orbs that need to be erased
    def find_matches(self, board):
        #total_matched_orbs is a 2D list of tuples corresponding to board coordinates
        total_matched_orbs = []
        
        #checks for vertical matches of at least 3 orbs, appends erased orbs to total list
        for y in range(BOARD_WIDTH):
            match_length = 1
            for x in range(1, BOARD_HEIGHT):
                if board[x][y] == board[x - 1][y] and board[x][y] != 'XX':
                    match_length += 1
                    if x == BOARD_HEIGHT - 1 and match_length >= 3:
                        matched_orbs = []
                        for row in range(x - (match_length - 1), x + 1):
                            matched_orbs.append((row, y))
                        total_matched_orbs.append(matched_orbs)
                else:
                    if match_length >= 3:
                        matched_orbs = []
                        for row in range(x - match_length, x):
                            matched_orbs.append((row, y))
                        total_matched_orbs.append(matched_orbs)
                    match_length = 1
        #checks for horizontal matches of at least 3 orbs, appends erased orbs to total list
        for x in range(BOARD_HEIGHT):
            match_length = 1
            for y in range(1, BOARD_WIDTH):
                if board[x][y] == board[x][y - 1] and board[x][y] != 'XX':
                    match_length += 1
                    if y == BOARD_WIDTH - 1 and match_length >= 3:
                        matched_orbs = []
                        for column in range(y - (match_length - 1), y + 1):
                            matched_orbs.append((x, column))
                        total_matched_orbs.append(matched_orbs)
                else:
                    if match_length >= 3:
                        matched_orbs = []
                        for column in range(y - match_length, y):
                            matched_orbs.append((x, column))
                        total_matched_orbs.append(matched_orbs)
                    match_length = 1
        #consolidates adjacent combos of the same color
        #compares every coordinate pair with every other one in total_matched_orbs
        #while loops are used because the length of the combo list and orbs per combo changes within the loop
        combo_index = 0
        combos = len(total_matched_orbs)
        while combo_index < combos - 1:
            orb_index = 0
            orbs = len(total_matched_orbs[combo_index])
            while orb_index < orbs:
                #stores tuple coordinates for easier reference
                x_coord1 = total_matched_orbs[combo_index][orb_index][0]
                y_coord1 = total_matched_orbs[combo_index][orb_index][1]
                iterator = combo_index + 1
                iterations = len(total_matched_orbs)
                while iterator < iterations:
                    for orb_coord2 in total_matched_orbs[iterator]:
                        x_coord2 = orb_coord2[0]
                        y_coord2 = orb_coord2[1]
                        #checks if orbs are adjacent and same color
                        if (board[x_coord1][y_coord1] == board[x_coord2][y_coord2] and
                            abs(x_coord1 - x_coord2) + abs(y_coord1 - y_coord2) < 2):
                            #removes duplicate orbs and consolidates into one "combo"
                            total_matched_orbs[combo_index] = list(set(total_matched_orbs[combo_index] +
                                                                       total_matched_orbs[iterator]))
                            total_matched_orbs.pop(iterator)
                            break
                    iterations = len(total_matched_orbs)
                    iterator += 1
                orbs = len(total_matched_orbs[combo_index])
                orb_index += 1
            combos = len(total_matched_orbs)
            combo_index += 1
        return total_matched_orbs

    #converts the board position into 7 2D lists, with each 2D list representing an
    #orb type, with the last 2D list representing the cursor location. Each 2D list
    #is a representation of the board, with a 1 in the locations of the specified orb
    #type, and a 0 in all other locations.
    def init_state(self):
        self.state = np.zeros((5, 6, 7))
        orb_names = ['Fr', 'Wt', 'Wd', 'Lt', 'Dk', 'Ht']
        for count, orb in enumerate(orb_names):
            for y in range(BOARD_WIDTH):
                for x in range(BOARD_HEIGHT):
                    if self.board_position[x][y] == orb:
                        self.state[x][y][count] = 1
        self.state[self.cursor[0]][self.cursor[1]][6] = 1
        return self.state

    #calculates reward function for training
    def get_reward(self):
        return 10 * self.calculate_combos() - self.moves / 5

if __name__ == "__main__":

    #convolutional neural network definition
    network = Sequential()

    #input is board state
    network.add(Conv2D(32, (3,3), input_shape=(5,6,7), padding='same', activation='relu'))
#    network.add(MaxPooling2D(pool_size=(2,2)))
    
    network.add(Conv2D(64, (3,3), padding='same', activation='relu'))
#    network.add(MaxPooling2D(pool_size=(2,2)))

    network.add(Conv2D(64, (1,1), padding='same', activation='relu'))
#    network.add(MaxPooling2D(pool_size=(2,2)))

    network.add(Flatten())
    network.add(Dense(5, kernel_initializer='lecun_uniform', activation='linear'))
    
    rms = RMSprop()
    network.compile(loss='mse', optimizer=rms)
    
    epochs = 1000000 #number of games run 1000000 takes a few hours
    gamma = 0.9 #discount factor for DQN
    epsilon = 1 #exploration rate for epsilon-greedy, linearly decreases to 0.1 over epochs
    batch_size = 100 #batch size for experience replay
    memory = [] #memory to store experiences for replay
    
    for i in range(epochs):
        #2D board array/list to be imported into the Board class
        init_board = [['Fr', 'Lt', 'Wd', 'Ht', 'Wd', 'Wt'],
                      ['Ht', 'Ht', 'Lt', 'Ht', 'Lt', 'Fr'],
                      ['Ht', 'Lt', 'Wd', 'Dk', 'Fr', 'Fr'],
                      ['Wt', 'Dk', 'Wd', 'Ht', 'Lt', 'Wd'],
                      ['Fr', 'Dk', 'Fr', 'Fr', 'Lt', 'Wt']]
        init_cursor = (0, 0)
        board = Board(init_cursor, init_board)
        game_exit = False
        while not game_exit:
            #another dimension is added to board.state to match the network input definitions
            q_value = network.predict(board.state[np.newaxis,...], batch_size=1)
            #takes random action if less than epsilon, acts according to network otherwise
            if random.random() < epsilon:
                action = np.random.randint(0, 5)
            else:
                action = np.argmax(q_value)
            #performs swap, action 4 ends game early
            old_state = board.state
            if action == 4:
                new_state = board.state
            else:
                new_state = board.swap(action)
            #exits game if action is 4 or >=20 moves are made
            if board.moves >= 20 or action == 4:
                game_exit = True
            reward = board.get_reward()
            #experience format: (previous state, action, reward, state, game exit?)
            memory.append((old_state, action, reward, new_state, game_exit))
        #experience replay every 100 games
        if i % 100 == 0:
            print("Game #: %s, Epsilon: %s" % (i, epsilon))
            x_batch, y_batch = [], []
            #chooses random batch of experiences from memory
            minibatch = random.sample(memory, min(len(memory), batch_size))
            for experience in minibatch:
                #adds dimension to match network input
                y_target = network.predict(experience[0][np.newaxis,...])
                if experience[4]:
                    #sets reward if game is in terminal state
                    y_target[0][experience[1]] = experience[2]
                else:
                    #uses Bellman equation to value potential rewards
                    y_target[0][experience[1]] = experience[2] + gamma * np.max(network.predict(experience[3][np.newaxis,...]))
                x_batch.append(experience[0])
                y_batch.append(y_target[0])
            #fits network to batch of experiences
            network.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        #decrements epsilon
        if epsilon > 0.1:
            epsilon -= 1 / epochs

    #tests the training
    init_board = [['Fr', 'Lt', 'Wd', 'Ht', 'Wd', 'Wt'],
                  ['Ht', 'Ht', 'Lt', 'Ht', 'Lt', 'Fr'],
                  ['Ht', 'Lt', 'Wd', 'Dk', 'Fr', 'Fr'],
                  ['Wt', 'Dk', 'Wd', 'Ht', 'Lt', 'Wd'],
                  ['Fr', 'Dk', 'Fr', 'Fr', 'Lt', 'Wt']]
    init_cursor = (0, 0)
    board = Board(init_cursor, init_board)
    game_exit = False
    print(board.board_position)
    #keeps track of actions taken
    action_list = []
    while not game_exit:
        q_value = network.predict(board.state[np.newaxis,...], batch_size=1)
        action = np.argmax(q_value)
        action_list.append(action)
        print('Move #: %s, Taking action: %s' % (board.moves, action))
        if action != 4:
            state = board.swap(action)
        print(board.board_position)
        reward = board.get_reward()
        if action == 4:
            game_exit = True
            print('Combos: %s, Moves: %s, Reward: %s' % (board.calculate_combos(), board.moves, reward))
        elif board.moves >= 20:
            print('Too many moves')
            print('Combos: %s, Moves: %s, Reward: %s' % (board.calculate_combos(), board.moves, reward))
            game_exit = True

    #initializes pygame
    pygame.init()
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    pygame.display.set_caption('PAD Orb Combo')

    #loads sprites
    orb_sprites = board.load_sprite_table("orbs.png", ORB_SIZE, ORB_SIZE)
    background_sprites = board.load_sprite_table("board.png", ORB_SIZE, ORB_SIZE)

    board.draw_board(background_sprites, orb_sprites)
    cursor = (0, 0)
    #x and y coords of actions are swapped because of pygame coordinate format
    possible_actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    #draws a line `through actions
    for action in action_list:
        if action == 4:
            break
        new_cursor = (cursor[0] + possible_actions[action][0], cursor[1] + possible_actions[action][1])
        pygame.draw.line(screen, (0,0,0), tuple(ORB_SIZE*x+ORB_SIZE/2 for x in cursor),
                         tuple(ORB_SIZE*x+ORB_SIZE/2 for x in new_cursor), 15)
        cursor = new_cursor
    pygame.display.update()
    #allows you to quit by pressing Q or closing window
    game_exit = False
    while not game_exit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                game_exit = True
    pygame.quit()
    quit()
