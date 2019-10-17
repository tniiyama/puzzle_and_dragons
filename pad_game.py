import pygame
import pygame.locals
import math

BOARD_WIDTH = 6
BOARD_HEIGHT = 5
ORB_SIZE = 110
DISPLAY_WIDTH = BOARD_WIDTH * ORB_SIZE
DISPLAY_HEIGHT = BOARD_HEIGHT * ORB_SIZE

#initializes board position, change edit this code to change the default board
#Fr = Fire, Wt = Water, Wd = Wood, Lt = Light, Dk = Dark, Ht = Heart
def init_board():
    return [['Fr', 'Ht', 'Fr', 'Wd', 'Wd', 'Lt'],
            ['Ht', 'Wd', 'Wt', 'Wd', 'Lt', 'Lt'],
            ['Dk', 'Ht', 'Fr', 'Lt', 'Fr', 'Wd'],
            ['Lt', 'Lt', 'Wd', 'Dk', 'Lt', 'Wt'],
            ['Wt', 'Wd', 'Ht', 'Wd', 'Dk', 'Wt']]

#loads orb sprites into a 2D list, orb_table
def load_orb_table(filename, width, height):
    #takes image grid from file
    image = pygame.image.load(filename).convert_alpha()
    image_width, image_height = image.get_size()
    orb_table = []
    #fills orb_table
    for orb_y in range(int(image_width / width)):
        line = []
        orb_table.append(line)
        for orb_x in range(int(image_height / height)):
            rect = (orb_y * width, orb_x * height, width, height)
            line.append(image.subsurface(rect))
    return orb_table

#swaps two adjacent orbs
def swap(board, action):
    global moves, cursor
    #list of possible movements (up, down, left, right)
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    new_cursor = (cursor[0] + actions[action][0], cursor[1] + actions[action][1])
    #checks if orb coordinates are within board dimensions
    if (new_cursor[0] < 0 or new_cursor[1] < 0 or
        new_cursor[0] > BOARD_HEIGHT - 1 or new_cursor[1] > BOARD_WIDTH - 1):
        print('Swap out of board dimensions')
        return
    #swaps and updates board argument
    temp = board[cursor[0]][cursor[1]]
    board[cursor[0]][cursor[1]] = board[new_cursor[0]][new_cursor[1]]
    board[new_cursor[0]][new_cursor[1]] = temp
    cursor = new_cursor
    moves += 1
    return

#draws background orb sprites
def draw_board(board, background, orbs, cursor):
    for x in range(BOARD_HEIGHT):
        for y in range(BOARD_WIDTH):
            screen.blit(background[(x + y) % 2][0], (y * ORB_SIZE, x * ORB_SIZE))
            if board[x][y] == 'Fr':
                screen.blit(orbs[0][0], (y * ORB_SIZE, x * ORB_SIZE))
            elif board[x][y] == 'Wt':
                screen.blit(orbs[1][0], (y * ORB_SIZE, x * ORB_SIZE))
            elif board[x][y] == 'Wd':
                screen.blit(orbs[2][0], (y * ORB_SIZE, x * ORB_SIZE))
            elif board[x][y] == 'Lt':
                screen.blit(orbs[3][0], (y * ORB_SIZE, x * ORB_SIZE))
            elif board[x][y] == 'Dk':
                screen.blit(orbs[4][0], (y * ORB_SIZE, x * ORB_SIZE))
            elif board[x][y] == 'Ht':
                screen.blit(orbs[5][0], (y * ORB_SIZE, x * ORB_SIZE))
            if cursor[0] == x and cursor[1] == y:
                screen.blit(background[2][0], (y * ORB_SIZE, x * ORB_SIZE))
    return

#calculates the number of combos
def calculate_combos(board):
    total_combos = 0
    #removed_orbs is 2D list of orbs that need to be cleared
    removed_orbs = find_matches(board)
    #iterates until all skyfall orb combos are resolved
    while len(removed_orbs) > 0:
        total_combos += len(removed_orbs)
        for combo in removed_orbs:
            for orb_coord in combo:
                #replaces board position with a "null" character
                board[orb_coord[0]][orb_coord[1]] = 'XX'
        #drops orbs down if orbs are erased
        for y in range(BOARD_WIDTH):
            for x in range(BOARD_HEIGHT - 1, 0, -1):
                if board[x][y] == 'XX':
                    for check_above in range(x - 1, -1, -1):
                        if board[check_above][y] != 'XX':
                            board[x][y] = board[check_above][y]
                            board[check_above][y] = 'XX'
                            break
        removed_orbs = find_matches(board)
    return total_combos

#creates a 2D list of orbs that need to be erased
def find_matches(board):
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
                        total_matched_orbs[combo_index] = list(set(total_matched_orbs[combo_index] + total_matched_orbs[iterator]))
                        total_matched_orbs.pop(iterator)
                        break
                iterations = len(total_matched_orbs)
                iterator += 1
            orbs = len(total_matched_orbs[combo_index])
            orb_index += 1
        combos = len(total_matched_orbs)
        combo_index += 1
    return total_matched_orbs

pygame.init()

screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))
pygame.display.set_caption('PAD Orb Combo')

clock = pygame.time.Clock()
    
orb_sprites = load_orb_table("orbs.png", ORB_SIZE, ORB_SIZE)
background_sprites = load_orb_table("board.png", ORB_SIZE, ORB_SIZE)
board_position = init_board()
cursor = (0, 0)
moves = 0

draw_board(board_position, background_sprites, orb_sprites, cursor)
pygame.display.update()

game_exit = False
reset = False 
while not game_exit:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_exit = True
        
        if reset == False and event.type == pygame.MOUSEBUTTONDOWN:
            moves = 0
            mouse_x, mouse_y = pygame.mouse.get_pos()
            cursor = (math.floor(mouse_y / ORB_SIZE), math.floor(mouse_x / ORB_SIZE))

        if event.type== pygame.KEYDOWN:
            if reset == False and event.key == pygame.K_UP:
                swap(board_position, 0)
            if reset == False and event.key == pygame.K_DOWN:
                swap(board_position, 1)
            if reset == False and event.key == pygame.K_LEFT:
                swap(board_position, 2)
            if reset == False and event.key == pygame.K_RIGHT:
                swap(board_position, 3)
            if event.key == pygame.K_SPACE:
                print('Combos: %s, Moves: %s' % (calculate_combos(board_position), moves))
                print('Press R to reset')
                reset = True
            if event.key == pygame.K_r:
                board_position = init_board()
                cursor = (0, 0)
                moves = 0
                reset = False
            if event.key == pygame.K_q:
                game_exit = True
        
    draw_board(board_position, background_sprites, orb_sprites, cursor)
    pygame.display.update()
    clock.tick(60)

pygame.quit()
quit()
