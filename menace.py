import numpy as np
import pandas as pd
from random import randint

def menace(playing = False, states = False, iterations = 300):
    """
    Based on Menace matchbox, tic tac toe machine
    
    known bugs: player can make an illegal move
                if an illegal move is made, the game will end prematurely

    Parameters
    ----------
    playing : bool, optional
        true if user wants to play game. The default is False.
    states : pd.DataFrame, optional
        if using a premade machine state insert here. The default is False.
    iterations : int, optional
        number of iterations the computer should make. The default is 300.

    Returns
    -------
    states : pd.DataFrame
        the final machine state before exiting the program. Only returns if 
        no machine state was input.

    """
    # generate menace model
    if states is False:
        init_state = False
        states = menace_learn(iterations)
    else:
        init_state = True
        
    # play game against a person
    if playing:
        print("starting game")
        currentgame = np.zeros(9,int)
        check = 0
        player_turn = randint(0,1)
        player = 1
        for turn in range(9):
            # If I get time replace 0 with ' ', 1 with 'X', and 2 with 'O'
            if player_turn:
                print(currentgame.reshape(3,3))
                print("player "+str(player)+"'s turn")
                player_move = int(input("your move \n x(0-8) top left to bottom right: "))
                currentgame[player_move] = player
                player_turn = 0
            else:
                computer_move, states = make_move(currentgame, states) 
                currentgame[computer_move[1]] = player
                player_turn = 1
            
            check = check_winner(currentgame)
            if check:
                break
            player = switch_player(player)
            
        if check == 0:
            print("draw")
        else:
            print("player "+str(check)+" won")
            
        print(currentgame.reshape(3,3))
        
    # return menace model
    if init_state is False:
        return states

def switch_player(player):
    """
    simple function to switch players
    """
    if player == 1:
        player = 2
    else:
        player = 1
    return player

def menace_learn(iterations):
    """
    very basic machine learning algorithm for tic tac toe

    Parameters
    ----------
    iterations : int
        number of iterations the computer learns for.

    Returns
    -------
    states : pd.DataFrame
        final machine state after learning from itself.

    """
    # training    
    currentgame = np.zeros(9,int)
    states = pd.DataFrame({"grid" : [currentgame.copy()],
                           "weights" : [np.full(9,10)]})

    for i in range(iterations):
        currentgame.fill(0)
        game_history = {}
        player = 1
        winner = 0
        for turn in range(9):
            choice, states = make_move(currentgame, states)
            currentgame[choice[1]] = player
            game_history[turn] = choice
            
            # check for a win
            check = check_winner(currentgame)
            if check:
                winner = check
                break
            
            # change player
            player = switch_player(player)
        # end for
        
        # adjust choices based on who won
        # subtract 1 from losing moves, add 3 to winning moves
        if winner == 1:
            win = True
        else:
            win = False
        
        if winner != 0:
            for turn in range(len(game_history)):
                # win
                if win:
                    states["weights"][game_history[turn][0]][game_history[turn][1]] += 3
                    win = False
                # loss
                else:
                    states["weights"][game_history[turn][0]][game_history[turn][1]] -= 1
                    win = True
            
        
    # end for
    return states

def make_move(currentgame, states):
    """
    have the computer make a move
    
    known bug: machine will sometimes make an illegal move

    Parameters
    ----------
    currentgame : np.array
        array of current gameboard.
    states : pd.DataFrame
        current machine state.

    Returns
    -------
    (indx,choice) : int
        index of DataFrame.
    int
        choice the machine made.
    states : pd.DataFrame
        machine state after making a move.

    """
    # check if position has been reached before
    gamegrid = currentgame.reshape(3,3)
    same_grid = (
        currentgame,
        gamegrid.transpose().flatten(),
        gamegrid[::-1].flatten(),
        gamegrid.transpose()[::-1].flatten(),
        gamegrid[::-1].transpose().flatten(),
        gamegrid.transpose()[::-1].transpose().flatten(),
        gamegrid[::-1].transpose()[::-1].flatten(),
        gamegrid[::-1].transpose()[::-1].transpose().flatten()
    )
    mask = states["grid"].map(lambda x : valid_grid(same_grid,x)) == True
    if not mask.any():
        valid_moves = currentgame == 0
        states = states.append(pd.DataFrame(
                      {"grid" : [currentgame.copy()],
                       "weights" : [np.full(9,10) * valid_moves]}),
                        ignore_index=True)
        indx = states.index[-1]
    else:
        indx = states[mask].index[0]
        currentgame = states[mask]['grid'][indx].copy()
        
    # check if any moves can be made (quits if it will lose)
    maximum = states["weights"][indx].sum()
    
    # make a move
    choice = states["weights"][indx].cumsum().searchsorted(randint(1,maximum))
    
    return ((indx,choice), states)

def valid_grid(same_grid, states):
    return any(all(states == this_grid) for this_grid in same_grid)
    
def check_winner(states):
    gamegrid= states.reshape(3,3)
    victory1 = np.array([1,1,1])
    victory2 = np.array([2,2,2])
    
    if np.all(gamegrid.diagonal() == victory1) or np.all(gamegrid[::-1].diagonal() == victory1):
        return 1
    elif np.all(gamegrid.diagonal() == victory2) or np.all(gamegrid[::-1].diagonal() == victory2):
        return 2
    
    for i in range(3):
        if np.all(gamegrid[i] == victory1) or np.all(gamegrid.transpose()[i] == victory1):
            return 1
        elif np.all(gamegrid[i] == victory2) or np.all(gamegrid.transpose()[i] == victory2):
            return 2
    
    return 0
