import numpy as np

def minKnightMoves(src, dest, moves=0):
    """
    function to find minimum number of moves for a knight to move from one
    position on a chess board to any other position.

    Parameters
    ----------
    src : int, list
        starting position on a chessboard (between 0 and 63, row major).
    dest : int
        target position on a chessboard (between 0 and 63, row major).
    moves : int, optional
        number of moves made. The default is 0.

    Returns
    -------
    moves : int
        number of moves made.

    """
    # check if starting on target location
    if src == dest:
        return 0
    
    # check if input location is an integer and make list
    if type(src) == type(0):
        src = [src]
    
    # increment and find all valid moves
    moves += 1
    valid = set(i for j in list(map(valid_moves, src)) for i in j)
    
    # check for target square, else drop a level in recursion
    if dest in valid:
        return moves
    else:
        return minKnightMoves(valid, dest, moves)


def valid_moves(src):
    """
    Find all valid moves a knight can make on a chessboard

    Parameters
    ----------
    src : int
        knight's position on chessboard.

    Returns
    -------
    valid : tuple
        all legal moves a knight can make from its current position.

    """
    valid = []
    # turn linear position to grid position
    position = (src//8,src%8)
    grid = np.array(range(64)).reshape(8,8)
    all_moves = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]
    
    # find all valid moves
    for i in all_moves:
        thismove = list(map(sum,zip(position,i)))
        if any(list(map(lambda x: 0>x or 7<x, thismove))):
            continue
        else:
            valid.append(grid[thismove[0]][thismove[1]])
    return valid
