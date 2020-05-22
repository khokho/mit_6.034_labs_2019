# MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1
import time

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    return board.num_rows*board.num_cols==board.count_pieces() or \
            len(max(board.get_all_chains(),key = len,default=[]))>=4

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []
    res = []
    for i in range(board.num_cols):
        if not board.is_column_full(i):
            res.append(board.add_piece(i))
    return res


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if len(max(board.get_all_chains(is_current_player_maximizer),key = len,default=[]))>=4:
        return 1000
    if len(max(board.get_all_chains(not is_current_player_maximizer),key = len,default=[]))>=4:
        return -1000
    return 0

def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    return int(endgame_score_connectfour(board, is_current_player_maximizer)*(2-board.count_pieces()/(board.num_rows*board.num_cols)))

def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    lens_me = list(map(lambda x:10**len(x),board.get_all_chains(is_current_player_maximizer)))
    lens_him = list(map(lambda x:10**len(x),board.get_all_chains(not is_current_player_maximizer)))
    return int(1000*(sum(lens_me)-sum(lens_him))/(1+sum(lens_me)+sum(lens_him)))

# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state) :
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    if state.is_game_over():
        return ([state], state.get_endgame_score(), 1)
    cnt=0
    max_best = None
    max_score = None
    for nxt in state.generate_next_states():
        best, score, curcnt = dfs_maximizing(nxt)
        cnt+=curcnt
        if max_score is None or max_score < score:
            max_score = score
            max_best = [state]+best
    return (max_best, max_score, cnt)





# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True) :
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    cnt=0
    max_best = None
    max_score = None
    for nxt in state.generate_next_states():
        best, score, curcnt = minimax_endgame_search(nxt, not maximize)
        if not maximize:
            score *= -1
        cnt+=curcnt
        if max_score is None or max_score < score:
            max_score = score
            max_best = [state]+best
    if not maximize:
        max_score *= -1
    return (max_best, max_score, cnt)


# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    if depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    cnt=0
    max_best = None
    max_score = None
    for nxt in state.generate_next_states():
        best, score, curcnt = minimax_search(nxt, heuristic_fn, depth_limit-1, not maximize)
        if not maximize:
            score *= -1
        cnt+=curcnt
        if max_score is None or max_score < score:
            max_score = score
            max_best = [state]+best
    if not maximize:
        max_score *= -1
    return (max_best, max_score, cnt)


# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=3))


def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type 
    as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    if depth_limit == 0:
        return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)
    cnt=0
    max_best = None
    max_score = None
    for nxt in state.generate_next_states():
        if alpha >= beta:
            break
        best, score, curcnt = minimax_search_alphabeta(nxt, alpha, beta, heuristic_fn, depth_limit-1, not maximize)
        if maximize:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)
        if not maximize:
            score *= -1
        cnt+=curcnt
        if max_score is None or max_score < score:
            max_score = score
            max_best = [state]+best
    if not maximize:
        max_score *= -1
    return (max_best, max_score, cnt)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    ret = AnytimeValue()
    tic = time.time()
    for i in range(1,depth_limit+1):
        cur = minimax_search_alphabeta(state,-INF,INF,heuristic_fn, i)
        ret.set_value(cur)
        toc = time.time()
        if toc - tic > time_limit:
            return ret
    return ret


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1000000, time_limit=10).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
