"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This should be the best heuristic function for your project submission.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # Legal moves of my move "Minus" legal moves of opponent move
    
    #Check if it is end game
    if game.is_winner(player):
        return float("inf")
    if game.is_loser(player):
        return float("-inf")
    #if it is not end game, calculate score
    number_of_my_moves_left = len(game.get_legal_moves(player)) #Use len to extract list of all legal moves
    number_of_opponent_moves_left  = len(game.get_legal_moves(game.get_opponent(player))) #Use .get_opponent to return opponent of the supplied player
    return float(number_of_my_moves_left - number_of_opponent_moves_left)




def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    raise NotImplementedError


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.
    ********************  DO NOT MODIFY THIS CLASS  ********************
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)
    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.
    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************
        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.
        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        def min_play(game,depth):
            if self.time_left() < self.TIMER_THRESHOLD: #Insert time constraint in each function
                raise SearchTimeout()
            if len(game.get_legal_moves())==0 or depth==0:
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()    #Retrieve Legal move
                bestscore=float("inf")                  #Initialized some initial value to inf
                for move in legal_moves:                #Find it in every move
                    score=max_play(game.forecast_move(move), depth-1) #Look it deeper by one level.
                    if score<bestscore:                 
                        bestscore=score                 #best score is the lower score
                return bestscore
        def max_play(game,depth):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if len(game.get_legal_moves())==0 or depth==0:
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()    #Retrieve Legal move
                bestscore=float("-inf")
                for move in legal_moves:
                    score=min_play(game.forecast_move(move), depth-1)
                    if score>bestscore:
                        bestscore=score
                return bestscore
        legal_moves = game.get_legal_moves()    #Retrieve Legal move
        bestmove = (-1,-1)                      #Set some initial tuples
        if len(game.get_legal_moves())==0:
            return bestmove
        bestscore=float("-inf")                 #Set some initial value
        for move in legal_moves:
            clone=game.forecast_move(move)      #Forecast move given each move
            score=min_play(clone,depth-1)             #Get the score of the shallowest depth 
            if score>=bestscore:               #update score for best move
                bestscore=score
                bestmove=move
        return bestmove

 
class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.
        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.
        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************
        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # TODO: finish this function!

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.alphabeta(game,self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # TODO: finish this function!

        def max_play(game,depth,alpha,beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if len(game.get_legal_moves())==0 or depth==0:
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()    #Retrieve Legal move
                bestscore=float("-inf")
                for move in legal_moves:
                    score=min_play(game.forecast_move(move),depth-1,alpha,beta)
                    bestscore=max(bestscore,score)
                    if bestscore>=beta:
                        return bestscore
                    alpha=max(alpha,bestscore)  #4. At max player, this alpha will be local max branch alpha. 
                    #It will not update upwards to the shallower depth. 
                    #Instead, this alpha value will be used for min player BELOW them in manner that if those min player could achieve their bestscore below this local alpha, 
                    #those resulting bestscore from min player below this local max branch will never be selected because
                    # those score min value will be lower than alpha (best score for max player at EITHER local or top level) and thus, the max player(Either this local max player or top level max player) will never choose it.
                return bestscore
        def min_play(game,depth,alpha,beta):
            if self.time_left() < self.TIMER_THRESHOLD: #Insert time constraint in each function
                raise SearchTimeout()
            if len(game.get_legal_moves())==0 or depth==0:
                return self.score(game,self)
            else:
                legal_moves = game.get_legal_moves()    #Retrieve Legal move
                bestscore=float("inf")                  #Initialized some initial value to inf
                for move in legal_moves:                #Find it in every move
                    score=max_play(game.forecast_move(move),depth-1,alpha,beta) #Look it deeper by one level.
                    bestscore=min(bestscore,score)
                    if bestscore<=alpha:            #2. At min player, if best score is less than alpha (which came from above max player), it means max player(which is above) will never choose this branch. So, prune out!!!
                        return bestscore
                    beta=min(beta,bestscore)        #3. At min player, beta value will  kept update and will be used for max player at its lower branch. 
                    #It will be used in case if max branch below it can get value equal or exceed beta, 
                    #those max branch will get value higher than beta (or best lowest score for min player). 
                    #Therefore, we can pruned out those max player beneath it.
                return bestscore
        legal_moves = game.get_legal_moves()    #Retrieve Legal move
        bestmove = (-1,-1)                      #Set some initial tuples
        if len(game.get_legal_moves())==0:
            return bestmove
        bestscore=float("-inf")                 #Set some initial value
        beta=float("inf")
        for move in legal_moves:
            clone=game.forecast_move(move)      #Forecast move given each move
            score=min_play(clone,depth-1,bestscore,beta)             #1.The best score will be plugged in as alpha(Permanent, increase only) from left most branch
            #Notably, depth is count as starting at 3, the going deep in will be 2 then 1 then 0.
            if score>=bestscore:               #update score for best move
                bestscore=score
                bestmove=move
        return bestmove
