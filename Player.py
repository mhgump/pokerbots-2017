# import site
# site.addsitedir('./venv/lib/python2.7/site-packages/')
import sys
import argparse
import socket
import numpy as np
import time
import operator
import sys
import glob
import math
from time import time
suit_index_dict = {"s": 0, "c": 1, "h": 2, "d": 3}
reverse_suit_index = ("s", "c", "h", "d")
val_string = "AKQJT98765432"
suit_value_dict = {"T": 10, "J": 11, "Q": 12, "K": 13, "A": 14}
for num in xrange(2, 10):
    suit_value_dict[str(num)] = num
"""
Simple example pokerbot, written in python.

This is an example of a bare bones pokerbot. It only sets up the socket
necessary to connect with the engine and then always returns the same action.
It is meant as an example of how a pokerbot should communicate with the engine.
"""
def sigfunc(x):
  return 1 / (1 + math.exp(-x))
sigmoid = np.vectorize(sigfunc)
class Card:
    # Takes in strings of the format: "As", "Tc", "6d"
    def __init__(self, card_string):
        value, self.suit = card_string[0], card_string[1]
        self.value = suit_value_dict[value]
        self.suit_index = suit_index_dict[self.suit]

    def __str__(self):
        return val_string[14 - self.value] + self.suit

    def __repr__(self):
        return val_string[14 - self.value] + self.suit

    def __eq__(self, other):
        return self.value == other.value and self.suit == other.suit

class GRUTheano:
    
    def __init__(self, x_dim,y_dim, path,hidden_dim=64, bptt_truncate=-1):
        # Assign instance variables
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        npzfile = np.load(path)
        E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
        # Theano: Created shared variables
        self.E = E
        self.U = U
        self.W = W
        self.V = V
        self.b = b
        self.c = c

    
    def forward_prop_step(self,x_t, s_t1_prev, s_t2_prev):
        # This is how we calculated the hidden state in a simple RNN. No longer!
        # s_t = T.tanh(U[:,x_t] + W.dot(s_t1_prev))
        
        # Word embedding layer
        x_e = self.E.dot(x_t)
        
        # GRU Layer 1
        z_t1 = sigmoid(self.U[0].dot(x_e) + self.W[0].dot(s_t1_prev) + self.b[0])
        r_t1 = sigmoid(self.U[1].dot(x_e) + self.W[1].dot(s_t1_prev) + self.b[1])
        c_t1 = np.tanh(self.U[2].dot(x_e) + self.W[2].dot(s_t1_prev * r_t1) + self.b[2])
        s_t1 = (np.ones(z_t1.shape) - z_t1) * c_t1 + z_t1 * s_t1_prev
        
        # GRU Layer 2
        z_t2 = sigmoid(self.U[3].dot(s_t1) + self.W[3].dot(s_t2_prev) + self.b[3])
        r_t2 = sigmoid(self.U[4].dot(s_t1) + self.W[4].dot(s_t2_prev) + self.b[4])
        c_t2 = np.tanh(self.U[5].dot(s_t1) + self.W[5].dot(s_t2_prev * r_t2) + self.b[5])
        s_t2 = (np.ones(z_t2.shape) - z_t2) * c_t2 + z_t2 * s_t2_prev
        
        # Final output calculation
        o_t = self.V.dot(s_t2) + self.c

        return [o_t, s_t1, s_t2]

def card_to_num(card):
    return 'sdch'.index(card[1])*13+'23456789TJQKA'.index(card[0])
def num_to_card(num):
    return '23456789TJQKA'[num%13]+'sdch'[num/13]
opp_bet_0_25        = np.eye(8)[0]
opp_bet_25_50      = np.eye(8)[1]
opp_bet_50_75    = np.eye(8)[2]
opp_bet_75_100   = np.eye(8)[3]
opp_bet_100_125   = np.eye(8)[4]
opp_bet_125_150  = np.eye(8)[5]
opp_bet_150_175 = np.eye(8)[6]
opp_bet_175_200 = np.eye(8)[7]

bet_0_25    = np.eye(8)[0]
bet_25_50    = np.eye(8)[1]
bet_50_75    = np.eye(8)[2]
bet_75_100   = np.eye(8)[3]
bet_100_125   = np.eye(8)[4]
bet_125_150  = np.eye(8)[5]
bet_150_175 = np.eye(8)[6]
bet_175_200 = np.eye(8)[7]
def bet_to_action_opp(bet):
    if bet<25:
        return opp_bet_0_25
    elif bet<50:
        return opp_bet_25_50
    elif bet<75:
        return opp_bet_50_75
    elif bet<100:
        return opp_bet_75_100
    elif bet<125:
        return opp_bet_100_125
    elif bet<150:
        return opp_bet_125_150
    elif bet<175:
        return opp_bet_150_175
    else:
        return opp_bet_175_200

def action_to_bet_range(index):
    if index==0:
        return (0,25)
    elif index==1:
        return (25,50)
    elif index==2:  
        return (50,75)
    elif index==3:
        return (75,100)
    elif index==4:
        return (100,125)
    elif index==5:
        return (125,150)
    elif index==6:
        return (150,175)
    else:
        return (175,205)    
    
class Player:
    def __init__(self):

        self.holeCard1 = None
        self.holeCard2 = None
        self.boardCards = None
        self.yourName = None
        self.otherName = None
        self.last_action = bet_0_25
        self.minBet = 0
        self.maxBet = 0
        self.last_discard = 0
        self.opponent_action = opp_bet_0_25
        self.states = []
        self.rewards = []
        self.last_board_size = 0
   
    def run(self, input_socket):
        # Get a file-object for reading packets from the socket.
        # Using this ensures that you get exactly one packet per read.
        f_in = input_socket.makefile()
        model = GRUTheano(120,8,'./model.npz')
        s_t1 = np.zeros(64)
        s_t2 = np.zeros(64)
        prob_random = 20
        while True:
            start_time = time()
            # Block until the engine sends us a packet.
            data = f_in.readline().strip()
            # If data is None, connection has closed.
            if not data:
                print "Gameover, engine disconnected."
                break

            # When appropriate, reply to the engine with a legal action.
            # The engine will ignore all spurious responses.
            # The engine will also check/fold for you if you return an
            # illegal action.
            # When sending responses, terminate each response with a newline
            # character (\n) or your bot will hang!
            word = data.split()[0]
            data = data.split()
            print data

            if word == "GETACTION":
                # Currently CHECK on every move. You'll want to change this.s

                cancall = False
                canfold = False
                cancheck = False
                canbet = False
                canraise = False
                to_call = 0
                
                # Get relevant information from engine
                self.potSize = data[1]
                self.numBoardCards = int(data[2])
                board_card_increment = 3+self.numBoardCards
                self.boardCards = [data[i] for i in range(3,board_card_increment)]
                self.numPrevActions = int(data[board_card_increment])
                prev_action_increment = board_card_increment+1+self.numPrevActions             
                self.prevActions = [data[i] for i in range(board_card_increment+1,prev_action_increment)]
                
                # Handle discards
                if self.last_discard != 0:
                    for action in self.prevActions:
                        action_split = action.split(':')
                        if len(action_split) == 4:
                            if self.last_discard==1:
                                self.holeCard1=action_split[2]
                            if self.last_discard==2:
                                self.holeCard2=action_split[2]
                
                # Get Previous Opponent Bet if there was one
                for action in self.prevActions:
                    action_split = action.split(':')
                    if self.otherName in action:
                        if 'BET' in action or 'RAISE' in action:
                            raised = int(action_split[1])
                            committed = float(int(self.potSize)-raised)/2
                            to_call = raised
                            self.opponent_action=bet_to_action_opp(committed+raised)
                            
                self.numLegalActions = int(data[prev_action_increment])
                legal_action_increment = prev_action_increment+1
                legal_action_increment_end = legal_action_increment+self.numLegalActions
                self.legalActions = [data[i] for i in range(legal_action_increment,legal_action_increment_end)]

                # Figure out legal actions
                discard_round = False
                for action in self.legalActions:
                    if 'BET' in action:
                        action_split = action.split(':')
                        canbet = True
                        self.bet = 1
                        self.minBet=int(action_split[1])
                        self.maxBet=int(action_split[2])
                    if 'RAISE' in action:
                        action_split = action.split(':')
                        canraise= True
                        self.bet = 2
                        self.minBet=int(action_split[1])
                        self.maxBet=int(action_split[2])
                    if 'DISCARD' in action:
                        discard_round = True
                    if 'FOLD' in action:
                        canfold = True
                    if 'CHECK' in action:
                        cancheck = True
                    if 'CALL' in action:
                        cancall = True
                        
                textaction = 'CHECK\n' # If all logic calls through the player should check
                if discard_round:
                    # Calculations for discard equity
                    hand_probs = return_probs([self.holeCard1,self.holeCard2]+self.boardCards)
                    drop_1_probs = return_probs([self.holeCard2]+self.boardCards)
                    drop_2_probs = return_probs([self.holeCard1]+self.boardCards)
                    opponent_probs =  return_probs(self.boardCards,taken=[self.holeCard1,self.holeCard2])

                    hand_equity = matchup(hand_probs,opponent_probs)
                    drop_1_equity = matchup(drop_1_probs,opponent_probs)
                    drop_2_equity = matchup(drop_2_probs,opponent_probs)

                    hand_equity_ratio = float(hand_equity[0]-hand_equity[1])+.1
                    drop_1_equity_ratio = float(drop_1_equity[0]-drop_1_equity[1])
                    drop_2_equity_ratio = float(drop_2_equity[0]-drop_2_equity[1]) 
                    decision = np.argmax([hand_equity_ratio,drop_1_equity_ratio,drop_2_equity_ratio])
                    if decision==2:
                        textaction = "DISCARD:{}\n".format(self.holeCard2)
                        self.last_discard=2
                    elif decision==1:
                        textaction = "DISCARD:{}\n".format(self.holeCard1)
                        self.last_discard=1
                    else:
                        textaction = "CHECK\n"
                        self.last_discard=0    
                else:
                    action = 0

                    # Initialize state
                    player_1_cards = np.zeros(52)
                    board          = np.zeros(52)
                    player_1_cards += np.eye(52)[card_to_num(self.holeCard1)]
                    player_1_cards += np.eye(52)[card_to_num(self.holeCard2)]

                    for card in self.boardCards:
                        board += np.eye(52)[card_to_num(card)]

                    opponent_action = self.opponent_action
                    state = np.append(np.append(np.append(player_1_cards,board),opponent_action),self.last_action)
                    prediction = np.ones(8)

                    prediction,s_t1,s_t2 = model.forward_prop_step(state,s_t1,s_t2)
                    action = np.argmax(prediction)
                    # Make prediction, during training, make a random action some percentage of the time
#                     if np.random.randint(100)<random_prob:
#                         action = np.random.randint(8)

                    self.last_action = np.eye(8)[action]
                    if len(self.states)>0:
                        self.states += [state]
                        left_in_hand = 0
                        # if self.last_board_size != len(self.boardCards) and self.last_board_size!=0:
                        #     left_in_hand = (200-float(self.potSize)/2)*(1./len(self.boardCards))
                        #     self.last_board_size==self.boardCards
                        self.rewards += [(0,action,False)]
                    else:
                        self.states = [state]
                        left_in_hand = 0
                        # if self.last_board_size != len(self.boardCards) and self.last_board_size!=0:
                        #     left_in_hand = (200-float(self.potSize)/2)*(1./len(self.boardCards))
                        #     self.last_board_size==self.boardCards
                        self.rewards = [(0,action,False)]
                    rangemin,rangemax = action_to_bet_range(action)
                    committed = float(int(self.potSize)-to_call)/2
                    print action
                    for e in prediction:
                        print e
                    # Logic to allow bot to commit on the position it has decided on
                    if cancall:
                        if canfold and (committed+float(to_call))>rangemax:
                            textaction = 'FOLD\n'
                        elif canraise and rangemax-committed-to_call>self.minBet:
                            bet_low = max(rangemin,committed)-committed
                            bet_high = rangemax-committed
                            bet_high = min(self.maxBet,bet_high)
                            bet_low = max(self.minBet,bet_low)
                            bet= np.random.randint(bet_low,bet_high) if bet_low!=bet_high else bet_low
                            textaction = 'RAISE:{}\n'.format(bet)
                        else:
                            textaction = 'CALL\n'
                    else:
                        if canbet and rangemax-committed-to_call>self.minBet:
                            bet_low = max(rangemin,committed)-committed
                            bet_high = rangemax-committed
                            bet_high = min(self.maxBet,bet_high)
                            bet_low = max(self.minBet,bet_low)
                            bet= np.random.randint(bet_low,bet_high) if bet_low!=bet_high else bet_low
                            textaction = 'BET:{}\n'.format(bet)
                        else:
                            textaction = 'CHECK\n'
                s.send(textaction)
            elif word == "NEWHAND":
                self.holeCard1 = data[3]
                self.holeCard2 = data[4]
                self.last_board_size = 0
                self.last_action = bet_0_25
                self.opponent_action= opp_bet_0_25
            elif word == "NEWGAME":
                self.yourName = data[1]
                self.otherName = data[2]
            elif word == "REQUESTKEYVALUES":
                # At the end, the engine will allow your bot save key/value pairs.
                # Send FINISH to indicate you're done.
                s.send("FINISH\n")
        # Clean up the socket.
        s.close()

def matchup(probs_1,probs_2):
    win = 0
    tie = 0
    for i in range(9):
        tie+=probs_1[i]*probs_2[i]
        win+=probs_1[i]*probs_2[:i].sum()
    return (win,1-win-tie,tie)
 
def return_probs(cards,taken=None,num_iterations=500):

    result_histograms = [0] * 9

    objectified_cards = [Card(arg) for arg in cards]

    deck = []
    for suit in reverse_suit_index:
        for ch in val_string:
            deck.append(Card(ch + suit))

    for card in objectified_cards:
        deck.remove(card)

    if taken != None:
        objectified_taken = [Card(arg) for arg in taken]
        for card in objectified_taken:
            deck.remove(card)

    for extra in generate_random_boards(deck, num_iterations, 7-len(cards)):
        board = extra+objectified_cards
        suit_histogram, histogram, max_suit = preprocess_board(board)
        result = fast_detect_hand(board,suit_histogram, histogram, max_suit)
        result_histograms[result]+=1

    return np.array(result_histograms,dtype=float)/num_iterations

def fast_detect_hand(board, suit_histogram,full_histogram, max_suit): 

    if max_suit >= 5:
        flush_index = suit_histogram.index(max_suit)
        flat_board = list(board)
        suit_board = generate_suit_board(flat_board, flush_index)
        result = detect_straight_flush(suit_board)
        if result[0]:
            return 8
        return 5

    histogram_board = preprocess(full_histogram)

    # Find which card value shows up the most and second most times
    current_max, max_val, second_max, second_max_val = 0, 0, 0, 0
    for item in histogram_board:
        val, frequency = item[0], item[1]
        if frequency > current_max:
            second_max, second_max_val = current_max, max_val
            current_max, max_val = frequency, val
        elif frequency > second_max:
            second_max, second_max_val = frequency, val

    # Check to see if there is a four of a kind
    if current_max == 4:
        return 7
    # Check to see if there is a full house
    if current_max == 3 and second_max >= 2:
        return 6
    # Check to see if there is a straight
    if len(histogram_board) >= 5:
        result = detect_straight(histogram_board)
        if result[0]:
            return 4
    # Check to see if there is a three of a kind
    if current_max == 3:
        return 3
    if current_max == 2:
        # Check to see if there is a two pair
        if second_max == 2:
            return 2
        # Return pair
        else:
            return 1
    # Check for high cards
    return 0  

# Takes an iterable sequence and returns two items in a tuple:
# 1: 4-long list showing how often each card suit appears in the sequence
# 2: 13-long list showing how often each card value appears in the sequence
def preprocess_board(flat_board):
    suit_histogram, histogram = [0] * 4, [0] * 13
    # Reversing the order in histogram so in the future, we can traverse
    # starting from index 0
    for card in flat_board:
        histogram[14 - card.value] += 1
        suit_histogram[card.suit_index] += 1
    return suit_histogram, histogram, max(suit_histogram)

def generate_random_boards(deck, num_iterations, remaining):
    import random
    import time
    random.seed(time.time())
    for iteration in xrange(num_iterations):
        yield random.sample(deck, remaining)
# Returns a board of cards all with suit = flush_index
def generate_suit_board(flat_board, flush_index):
    histogram = [card.value for card in flat_board
                                if card.suit_index == flush_index]
    histogram.sort(reverse=True)
    return histogram
# Returns tuple: (Is there a straight flush?, high card)
def detect_straight_flush(suit_board):
    contiguous_length, fail_index = 1, len(suit_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(suit_board):
        current_val, next_val = elem, suit_board[index + 1]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 5 and
                                                    suit_board[0] == 14):
                    return True, 5
                break
            contiguous_length = 1
    return False,

# Returns a list of two tuples of the form: (value of card, frequency of card)
def preprocess(histogram):
    return [(14 - index, frequency) for index, frequency in
                                        enumerate(histogram) if frequency]

def detect_straight(histogram_board):
    contiguous_length, fail_index = 1, len(histogram_board) - 5
    # Won't overflow list because we fail fast and check ahead
    for index, elem in enumerate(histogram_board):
        current_val, next_val = elem[0], histogram_board[index + 1][0]
        if next_val == current_val - 1:
            contiguous_length += 1
            if contiguous_length == 5:
                return True, current_val + 3
        else:
            # Fail fast if straight not possible
            if index >= fail_index:
                if (index == fail_index and next_val == 5 and
                                        histogram_board[0][0] == 14):
                    return True, 5
                break
            contiguous_length = 1
    return False,

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A Pokerbot.', add_help=False, prog='pokerbot')
    parser.add_argument('-h', dest='host', type=str, default='localhost', help='Host to connect to, defaults to localhost')
    parser.add_argument('port', metavar='PORT', type=int, help='Port on host to connect to')
    args = parser.parse_args()

    # Create a socket connection to the engine.
    print 'Connecting to %s:%d' % (args.host, args.port)
    try:
        s = socket.create_connection((args.host, args.port))
    except socket.error as e:
        print 'Error connecting! Aborting'
        exit()

    
    
    bot = Player()
    bot.run(s)
