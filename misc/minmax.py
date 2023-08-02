import numpy as np

MARKS = {0: "X", 1: "O"}


class Board:
    def __init__(self):
        self.state = [None] * 9
        self.counter = 0

    def render(self):
        text = """
0|1|2
-----
3|4|5
-----
6|7|8
"""
        for idx, x in enumerate(self.state):
            if x is not None:
                text = text.replace(str(idx), MARKS[x])  # 4 -> X
        print(text)

    def move(self, idx):
        if self.state[idx] is not None:
            return False

        player = self.counter % 2
        self.state[idx] = player
        self.counter += 1
        return True

    def unmove(self, idx):
        self.counter -= 1
        self.state[idx] = None

    def is_win(self, player):
        s = self.state
        if (
            s[0] == s[1] == s[2] == player
            or s[3] == s[4] == s[5] == player
            or s[6] == s[7] == s[8] == player
            or s[0] == s[3] == s[6] == player
            or s[1] == s[4] == s[7] == player
            or s[2] == s[5] == s[8] == player
            or s[0] == s[4] == s[8] == player
            or s[2] == s[4] == s[6] == player
        ):
            return True
        return False

    def is_end(self):
        if None in self.state:
            return False
        return True

    def valid_moves(self):
        moves = []
        for idx, player in enumerate(self.state):
            if player is None:
                moves.append(idx)
        return moves


class RandomPlayer:
    def play(self, board):
        moves = board.valid_moves()
        idx = np.random.choice(moves)
        print("ランダムプレイヤー：", idx)
        board.move(idx)


class BetterPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, board):
        moves = board.valid_moves()

        for idx in moves:
            board.move(idx)
            if board.is_win(self.player):
                return
            # unmove
            board.unmove(idx)

        idx = np.random.choice(moves)
        print("少し賢いプレイヤー：", idx)
        board.move(idx)


def minimax(board, player):
    maximize_player = 0
    minimize_player = 1

    if board.is_win(maximize_player):
        return (1, None)
    elif board.is_win(minimize_player):
        return (-1, None)
    elif board.is_end():
        return (0, None)

    opp = 1 if player == 0 else 0

    if player == maximize_player:
        max_score = -np.inf
        max_idx = None

        for idx in board.valid_moves():
            board.move(idx)
            score, next_idx = minimax(board, opp)
            if max_score < score:
                max_score = score
                max_idx = idx
            board.unmove(idx)

        return (max_score, max_idx)
    else:
        min_score = np.inf
        min_idx = None

        for idx in board.valid_moves():
            board.move(idx)
            score, next_idx = minimax(board, opp)
            if min_score > score:
                min_score = score
                min_idx = idx
            board.unmove(idx)

        return (min_score, min_idx)


class BestPlayer:
    def __init__(self, player):
        self.player = player

    def play(self, board):
        score, idx = minimax(board, self.player)
        print("最強のAI：", idx)
        board.move(idx)


class HumanPlayer:
    def play(self, board):
        while True:
            print("0~8の数字を入力してください：", end="")
            idx = input()

            try:
                idx = int(idx)
                success = board.move(idx)
                if success:
                    break
                else:
                    print("適切な数字を入力してください")
            except ValueError:
                pass


board = Board()
players = [BestPlayer(0), HumanPlayer()]
player = 0  # 0 or 1

while True:
    p = players[player]
    p.play(board)
    board.render()

    if board.is_win(player):
        print(MARKS[player] + "の勝ち！")
        break
    elif board.is_end():
        print("引き分け")
        break

    player = 1 if player == 0 else 0
