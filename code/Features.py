import chess
import numpy as np
import pandas as pd
import torch


class board_features():

    def __init__(self, fen):
        self.position = chess.Board(fen)

    def feature_legalcaptures(self, board):
        x = board.legal_moves
        counter = 0
        for m in x:
            if board.is_capture(m):
                counter += 1
        return counter

    def collect_pieces(self, board, color):
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        p = set()
        p.update(board.pieces(chess.KNIGHT, color))
        p.update(board.pieces(chess.BISHOP, color))
        p.update(board.pieces(chess.ROOK, color))
        p.update(board.pieces(chess.PAWN, color))
        p.update(board.pieces(chess.QUEEN, color))
        p.update(board.pieces(chess.KING, color))
        return p

    def feature_pieces(self, board, color):
        pieces = self.collect_pieces(board, color)
        return len(pieces)

    def feature_material(self, board, color):
        pieces = self.collect_pieces(board, color)
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        points = 0
        for p in pieces:
            if board.piece_type_at(p) == 1:
                points += 1
            elif board.piece_type_at(p) == 4:
                points += 5
            elif board.piece_type_at(p) == 2 or board.piece_type_at(p) == 3:
                points += 3
            elif board.piece_type_at(p) == 5:
                points += 10
        return points

    def feature_kingfreedom(self, board, color):
        # counts legal moves by king
        count = 0
        board.turn = color
        moves = board.legal_moves
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        k = board.king(color)
        for i in moves:
            if i.from_square == k:
                count += 1
        return count

    def feature_kingquadrant_attacks(self, board, color, other):
        moves_player = set([x.to_square for x in board.legal_moves])
        board.turn = other
        moves_adversary = set([x.to_square for x in board.legal_moves])
        board.turn = color
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        k = board.king(color)
        p = set()
        a = set()
        for move in moves_player:
            if chess.square_distance(k, move) <= 2:
                p.add(move)
        for move in moves_adversary:
            if chess.square_distance(k, move) <= 2:
                a.add(move)
        #c = p.intersection(a)
        return len(p), len(a) #, len(c)

    def feature_defended(self, board, color):
        # function to check what proportion of pieces are defended
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        pieces = self.collect_pieces(board, color)
        count = 0
        for p in pieces:
            if board.is_attacked_by(color, p):
                count += 1
        count = count / len(list(pieces))
        return count

    def feature_attackbalance(self, board, color, other):
        board.turn = color
        moves_player = set([x.to_square for x in board.legal_moves])
        board.turn = other
        moves_adversary = set([x.to_square for x in board.legal_moves])
        board.turn = color
        contested = moves_player.intersection(moves_adversary)
        return len(moves_player), len(moves_adversary), len(contested)

    def feature_pawnstructure(self, board, color):
        # an algorithm for counting pawn islands by parsing the pawns from left to right,
        # and counting only when we get to the end of a protective structure
        #base = chess.BaseBoard(board.fen().split(" ", 1)[0])
        count = 0
        pawns = list(board.pieces(chess.PAWN, color))
        if len(pawns) > 0:
            count = 1
            pawns.sort(key=lambda x: x % 8)
            for p in pawns:
                if (p + 9 not in pawns) and (p - 7 not in pawns) and (p + 1 not in pawns) and (p % 8 != 7):
                    count += 1
        return count

    def feature_concentration(self, board, color):
        # function to take average distances between pieces of a side
        # adds up the distances and divides by {n choose 2}
        if color == "both":
            x = list(self.collect_pieces(board, chess.WHITE)) + list(self.collect_pieces(board, chess.BLACK))
        else:
            x = list(self.collect_pieces(board, color))
        count = 0
        for w in x:
            for z in x:
                count = count + chess.square_distance(w, z)
        if len(x) > 1:
            count = count / (len(x) * (len(x) - 1))
        else:
            count = 0
        return count

    def extract(self):
        board = self.position
        color = board.turn
        if color == chess.WHITE:
            other = chess.BLACK
        else:
            other = chess.WHITE
        features = []
        # features 0, 1: color indicator one-hot, first black then white
        #features.append(int(color == chess.BLACK))
        features.append(int(color == chess.WHITE))
        # Concept: "Stage of Game"
        # feature 2: obtain plies from start
        features.append(board.ply())
        # Concept: "Material Advantage"
        # features 3, 4 obtain number of pieces on board for each side, absolute
        features.append(self.feature_pieces(board, color))
        features.append(self.feature_pieces(board, other))
        # features 5, 6 obtain material balance by points
        features.append(self.feature_material(board, color))
        features.append(self.feature_material(board, other))
        # Concept: "Positional Advantage"
        # features 7, 8: number of pawn islands on each side
        features.append(self.feature_pawnstructure(board, color))
        features.append(self.feature_pawnstructure(board, other))
        # feature 9, 10: what proportion of pieces are defended
        features.append(self.feature_defended(board, color))
        features.append(self.feature_defended(board, other))
        # features 11, 12: concentration of forces (average distance between pieces), on each side and in general
        features.append(self.feature_concentration(board, color))
        features.append(self.feature_concentration(board, other))
        #features.append(self.feature_concentration(board, "both"))
        # Concept: "Maneuverability"
        # feature 13, 14: number of legal moves for each side
        features.append(board.legal_moves.count())
        board.turn = other
        features.append(board.legal_moves.count())
        board.turn = color
        # Concept: "Control"
        # features 15-16: contested squares and number of squares that each side attacks
        x, y, z = self.feature_attackbalance(board, color, other)
        features.append(x)
        features.append(y)
        # Concept: "King Safety"
        # features 17, 18: attacks near opponent king
        x, y = self.feature_kingquadrant_attacks(board, color, other)
        #features.append(x)
        features.append(y)
        x, y = self.feature_kingquadrant_attacks(board, other, color)
        #features.append(x)
        features.append(y)
        # features 19, 20: king freedom
        features.append(self.feature_kingfreedom(board, color))
        features.append(self.feature_kingfreedom(board, other))
        return features


class move_features():

    def __init__(self, board, move):
        self.board = chess.Board(board)
        self.uci = move
        self.move = chess.Move.from_uci(move)
        self.color = self.board.turn

    def extract(self):
        # extract features to do with engines' move recommendations
        board = self.board
        move = self.move
        features = []
        # target square in coordinates
        features.append(chess.square_rank(move.to_square))
        features.append(chess.square_file(move.to_square))
        # feature: the distance of the player's move suggestions
        features.append(chess.square_distance(move.from_square, move.to_square))
        # feature: how much move approaches opposing king
        if self.color == chess.WHITE:
            features.append(chess.square_distance(move.to_square, board.king(chess.BLACK)) - chess.square_distance(
                move.from_square, board.king(chess.BLACK)))
        else:
            features.append(chess.square_distance(move.to_square, board.king(chess.WHITE)) - chess.square_distance(
                move.from_square, board.king(chess.WHITE)))
        # feature: whether the recommended move is backward or not
        if self.color == chess.WHITE:
            features.append(1 if (chess.square_rank(move.to_square) - chess.square_rank(move.from_square) < 0) else 0)
        else:
            features.append(1 if (chess.square_rank(move.from_square) - chess.square_rank(move.to_square) < 0) else 0)
        # feature: whether it is a horizontal move
        features.append(1 if (chess.square_rank(move.from_square) == chess.square_rank(move.to_square)) else 0)
        # feature: whether the recommended moves are flanking actions
        features.append(int((chess.square_file(move.from_square) <= 3 and chess.square_file(move.to_square) <= 3) or
                            (chess.square_file(move.from_square) >= 6 and chess.square_file(move.to_square) >= 6)))
        # feature: which piece is recommended to move, as one hot
        # pawn
        if board.piece_at(move.from_square).piece_type == 1:
            features.append(1)
        else:
            features.append(0)
        # rook
        if board.piece_at(move.from_square).piece_type == 4:
            features.append(1)
        else:
            features.append(0)
        # knight
        if board.piece_at(move.from_square).piece_type == 2:
            features.append(1)
        else:
            features.append(0)
        # bishop
        if board.piece_at(move.from_square).piece_type == 3:
            features.append(1)
        else:
            features.append(0)
        # queen
        if board.piece_at(move.from_square).piece_type == 5:
            features.append(1)
        else:
            features.append(0)
        # king
        if board.piece_at(move.from_square).piece_type == 6:
            features.append(1)
        else:
            features.append(0)
        # feature: whether move puts opponent in check
        features.append(int(board.gives_check(move)))
        # feature: whether move is a capture
        features.append(int(board.is_capture(move)))
        # features: whether move is a castling
        if board.is_castling(move):
            features.append(board.ply())
        else:
            features.append(0)
        return features


def make_features(file, player1, player2):
    df = pd.read_csv(f"{file}.csv")
    #df = df.sample(n=10000)
    print("data points: ", len(df.index))
    print("extracting features")
    data = df.progress_apply(lambda z: [z["position"]] + board_features(z['position']).extract() + [1 if (z[player1] > z[player2]) else 0], axis=1).to_list()
    data = pd.DataFrame([{z: w[z] for z in range(len(w))} for w in data])
    data.rename(columns={"0": "position", "1": "color", "2": "ply", "3": "num_pieces", "4": "num_pieces_opponent",
                       "5": "material_points", "6": "material_points_opponent",
                       "7": "pawn_islands", "8": "pawn_islands_opponent",
                       "9": "defended", "10": "defended_opponent", 
                       "11": "concentration", "12": "concentration_opponent",
                       "13": "legal_moves", "14": "legal_moves_opponent",
                       "15": "attacks", "16": "attacks_opponent",
                       "17": "attacks_nearking", "18": "attacks_nearking_opponent",
                       "19": "king_freedom", "20": "king_freedom_opponent", "21": "eval"}, inplace=True)
    data.to_csv(f"{file}_features.csv", index=False)
    return

