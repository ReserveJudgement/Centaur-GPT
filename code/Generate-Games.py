import chess
import chess.pgn
import chess.svg
from chess import engine
import numpy as np
import pandas as pd


# a global dictionary of engines and their paths
players = {"stockfish": "path_to_stockfish_engine",
           "lc0: "path_to_lc0_engine"}

# list of opening positions as FEN strings
openings = [
    "r1bq1rk1/pp2ppbp/2n2np1/2pp4/5P2/1P2PN2/PBPPB1PP/RN1Q1RK1 w - -",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/4P3/3P1NP1/PPP2PBP/RNBQ1RK1 b - -",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq -",
    "rn1qk2r/1b2bppp/pp1ppn2/8/2PQ4/2N2NP1/PP2PPBP/R1BR2K1 w kq -",
    "rn1qkb1r/3ppp1p/b4np1/2pP4/8/2N5/PP2PPPP/R1BQKBNR w KQkq -",
    "rnbqkb1r/pp3p1p/3p1np1/2pP4/4P3/2N5/PP3PPP/R1BQKBNR w KQkq -",
    "rnb1qrk1/ppp1p1bp/3p1np1/3P1p2/2P5/2N2NP1/PP2PPBP/R1BQ1RK1 b - -",
    "rnb1k2r/pp2q1pp/2pbpn2/3p1p2/2PP4/1P3NP1/P3PPBP/RNBQ1RK1 w kq -",
    "rnb1kb1r/pp2pppp/2p2n2/q7/2BP4/2N2N2/PPP2PPP/R1BQK2R b KQkq -",
    "rnbqkb1r/ppp1pppp/3p4/3nP3/3P4/5N2/PPP2PPP/RNBQKB1R b KQkq -",
    "rnbq1rk1/pp2ppbp/2pp1np1/8/3PP3/2N2N2/PPP1BPPP/R1BQ1RK1 w - -",
    "rnbq1rk1/ppp1ppbp/3p1np1/8/3PPP2/2N2N2/PPP3PP/R1BQKB1R w KQ -",
    "rn1qkbnr/pp3ppp/4p3/2ppPb2/3P4/5N2/PPP1BPPP/RNBQK2R w KQkq -",
    "rnbqkb1r/pp3ppp/4pn2/3p4/2PP4/2N2N2/PP3PPP/R1BQKB1R b KQkq -",
    "r2qkbnr/pp1nppp1/2p4p/7P/3P4/3Q1NN1/PPP2PP1/R1B1K2R b KQkq -",
    "rnb1kb1r/pp3ppp/4pn2/2pq4/3P4/2P2N2/PP3PPP/RNBQKB1R w KQkq -",
    "r1bq1rk1/pp2npbp/2npp1p1/2p5/4PP2/2NP1NP1/PPP3BP/R1BQ1RK1 w - -",
    "r1bqkb1r/5p1p/p1np4/1p1Npp2/4P3/N7/PPP2PPP/R2QKB1R w KQkq -",
    "r1bqkb1r/pp2pp1p/2np1np1/8/2PNP3/2N5/PP2BPPP/R1BQK2R b KQkq -",
    "r1bqkbnr/1p1p1ppp/p1n1p3/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq -",
    "r1bqkb1r/1p3pp1/p1nppn1p/6B1/3NP3/2N5/PPPQ1PPP/2KR1B1R w kq -",
    "2rq1rk1/pp1bppbp/3p1np1/4n3/3NP3/1BN1BP2/PPPQ2PP/2KR3R w - -",
    "rnbq1rk1/1p2bppp/p2ppn2/8/3NPP2/2N5/PPP1B1PP/R1BQ1RK1 w - -",
    "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq -",
    "r1b1kbnr/pp3ppp/1qn1p3/3pP3/2pP4/P1P2N2/1P3PPP/RNBQKB1R w KQkq -",
    "r1bqkb1r/pp1n1ppp/2n1p3/2ppP3/3P4/2PB4/PP1N1PPP/R1BQK1NR w KQkq -",
    "rnbqk2r/pp2nppp/4p3/2ppP3/3P4/P1P5/2P2PPP/R1BQKBNR w KQkq -",
    "rnbqk2r/ppp1bppp/3p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq -",
    "rnbqkb1r/ppp2ppp/8/3p4/3Pn3/3B1N2/PPP2PPP/RNBQK2R b KQkq -",
    "r1bqkbnr/pppp1ppp/2n5/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq -",
    "r1bqk2r/1pp2ppp/pbnp1n2/4p3/PPB1P3/2PP1N2/5PPP/RNBQK2R w KQkq -",
    "r1b1kbnr/1pp2ppp/p1p5/8/3NP3/8/PPP2PPP/RNB1K2R b KQkq -",
    "r1bqk2r/2pp1ppp/p1n2n2/1pb1p3/4P3/1B3N2/PPPP1PPP/RNBQ1RK1 w kq -",
    "r2qkb1r/2p2ppp/p1n1b3/1p1pP3/4n3/1B3N2/PPP2PPP/RNBQ1RK1 w kq -",
    "r2qr1k1/1bp1bppp/p1np1n2/1p2p3/3PP3/1BP2N1P/PP1N1PP1/R1BQR1K1 b - -",
    "rnbqkb1r/pp3ppp/4pn2/2pp4/3P4/2PBPN2/PP3PPP/RNBQK2R b KQkq -",
    "rn1qkb1r/pp2pppp/2p2n2/5b2/P1pP4/2N2N2/1P2PPPP/R1BQKB1R w KQkq -",
    "r1bq1rk1/pp2bppp/2n2n2/2pp2B1/3P4/2N2NP1/PP2PPBP/R2Q1RK1 b - -",
    "r1bqkb1r/pp1n1ppp/2p1pn2/3p4/2PP4/2NBPN2/PP3PPP/R1BQK2R b KQkq -",
    "rnbq1rk1/p1p1bpp1/1p2pn1p/3p4/2PP3B/2N1PN2/PP3PPP/R2QKB1R w KQ -",
    "rnbq1rk1/ppp1ppbp/6p1/3n4/3P4/5NP1/PP2PPBP/RNBQ1RK1 b - -",
    "rnbqk2r/pp2ppbp/6p1/2p5/2BPP3/2P5/P3NPPP/R1BQK2R b KQkq -",
    "rnbq1rk1/ppp1bppp/4pn2/3p4/2PP4/5NP1/PP2PPBP/RNBQK2R w KQ -",
    "rnb1k2r/ppppqppp/4pn2/8/1bPP4/5NP1/PP1BPP1P/RN1QKB1R b KQkq -",
    "rnbqkb1r/p1pp1ppp/1p2pn2/8/2PP4/5NP1/PP2PP1P/RNBQKB1R b KQkq -",
    "rnbq1rk1/pppp1ppp/4pn2/8/2PP4/P1Q5/1P2PPPP/R1B1KBNR b KQ -",
    "rnbq1rk1/pp3ppp/4pn2/2pp4/1bPP4/2NBPN2/PP3PPP/R1BQ1RK1 b - -",
    "r1bq1rk1/pppn1pbp/3p1np1/4p3/2PPP3/2N2NP1/PP3PBP/R1BQ1RK1 b - -",
    "rnbq1rk1/pp3pbp/2pp1np1/4p3/2PPP3/2N1BP2/PP2N1PP/R2QKB1R w KQ -",
    "r1bq1rk1/pppnn1bp/3p4/3Pp1p1/2P1Pp2/2N2P2/PP2BBPP/R2QNRK1 w - -"
]


def compete(first, second, games, seconds, d1, d2, starts, colors="both"):
    """
    # function to run a tournament between two engines
    # first and second - strings, specifies names of players
    # games - int, specifies number of games to play per opening position
    # seconds, depth(d1, d2) - int, designate time limit and search depths for engines
    # starts - list of positions in fen form from which to start games
    # colors - either string "both" specifying that each position is to be played out from both colors
    # or it's a list of chess.color objects corresponding to the color of the first player in each starting position
    # the function returns numbers of wins and draws and two dictionaries of stats for each move
    # each dictionary has lists of positions, moves by one of the engines, and basic game stats, from that engine's point of view
    """
    engine1 = chess.engine.SimpleEngine.popen_uci(players[first])
    print(f"engine {first} ready")
    engine2 = chess.engine.SimpleEngine.popen_uci(players[second])
    print(f"engine {second} ready")
    # set up counters of wins for each player and number of draws
    player1wins = 0
    player2wins = 0
    draws = 0
    # set up recording of games
    record1 = {
        "game": [],
        "ply": [],
        "player": [],
        "adversary": [],
        "color": [],
        "position": [],
        "evaluation1": [],
        "move1": [],
        "result": []
    }
    record2 = {
        "game": [],
        "ply": [],
        "player": [],
        "adversary": [],
        "color": [],
        "position": [],
        "evaluation": [],
        "move": [],
        "result": []
    }
    if colors == "both":
        colors = (len(starts) * [chess.WHITE]) + (len(starts) * [chess.BLACK])
        starts = 2 * starts
    position_scores1 = [{"wins": 0, "draws": 0, "losses": 0} for x in range(len(starts))]
    position_scores2 = [{"wins": 0, "draws": 0, "losses": 0} for x in range(len(starts))]
    unclear = 0
    # run tournament
    uniques = [[""] for x in range(len(starts))]
    sofar = 1
    # iterate over number of games per position
    for i in range(games):
        # iterate over game opening positions
        counter = 0
        timeout = 0
        while counter < len(starts) and timeout < 6:
            opening = starts[counter]
            color = colors[counter]
            if color == chess.WHITE:
                other = chess.BLACK
            else:
                other = chess.WHITE
            board1 = []
            player1 = []
            adversary1 = []
            eval1 = []
            move1 = []
            board2 = []
            player2 = []
            adversary2 = []
            eval2 = []
            move2 = []
            game1 = []
            game2 = []
            ply1 = []
            ply2 = []
            color1 = []
            color2 = []
            position = chess.Board(opening)
            # get decision from engine according to turn (first player is white, second player is black)
            while not position.is_game_over(claim_draw=False):
                if position.turn == color:
                    game1.append(sofar)
                    ply1.append(position.ply())
                    player1.append(first)
                    adversary1.append(second)
                    color1.append("white" if color == chess.WHITE else "black")
                    board1.append(position.fen())
                    if color == chess.WHITE:
                        evaluation = engine1.analyse(position, chess.engine.Limit(time=seconds, depth=d1))[
                            "score"].white().wdl(model="lichess").expectation()
                    else:
                        evaluation = engine1.analyse(position, chess.engine.Limit(time=seconds, depth=d1))[
                            "score"].black().wdl(model="lichess").expectation()
                    moves = []
                    for j in range(samples):
                        moves.append(engine1.play(position, chess.engine.Limit(time=seconds, depth=d1)).move.uci())
                    decision = max(moves, key=moves.count)
                    position.push_uci(decision)
                    eval1.append(evaluation)
                    move1.append(decision)
                    # print(f"{first} plays: {decision.move}")
                else:
                    game2.append(sofar)
                    ply2.append(position.ply())
                    player2.append(second)
                    adversary2.append(first)
                    color2.append("white" if other == chess.WHITE else "black")
                    board2.append(position.fen())
                    if other == chess.BLACK:
                        evaluation = engine2.analyse(position, chess.engine.Limit(time=seconds, depth=d2))[
                            "score"].black().wdl(model="lichess").expectation()
                    else:
                        evaluation = engine2.analyse(position, chess.engine.Limit(time=seconds, depth=d2))[
                            "score"].white().wdl(model="lichess").expectation()
                    decision = engine2.play(position, chess.engine.Limit(time=seconds, depth=d2)).move.uci()
                    position.push_uci(decision)
                    eval2.append(evaluation)
                    move2.append(decision)
                    # print("{second} plays: {decision.move}")
                # cycle to next turn in game
            # handle end of game
            if position.fen() not in uniques[counter]:
                if position.outcome().winner == color:  # or (position.is_checkmate() and position.turn == other):
                    print(f"{first} wins")
                    print(f"closing position: {position.fen()}")
                    result1 = "win"
                    result2 = "lose"
                    player1wins += 1
                    position_scores1[counter]["wins"] += 1
                    position_scores2[counter]["losses"] += 1
                elif position.outcome().winner == other:  # or (position.is_checkmate() and position.turn == color):
                    print(f"{second} wins")
                    print(f"closing position: {position.fen()}")
                    result1 = "lose"
                    result2 = "win"
                    player2wins += 1
                    position_scores2[counter]["wins"] += 1
                    position_scores1[counter]["losses"] += 1
                elif position.outcome().winner == None or position.is_stalemate() or position.is_fivefold_repetition() or position.is_insufficient_material():
                    print("draw")
                    result1 = "draw"
                    result2 = "draw"
                    draws += 1
                    position_scores1[counter]["draws"] += 1
                    position_scores2[counter]["draws"] += 1
                else:
                    print("result unclear")
                    result1 = "unclear"
                    result2 = "unclear"
                    print(position.outcome())
                    unclear += 1
                # store game
                sofar += 1
                timeout = 0
                print("finished game ", sofar)
                record1["game"].extend(game1)
                record1["player"].extend(player1)
                record1["color"].extend(color1)
                record1["adversary"].extend(adversary1)
                record1["ply"].extend(ply1)
                record1["position"].extend(board1)
                record1["evaluation1"].extend(eval1)
                record1["move1"].extend(move1)
                record1["result"].extend([result1 for _ in range(len(board1))])
                record2["game"].extend(game2)
                record2["player"].extend(player2)
                record2["color"].extend(color2)
                record2["adversary"].extend(adversary2)
                record2["ply"].extend(ply2)
                record2["position"].extend(board2)
                record2["evaluation"].extend(eval2)
                record2["move"].extend(move2)
                record2["result"].extend([result2 for _ in range(len(board2))])
            else:
                timeout += 1
            # handle repeated games
            uniques[counter].append(position.fen())
            if uniques[counter][-3:].count(position.fen()) == 3:
                print("error game loop, rebooting engines")
                engine1.quit()
                engine2.quit()
                engine1 = chess.engine.SimpleEngine.popen_uci(players[first])
                print(f"engine {first} ready")
                engine2 = chess.engine.SimpleEngine.popen_uci(players[second])
                print(f"engine {second} ready")
            if timeout == 0:
                counter += 1
            # cycle back and play from next opening position
        # do another round of plays
    # end tournament and return results
    engine1.quit()
    engine2.quit()
    print(f"{first} won {player1wins} times")
    print(f"{second} won {player2wins} times")
    print(f"{first} and {second} drew {draws} times")
    print(f"wdl: {(player1wins + (0.5 * draws)) / (player1wins + player2wins + draws)}")
    print(f"{unclear} unclear outcomes")
    u = []
    for i in uniques:
        u += i[1:]
    print("unique games: ", len(set(u)))
    maxreps = max(u, key=u.count)
    print(f"maximum repetitions: {u.count(maxreps)} in game: {maxreps}")
    return player1wins, player2wins, draws, position_scores1, position_scores2, record1, record2


if __name__ == '__main__':

    player1wins, player2wins, draws, scores1, scores2, record1, record2 = compete("stockfish", "lc0", games=100, seconds=1, d1=15, d2=15, starts=openings)
    df = pd.DataFrame(record1)
    df = df[df['move'] != 'error']
    df.to_csv("path_to_save_file", index=False)
