import random
import sys
import chess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import CentaurGPT-trainer


# a global dictionary of engines and their paths
players = {"stockfish": r".stockfish/stockfish_14_x64_popcnt",
           "leela": r"./leela/lc0_lnx_cpu.exe",
           "maia": r"./maia/lc0_lnx_cpu.exe"}


class TeamPlay:
    def __init__(self, team, adv):
        self.team = team
        self.adv = adv

    def evaluate_team(self, model, starts, colors="both", games=1, seconds=1, depth1=1, depth2=1):
        # function to run a tournament between a team of engines and an adversary, using a classifier
        # self.engines is a dictionary of player names and their spun-up engines, self.adversayr is adv engine.
        # classifier is a classifier function. It is sent model info which may be any object, and the game circumstances.
        # games - int, specifies number of games to play per opening position
        # seconds, depth1, depth2 - int, designate time limit and search depth for engines
        # starts - list of positions in fen form from which to start games
        # colors - either string "both" specifying that each position is to be played out from both colors
        # or it's a list of chess.color objects corresponding to the color of the first player in each starting position
        # the function returns numbers of wins and draws and a dictionary of stats for each move
        # the dictionary has lists of positions, moves and basic game stats
        # set up counters of wins for each player and number of draws
        player1wins = 0
        player2wins = 0
        draws = 0
        # set up recording of games
        record = {
            "startstate": [],
            "game": [],
            "ply": [],
            "move": [],
            "contributor": [],
            "adversary": [],
            "color": [],
            "position": [],
            "result": []
        }
        for player in self.team:
            record[f"{player}_move"] = []
            record[f"{player}_eval"] = []
        if colors == "both":
            colors = (len(starts) * [chess.WHITE]) + (len(starts) * [chess.BLACK])
            starts = 2 * starts
        position_scores = [{"wins": 0, "draws": 0, "losses": 0} for _ in range(len(starts))]
        unclear = 0
        # run tournament
        uniques = [[""] for _ in range(len(starts))]
        sofar = 1
        # iterate over number of games per position
        for i in range(games):
            # iterate over game opening positions
            counter = 0
            timeout = 0
            while counter < len(starts) and timeout < 6:
                self.engines = {}
                for player in self.team:
                    self.engines[player] = chess.engine.SimpleEngine.popen_uci(players[player])
                    print(f"engine {player} ready")
                self.adversary = chess.engine.SimpleEngine.popen_uci(players[self.adv])
                opening = starts[counter]
                print("starting from: ", opening)
                color = colors[counter]
                if color == chess.WHITE:
                    other = chess.BLACK
                else:
                    other = chess.WHITE
                begin =[]
                board = []
                against = []
                move = []
                game = []
                ply = []
                side = []
                contributor = []
                members = {}
                evals = {}
                for player in self.team:
                    members[f"{player}_move"] = []
                    evals[f"{player}_eval"] = []
                    #conf[f"{player}_conf"] = []
                position = chess.Board(opening)
                # get decision from engine according to turn (first player is white, second player is black)
                while not position.is_game_over(claim_draw=False):
                    if position.turn == color:
                        begin.append(opening)
                        game.append(sofar)
                        ply.append(position.ply())
                        against.append(self.adv)
                        side.append("white" if color == chess.WHITE else "black")
                        board.append(position.fen())
                        recs = []
                        for n, player in enumerate(self.team):
                            mv = self.engines[player].play(position, chess.engine.Limit(time=seconds, depth=depth1[n]), game=object()).move.uci()
                            recs.append(mv)
                            members[f"{player}_move"].append(mv)
                        # if there is agreement then just do the agreement
                        # otherwise classifier should return best move and selected player in team
                        if all([x == recs[0] for x in recs]):
                        #if recs[0] == recs[1]:
                            decision = recs[0]
                            member = "agreed"
                            scores = [-1 for _ in range(len(self.team))]
                        else:
                            ### *** GET DECISION FROM MODEL ***
                            ### _______________________________
                            decision, member, scores = model.classify(position, recs, self.engines, self.adv, self.team)
                            ### _______________________________
                        move.append(decision)
                        contributor.append(member)
                        for idx, player in enumerate(self.team):
                            evals[f"{player}_eval"].append(scores[idx])
                        position.push_uci(decision)
                    # adversary turn
                    else:
                        decision = self.adversary.play(position, chess.engine.Limit(time=seconds, depth=depth2), game=object()).move.uci()
                        position.push_uci(decision)
                    # cycle to next turn in game
                # handle end of game
                if position.fen() not in uniques[counter]:
                    if position.outcome().winner == color or (position.is_checkmate() and position.turn == other):
                        print(f"team wins")
                        # print(position.outcome())
                        print(f"closing position: {position.fen()}")
                        result = "win"
                        player1wins += 1
                        position_scores[counter]["wins"] += 1
                    elif position.outcome().winner == other or (position.is_checkmate() and position.turn == color):
                        print(f"{self.adv} wins")
                        # print(position.outcome())
                        print(f"closing position: {position.fen()}")
                        result = "lose"
                        player2wins += 1
                        position_scores[counter]["losses"] += 1
                    elif position.is_stalemate() or position.is_fivefold_repetition() or position.is_insufficient_material() or position.outcome() is None:
                        print("draw")
                        # print(position.outcome())
                        result = "draw"
                        draws += 1
                        position_scores[counter]["draws"] += 1
                    else:
                        print("result unclear")
                        result = "unclear"
                        print(position.outcome())
                        unclear += 1
                    # store game
                    sofar += 1
                    timeout = 0
                    print("finished game ", sofar)
                    record["startstate"].extend(begin)
                    record["game"].extend(game)
                    record["color"].extend(side)
                    record["adversary"].extend(against)
                    record["ply"].extend(ply)
                    record["position"].extend(board)
                    record["move"].extend(move)
                    record["contributor"].extend(contributor)
                    for player in self.team:
                        record[f"{player}_move"].extend(members[f"{player}_move"])
                        record[f"{player}_eval"].extend(evals[f"{player}_eval"])
                    record["result"].extend([result for _ in range(len(board))])
                else:
                    timeout += 1
                # handle repeated games
                uniques[counter].append(position.fen())
                if uniques[counter][-3:].count(position.fen()) == 3:
                    print("error game loop, rebooting engines")
                    for name, engine in self.engines.items():
                        engine.quit()
                        self.engines[name] = chess.engine.SimpleEngine.popen_uci(players[name])
                if timeout == 0:
                    counter += 1
                for name, engine in self.engines.items():
                    engine.quit()
                self.adversary.quit()
                # cycle back and play from next opening position
            # do another round of plays
        # end tournament and return results
        model.quit()
        print(f"team won {player1wins} times")
        print(f"adversary won {player2wins} times")
        print(f"they drew {draws} times")
        print(f"{unclear} unclear outcomes")
        for p in self.team:
            print(f"{record['contributor'].count(p)} contributions by {p}")
        print(f"{record['contributor'].count('agreed')} agreements between team members")
        print(f"{record['contributor'].count('random')} random choices between team members")
        u = []
        for i in uniques:
            u += i[1:]
        print("unique games: ", len(set(u)))
        maxreps = max(u, key=u.count)
        print(f"maximum repetitions: {u.count(maxreps)} in game: {maxreps}")
        return player1wins, player2wins, draws, position_scores, record


class CentaurModel:
    def __init__(self, modelpathname, fc=None):
        if fc is not None:
            import models
            self.fc = True
            self.model = train-fc.py_model(fc)
            loaded_dict = torch.load(f'{modelpathname}.pt')
            self.model.load_state_dict(loaded_dict)
            self.model.eval().to("cuda")
        if fc is None:
            self.fc = False
            self.model = CentaurGPT-trainer.init_model("CentaurGPT")
            dims = self.model.config.n_embd
            loaded_dict = torch.load(f'{modelpathname}Encoder.pt')
            self.model.load_state_dict(loaded_dict)
            self.model.eval().to("cuda")
            self.clf = nn.Sequential(nn.Linear(dims, dims),
                                             nn.BatchNorm1d(dims),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(dims, dims),
                                             nn.BatchNorm1d(dims),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(dims, dims),
                                             nn.BatchNorm1d(dims),
                                             nn.LeakyReLU(inplace=True),
                                             nn.Linear(dims, 1)).eval().to("cuda")
            loaded_dict = torch.load(f'{modelpathname}Clf.pt')
            self.clf.load_state_dict(loaded_dict)
            self.clf.eval().to("cuda")

    def classify(self, position, moves, engines, adv, team):
        scores = []
        with torch.no_grad():
            if self.fc is False:
                pos = chessGPT.board_encoder(position.fen())
                x = torch.tensor(pos, dtype=torch.long).unsqueeze(0).to("cuda")
                encoding = self.model(x)
                proba = torch.sigmoid(self.clf(encoding[:, -1, :]))
                scores = [proba.item(), 1 - proba.item()]
            elif self.fc is True:
                x = features.board_features(position.fen()).extract_reduced()
                x = torch.tensor(x, dtype=torch.float).unsqueeze(0).to("cuda")
                proba = torch.sigmoid(self.model(x).squeeze())
                scores = [proba.item(), 1 - proba.item()]
        if scores[0] > scores[1]:
            idx = 0
        elif scores[1] > scores[0]:
            idx = 1
        else:
            idx = random.choice([0, 1])
        move = moves[idx]
        member = team[idx]
        return move, member, scores

    def quit(self):
        return


class PolicyIterate:
    def __init__(self, modelname, device, p=1.0):
        self.device = device
        self.model = CentaurGPT-trainer.init_model("CentaurGPT").to(self.device)
        dims = self.model.config.n_embd
        self.clf = cls(dims).to(self.device)
        model_dict = torch.load(f"{modelname}Encoder.pt", map_location=torch.device(self.device))
        self.model.load_state_dict(model_dict)
        model_dict = torch.load(f"{modelname}Clf.pt", map_location=torch.device(self.device))
        self.clf.load_state_dict(model_dict)
        self.model.eval()
        self.clf.eval()
        self.p = p

    def classify(self, position, recs, eggs, advname, team):
        if position.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK
        results = []
        for i, player in enumerate(team):
            adversary = chess.engine.SimpleEngine.popen_uci(players[advname])
            engine1 = chess.engine.SimpleEngine.popen_uci(players[team[0]])
            engine2 = chess.engine.SimpleEngine.popen_uci(players[team[1]])
            board = chess.Board(position.fen())
            board.push_uci(recs[i])
            done, result = self.check_end(board, color)
            while not done:
                board.push(adversary.play(board, chess.engine.Limit(depth=1), game=object()).move)
                done, result = self.check_end(board, color)
                if not done:
                    bidx = CentaurGPT.board_encoder(board.fen())
                    x = torch.tensor(bidx + [19], dtype=torch.long, device=self.device).unsqueeze(0)
                    encoding = self.model(x)[:, -1, :]
                    proba = torch.sigmoid(self.clf(encoding)).to(self.device)
                    predict = int(proba.round().squeeze().item())
                    if predict == 1:
                        mv = engine1.play(board, chess.engine.Limit(depth=1), game=object()).move.uci()
                    elif predict == 0:
                        mv = engine2.play(board, chess.engine.Limit(depth=1), game=object()).move.uci()
                    board.push_uci(mv)
                    done, result = self.check_end(board, color)
                if result is None:
                    print("error getting result")
                    result = 0
            results.append(result)
            adversary.quit()
            engine1.quit()
            engine2.quit()
        if results.count(max(results)) == len(results):
            idx = random.choice(range(len(recs)))
            move = recs[idx]
            member = team[idx]
        else:
            r = np.random.choice([0, 1], p=[self.p, 1 - self.p])
            if r == 0:
                idx = results.index(max(results))
            elif r == 1:
                idx = results.index(min(results))
            move = recs[idx]
            member = team[idx]
        return move, member, results

    def check_end(self, board, color):
        terminate = False
        reward = None
        other = chess.BLACK if color == chess.WHITE else chess.WHITE
        if board.outcome(claim_draw=False) is not None:
            terminate = True
            if board.outcome().winner == color:
                reward = 1
            elif board.outcome().winner == other:
                reward = 0
            else:
                reward = 0.5
        return terminate, reward

    def quit(self):
        return


class FCIterate:
    def __init__(self, modelname, dims, device, p=1.0):
        self.device = device
        import models
        self.model = models.py_model(dims).to(self.device)
        model_dict = torch.load(f"{modelname}.pt", map_location=torch.device(self.device))
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.p = p

    def classify(self, position, recs, eggs, advname, team):
        if position.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK
        results = []
        for i, player in enumerate(team):
            adversary = chess.engine.SimpleEngine.popen_uci(players[advname])
            engine1 = chess.engine.SimpleEngine.popen_uci(players[team[0]])
            engine2 = chess.engine.SimpleEngine.popen_uci(players[team[1]])
            board = chess.Board(position.fen())
            board.push_uci(recs[i])
            done, result = self.check_end(board, color)
            while not done:
                board.push(adversary.play(board, chess.engine.Limit(depth=1), game=object()).move)
                done, result = self.check_end(board, color)
                if not done:
                    bidx = board_features(board.fen()).extract_reduced()
                    x = torch.tensor(bidx, dtype=torch.float, device=self.device).unsqueeze(0)
                    encoding = self.model(x).squeeze()
                    proba = torch.sigmoid(encoding).to(self.device)
                    predict = int(proba.round().squeeze().item())
                    if predict == 1:
                        mv = engine1.play(board, chess.engine.Limit(depth=1), game=object()).move.uci()
                    elif predict == 0:
                        mv = engine2.play(board, chess.engine.Limit(depth=1), game=object()).move.uci()
                    board.push_uci(mv)
                    done, result = self.check_end(board, color)
            if result is None:
                print("error getting result")
                result = 0
            results.append(result)
            adversary.quit()
            engine1.quit()
            engine2.quit()
        if results.count(max(results)) == len(results):
            idx = random.choice(range(len(recs)))
            move = recs[idx]
            member = team[idx]
        else:
            r = np.random.choice([0, 1], p=[self.p, 1 - self.p])
            if r == 0:
                idx = results.index(max(results))
            elif r == 1:
                idx = results.index(min(results))
            move = recs[idx]
            member = team[idx]
        return move, member, results

    def check_end(self, board, color):
        terminate = False
        reward = None
        other = chess.BLACK if color == chess.WHITE else chess.WHITE
        if board.outcome(claim_draw=False) is not None:
            terminate = True
            if board.outcome().winner == color:
                reward = 1
            elif board.outcome().winner == other:
                reward = 0
            else:
                reward = 0.5
        return terminate, reward

    def quit(self):
        return


class RandomChoice():
    def __init__(self, team, p):
        self.team = team
        self.p = p

    def classify(self, board, recs, engines, adv, team):
        scores = []
        if board.turn is True or board.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK
        for player in self.team:
            if color == chess.WHITE:
                evaluation = engines[player].analyse(board, chess.engine.Limit(depth=1))["score"].white().wdl(model="lichess").expectation()
            else:
                evaluation = engines[player].analyse(board, chess.engine.Limit(depth=1))["score"].black().wdl(model="lichess").expectation()
            scores.append(evaluation)
        member = np.random.choice(self.team, p=[self.p, 1 - self.p])
        r = self.team.index(member)
        #r = random.randint(0, len(recs)-1)
        move = recs[r]
        return move, member, scores


class GreedyRollouts():
    def __init__(self, depth=1, rollouts=1):
        #super().__init__(team, adv)
        self.depth = depth
        self.rollouts = rollouts

    def classify(self, position, recs, engines, advname, team):
        if position.turn is True or position.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK
        results = []
        for i, player in enumerate(team):
            adversary = chess.engine.SimpleEngine.popen_uci(players[advname])
            engine = chess.engine.SimpleEngine.popen_uci(players[player])
            score = 0
            rounds = 0
            for j in range(self.rollouts):
                board = chess.Board(position.fen())
                board.push_uci(recs[i])
                done, result = self.check_end(board, color)
                while not done:
                    board.push(adversary.play(board, chess.engine.Limit(depth=self.depth), game=object()).move)
                    done, result = self.check_end(board, color)
                    if not done:
                        move = engine.play(board, chess.engine.Limit(depth=1), game=object()).move.uci()
                        board.push_uci(move)
                        done, result = self.check_end(board, color)
                if result is None:
                    print("error getting result")
                    result = 0
                else:
                    rounds += 1
                score += result
            results.append(score/rounds)
            adversary.quit()
            engine.quit()
        if results.count(max(results)) == len(results):
            idx = random.choice([0, 1])
            move = recs[idx]
            member = team[idx]
        else:
            idx = results.index(max(results))
            move = recs[idx]
            member = team[idx]
        return move, member, results

    def check_end(self, board, color):
        terminate = False
        reward = None
        other = chess.BLACK if color == chess.WHITE else chess.WHITE
        if board.is_game_over(claim_draw=False):
            terminate = True
            if board.outcome().winner == color:
                reward = 1
            elif board.outcome().winner == other:
                reward = 0
            else:
                reward = 0.5
        return terminate, reward


class Teacher():
    def __init__(self, modelname, depth):
        self.scorer = chess.engine.SimpleEngine.popen_uci(players[modelname])
        self.depth = depth

    def classify(self, position, recs, engines, adv, team):
        # method receives teacher to serve as a classifier
        # model_info has name of teacher engine, depth
        board = chess.Board(position.fen())
        scores = []
        if board.turn == chess.WHITE:
            color = chess.WHITE
        else:
            color = chess.BLACK
        for i, move in enumerate(recs):
            board.push_uci(move)
            if color == chess.WHITE:
                evaluation = self.scorer.analyse(board, chess.engine.Limit(time=60, depth=self.depth), game=object())["score"].white().wdl(model="sf14").expectation()
            else:
                evaluation = self.scorer.analyse(board, chess.engine.Limit(time=60, depth=self.depth), game=object())["score"].black().wdl(model="sf14").expectation()
            scores.append(evaluation)
            board.pop()
        if scores.count(max(scores)) == len(scores):
            m = random.randint(0, len(recs) - 1)
            move = recs[m]
            member = "random"
        else:
            m = scores.index(max(scores))
            move = recs[m]
            member = team[m]
        return move, member, scores

    def quit(self):
        self.scorer.quit()
        return

if __name__ == '__main__':
    df = pd.read_csv("train-opening-positions.csv")
    openings = df['positions'].tolist()
    team = ["maia", "leela"]
    adv = "stockfish"
    gamerun = TeamPlay(team, adv)
    model = "CentaurEncoder"
    manager = CentaurModel(model)
    _, _, _, _, record = gamerun.evaluate_team(manager, starts, colors="both", games=1, seconds=1, depth1=1, depth2=1)
    df = pd.DataFrame(record)
    df.to_csv("./EvaluationGames.csv", index=False)
    
