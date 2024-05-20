


class TeamPlay:
    def __init__(self, team, adv):
        self.team = team
        self.adv = adv

    def evaluate_team(self, model, starts, colors="both", games=1, seconds=1, depth1=None, depth2=1):
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
                        #record[f"{player}_conf"].extend(conf[f"{player}_conf"])
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
        for name, engine in self.engines.items():
            engine.quit()
        self.adversary.quit()
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
    def __init__(self, modelpathname, t, fc=None):
        if fc is not None:
            import models
            self.fc = True
            self.model = train-fc.py_model(fc)
            loaded_dict = torch.load(f'{modelpathname}.pt')
            self.model.load_state_dict(loaded_dict)
            self.model.eval().to("cuda")
        if fc is None:
            self.fc = False
            self.model = CentaurGPT.init_chessGPT(t)
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


