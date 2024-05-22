import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import cairosvg
from PIL import Image
import captum
import torch
import torch.nn as nn
import chess
from chess import svg as svg
import numpy as np
import pandas as pd
import CentaurGPT-trainer


def make_cls(dims):
    classifier = nn.Sequential(nn.Linear(dims, dims),
                               nn.BatchNorm1d(dims),
                               nn.ReLU(inplace=True),
                               nn.Linear(dims, dims),
                               nn.BatchNorm1d(dims),
                               nn.ReLU(inplace=True),
                               nn.Linear(dims, dims),
                               nn.BatchNorm1d(dims),
                               nn.ReLU(inplace=True),
                               nn.Linear(dims, 1)).to("cuda")
    return classifier

def get_embeddings(filename, modelname):
    # load model
    data = pd.read_csv(f"{filename}.csv")
    model = CentaurGPT.init_model("CentaurGPT").to("cuda")
    dims = model.config.n_embd
    model_dict = torch.load(f'{modelname}.pt')
    model.load_state_dict(model_dict)
    model.eval()
    emb = []
    for i, row in tqdm(data.iterrows()):
        board = CentaurGPT.board_encoder(row['position'])
        with torch.no_grad():
            x = torch.tensor(board + [19], dtype=torch.long, device="cuda").unsqueeze(0)
            vec = model(x)[:, -1, :]
            vec = vec.squeeze().tolist()
            emb.append({f"{z}": vec[z] for z in range(len(vec))})
    emb = pd.DataFrame(emb)
    emb.to_csv(f"{filename}_{modelname}_embeddings.csv", index=False)
    return


def get_scores(filename, modelname):
    # get model scores
    # load model
    data = pd.read_csv(f"{filename}.csv")
    model = CentaurGPT.init_model("CentaurGPT").to("cuda")
    dims = model.config.n_embd
    model_dict = torch.load(f'{modelname}Encoder.pt')
    model.load_state_dict(model_dict)
    model.eval()
    clf = make_cls(dims)
    model_dict = torch.load(f'{modelname}Clf.pt')
    clf.load_state_dict(model_dict)
    clf.eval()
    maiascores = []
    leelascores = []
    emb = []
    for i, row in tqdm(data.iterrows()):
        board = CentaurGPT-trainer.board_encoder(row['position'])
        with torch.no_grad():
            x = torch.tensor(board + [19], dtype=torch.long, device="cuda").unsqueeze(0)
            vec = model(x)[:, -1, :]
            score = float(torch.sigmoid(clf(vec).squeeze()).item())
            if score == 0.5:
                continue
            maiascores.append(score)
            leelascores.append(1 - score)
            vec = vec.squeeze().tolist()
            emb.append({f"{z}": vec[z] for z in range(len(vec))})
    emb = pd.DataFrame(emb)
    emb["eval"] = [1 if maiascores[x] > leelascores[x] else 0 for x in range(len(maiascores))]
    emb.to_csv(f"{filename}_scored.csv", index=False)
    return


def get_att_scores(filename, modelname, player1, player2, sample=10000, attype="mean"):
    """
    :param filename: path to .csv file with chess positions and move recommendations (string)
    :param modelpath: path to .pt file of model we want attention scores from (string)
    :param player1/2: names of team members (strings)
    :param sample: how many random samples to extract attentions scores for (int)
    :param attype: indicator of whether we want attention scores in "last" layer or "mean" across layers (string) or integer for number of layer to examine
    :return: pd dataframe with attention scores per token
    """
    data = pd.read_csv(f"{filename}.csv")
    data = data[data[f"{player1}_eval"] != data[f"{player2}_eval"]]
    data = data.sample(n=sample)
    # load model
    model = CentaurGPT.init_model("CentaurGPT").to("cuda")
    model_dict = torch.load(f'{modelname}.pt')
    model.load_state_dict(model_dict)
    model.eval()
    df = []
    for i, row in tqdm(data.iterrows()):
        board = CentaurGPT-trainer.board_encoder(row['position'])
        with torch.no_grad():
            x = torch.tensor(board + [19], dtype=torch.long, device="cuda").unsqueeze(0)
            model.forward(x)
        if attype == "last":
            # get attention scores in last transformer block
            attention_scores = model.att_scores[-1]
            # get the scores at last token
            last_token_attention = attention_scores[0, :, -1, :]
            # average over attention heads
            attention = torch.mean(last_token_attention, dim=0)
            attn_dict = {f"token_{x}": float(attention[x].item()) for x in range(68)}
        elif attype == "mean":
            cumulative_att = np.zeros(68)
            for j in range(model.config.n_layer):
                attention_scores = model.att_scores[j]
                last_token_attention = attention_scores[0, :, -1, :]
                attention = torch.mean(last_token_attention, dim=0)
                cumulative_att += np.asarray([float(attention[x].item()) for x in range(68)])
            attention = cumulative_att / model.config.n_layer
            attn_dict = {f"token_{x}": float(attention[x]) for x in range(68)}
        else:
            k = int(attype)
            attention_scores = model.att_scores[k]
            # get the scores at last token
            last_token_attention = attention_scores[0, :, -1, :]
            # average over attention heads
            attention = torch.mean(last_token_attention, dim=0)
            attn_dict = {f"token_{x}": float(attention[x].item()) for x in range(68)}
        df.append({"position": row['position'],
                   f"{player1}_move": row[f"{player1}_move"],
                   f"{player2}_move": row[f"{player2}_move"],
                   "model_choice": row[f"{row['contributor']}_move"],
                   **attn_dict
                   })
    df = pd.DataFrame(df)
    df.to_csv(f"Attention-{filename}-{modelname}-{attype}.csv", index=False)
    return df


def analysis_attention(filename, player1, player2, controlengine-path):
    control = chess.engine.SimpleEngine.popen_uci(controlengine-path)
    data = pd.read_csv(f"{filename}.csv")
    attacked = []
    notattacked = []
    maiafromatt = []
    maiatoatt = []
    leelafromatt = []
    leelatoatt = []
    randomfromatt = []
    randomtoatt = []
    controlfromatt = []
    controltoatt = []
    meanattpieces = []
    meanattempty = []
    for i, row in tqdm(data.iterrows()):
        board = chess.Board(row["position"])
        color = board.turn
        other = chess.BLACK if color == chess.WHITE else chess.WHITE
      
        # attention for pieces and empty squares
        meanattempty.append(np.asarray([row[f"token_{x}"] for x in range(64) if board.piece_at(x) is None]).mean())
        meanattpieces.append(np.asarray([row[f"token_{x}"] for x in range(64) if board.piece_at(x) is not None]).mean())  
        
        # attentions for attacked and not attacked pieces
        globalthreats = []
        for square in range(64):
            if board.piece_at(square) is not None:
                c = board.piece_at(square).color
                globalthreats.extend(list(board.attackers(color=(chess.WHITE if c == chess.BLACK else chess.BLACK), square=square)))
        globalthreats = list(set(globalthreats))
        friendlypieces = [x for x in range(64) if ((board.piece_at(x) is not None) and (board.piece_at(x).color == color))]
        enemypieces = [x for x in range(64) if ((board.piece_at(x) is not None) and (board.piece_at(x).color == other))]
        globalnotattacked = [x for x in (friendlypieces + enemypieces) if x not in globalthreats]
        attacked.append(np.asarray([row[f"token_{x}"] for x in globalthreats]).mean())
        notattacked.append(np.asarray([row[f"token_{x}"] for x in globalnotattacked]).mean())

        # attention for moves
        legals = list(set(board.legal_moves))
        random.shuffle(legals)
        maiamove = chess.Move.from_uci(row[f"{player1}_move"])
        leelamove = chess.Move.from_uci(row[f"{player2}_move"])
        controlmove = control.play(board, chess.engine.Limit(depth=5), game=object()).move#.uci()
        randmove = random.choice(legals)
        scoresfrom = []
        scoresto = []
        scores = []
        legalorigins = list(set([x.from_square for x in legals]))
        legaldestinations = list(set(x.to_square for x in legals))
        for move in legalorigins:
            scoresfrom.append({"from_square": move, "from_att": row[f"token_{move}"]})
        for move in legaldestinations:
            scoresto.append({"to_square": move, "to_att": row[f"token_{move}"]})
        scoresfrom = pd.DataFrame(scoresfrom)
        scoresfrom.drop_duplicates(inplace=True)
        meanfrom.append(scoresfrom["from_att"].mean())
        scoresto = pd.DataFrame(scoresto)
        scoresto.drop_duplicates(inplace=True)
        meanto.append(scoresto["to_att"].mean())
        maiafromatt.append(float(scoresfrom[scoresfrom["from_square"] == maiamove.from_square].drop_duplicates()["from_att"].item()))
        maiatoatt.append(float(scoresto[scoresto["to_square"] == maiamove.to_square].drop_duplicates()["to_att"].item()))
        randomfromatt.append(float(scoresfrom[scoresfrom["from_square"] == randmove.from_square].drop_duplicates()["from_att"].item()))
        randomtoatt.append(float(scoresto[scoresto["to_square"] == randmove.to_square].drop_duplicates()["to_att"].item()))
        controlfromatt.append(float(scoresfrom[scoresfrom["from_square"] == controlmove.from_square].drop_duplicates()["from_att"].item()))
        controltoatt.append(float(scoresto[scoresto["to_square"] == controlmove.to_square].drop_duplicates()["to_att"].item()))
        leelafromatt.append(float(scoresfrom[scoresfrom["from_square"] == leelamove.from_square].drop_duplicates()["from_att"].item()))
        leelatoatt.append(float(scoresto[scoresto["to_square"] == leelamove.to_square].drop_duplicates()["to_att"].item()))
    ranks = pd.DataFrame({
                          f"{player1}_origin_att": maiafromatt, f"{player2}_origin_att": leelafromatt,
                          f"{player1}_destination_att": maiatoatt, f"{player2}_destination_att": leelatoatt,
                          "control_origin_att": controlfromatt, "control_destination_att": controltoatt,
                          "random_origin_att": randomfromatt, "random_destination_att": randomtoatt,
                          "mean_pieces_attention": meanattpieces, "mean_empty_attention": meanattempty,
                          "mean_attacked_attention": attacked, "mean_notattacked_attention": notattacked
                          })
    ranks.to_csv(f"{filename}_attqueries.csv", index=False)
    return ranks


def visualize_attention(filename, player1, player2):
    df = pd.read_csv(f"{filename}.csv")
    samp = df.sample(n=1)
    idx = samp.index[0]
    board = []
    for i in range(8):
        board.append([samp[f"token_{(i * 8) + j}"] for j in range(8)])
    ax = sns.heatmap(np.asarray(board).squeeze(), cmap="plasma", cbar=False, xticklabels=False, yticklabels=False)
    ax.invert_yaxis()
    plt.autoscale()
    plt.tight_layout()
    plt.savefig(f"{filename}_heatmap.png")
    plt.show()
    pos = chess.Board(fen=samp.at[idx, "position"])
    move1 = chess.Move.from_uci(samp.at[idx, f"{player1}_move"])
    arrow1 = chess.svg.Arrow(tail=move1.from_square, head=move1.to_square, color="black")
    move2 = chess.Move.from_uci(samp.at[idx, f"{player2}_move"])
    arrow2 = chess.svg.Arrow(tail=move2.from_square, head=move2.to_square, color="red")
    img2 = svg.board(pos, coordinates=True, arrows=[arrow1, arrow2])
    img2 = cairosvg.svg2png(bytestring=img2, write_to=f"{filename}_board.png")
    img1 = np.asarray(Image.open(f"{filename}_heatmap.png"))
    img2 = np.asarray(Image.open(f"{filename}_board.png"))
    extent1 = 0, 10, 0, 10
    extent2 = -10, 20, -10, 20
    # extent1 = extent2 = 1, 8, 1, 8
    fig = plt.figure(frameon=False)
    ax1 = fig.add_subplot(111, frameon=False)
    ax2 = fig.add_axes(ax1.get_position(), frameon=False)
    ax1.imshow(img1, cmap="gray", extent=extent2, alpha=1.0, interpolation="nearest")
    ax2.imshow(img2, cmap="plasma", extent=extent1, alpha=0.5, interpolation="nearest")
    plt.show()
    return


def Aw(col1, col2, player2="leela10b2500"):
    n1 = len(col1.index)
    n2 = len(col2.index)
    c = 0
    for p in col1.tolist():
        g = len(col2[p > col2].index)
        e = len(col2[p == col2].index)
        c += g + (0.5 * e)
    ES = c/(n1*n2)
    #print("Aw effect size: ", ES)
    return ES


def feature_importance(modelpath, num_features, hiddenlayers, layerwidth, datapath, samples=5000, attrib_method="intgrad", baseline="zero"): 
    # feature importance attribution using captum
    device = "cuda"
    model = models.py_model([num_features] + (hiddenlayers*[layerwidth]) + [1]).to(device)
    loaded_dict = torch.load(f"{modelpath}.pt")
    model.load_state_dict(loaded_dict)
    model.eval()
    df = pd.read_csv(f"{datapath}.csv")
    df = df.sample(samples)
    # drop position column at beginning and eval column at end
    df = df.iloc[:, 1:-1]
    featurenames = list(df.columns)
    if atrrib_method == "intgrad":
        fp = captum.attr.Lime(lambda z: torch.sigmoid(model(z)))#, multiply_by_inputs=True)
    elif attrib_method == "lime":
        fp = captum.attr.Lime(lambda z: torch.sigmoid(model(z)))#, multiply_by_inputs=True)
    data = torch.tensor(df.values, dtype=torch.float, device=device, requires_grad=True)
    if baseline == "gamestart":
        z = features.board_features(chess.Board().fen()).extract()
    elif baseline == "mean":
        z = [df[x].mean() for x in df.columns]
    elif baseline == "zero":
        z = 20 * [0]
    baseline = torch.tensor(z, dtype=torch.float, device=device).unsqueeze(0)
    attributions = fp.attribute(inputs=data), baselines=baseline).squeeze()
    attributions = pd.DataFrame(attributions.detach().cpu().numpy(), columns=featurenames)
    attributions.to_csv(f"{datapath}_attributions.csv", index=False)
    plt.title("Feature Importances", fontsize=18)
    plt.bar(featurenames, [abs(attributions[f].mean()) for f in featurenames])
    plt.errorbar(featurenames, [abs(attributions[f].mean()) for f in featurenames], [attributions[f].std()/(len(df.index)**0.5) for f in featurenames], fmt="none", ecolor="black")
    plt.ylabel("Attributions", fontsize=16)
    plt.xlabel("Features", fontsize=16)
    plt.xticks(rotation=25, ha="right", fontsize=14)
    plt.tight_layout()
    plt.show()
    return attributions


