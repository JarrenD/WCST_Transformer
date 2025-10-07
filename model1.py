import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dModel, numHeads, dropout=0.1):
        super().__init__()
        assert dModel % numHeads == 0, "dModel must be divisible by numHeads"
        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads

        self.Wq = nn.Linear(dModel, dModel)
        self.Wk = nn.Linear(dModel, dModel)
        self.Wv = nn.Linear(dModel, dModel)
        self.Wo = nn.Linear(dModel, dModel)
        self.dropout = nn.Dropout(dropout)

    def splitHeads(self, x):
        batchSize, seqLen, _ = x.size()
        return x.view(batchSize, seqLen, self.numHeads, self.dK).transpose(1, 2)

    def combineHeads(self, x):
        batchSize, numHeads, seqLen, dK = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, seqLen, self.dModel)

    def forward(self, x, mask=None, returnAttention=False):
        Q = self.splitHeads(self.Wq(x))
        K = self.splitHeads(self.Wk(x))
        V = self.splitHeads(self.Wv(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dK, dtype=torch.float32, device=x.device)
        )
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attentionWeights = F.softmax(scores, dim=-1)
        attentionWeights = self.dropout(attentionWeights)
        context = torch.matmul(attentionWeights, V)
        output = self.Wo(self.combineHeads(context))
        if returnAttention:
            return output, attentionWeights
        return output


class FeedForward(nn.Module):
    def __init__(self, dModel, dFF, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dModel, dFF)
        self.linear2 = nn.Linear(dFF, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dFF, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dModel, numHeads, dropout)
        self.feedForward = FeedForward(dModel, dFF, dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnAttention=False):
        if returnAttention:
            attnOutput, attnWeights = self.attention(self.norm1(x), mask, returnAttention=True)
            x = x + self.dropout(attnOutput)
            x = x + self.dropout(self.feedForward(self.norm2(x)))
            return x, attnWeights
        else:
            attnOutput = self.attention(self.norm1(x), mask)
            x = x + self.dropout(attnOutput)
            x = x + self.dropout(self.feedForward(self.norm2(x)))
            return x


class WCSTTransformer(nn.Module):
    def __init__(self, vocabSize=70, dModel=128, numHeads=4, numLayers=4, dFF=512, maxSeqLen=512, dropout=0.1):
        super().__init__()
        self.dModel = dModel
        self.tokenEmbedding = nn.Embedding(vocabSize, dModel, padding_idx=0)
        self.positionEmbedding = nn.Embedding(maxSeqLen, dModel)
        self.blocks = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFF, dropout) for _ in range(numLayers)
        ])
        self.outputProjection = nn.Linear(dModel, vocabSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, returnAttention=False):
        batchSize, seqLen = x.size()
        mask = torch.tril(torch.ones(seqLen, seqLen, device=x.device)).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batchSize, self.blocks[0].attention.numHeads, seqLen, seqLen)
        positions = torch.arange(seqLen, device=x.device).unsqueeze(0).expand(batchSize, -1)
        x = self.tokenEmbedding(x) + self.positionEmbedding(positions)
        x = self.dropout(x)

        attentionWeightsList = []
        for block in self.blocks:
            if returnAttention:
                x, attnWeights = block(x, mask, returnAttention=True)
                attentionWeightsList.append(attnWeights)
            else:
                x = block(x, mask)

        logits = self.outputProjection(x)
        if returnAttention:
            return logits, attentionWeightsList
        return logits


# ============================================================================
# WCST DATASET AND COLLATE FUNCTION
# ============================================================================

class WCSTDataset(Dataset):
    def __init__(self, wcstGen, numBatches):
        self.data = []
        for _ in range(numBatches):
            batchInputs, batchTargets = next(wcstGen)
            for inp, tgt in zip(batchInputs, batchTargets):
                # inp contains: [4 category cards, 1 example card, SEP, example_label, EOS]
                # tgt contains: [question card, SEP, question_label]
                
                # Full sequence: inp + tgt
                fullSeq = np.concatenate([inp, tgt])
                
                # Create input (all tokens except the last one for next-token prediction)
                inputSeq = fullSeq[:-1]
                
                # Create target (shifted by 1, predicting next token)
                targetSeq = fullSeq[1:]
                
                # Mask out positions where we don't want to compute loss
                # We only want to predict: example_label (after first SEP), 
                # question card, SEP, and question_label
                targetMask = np.full_like(targetSeq, -100)
                
                # Find positions to predict
                # The structure is: [cat1, cat2, cat3, cat4, ex_card, SEP, ex_label, EOS, q_card, SEP, q_label]
                # After removing last token: [cat1, cat2, cat3, cat4, ex_card, SEP, ex_label, EOS, q_card, SEP]
                # We want to predict at positions: 6(ex_label), 7(EOS), 8(q_card), 9(SEP), 10(q_label if exists)
                
                # Position 6 should predict example_label (position 7 in original)
                if len(targetSeq) > 6:
                    targetMask[6:] = targetSeq[6:]
                
                self.data.append({
                    'input': torch.tensor(inputSeq, dtype=torch.long),
                    'target': torch.tensor(targetMask, dtype=torch.long)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['target']


def collateFn(batch):
    inputs, targets = zip(*batch)
    inputsPadded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targetsPadded = pad_sequence(targets, batch_first=True, padding_value=-100)
    return inputsPadded, targetsPadded


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def trainEpoch(model, dataloader, optimizer, device, maxGradNorm=1.0):
    model.train()
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0

    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        logitsFlat = logits.reshape(-1, logits.size(-1))
        targetsFlat = targets.reshape(-1)
        loss = F.cross_entropy(logitsFlat, targetsFlat, ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)
        optimizer.step()

        mask = targetsFlat != -100
        totalCorrect += (logitsFlat.argmax(dim=-1)[mask] == targetsFlat[mask]).sum().item()
        totalTokens += mask.sum().item()
        totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / totalTokens


def evaluate(model, dataloader, device):
    model.eval()
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            logitsFlat = logits.reshape(-1, logits.size(-1))
            targetsFlat = targets.reshape(-1)
            loss = F.cross_entropy(logitsFlat, targetsFlat, ignore_index=-100)

            mask = targetsFlat != -100
            totalCorrect += (logitsFlat.argmax(dim=-1)[mask] == targetsFlat[mask]).sum().item()
            totalTokens += mask.sum().item()
            totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / totalTokens


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    import numpy as np
    from wcst import WCST  # <-- your WCST generator code
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 8
    wcst = WCST(batch_size=batch_size)
    trainGen = wcst.gen_batch()
    valGen = wcst.gen_batch()
    testGen = wcst.gen_batch()

    trainDataset = WCSTDataset(trainGen, numBatches=10000)
    valDataset = WCSTDataset(valGen, numBatches=2000)
    testDataset = WCSTDataset(testGen, numBatches=2000)

    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=collateFn)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, collate_fn=collateFn)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False, collate_fn=collateFn)

    print("Initializing model...")
    model = WCSTTransformer(vocabSize=70, dModel=128, numHeads=4, numLayers=4, dFF=512, dropout=0.1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    numEpochs = 10
    bestValLoss = float('inf')
    history = {'trainLoss': [], 'trainAcc': [], 'valLoss': [], 'valAcc': []}

    for epoch in range(numEpochs):
        trainLoss, trainAcc = trainEpoch(model, trainLoader, optimizer, device)
        valLoss, valAcc = evaluate(model, valLoader, device)
        scheduler.step()

        history['trainLoss'].append(trainLoss)
        history['trainAcc'].append(trainAcc)
        history['valLoss'].append(valLoss)
        history['valAcc'].append(valAcc)

        print(f"Epoch {epoch+1}/{numEpochs} | "
              f"Train Loss: {trainLoss:.4f} | Train Acc: {trainAcc:.4f} | "
              f"Val Loss: {valLoss:.4f} | Val Acc: {valAcc:.4f}")

        if valLoss < bestValLoss:
            bestValLoss = valLoss
            torch.save(model.state_dict(), 'model1.pt')

    model.load_state_dict(torch.load('model1.pt'))
    testLoss, testAcc = evaluate(model, testLoader, device)
    print(f"\nTest Loss: {testLoss:.4f} | Test Acc: {testAcc:.4f}")

    with open('trainingHistory.json', 'w') as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    model, history = main()