import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import builtins
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

PAD_ID = 70
SEP_ID = 68
C1,C2,C3,C4 = 64,65,66,67

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
            scores = scores.masked_fill(~mask, float('-inf'))
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
    def __init__(self, vocabSize=71, dModel=128, numHeads=4, numLayers=4, dFF=512, maxSeqLen=512, dropout=0.1):
        super().__init__()
        self.dModel = dModel
        self.tokenEmbedding = nn.Embedding(vocabSize, dModel, padding_idx=PAD_ID)
        self.positionEmbedding = nn.Embedding(maxSeqLen, dModel)
        self.segment_embedding = nn.Embedding(2, dModel)
        self.blocks = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFF, dropout) for _ in range(numLayers)
        ])
        self.outputProjection = nn.Linear(dModel, vocabSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_type=None, returnAttention=False):
        batchSize, seqLen = x.size()
        mask = torch.tril(torch.ones(seqLen, seqLen, device=x.device, dtype = torch.bool)).unsqueeze(0).unsqueeze(0)
        keys_with_no_padding = (x != PAD_ID).unsqueeze(1).unsqueeze(2)
        combination = mask & keys_with_no_padding
        mask = combination.expand(batchSize, self.blocks[0].attention.numHeads, seqLen, seqLen)
        
        positions = torch.arange(seqLen, device=x.device).unsqueeze(0).expand(batchSize, -1)
        tok = self.tokenEmbedding(x) 
        pos = self.positionEmbedding(positions)
        if token_type is None:
            seg = 0
        else:
            seg = self.segment_embedding(token_type)
        x = self.dropout(tok + pos + seg)

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
    def __init__(self, wcst, wcstGen, numBatches, switch_period=None):
        self.data = []
        self.inputs = []
        self.targets = []
        self.lengths = []
        steps = 0
        for _ in range(numBatches):
            if switch_period is not None and steps > 0 and (steps % switch_period) ==0:
                wcst.context_switch()
                print(f"[Dataset] Context switched after {steps} batches.")
                

            batchInputs, batchTargets = builtins.next(wcstGen)
            steps += 1
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
                sep_position = np.where(inputSeq == SEP_ID)[0]
                if len(sep_position) > 0:
                    p = sep_position[-1]
                    next_value = targetSeq[p]
                    if next_value in (C1,C2,C3,C4):
                        targetMask[p] = next_value
                
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
    maxLen = max(len(x) for x in inputs)

    batchInputs  = torch.full((len(batch), maxLen), PAD_ID, dtype=torch.long)
    batchTargets = torch.full((len(batch), maxLen), -100,   dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        batchInputs[i, :L]  = torch.as_tensor(inp, dtype=torch.long)
        batchTargets[i, :L] = torch.as_tensor(tgt, dtype=torch.long)

    # build token_type (0=examples, 1=query/answer block)
    B, T = batchInputs.shape
    token_type_ids = torch.zeros((B, T), dtype=torch.long)
    for i in range(B):
        seq = batchInputs[i]
        valid = (seq != PAD_ID).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        last_t = valid[-1].item()
        seps = (seq[:last_t+1] == SEP_ID).nonzero(as_tuple=True)[0]
        last_sep = seps[-1].item() if len(seps) > 0 else 0
        token_type_ids[i, last_sep:] = 1   # mark query/answer segment

    return batchInputs, batchTargets, token_type_ids


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def trainEpoch(model, dataloader, optimizer, device, maxGradNorm=1.0, warmup_steps=500, base_lr=3e-4, global_step_start=0):
    model.train()
    totalLoss = 0.0
    totalCorrect = 0
    totalTokens = 0
    global_step = global_step_start
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) ==3:
            inputs, targets, ttids = batch
        else:
            inputs, targets = batch
            ttids = torch.zeros_like(inputs)

        inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)

        # sanity: exactly one supervised label per seq
        assert (targets != -100).sum().item() == inputs.size(0), "Expected exactly 1 supervised label per seq"

        optimizer.zero_grad()
        logits = model(inputs)

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        # clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)

        # warmup (linear)
        global_step += 1
        if global_step <= warmup_steps:
            warm_lr = base_lr * (global_step / float(warmup_steps))
            for g in optimizer.param_groups:
                g['lr'] = warm_lr

        optimizer.step()

        # accuracy on the single supervised position per seq
        logitsFlat = logits.reshape(-1, logits.size(-1))
        targetsFlat = targets.reshape(-1)
        mask = targetsFlat != -100
        totalCorrect += (logitsFlat.argmax(dim=-1)[mask] == targetsFlat[mask]).sum().item()
        totalTokens += mask.sum().item()
        totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / totalTokens, global_step


def evaluate(model, dataloader, device):
    model.eval()
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, ttids = batch
            else:
                inputs, targets = batch
                ttids = torch.zeros_like(inputs)

            inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)
            logits = model(inputs, token_type =ttids)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            logitsFlat = logits.reshape(-1, logits.size(-1))
            targetsFlat = targets.reshape(-1)

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
    N = 64 # switches every 64 batches for training
    wcst_train = WCST(batch_size=batch_size)
    wcst_val = WCST(batch_size=batch_size+1)
    wcst_test = WCST(batch_size=batch_size+2)
    trainGen = wcst_train.gen_batch()
    valGen = wcst_val.gen_batch()
    testGen = wcst_test.gen_batch()

    trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=2000, switch_period = None) #was 10 000
    valDataset = WCSTDataset(wcst_val, valGen, numBatches=300)      #was 2000
    testDataset = WCSTDataset(wcst_test, testGen, numBatches=300)    #was 2000

    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=collateFn)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, collate_fn=collateFn)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False, collate_fn=collateFn)

    print("Initializing model...")
    model = WCSTTransformer(vocabSize=71, dModel=128, numHeads=4, numLayers=4, dFF=512, dropout=0.1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = None

    numEpochs = 10
    bestValLoss = float('inf')
    history = {'trainLoss': [], 'trainAcc': [], 'valLoss': [], 'valAcc': []}

    from collections import Counter
    cnt = Counter()
    for batch in valLoader:  # or trainLoader; val is fine and small
        if len(batch) == 3:
            _, t, _ = batch
        else:
            _, t = batch
        lab = t[t != -100].flatten()
        for c in (64, 65, 66, 67):
            cnt[c] += (lab == c).sum().item()
    print("Label counts (C1..C4):", dict(cnt))

    patience = 3
    bad = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    global_step = 0

    for epoch in range(numEpochs):
        # Rebuild train dataset each epoch (new rule draws, new samples)
        wcst_train = WCST(batch_size=batch_size)
        trainGen   = wcst_train.gen_batch()
        trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=2000, switch_period=None)
        trainLoader  = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=collateFn)

        trainLoss, trainAcc, global_step = trainEpoch(
            model, trainLoader, optimizer, device,
            maxGradNorm=1.0, warmup_steps=500, base_lr=3e-4, global_step_start=global_step
        )
        valLoss, valAcc = evaluate(model, valLoader, device)

        # ↓ scheduler + early stopping
        scheduler.step(valLoss)  # adjust LR if plateau detected

        if valLoss < bestValLoss - 1e-4:
            bestValLoss = valLoss
            bad = 0
            torch.save(model.state_dict(), 'model1.pt')
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

        history['trainLoss'].append(trainLoss)
        history['trainAcc'].append(trainAcc)
        history['valLoss'].append(valLoss)
        history['valAcc'].append(valAcc)

        print(f"Epoch {epoch+1}/{numEpochs} | "
            f"Train Loss: {trainLoss:.4f} | Train Acc: {trainAcc:.4f} | "
            f"Val Loss: {valLoss:.4f} | Val Acc: {valAcc:.4f}")

    model.load_state_dict(torch.load('model1.pt'))
    testLoss, testAcc = evaluate(model, testLoader, device)
    print(f"\nTest Loss: {testLoss:.4f} | Test Acc: {testAcc:.4f}")

    # record where the dataset switches happened
    history["switch_points"] = list(range(64, 10000, 64))  # adjust for your dataset size

    with open('trainingHistory.json', 'w') as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    model, history = main()