import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, n_layers=1, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers,
                             bidirectional=True, batch_first=True,
                             dropout=dropout if n_layers > 1 else 0.0)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, (h, c) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        c_cat = torch.cat([c[-2], c[-1]], dim=1)
        h0 = torch.tanh(self.fc_h(h_cat)).unsqueeze(0) 
        c0 = torch.tanh(self.fc_c(c_cat)).unsqueeze(0)
        return outputs, (h0, c0)
 
 
class Attention(nn.Module):
    """Bahdanau additive attention over encoder outputs."""
 
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
 
    def forward(self, dec_hidden, enc_outputs, mask):
        B, S, _ = enc_outputs.shape
        dec_hidden = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, S, 1) 
        energy = torch.tanh(self.attn(torch.cat([dec_hidden, enc_outputs], dim=2)))
        scores = self.v(energy).squeeze(2) 
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return F.softmax(scores, dim=1) 
 
 
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, hid_dim=512, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hid_dim)
        self.lstm = nn.LSTM(emb_dim + hid_dim * 2, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, input_tok, hidden, cell, enc_outputs, mask):
        input_tok = input_tok.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tok))
 
        attn_weights = self.attention(hidden, enc_outputs, mask) 
        context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs) 
 
        lstm_input = torch.cat([embedded, context], dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
 
        output = output.squeeze(1)
        context = context.squeeze(1)
        embedded = embedded.squeeze(1)
 
        pred = self.fc_out(torch.cat([output, context, embedded], dim=1))
        return pred, hidden, cell, attn_weights
 
 
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, sos_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.device = device
 
    def make_mask(self, src, src_lens):
        B, S = src.shape
        idx = torch.arange(S, device=src.device).unsqueeze(0)
        return (idx < src_lens.unsqueeze(1).to(src.device)).long()
 
    def forward(self, src, src_lens, tgt, teacher_forcing_ratio=0.5):
        """Training forward pass with scheduled teacher forcing."""
        B, T = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(B, T, vocab_size, device=self.device)
 
        enc_outputs, (hidden, cell) = self.encoder(src, src_lens)
        mask = self.make_mask(src, src_lens)
 
        input_tok = tgt[:, 0]
        for t in range(1, T):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, enc_outputs, mask)
            outputs[:, t] = pred
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_tok = tgt[:, t] if teacher_force else top1
        return outputs
 
    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len=20):
        self.eval()
        enc_outputs, (hidden, cell) = self.encoder(src, src_lens)
        mask = self.make_mask(src, src_lens)
        B = src.size(0)
        input_tok = torch.full((B,), self.sos_idx, dtype=torch.long, device=self.device)
 
        sequences = torch.full((B, max_len), self.pad_idx, dtype=torch.long, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        for t in range(max_len):
            pred, hidden, cell, _ = self.decoder(input_tok, hidden, cell, enc_outputs, mask)
            top1 = pred.argmax(1)
            sequences[:, t] = torch.where(finished, torch.full_like(top1, self.pad_idx), top1)
            finished = finished | (top1 == self.eos_idx)
            input_tok = top1
            if finished.all():
                break
        return sequences
 
    @torch.no_grad()
    def beam_search_decode(self, src, src_lens, beam_width=4, max_len=20, len_penalty=0.7):
        """Beam search for a single example (expects batch size 1)."""
        assert src.size(0) == 1, "beam_search_decode expects batch size 1"
        self.eval()
        enc_outputs, (hidden, cell) = self.encoder(src, src_lens)
        mask = self.make_mask(src, src_lens)
 
        beams = [([self.sos_idx], 0.0, hidden, cell, False)]
        for _ in range(max_len):
            candidates = []
            for tokens, score, h, c, finished in beams:
                if finished:
                    candidates.append((tokens, score, h, c, finished))
                    continue
                input_tok = torch.tensor([tokens[-1]], device=self.device)
                pred, h2, c2, _ = self.decoder(input_tok, h, c, enc_outputs, mask)
                log_probs = F.log_softmax(pred, dim=1).squeeze(0)
                topk_logp, topk_idx = log_probs.topk(beam_width)
                for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                    candidates.append((tokens + [idx], score + lp, h2, c2, idx == self.eos_idx))
            candidates.sort(key=lambda x: x[1] / (len(x[0]) ** len_penalty), reverse=True)
            beams = candidates[:beam_width]
            if all(b[4] for b in beams):
                break
        best = max(beams, key=lambda x: x[1] / (len(x[0]) ** len_penalty))
        return best[0]
 