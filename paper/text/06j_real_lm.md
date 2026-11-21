### 6.5 Character-level language modelling

A sequence is a one-dimensional arrangement of entities. Each position `t` reads its previous 64 characters (`t − 1` down to `t − 64`) as its partner list, with seven functions of the integer offset `d ∈ {1, …, 64}` as the relation: `d/64`, `log(d)/log(64)`, `1/d`, and sine and cosine at periods 8
and 32. There is no positional encoding, and the partner list is causal by construction (position `t = 0` of each window has no prior partners, so the LESNet stack outputs zero there). The model comprises a character embedding, six LESNet layers of width 252 and a linear output head. The baseline is a small GPT (Radford et al., 2019; six pre-norm blocks of width 104 with feed-forward width 416, four heads and learned positional embeddings) at the same
parameter count (about 800k). Both are trained on Tiny Shakespeare (Karpathy, 2015; 1.1M characters, 90% train / 10% validation) using 256-character windows, batch size 64 and 5000 steps of AdamW (Loshchilov and Hutter, 2019; weight decay 0.01) with a cosine schedule. Learning rates are selected from
{3e-4, 1e-3, 3e-3, 1e-2, 3e-2} by lowest validation bits per character in a 3000-step sweep, and we report the mean minimum validation bits per character and 95% interval over three seeds. The same 10% split is used for learning-rate selection and evaluation, with no separate test split.

{{fig:bench_charlm.png|Tiny Shakespeare: validation bits per character (lower is better) against training step (left) and relative compute (right); line: mean, band: range over three seeds. Both models overfit; the results table of §6.6 reports each run's minimum.}}

Both models overfit the corpus within 5000 steps, LESNet earlier than GPT, so {{ref:realdata}} reports each run's best validation value. LESNet reaches 2.313 bits per character against 2.237 for GPT, trailing by 0.08 bits (3.4%) at matched parameter count.
