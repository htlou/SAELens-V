#  result

## pliesae

average_l0 it2t 94
average_l0 t2t 87

### reconstruction loss in t2t task

Orig 2.46400785446167
reconstr 6.798471450805664
Zero 10.37548828125

### reconstruction loss in it2t task

Orig 2.6099588871002197
reconstr 7.340147018432617
Zero 10.375489234924316

### Mary ioi

prepend_bosTokenized prompt: ['<s>', 'When', 'John', 'and', 'Mary', 'went', 'to', 'the', 'shops', ',', 'John', 'gave', 'the', 'bag', 'to']
Tokenized answer: ['', 'Mary']

probs size: torch.Size([17, 32064]), tokens size: torch.Size([1, 17])
Performance on answer token:(original model)
Rank: 22       Logit:  7.00 Prob:  0.01% Token: ||
Top 0th token. Logit: 15.97 Prob: 94.38% Token: |Mary|
Top 1th token. Logit: 12.54 Prob:  3.05% Token: |the|
Top 2th token. Logit: 11.05 Prob:  0.69% Token: |a|
Top 3th token. Logit: 10.00 Prob:  0.24% Token: |carry|
Top 4th token. Logit:  9.88 Prob:  0.21% Token: |his|
Top 5th token. Logit:  9.72 Prob:  0.18% Token: |John|
Top 6th token. Logit:  9.30 Prob:  0.12% Token: |me|
Top 7th token. Logit:  8.50 Prob:  0.05% Token: |an|
Top 8th token. Logit:  8.36 Prob:  0.05% Token: |her|
Top 9th token. Logit:  8.34 Prob:  0.05% Token: |their|=True)

Performance on answer token:(sae model)
Rank: 1        Logit: 10.15 Prob: 14.45% Token: |Mary|
Top 0th token. Logit: 10.99 Prob: 33.43% Token: |</s>|
Top 1th token. Logit: 10.15 Prob: 14.45% Token: |Mary|
Top 2th token. Logit:  9.13 Prob:  5.18% Token: |1|
Top 3th token. Logit:  8.98 Prob:  4.48% Token: |
|
Top 4th token. Logit:  8.77 Prob:  3.61% Token: |5|
Top 5th token. Logit:  8.46 Prob:  2.67% Token: |6|
Top 6th token. Logit:  8.43 Prob:  2.58% Token: |mary|
Top 7th token. Logit:  8.42 Prob:  2.57% Token: |8|
Top 8th token. Logit:  8.24 Prob:  2.14% Token: |7|
Top 9th token. Logit:  8.18 Prob:  2.00% Token: |2|
Ranks of the answer tokens: [('', 22), ('Mary', 1)]
Orig 4.05479621887207
reconstr 6.5594892501831055
Zero 10.375490188598633
Performance on answer token:(zero model)
Rank: 8        Logit:  3.16 Prob:  0.24% Token: ||
Top 0th token. Logit:  4.09 Prob:  0.61% Token: |the|
Top 1th token. Logit:  3.94 Prob:  0.52% Token: |，|
Top 2th token. Logit:  3.91 Prob:  0.51% Token: |。|
Top 3th token. Logit:  3.81 Prob:  0.46% Token: |.|
Top 4th token. Logit:  3.48 Prob:  0.33% Token: |his|
Top 5th token. Logit:  3.48 Prob:  0.33% Token: |him|
Top 6th token. Logit:  3.25 Prob:  0.26% Token: |my|
Top 7th token. Logit:  3.16 Prob:  0.24% Token: |
|
Top 8th token. Logit:  3.16 Prob:  0.24% Token: ||
Top 9th token. Logit:  3.10 Prob:  0.23% Token: |的|
Ranks of the answer tokens: [('', 8)] 
Tokenized prompt: ['<s>', 'When', 'John', 'and', 'Mary', 'went', 'to', 'the', 'shops', ',', 'John', 'gave', 'the', 'bag', 'to']
Tokenized answer: ['', 'Mary']
probs size: torch.Size([15, 32064]), tokens size: torch.Size([1, 17])

## obelicssae

average l0 it2t 160
average l0 t2t 204

### reconstruction loss in it2t task

Orig 2.6099588871002197
reconstr 2.775998115539551
Zero 10.375489234924316

### reconstruction loss in t2t task

Orig 2.0448360443115234
reconstr 2.86653733253479
Zero 10.375490188598633

### Mary ioi

Orig 4.054447174072266
reconstr 4.224089622497559
Zero 10.375490188598633

