import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('train_bpe_mod', pathlib.Path('cs336_basics/train_bpe.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
train_bpe = mod.train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode
input_path = FIXTURES_PATH / 'corpus.en'
_, merges = train_bpe(str(input_path), 500, ['<|endoftext|>'])
reference_merges_path = FIXTURES_PATH / 'train-bpe-reference-merges.txt'
gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
with open(reference_merges_path, encoding='utf-8') as f:
    gpt2_reference_merges = [tuple(line.rstrip().split(' ')) for line in f]
reference_merges = [(
    bytes([gpt2_byte_decoder[token] for token in m1]),
    bytes([gpt2_byte_decoder[token] for token in m2]),
) for m1, m2 in gpt2_reference_merges]
for i,(m,r) in enumerate(zip(merges, reference_merges)):
    if m!=r:
        print('mismatch', i, m, r)
        break
else:
    print('all match', len(merges))
