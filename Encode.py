from bpe import BPE

TRAINING_DIR = 'training_data'

with open(TRAINING_DIR + '/titles.txt') as fin:
	my_text = fin.read().lower()

bpe = BPE()
bpe.add_seq(my_text)
bpe.set_merges('abcdefghijklmnopqrstuvwxyz0123456789-')
for dict_size in [200, 400, 600, 800, 1000]:
	bpe.embed(dict_size)
	bpe.save(TRAINING_DIR + '/words' + str(dict_size) + '.bpe')
