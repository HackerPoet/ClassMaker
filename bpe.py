import struct

class BPE:
	def __init__(self):
		self.all_seq = []
		self.token_to_str = {}
		self.str_to_token = {}
		self.splits = set()

	def save(self, fname):
		num_entries = len(self.token_to_str)
		with open(fname, 'wb') as fout:
			fout.write(struct.pack('i', num_entries))
			for t in self.token_to_str:
				s = self.token_to_str[t]
				fout.write(struct.pack('ii', t, len(s)))
				fout.write(s.encode('charmap'))

	def load(self, fname):
		self.token_to_str = {}
		self.str_to_token = {}
		with open(fname, 'rb') as fin:
			num_entries = struct.unpack('i', fin.read(4))[0]
			for e in range(num_entries):
				t, s_len = struct.unpack('ii', fin.read(8))
				s = fin.read(s_len).decode('charmap')
				self.token_to_str[t] = s
				self.str_to_token[s] = t

	def encode(self, sentence):
		all_strs = sorted(self.str_to_token.keys(), key=len, reverse=True)
		tokens = [sentence]
		for s in all_strs:
			needs_loop = True
			while needs_loop:
				needs_loop = False
				new_tokens = []
				for t in tokens:
					if (type(t) is str) and (s in t):
						i = t.index(s)
						j = i + len(s)
						if i > 0:
							new_tokens.append(t[:i])
						new_tokens.append(self.str_to_token[s])
						if j < len(t):
							new_tokens.append(t[j:])
						needs_loop = True
					else:
						new_tokens.append(t)
				tokens = new_tokens
		return tokens

	def decode(self, tokens):
		return ''.join([self.token_to_str[t] for t in tokens])

	def add_seq(self, seq):
		tokens = []
		for c in seq:
			if c not in self.str_to_token:
				num_tokens = len(self.token_to_str)
				self.str_to_token[c] = num_tokens
				self.token_to_str[num_tokens] = c
			tokens.append(self.str_to_token[c])
		self.all_seq.append(tokens)

	def set_merges(self, merges):
		self.splits = set()
		for s in self.str_to_token:
			if s not in merges:
				self.splits.add(self.str_to_token[s])

	def count_pairs(self):
		pair_counts = {}
		for seq in self.all_seq:
			for i in range(len(seq) - 1):
				a = seq[i]
				if a in self.splits:
					continue
				b = seq[i + 1]
				if b in self.splits:
					continue
				pair = (a, b)
				if pair not in pair_counts:
					pair_counts[pair] = 1
				else:
					pair_counts[pair] += 1
		return pair_counts

	def max_pair(self, pairs):
		v = list(pairs.values())
		k = list(pairs.keys())
		return k[v.index(max(v))]

	def pair_to_str(self, pair):
		return self.token_to_str[pair[0]] + self.token_to_str[pair[1]]

	def apply_pair_encode(self, pair):
		new_token = len(self.token_to_str)
		new_str = self.pair_to_str(pair)
		self.token_to_str[new_token] = new_str
		self.str_to_token[new_str] = new_token
		for i in range(len(self.all_seq)):
			seq = self.all_seq[i]
			new_seq = []
			j = 0
			while j < len(seq):
				if j < len(seq) - 1 and pair == (seq[j], seq[j + 1]):
					new_seq.append(new_token)
					j += 1
				else:
					new_seq.append(seq[j])
				j += 1
			self.all_seq[i] = new_seq

	def embed_step(self):
		pairs = self.count_pairs()
		pair = self.max_pair(pairs)
		print(self.pair_to_str(pair), pairs[pair])
		if pairs[pair] > 1:
			self.apply_pair_encode(pair)
			return True
		else:
			return False

	def embed(self, num_tokens):
		while len(self.token_to_str) < num_tokens:
			if not self.embed_step():
				break
		return len(self.token_to_str)
