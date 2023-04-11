import numpy as np

def radix4_bit_reversal(stages):
	
	def bit_reversal(num):
		num_rev = 0
		
		for i in range(stages):

			mask = (3 & (num >> 2*i)) << 2*(stages-i-1)
			
			num_rev |= mask
		
		return num_rev

	fields = [(i,bit_reversal(i)) for i in range(4**stages)]

	for idx, field in enumerate(fields):
	
		if field[1] < field[0]:
			fields[idx] = (field[1], field[0])

	return set(filter(
		lambda x : x[0] != x[1],
		fields
	))

def radix4_fft(x):

	X = np.copy(x)

	N = len(x)

	# number of stages in the FFT
	num_stages = round(np.log2(N) / 2)

	# butterfly matrix
	butterfly = np.array([
		[1,  1 ,  1,  1 ],
		[1, -1j, -1,  1j],
		[1, -1 ,  1, -1 ],
		[1,  1j, -1, -1j],
	])

	for stage in range(num_stages):

		partitions = 1 << ( stage               << 1) # 4^stage
		stride     = 1 << ((num_stages-stage-1) << 1) # 4^(num_stages-stage-1)

		base_idx = stride * np.arange(0,4)

		for i in range(partitions): # 4^stage
			for j in range(stride): # 4^(num_stages-stage-1)

				# determines the index mask for the butterfly
				idx = base_idx + j + i*(4*stride)

				# calculate twiddle factors
				twiddle = np.exp(-2j * np.pi * j / (4*stride) * np.arange(1,4))

				# applyies butterfly (verified)
				X[idx] = np.dot(butterfly, X[idx])

				# applies twiddle factors
				X[idx[1:]] *= twiddle

	# bit reversed order
	for idx1, idx2 in radix4_bit_reversal(num_stages):
		X[[idx1,idx2]] = X[[idx2,idx1]]

	return X

if __name__ =='__main__':

	import fft

	def error(sig,ref):
		return abs(sig-ref).sum()

	for i in range(1_000):
		N   = np.random.randint(3,8)
		sig = np.random.rand(4**N).astype(np.complex128)

		SIG = radix4_fft(sig)
		REF = np.fft.fft(sig)

		try:
			err = error(SIG,REF)
			assert err < (4**N * 1e-12)
		except AssertionError:
			print(f'error: {err}')

	pass