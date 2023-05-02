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

def radix4_fft(x,dim=0,copy=True):

	if copy:
		X = np.copy(x)
	else:
		X = x

	N = x.shape[dim]

	# number of stages in the FFT
	num_stages = round(np.log2(N) / 2)

	# butterfly matrix
	butterfly = np.array([
		[1,  1 ,  1,  1 ],
		[1, -1j, -1,  1j],
		[1, -1 ,  1, -1 ],
		[1,  1j, -1, -1j],
	])

	twiddles = np.exp(-2j * np.pi / N * np.arange(N))

	for stage in range(num_stages):

		partitions = 1 << ( stage               << 1) # 4^stage
		stride     = 1 << ((num_stages-stage-1) << 1) # 4^(num_stages-stage-1)

		base_idx = stride * np.arange(0,4)

		for i in range(partitions): # 4^stage
			for j in range(stride): # 4^(num_stages-stage-1)

				# determines the index mask for the butterfly
				idx = base_idx + j + i*(4*stride)

				# calculate twiddle factors
				twiddle = twiddles[np.arange(1,4) * partitions * j]

				# applyies butterfly (verified)
				if dim == 0:
					X[idx    ,:]  = np.dot(butterfly, X[idx,:])
					X[idx[1:],:] *= np.expand_dims(twiddle, axis=1)
				elif dim == 1:
					X[:, idx    ]  = np.dot(butterfly, X[:,idx].T).T
					X[:, idx[1:]] *= np.expand_dims(twiddle, axis=0)

	# bit reversed order
	if dim == 0:
		for idx1, idx2 in radix4_bit_reversal(num_stages):
			X[[idx1,idx2],:] = X[[idx2,idx1],:]
	elif dim == 1:
		for idx1, idx2 in radix4_bit_reversal(num_stages):
			X[:,[idx1,idx2]] = X[:,[idx2,idx1]]

	return X

def radix4_fft2d(x):
    X = np.copy(x)
    
    return radix4_fft(
        radix4_fft(
            X,dim=0,copy=False
        )
        ,dim=1,copy=False
    )

if __name__ =='__main__':

	import fft

	def error(sig,ref):
		return abs(sig-ref).sum()

	for i in range(1_000):

		N = np.random.randint(2,4)
		M = np.random.randint(2,4)
		sig = np.random.rand(4**N,4**M).astype(np.complex128)

		SIG = radix4_fft2d(sig)
		REF = np.fft.fft2(sig)

		try:
			err = error(SIG,REF)
			assert err < (64*4 * 1e-12)
		except AssertionError:
			print(f'error: {err}')

	pass