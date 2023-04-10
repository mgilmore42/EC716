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

	num_stages = round(np.log2(N) / 2)

	num_butterflies = N//4

	butterfly = np.array([
		[1,  1 ,  1,  1 ],
		[1, -1j, -1,  1j],
		[1, -1 ,  1, -1 ],
		[1,  1j, -1, -1j],
	])

	for stage in range(num_stages):

		base_idx = 4**(num_stages-stage-1) * np.arange(0,4)

		# twiddles = [
		# 	np.exp(-2j*np.pi*np.arange(4)*k / (4**(stage+1))) for k in range(4**stage)
		# ]

		for i in range(num_butterflies >> 2*stage):
			for j in range(4**stage):

				idx = base_idx + j + k*(1 << 2*(stage))

				X[idx] = np.dot(butterfly, X[idx])

	# bit reversed order
	for idx1, idx2 in radix4_bit_reversal(num_stages):
		X[[idx1,idx2]] = X[[idx2,idx1]]

	return X

if __name__ =='__main__':

	import fft

	def error(sig,ref):
		return abs(sig-ref).sum()

	for i in range(1_000):
		N   = np.random.randint(3,5)
		sig = np.random.rand(4**N).astype(np.complex128)

		SIG = radix4_fft(sig)
		REF = fft.radix4_fft(sig)

		abs(SIG).tofile('SIG.csv', sep = ',')
		abs(REF).tofile('REF.csv', sep = ',')

		print(error(SIG,REF))
	pass