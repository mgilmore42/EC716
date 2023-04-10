import numpy as np

def radix4_fft(x):

	N = len(x)

	if N == 1:
		return x

	# Split the input into four parts
	x0 = x[ ::4]
	x1 = x[1::4]
	x2 = x[2::4]
	x3 = x[3::4]

	# Recursively compute the FFT of each part
	X0 = radix4_fft(x0)
	X1 = radix4_fft(x1)
	X2 = radix4_fft(x2)
	X3 = radix4_fft(x3)

	# Combine the results using twiddle factors
	X = np.zeros(N, dtype=np.complex128)
	for k in range(N//4):
		# twiddle1 = np.exp(-2j * np.pi *   k / N)
		# twiddle2 = np.exp(-2j * np.pi * 2*k / N)
		# twiddle3 = np.exp(-2j * np.pi * 3*k / N)
		twiddle1 = 1
		twiddle2 = 1
		twiddle3 = 1
		t1 = X1[k] * twiddle1
		t2 = X2[k] * twiddle2
		t3 = X3[k] * twiddle3
		X[k         ] = X0[k] +    t1 + t2 +    t3
		X[k +   N//4] = X0[k] - 1j*t1 - t2 + 1j*t3
		X[k +   N//2] = X0[k] -    t1 + t2 -    t3
		X[k + 3*N//4] = X0[k] + 1j*t1 - t2 - 1j*t3

	return X

if __name__ =='__main__':

	def error(sig,ref):
		return abs(sig-ref).sum()

	for i in range(1_000):
		N   = np.random.randint(3,5)
		sig = np.random.rand(4**N).astype(np.complex128)

		SIG = radix4_fft(sig)
		REF = np.fft.fft(sig)

		assert error(SIG,REF) < 1e-12
	pass