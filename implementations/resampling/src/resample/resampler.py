from fractions import Fraction

import numpy as np

class resampler:

	def __init__(
			self, 
			origin_Fs : int, # sampling frequency of the original signal
			target_Fs : int, # sampling frequency we want to have
			filter    : list[float] # filter for anti-aliasing
		):

		# stores the inputs
		self.origin_Fs = origin_Fs
		self.target_Fs = target_Fs
		self.filter    = filter

		# finds minimal L and M
		self.L, self.M = self._get_min_sample_factors()

		# partitions the original filter into filter banks
		self.filter_banks = self._get_filter_banks()

		if self.L > self.M:
			self.FSM = self._build_FSM_upsample()
		elif self.L < self.M:
			raise NotImplementedError("Have not made a down sampling case yet")
		else:
			raise NotImplementedError("Trivial case of target == origin will not be supported")


	def _get_min_sample_factors(self) -> tuple[int]:

		# used to find simplest terms
		ratio = Fraction(self.target_Fs, self.origin_Fs)

		return ratio.numerator, ratio.denominator

	def _get_filter_banks(self):

		# cached to local scope
		L = self.L

		# seperates the original filter into banks of stride L
		banks = [
			self.filter[i::L] for i in range(L)
		]

		# gets the largest filter length to pad other filter to
		# NOTE: it may be feasible to check the first filter for
		#		the max length but would need to be proven for all
		#		corner cases.
		largest_filter = max(
			len(filter) for filter in banks
		)

		# pads filters to uniform length
		for idx, bank in enumerate(banks):
			while len(bank) < largest_filter:
				bank.append(0)

			banks[idx] = np.array(bank)

		return banks

	def _build_FSM_upsample(self):
		'''
			Assumes L > M. This implies for every sample passed you will get at least
			1 sample for the target system
		'''

		# brings filter into main scope
		filter_banks = self.filter_banks
		L = self.L
		M = self.M

		# creates list of filters
		head_filter = [filter_banks[idx] for idx in range(0,L,M)]
		filters = [head_filter]

		# sets the m index for the next 
		m_idx = M*len(head_filter) - L

		while m_idx != 0:

			# sets m_idx to a positive value
			m_idx = m_idx + M

			new_filters = [
				filter_banks[idx] for idx in range(m_idx,L,M)
			]

			filters.append(new_filters)

		filter_list = [linked_list(filter) for filter in filters]
		filter_next = filter_list[1:].append(filter_list[0])

		for head, next in zip(filter_list, filter_next):
			head.next = next

		return filter_list[0]

	# sub for use later
	def __call__(self, data):
		pass

class linked_list:
	def __init__(self,filter):
		self.filter = filter
		self.next = None