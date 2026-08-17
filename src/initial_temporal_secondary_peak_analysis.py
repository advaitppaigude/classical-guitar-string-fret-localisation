import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import wavfile
from scipy.signal import find_peaks

# measured experimentally


#FOR Ei
start_times = [1.948, 2.88, 0.66, 2.56, 1.44]
end_times = [5.1, 7.5, 4.22, 6.73, 6.35]


"""
#FOR Eii
start_times = [3.34, 2.41, 2.84, 2.91, 3.56]
end_times = [8.27, 6.12, 8.08, 8.34, 8.69]
"""

"""
#FOR Eiii
start_times = [3.13, 3.49, 2.36, 4.21, 3.24]
end_times = [7.05, 7.09, 7.11, 8.34, 6.59]
"""


def get_normalised_waveform(filename):
	rate, data = wavfile.read(filename)
	start_sample = int(start_times[i] * rate)
	end_sample = int(end_times[i] * rate)
	aligned_data = data[start_sample:end_sample]
	max_amplitude = np.max(np.abs(aligned_data))
	normalised_data = np.divide(aligned_data, max_amplitude)
	return rate, normalised_data



def get_RMS_envelope(rate, data):
	RMS_data = []
	RMS_times = []
	window_samples = int(0.05 * rate)
	hop_samples = int(0.010 * rate)
	start_sample_index = 0
	while start_sample_index + window_samples < len(data):
		temp_RMS = np.sqrt(np.mean(np.square(data[start_sample_index:start_sample_index + window_samples])))
		RMS_data.append(temp_RMS)
		RMS_times.append((start_sample_index + window_samples/2)/rate)
		start_sample_index += hop_samples

	return RMS_times, RMS_data

def get_normalised_RMS_envelope(rate, data):
	times, RMS_data = get_RMS_envelope(rate, data)
	max_amplitude = np.max(np.abs(RMS_data))
	normalised_data = np.divide(RMS_data, max_amplitude)
	return times, normalised_data


peak_times = []
secondary_peak_times = []
secondary_peak_heights = []
secondary_peak_prominences = []

for i in range(5):
	rate, normalised_data = get_normalised_waveform(f"Ei_{i+1}.wav")
	RMS_times, RMS_data = get_normalised_RMS_envelope(rate, normalised_data)
	plt.plot(RMS_times, RMS_data, label = f"String 1 Fret 0 E {i+1}")

	plt.axhline(0.7, linestyle="--")

	# featurising decay properties
	peak_index = np.argmax(RMS_data)
	peak_time = RMS_times[peak_index]
	peak_times.append(peak_time)
	peak_value = RMS_data[peak_index]

	after_peak = RMS_data[peak_index:]
	below_70 = np.where(after_peak <= 0.7 * peak_value)[0]

	if len(below_70) == 0:
		continue

	start_of_secondary_peak_region_index = below_70[0] + peak_index
	end_of_secondary_peak_region_time = RMS_times[start_of_secondary_peak_region_index] + 0.75
	end_of_secondary_peak_region_index = np.searchsorted(RMS_times, end_of_secondary_peak_region_time)
	end_of_secondary_peak_region_index = min(end_of_secondary_peak_region_index, len(RMS_data))

	secondary_peak_region = RMS_data[start_of_secondary_peak_region_index:end_of_secondary_peak_region_index]

	peaks, properties = find_peaks(secondary_peak_region, prominence=(None, None))

	peaks_global = peaks + start_of_secondary_peak_region_index

	best_local = np.argmax(properties["prominences"])
	best_global_index = peaks_global[best_local]

	secondary_peak_time = RMS_times[best_global_index]
	secondary_peak_height = RMS_data[best_global_index]
	secondary_peak_prominence = properties["prominences"][best_local]

	secondary_peak_times.append(secondary_peak_time)
	secondary_peak_heights.append(secondary_peak_height)
	secondary_peak_prominences.append(secondary_peak_prominence)

	print(f"Recording {i+1}")
	print(secondary_peak_time)
	print(secondary_peak_height)
	print(secondary_peak_prominence)
	print()

	
	plt.scatter(
	    np.array(RMS_times)[best_global_index],
	    np.array(RMS_data)[best_global_index]
	)


print("Secondary Peak Times")
print("Mean: ", np.mean(secondary_peak_times))
print("Range: ", np.ptp(secondary_peak_times))
print("Standard deviation: ", np.std(secondary_peak_times))
print()

print("Secondary Peak Heights")
print("Mean: ", np.mean(secondary_peak_heights))
print("Range: ", np.ptp(secondary_peak_heights))
print("Standard deviation: ", np.std(secondary_peak_heights))
print()

print("Secondary Peak Prominences")
print("Mean: ", np.mean(secondary_peak_prominences))
print("Range: ", np.ptp(secondary_peak_prominences))
print("Standard deviation: ", np.std(secondary_peak_prominences))
print()



plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()
