import ClassRMMachine as rm
import numpy as np

channel_freq_GHz = np.array([1.4 , 1.42, 1.44, 1.46, 1.48, 1.5 , 1.52, 1.54, 1.56, 1.58])

t = rm.ClassRMMachine(channel_freq_GHz,channel_freq_GHz[1]-channel_freq_GHz[0],channel_freq_GHz[0])

t.read_from_fits('test-dimg.fits')
t.make_Q2U2()
t.find_peak()
t.calc_rmsf()
t.calc_fdf()
t.plot_fdf_rmsf()
t.check_fdf()

# the peak in fdf will be saved in t.abs_fdf_peak_pos
# the pixel position is saved in t.peak_x and t.peak_y
