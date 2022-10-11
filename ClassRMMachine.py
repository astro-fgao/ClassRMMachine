class ClassRMMachine():
    """
    - input: I_array, Q_array, U_array, frequency list, channel width, reference frequency
    - I/Q/U array is the stokes I/Q/U intensity over frequency axis at the working pixel position
    - currently frequencys are in the unit of GHz
    """

    
    def __init__(self, I_array, Q_array, U_array, freq_array, chan_width, ref_freq, rm_min=-1000, rm_max = 1000, rm_step=1):
        self.I_array = I_array
        self.Q_array = Q_array
        self.U_array = U_array
        self.freq_array = freq_array
        self.chan_width = chan_width
        self.ref_freq = ref_freq
        self.n_chan = len(freq_array)
        self.wave_array = speed_of_light/self.freq_array/1e9
        self.wave_array2 = self.wave_array**2
        self.wave_array2_ref = (speed_of_light/self.ref_freq/1e9)**2
        self.rm_min = rm_min
        self.rm_max = rm_max
        self.rm_step = rm_step
        self.rm_chunk_n = int((self.rm_max - self.rm_min)/self.rm_step)

    def gauss(self,x,a,b,c):
        return a*np.exp(-(x-b)**2/2/c**2)
        
    
    def calc_rmsf(self,rm_min=-1000,rm_max= 1000,rm_step=1):
        """
        return the rmsf and the fwhm
        """
        # CHECK ME: should the freq_array and chan_freq be the same???
        
        rmsf = np.zeros(self.rm_chunk_n*2,dtype=complex).reshape(2,self.rm_chunk_n)

        for i in range(self.rm_chunk_n):
            rm_tmp = self.rm_min + i * self.rm_step
            rmsf_tmp = 0+0j
            for j in range(self.n_chan):
                rmsf_tmp += np.exp(-2*1j*rm_tmp*(self.wave_array2[j] - self.wave_array2_ref))
            rmsf[0,i] = rm_tmp
            rmsf[1,i] = rmsf_tmp
        popt,pcov = curve_fit(self.gauss,rmsf[0,:].real, rmsf[1,:].real)
        FWHM = 2*np.sqrt(2*np.log(2))*popt[2]
        print("calculating rmsf have successfully finished!")
        self.rmsf = rmsf
        self.fwhm = FWHM
        
    def calc_fdf(self):
        
        #chan_width = chan_width/1000. # convert from MHz to GHz
        #n_chan = 1000
        #chan_start_freq = 0.6

        #chan_freq = np.arange(n_chan)*chan_width + chan_start_freq # in GHz
        #chan_lambda = speed_of_light/chan_freq/1e9 # in meters
        #chan_lambda2 = chan_lambda**2
        #chan_lambda2_ref = np.mean(chan_lambda2)

        p_chan_frac = np.sqrt(self.Q_array**2+self.U_array**2)/self.I_array

        p_chan_angle = 0.5*np.arctan2(self.U_array,self.Q_array)

        
        #complex polarization per channel
        p_chan = p_chan_frac*np.exp(2*1j*p_chan_angle)          #checked with simple case already

        fdf = np.zeros(self.rm_chunk_n*2,dtype=complex).reshape(2,self.rm_chunk_n)
        #rmsf = np.zeros(rm_chunk_n*2,dtype=complex).reshape(2,rm_chunk_n)

        for i in range(self.rm_chunk_n):
            rm_tmp = self.rm_min + i*self.rm_step
            fdf_tmp = 0+0j
            #rmsf_tmp = 0+0j
            for j in range(self.n_chan):
                fdf_tmp +=  p_chan[j]*np.exp(-2*1j*rm_tmp*(self.wave_array2[j] - self.wave_array2_ref))
            fdf[1,i] = fdf_tmp
            fdf[0,i] = rm_tmp

        print("calculating fdf have successfully finished!!")
        self.fdf = fdf
        #return fdf, rmsf, rm_min, rm_max, rm_step, chan_lambda2, p_chan_angle, n_chan, chan_width, chan_start_freq

    def plot_rmsf(self, file_out_name='test_rmsf.png'):
        rm_axis = np.arange(self.rm_chunk_n)*self.rm_step + self.rm_min
        
        fig = plt.figure()

        ax2 = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        
        ax2.plot(self.rmsf[0,:],self.rmsf[1,:],color='C0')
            
        ax2.set_xlim([self.rm_min,self.rm_max])
        
        ax2.set_ylabel("RM spread function")
        
        ax2.set_xlabel(''r'$\phi$  [rad  'r'$m^{-2}$]')
        
        popt,pcov = curve_fit(self.gauss,self.rmsf[0,:].real, self.rmsf[1,:].real)
        FWHM = 2*np.sqrt(2*np.log(2))*popt[2]
        print("popt=",popt)
        fake_y=self.gauss(rm_axis,popt[0],popt[1],popt[2])
        ax2.plot(rm_axis,fake_y,color='C1')
        ax2.annotate('FWHM (rad  'r'$m^{-2}$) =',color='C1',xy=(0.65,0.85),xycoords='axes fraction',size=10)
        ax2.annotate(r'${%d}$'%(FWHM),color='C1',xy=(0.92,0.85),xycoords='axes fraction',size=10)

        plt.savefig(file_out_name)
        plt.close()
        print("saving plots in", file_out_name)
        
    def check_fdf(self):
        '''
        check whether the fdf contains significant peak.
        Input:
                fdf --- the calculated Faraday depth function
                fwhm --- the fitted FWHM of the RMSF
        If yes, return a boolean and also the peak of the fdf
        If no, just return a boolean.
        '''
        abs_fdf = np.absolute(self.fdf)
        abs_fdf_peak = np.max(abs_fdf[1,:])
        abs_fdf_peak_pos = np.argmax(abs_fdf[1,:])
        abs_fdf_std = np.std(np.concatenate((abs_fdf[1,:abs_fdf_peak_pos-int(self.fwhm/2)],abs_fdf[1,abs_fdf_peak_pos+int(self.fwhm/2):])))
        fdf_peak_snr = abs_fdf_peak/abs_fdf_std
        if fdf_peak_snr >=3:
            self.fdf_peak_sign = True
            self.abs_fdf_peak_pos = abs_fdf_peak_pos
        else:
            self.fdf_peak_sign = False
            self.abs_fdf_peak_pos = -999
