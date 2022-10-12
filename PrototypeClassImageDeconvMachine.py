import numpy as np

channel_freq_GHz = np.array([1.4 , 1.42, 1.44, 1.46, 1.48, 1.5 , 1.52, 1.54, 1.56, 1.58])

t = rm.ClassRMMachine(channel_freq_GHz,channel_freq_GHz[1]-channel_freq_GHz[0],channel_freq_GHz[0])

DirtyCube=t.read_from_fits('test-dimg.fits') # shape = (4, nch, nx,ny)
SpectralPSF=t.read_from_fits('psf.fits') # shape = (1, nch, NpixPSF_x,NpixPSF_y)

DirtyCube=DirtyCube[1:3,...]

t.make_Q2U2()

# initialisation
t.calc_rmsf()

ResidualCube=DirtyCube.copy()


while True:
    x,y=t.find_peak(ResidualCube)
    # SpectralPSF=self.PSFServer.givePSF(x,y)
    
    #RM_1d=t.calc_fdf(x,y)
    #t.plot_fdf_rmsf()
    DeconvComponant=t.deconv_1d(x,y) # {"pos":(x,y),RM,QU_0,QU_1,QU_etc}
    IQUV_1D=t.RMPixelComponant_to_1D_sprecta(DeconvComponant)
    # IQUV_1D.shape should be npol,nchan

    # append componant to model (for bookmarking) - that 
    # ModelMachine.append_componant(DeconvComponant)
    

    # subtract spectral componant convolved by spectral psf (centered at x,y) to spectral dirty   
    _,_,nxDirty,nyDirty=ResidualCube.shape
    _,_,NpixPSF_x,NpixPSF_y=SpectralPSF.shape
    Aedge,Bedge=GiveEdgesDissymetric(x,y,nxDirty,nyDirty, NpixPSF_x//2,NpixPSF_y//2,NpixPSF_x,NpixPSF_y)
    x0d,x1d,y0d,y1d=Aedge
    x0p,x1p,y0p,y1p=Bedge
    ResidualCube_sub, SpectralPSF_sub = ResidualCube[:,:,x0d:x1d,y0d:y1d], SpectralPSF[:,:,x0p:x1p,y0p:y1p]
    
    ConvolvedSpectralComponant_sub=gain*IQUV_1D.reshape((npol,nchan,1,1))*SpectralPSF_sub
    ResidualCube_sub-=ResidualCube_sub
    
# Output: ModelMachine



# class ClassImageDeconvMachine():
#     def __init__(self):
#         pass
#     def Init(self, **kwargs):
#         self.SetPSF(kwargs["PSFVar"])
#         self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
#         self.Freqs = kwargs["GridFreqs"]
#         AllDegridFreqs = []
#         for i in kwargs["DegridFreqs"].keys():
#             AllDegridFreqs.append(kwargs["DegridFreqs"][i])
#         self.Freqs_degrid = np.unique(np.concatenate(AllDegridFreqs).flatten())
#         self.SetPSF(kwargs["PSFVar"])
#         self.setSideLobeLevel(kwargs["PSFAve"][0], kwargs["PSFAve"][1])
#         self.ModelMachine.setPSFServer(self.PSFServer)
#         self.ModelMachine.setFreqMachine(self.Freqs, self.Freqs_degrid,
#                                          weights=kwargs["PSFVar"]["WeightChansImages"], PSFServer=self.PSFServer)
    
#     def setMaskMachine(self,MaskMachine):
#         self.MaskMachine=MaskMachine
