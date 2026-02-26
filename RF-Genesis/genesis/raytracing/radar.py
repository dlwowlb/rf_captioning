import torch
import matplotlib.pyplot as plt
import numpy as np
import json

def FSPL(distance, fc_hz=77e9):
    """Free Space Path Loss.
    
    Args:
        distance: path length (meters)
        fc_hz: carrier frequency in Hz (default 77GHz)
    """
    wavelength = 3e8 / fc_hz
    return (wavelength / (4 * 3.14159265358979323846 * distance)) ** 2

def dechirp(x,xref):
    return xref * torch.conj(x)

class Radar:

    def __init__(self, config="config.json"):
        self.c0 = 299792458

        if isinstance(config, str):
            with open(config, 'r') as f:
                config = json.load(f)
        elif not isinstance(config, dict):
            raise ValueError("config must be a filename or dict")


        self.num_tx = config["num_tx"]
        self.num_rx = config["num_rx"]
        self.slope = config["slope"]
        self.adc_samples = config["adc_samples"]
        self.adc_start_time = config["adc_start_time"]
        self.sample_rate = config["sample_rate"]
        self.idle_time = config["idle_time"]
        self.ramp_end_time = config["ramp_end_time"]
        self.chirp_per_frame = config["chirp_per_frame"]
        self.frame_per_second = config["frame_per_second"]
        self.num_doppler_bins = config["num_doppler_bins"]
        self.num_range_bins = config["num_range_bins"]
        self.num_angle_bins = config["num_angle_bins"]
        self.power = config["power"]

        # ============================================================
        # FIX: Normalize fc to Hz internally, regardless of config unit
        # TI configs typically use GHz (e.g. fc=77)
        # Some configs may use Hz (e.g. fc=77e9)
        # ============================================================
        fc_raw = config["fc"]
        if fc_raw < 1e6:       # GHz (e.g. 77)
            self.fc_hz = fc_raw * 1e9
        else:                   # already Hz (e.g. 77e9)
            self.fc_hz = fc_raw

        self._lambda = self.c0 / self.fc_hz
        antenna_spacing = self._lambda / 2   # correct: ~1.95mm for 77GHz

        self.tx_loc = np.array(config["tx_loc"],dtype = np.float32)*antenna_spacing
        self.rx_loc = np.array(config["rx_loc"],dtype = np.float32)*antenna_spacing

        self.tx_pos = torch.tensor(self.tx_loc)
        self.rx_pos = torch.tensor(self.rx_loc)

        # slope is in MHz/μs -> Hz/s = slope * 1e12
        # sample_rate is in ksps -> sps = sample_rate * 1e3
        # idle_time, ramp_end_time in μs
        self.range_resolution = (self.c0 * self.sample_rate * 1e3) / (2 * self.slope * 1e12 * self.adc_samples)
        self.max_range = (self.c0 * self.sample_rate * 1e3) / (2 * self.slope * 1e12) / 2

        chirp_period_s = (self.idle_time + self.ramp_end_time) * 1e-6
        self.doppler_resolution = self.c0 / (2 * self.fc_hz * chirp_period_s * self.num_doppler_bins * self.num_tx)
        self.max_doppler = self.c0 / (4 * self.fc_hz * chirp_period_s * self.num_tx)

    
    def waveform(self,t,phi=0): 
        # FIX: use self.fc_hz (Hz) consistently
        fc = (self.fc_hz * t + 0.5 * (self.slope * 1e12) *  t * t )
        yI = torch.cos( 2 * torch.pi * fc +phi)
        yQ = torch.sin( 2 * torch.pi  * fc +phi)
        y =  yI+yQ*1j
        return y

    def chirp(self,distance,intensity):
        t_sample = torch.arange(0,self.adc_samples,dtype=torch.float64)/(self.sample_rate*1e3) +self.adc_start_time*1e-6
        # FIX: pass fc_hz to FSPL instead of hardcoded 77e9
        loss = FSPL(distance, fc_hz=self.fc_hz).view(-1,1)
        intensity = intensity.view(-1,1)
        tof = distance / self.c0
        
        tx = self.waveform(t_sample)
        rx = self.waveform(t_sample-tof)*  loss * intensity

        rx_combined = torch.sum(rx,axis=0)
        sig = dechirp(rx_combined,tx)
        return sig
    
    def frame(self,interpolator,t0, n_ray = 1, sensor_origin=None):
        frame = torch.zeros((self.chirp_per_frame, self.adc_samples),dtype = torch.complex128)
        
        
        # Offset antenna positions to sensor location in world coordinates
        origin_offset = torch.zeros(3, dtype=self.tx_pos.dtype, device=self.tx_pos.device)
        if sensor_origin is not None:
            origin_offset = torch.tensor(sensor_origin, dtype=self.tx_pos.dtype, device=self.tx_pos.device)

        
        for chirp_id in range(self.chirp_per_frame):
            time_in_frame = chirp_id / self.chirp_per_frame / self.frame_per_second
            intensity,loc =  interpolator(t0+time_in_frame)
            
            
            tx_pos = self.tx_pos[0].unsqueeze(0) + origin_offset.unsqueeze(0)
            rx_pos = self.rx_pos[0].unsqueeze(0) + origin_offset.unsqueeze(0)
            tof = torch.cdist(loc,tx_pos)+ torch.cdist(loc,rx_pos) 

            chirp =  self.chirp(tof,intensity)
            frame[chirp_id,:] =chirp
        return frame
    

    def frameMIMO(self,interpolator,t0=0, n_ray = 1, sensor_origin=None):
        frame = torch.zeros((self.num_tx, self.num_rx, self.chirp_per_frame, self.adc_samples),dtype = torch.complex128)
        
        # Offset antenna positions to sensor location in world coordinates.
        # tx_pos/rx_pos are mm-scale offsets around (0,0,0); sensor_origin is
        # the world-coordinate position where the radar physically sits.
        origin_offset = torch.zeros(3, dtype=self.tx_pos.dtype, device=self.tx_pos.device)
        if sensor_origin is not None:
            origin_offset = torch.tensor(sensor_origin, dtype=self.tx_pos.dtype, device=self.tx_pos.device)
 
        for chirp_id in range(self.chirp_per_frame): 
            
            # it is inefficient to calculate the location for every sample point, so we do that for every chirp
            # this is a tradeoff between accuracy and computation
            
            
            time_in_frame = chirp_id / self.chirp_per_frame  / self.frame_per_second
            intensity,loc = interpolator(t0+time_in_frame) # loc is a set of reflection points (N,3)
            for tx_id in range(self.num_tx):
                for rx_id in range(self.num_rx):
                    tx_pos = self.tx_pos[tx_id].unsqueeze(0) + origin_offset.unsqueeze(0)
                    rx_pos = self.rx_pos[rx_id].unsqueeze(0) + origin_offset.unsqueeze(0)  # Convert shape from (3,) to (1, 3)
                    tof = torch.cdist(loc,tx_pos)+ torch.cdist(loc,rx_pos)    # Tx - Surface - Rx
                    frame[tx_id,rx_id,chirp_id,:] = self.chirp(tof,intensity)
        return frame