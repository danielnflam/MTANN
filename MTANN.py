import torch
import torch.nn as nn
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import os, sys, time, datetime, pathlib, random, math

import transforms

import skimage.transform as sktf
##########################################
# MTANN model in K. Suzuki, H. Abe, H. MacMahon and K. Doi, "Image-processing technique for suppressing ribs in chest radiographs by means of massive training artificial neural network 
# (MTANN)," in IEEE Transactions on Medical Imaging, vol. 25, no. 4, pp. 406-416, April 2006, doi: 10.1109/TMI.2006.871549.
##########################################

# HELPER FUNCTIONS
def load_networks(first_layer=81, second_layer=20, path_HR=None, path_MR=None, path_LR=None):
    netHR = linearOutputANN(first_layer=81, second_layer=20)
    netMR = linearOutputANN(first_layer=81, second_layer=20)
    netLR = linearOutputANN(first_layer=81, second_layer=20)
    
    if path_HR is not None and path_MR is not None and path_LR is not None:
        netHR.load_state_dict(torch.load('./netHR.pt')["model_state_dict"])
        netMR.load_state_dict(torch.load('./netMR.pt')["model_state_dict"])
        netLR.load_state_dict(torch.load('./netLR.pt')["model_state_dict"])
    return netHR, netMR, netLR


def save_model(path_to_save, net, optimizer, loss, reals_shown_now):
    torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'reals_shown': reals_shown_now
            }, path_to_save)
    print("Saved File at: "+ str(path_to_save))

class ImageReconstruction(object):
    def __init__(self, sample, key_source, key_boneless=None, matrix_size=9, stride=1):
        self.key_source = key_source
        self.key_boneless = key_boneless
        self.matrix_size=matrix_size
        self.stride = stride
        
        self.OverlappingSubregions = OverlappingSubregions(self.key_source, self.key_boneless, self.matrix_size, self.stride)
        if key_boneless is not None:
            keys_sample = [self.key_source, self.key_boneless]
        else:
            keys_sample = [self.key_source]
        self.dataPreprocessing = dataPreprocessing(sample, keys_sample)
        self.image_shape_HR = self.dataPreprocessing.HR[self.key_source].shape
    
    def reconstruct_multiresolution(self,model_HR,model_MR,model_LR):
        image_LR = self.reconstruct_from_model( model_LR, self.dataPreprocessing.LR)
        image_MR = self.reconstruct_from_model( model_MR, self.dataPreprocessing.MR)
        image_HR = self.reconstruct_from_model( model_HR, self.dataPreprocessing.HR)
        
        image_MR_upscaled_from_LR = sktf.resize(image_LR, output_shape=(self.image_shape_HR[0]//2, self.image_shape_HR[1]//2), order=0, anti_aliasing=False)
        image_MR_upscaled_from_LR = image_MR_upscaled_from_LR + self.dataPreprocessing.MR2LR_difference[self.key_source]
        image_HR_upscaled_from_LR = sktf.resize(image_MR_upscaled_from_LR, output_shape=(self.image_shape_HR[0], self.image_shape_HR[1]), order=0, anti_aliasing=False)
        image_HR_upscaled_from_LR = image_HR_upscaled_from_LR + self.dataPreprocessing.HR2MR_difference[self.key_source]
        
        image_HR_upscaled_from_MR = sktf.resize(image_MR, output_shape=(self.image_shape_HR[0], self.image_shape_HR[1]), order=0, anti_aliasing=False)
        image_HR_upscaled_from_MR = image_HR_upscaled_from_MR + self.dataPreprocessing.HR2MR_difference[self.key_source]
        
        # Composition of HR upscaled from MR and from LR, as well as with HR native analysis
        image = image_HR + image_HR_upscaled_from_MR + image_HR_upscaled_from_MR
        return image, image_HR_upscaled_from_LR, image_HR_upscaled_from_MR, image_HR
        
    # Reconstruction from model for a single image
    def reconstruct_from_model(self, model, resolutionImageSample):
        LR_image = np.zeros(resolutionImageSample[self.key_source].shape)
        side = self.matrix_size//2
        for ii, data in enumerate(self.OverlappingSubregions.execute(resolutionImageSample)):
            subregion, _, center_pixel = data
            pixel_output = model(subregion)
            LR_image[center_pixel[0]-side , center_pixel[1]-side] = pixel_output
        return LR_image
        
    
class OverlappingSubregions(object):
    """
    SOURCE IMAGE: this is the original radiograph
    BONELESS IMAGE: this is the ideal, bone-suppressed training image counterpart to the SOURCE IMAGE
    """
    def __init__(self, key_source, key_boneless=None, matrix_size=9, stride=1):
        self.key_source = key_source
        self.key_boneless = key_boneless
        self.matrix_size=matrix_size
        self.stride = stride
        
    def execute(self, sample):
        """
        Generator sweeps from left-to-right (image.shape[1]), then from top to bottom (image.shape[0]) of the image.
        Run this execute function in a for loop, e.g. for i, data in enumerate(OverlappingSubregions.execute(sample)): ...
        
        Input:
            sample: a dict with keys that include key_source and key_boneless. The value for key_source and key_boneless are numpy ndarrays of dimension [HxW].
        Output:
            Generator that yields:
                subregion: a matrix_size x matrix_size square of the image
                target_pixel: the pixel value at the center of the square for the CORRESPONDING BONELESS IMAGE
    
        """
        side = self.matrix_size//2 # floor division
        image = np.pad(sample[self.key_source], side, mode='reflect')
        if self.key_boneless is not None:
            boneless = np.pad(sample[self.key_boneless], side, mode='reflect')
        
        for h in range(side, image.shape[0]-side, self.stride):
            for w in range(side, image.shape[1]-side, self.stride):
                center_pixel = (h,w)
                subregion = image[h-side:h+side+1,w-side:w+side+1]
                
                # Transform outputs into torch tensors
                subregion = torch.from_numpy(np.expand_dims(subregion,(0,1)))
                if self.key_boneless is not None:
                    target_pixel = boneless[h,w]
                    target_pixel = torch.from_numpy(np.expand_dims(target_pixel,(0,1)))
                else:
                    target_pixel = None
                yield subregion, target_pixel, center_pixel
        

class dataPreprocessing(object):
    def __init__(self, sample, sample_keys_images):
        """
        This class is used to store the original (HR) image, the downsampled images (MR and LR), and the difference images (HR2MR and MR2LR).
        Images are numpy ndarrays.
        Difference images are higher-res minus lower-res.
        """
        self.sample_keys_images = sample_keys_images
        
        # For training
        self.HR = {}
        self.MR = {}
        self.LR = {}
        self.HR2MR_difference = {}
        self.MR2LR_difference = {}
        for key_idx in self.sample_keys_images:
            image = sample[key_idx]
            self.HR[key_idx] = image
            HR_h = image.shape[-2]
            HR_w = image.shape[-1]
            self.MR[key_idx] = sktf.resize(image, output_shape=(HR_h//2, HR_w//2), order=1, anti_aliasing=False)
            MR_h = self.MR[key_idx].shape[-2]
            MR_w = self.MR[key_idx].shape[-1]
            self.LR[key_idx] = sktf.resize(self.MR[key_idx], output_shape=(MR_h//2, MR_w//2), order=1, anti_aliasing=False)
            self.HR2MR_difference[key_idx] = self.HR[key_idx] - sktf.resize(self.MR[key_idx], (HR_h, HR_w), order=0)
            self.MR2LR_difference[key_idx] = self.MR[key_idx] - sktf.resize(self.LR[key_idx], (MR_h, MR_w), order=0)

"""    def execute(self, key_source, key_boneless):
        #Train 3 MTANNs -- 1 for each resolution.
        #This behaves like a generator, otherwise we get OOM errors
        
        if key_source not in self.sample_keys_images:
            RuntimeError("key_source is not in the self.sample_keys_images")
        if key_boneless not in self.sample_keys_images:
            RuntimeError("key_boneless is not in the self.sample_keys_images")
        
        # Transform the source images into subregions
        HR_subregions, HR_center_pixels = self._transformToSubregions( self.HR, key_source, key_boneless)
        MR_subregions, MR_center_pixels = self._transformToSubregions( self.MR, key_source, key_boneless)
        LR_subregions, LR_center_pixels = self._transformToSubregions( self.LR, key_source, key_boneless)
        return HR_subregions, HR_center_pixels, MR_subregions, MR_center_pixels, LR_subregions, LR_center_pixels
        
        
    def _transformToSubregions(self, image_dict, key_source, key_boneless, matrix_size=9, stride=1):
        ovs = OverlappingSubregions([key_source], matrix_size, stride) 
        subregions, center_pixels = ovs.execute(image_dict[key_source]) 
        # subregions is [NxHxW] where N is the number of subregions
        # center_pixels is a list of [Nx1] tuples, each tuple is (h,w)
        
        # Subregion
        subregions = np.expand_dims(subregions, axis=1) # Subregions are the batches -- 1 channel
        subregions = torch.from_numpy(subregions) # to torch tensor
        
        # Boneless center pixel
        center_pixels = np.asarray(center_pixels)
        h = center_pixels[:,0]
        w = center_pixels[:,1]
        center_pixels = image_dict[key_boneless][h,w]
        center_pixels = torch.from_numpy(center_pixels)
        #print(subregions.shape)
        #print(center_pixels.shape)
        return subregions, center_pixels"""
        

##################
# TORCH network to be trained
##################
class linearOutputANN(nn.Module):
    def __init__(self, first_layer=81, second_layer=20):
        super().__init__()
        self.first_layer_nodes = first_layer
        self.second_layer_nodes = second_layer
        self.third_layer_nodes = 1
        
        self.flatten = nn.Flatten()
        self.first_layer = nn.Linear(self.first_layer_nodes, self.second_layer_nodes)
        self.second_layer = nn.Linear(self.second_layer_nodes, self.third_layer_nodes)
        
        self.sigmoidFunction = nn.Sigmoid()
        self.linearFunction = nn.ReLU()
        
    def forward(self, x):
        out = self.flatten(x)
        # Activate the input layer
        out = self.linearFunction(out)
        # Input2Hidden
        out = self.first_layer(out)
        out = self.sigmoidFunction(out)
        # Hidden2Output
        out = self.second_layer(out)
        out = self.linearFunction(out)
        return out
        
class linearActivationFunction(nn.Module):
    def __init__(self, a, b):
        self.gradient = a
        self.offset = b
    def forward(self, x):
        return self.gradient*a + self.offset
