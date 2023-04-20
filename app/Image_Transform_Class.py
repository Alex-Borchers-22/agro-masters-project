# -*- coding: utf-8 -*-
"""
Created on Thu Apr 6 2023
@author: Alex Borcherss
"""

import numpy as np
from PIL import Image #to open images
from PIL import ImageOps #to open images

class Image_MR:
    """
    This class performs metamorphic relations on a bitmap of an image

    """

    def __init__(self, image_data):
        """
        This function controls the initial creation of image.

        Parameters
        ----------
        image_data : array
            bitmap of an image

        Attributes
        -------
        image_data : array
            The current state of the image data
        """

        # Set image_data
        self.image_data = image_data
    
    #def getImageData(self, reshape=True):
        """
        This function returns the image data as an array

        Parameters
        ----------
        reshape : boolean
            True if image needs to be flatted (reshape(-1)).

        Returns
        -------
        image_data : list
            current state of bitmap for the given image.
        """
        #return self.image_data
        
    def reshapeBitmap(self):
        """
        This function reshapes the input data using np.array().reshape (flattens to 1-D list)

        """
        self.image_data = np.array(self.image_data).reshape(-1)

    def modifyRGB(self, rgb_channel):
        """
        This function modifies the RGB channel of the image as defined by the user

        Parameters
        ----------
        rgb_channel : string
            The new order of the rgb channel of an image

        """

        # Based case, channel is rgb (do nothing)
        if (rgb_channel == "rgb"):
            return

        # Split into 3 channels
        r, g, b = self.image_data.split()

        # Make decision on based on user input
        if rgb_channel == "bgr":
            self.image_data = Image.merge('RGB', (b, g, r))
        elif rgb_channel == "brg":
            self.image_data = Image.merge('RGB', (b, r, g))
        elif rgb_channel == "gbr":
            self.image_data = Image.merge('RGB', (g, b, r))
        elif rgb_channel == "grb":
            self.image_data = Image.merge('RGB', (g, r, b))
        elif rgb_channel == "rbg":
            self.image_data = Image.merge('RGB', (r, b, g))

    def modifyByConstant(self, constant):
        """
        This function modifies the bitmap by multiplying by a constant

        Parameters
        ----------
        constant : float
            The value to multiply the bitmap by

        """
        self.image_data = self.image_data * float(constant)

    def modifyByTransform(self, transform):
        """
        This function modifies the bitmap by performing some transformation
            ex. 90 degree turn, 180 degree turn, 270 degree turn, mirror

        Parameters
        ----------
        transform : string
            The type of transformation that will take place

        """

        if transform == "rotate90":
            self.image_data = self.image_data.rotate(90)
        elif transform == "rotate180":
            self.image_data = self.image_data.rotate(180)
        elif transform == "rotate270":
            self.image_data = self.image_data.rotate(270)
        elif transform == "mirror":
            self.image_data = self.image_data.transpose(method=Image.FLIP_LEFT_RIGHT)
    
    def modifyByNormalizing(self, modify):
        """
        This function modifies the bitmap by normalizing the data (dividing by standard deviation of the data set)

        Parameters
        ----------
        modify : String
            User specified decision variable

        """

        if modify != "None":
            st_dev = np.std(self.image_data)
            self.image_data = self.image_data / float(st_dev)

    def modifyByInverting(self, modify):
        """
        This function modifies the bitmap by inverting the data (0 => 255, 1 => 254,..., 255 => 0)

        Parameters
        ----------
        modify : String
            User specified decision variable

        """

        if modify != "None":
            self.image_data = ImageOps.invert(self.image_data)