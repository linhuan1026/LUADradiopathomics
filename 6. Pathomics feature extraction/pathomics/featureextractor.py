# -*- coding: utf-8 -*-
from __future__ import print_function

import collections
from itertools import chain
import json
import logging
import os
import pathlib

import pykwalify.core
import SimpleITK as sitk
import six
import numpy as np

from pathomics import generalinfo, getFeatureClasses, getImageTypes, getParameterValidationFiles, imageoperations

#
import multiprocessing
from multiprocessing import Pool, Manager
from itertools import repeat
import scipy.io as sio
from .mask import *
from .image import *
#from PIL import Image

#Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
geometryTolerance = None


class PathomicsFeatureExtractor:
    r"""
    Wrapper class for calculation of a pathomics signature.
    At and after initialisation various settings can be used to customize the resultant signature.
    This includes which classes and features to use, as well as what should be done in terms of preprocessing the image.

    Then a call to :py:func:`execute` generates the pathomics
    signature specified by these settings for the passed image and mask (or auto mask type) combination. This function can be called
    repeatedly in a batch process to calculate the pathomics signature for all image and mask combinations.

    At initialization, a parameters file (string pointing to yaml or json structured file) or dictionary can be provided
    containing all necessary settings (top level containing keys "setting", "imageType" and/or "featureClass). This is
    done by passing it as the first positional argument. If no positional argument is supplied, or the argument is not
    either a dictionary or a string pointing to a valid file, defaults will be applied.
    Moreover, at initialisation, custom settings (*NOT enabled image types and/or feature classes*) can be provided
    as keyword arguments, with the setting name as key and its value as the argument value (e.g. ``mask=histoqc``).
    Settings specified here will override those in the parameter file/dict/default settings.
    For more information on possible settings and customization, see
    :ref:`Customizing the Extraction <pathomics-customization-label>`.

    By default, all features in all feature classes are enabled.
    """

    def __init__(self, *args, **kwargs):
        global logger

        self.settings = {}
        self.enabledImagetypes = {}
        self.enabledFeatures = {}

        self.featureClassNames = list(getFeatureClasses().keys())

        if len(args) == 1 and isinstance(args[0], dict):
            logger.info("Loading parameter dictionary")
            self._applyParams(paramsDict=args[0])
        elif len(args) == 1 and (isinstance(args[0], six.string_types)
                                 or isinstance(args[0], pathlib.PurePath)):
            if not os.path.isfile(args[0]):
                raise IOError("Parameter file %s does not exist." % args[0])
            logger.info("Loading parameter file %s", str(args[0]))
            self._applyParams(paramsFile=args[0])
        else:
            # Set default settings and update with and changed settings contained in kwargs
            self.settings = self._getDefaultSettings()
            logger.info('No valid config parameter, using defaults: %s',
                        self.settings)

            self.enabledImagetypes = {'Original': {}}
            logger.info('Enabled image types: %s', self.enabledImagetypes)

            for featureClassName in self.featureClassNames:
                self.enabledFeatures[featureClassName] = []
            logger.info('Enabled features: %s', self.enabledFeatures)

        if len(kwargs) > 0:
            logger.info('Applying custom setting overrides: %s', kwargs)
            self.settings.update(kwargs)
            logger.debug("Settings: %s", self.settings)

        self._setTolerance()

    def _setTolerance(self):
        global geometryTolerance, logger
        geometryTolerance = self.settings.get('geometryTolerance')
        if geometryTolerance is not None:
            logger.debug('Setting SimpleITK tolerance to %s',
                         geometryTolerance)
            sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(
                geometryTolerance)
            sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(
                geometryTolerance)

    def addProvenance(self, provenance_on=True):
        """
        Enable or disable reporting of additional information on the extraction. This information includes toolbox version,
        enabled input images and applied settings. Furthermore, additional information on the image and region of interest
        (ROI) is also provided, including original image spacing, total number of voxels in the ROI and total number of
        fully connected volumes in the ROI.

        To disable this, call ``addProvenance(False)``.
        """
        self.settings['additionalInfo'] = provenance_on

    @staticmethod
    def _getDefaultSettings():
        """
        Returns a dictionary containg the default settings specified in this class. These settings cover global settings,
        such as ``additionalInfo``, as well as the image pre-processing settings (e.g. resampling). Feature class specific
        are defined in the respective feature classes and and not included here. Similarly, filter specific settings are
        defined in ``imageoperations.py`` and also not included here.
        """
        # yuxin, need to check here after some modifications.
        return {
            'minimumROIDimensions': 2,
            'minimumROISize': None,  # Skip testing the ROI size by default
            'normalize': False,
            'normalizeScale': 1,
            'removeOutliers': None,
            'resampledPixelSpacing': None,  # No resampling by default
            'interpolator': 'sitkBSpline',  # Alternative: sitk.sitkBSpline
            'preCrop': False,
            'padDistance': 5,
            'distances': [1],
            'force2D': False,
            'force2Ddimension': 0,
            'resegmentRange': None,  # No resegmentation by default
            'label': 1,
            'additionalInfo': True
        }

    def loadParams(self, paramsFile):
        """
        Parse specified parameters file and use it to update settings, enabled feature(Classes) and image types. For more
        information on the structure of the parameter file, see
        :ref:`Customizing the extraction <pathomics-customization-label>`.

        If supplied file does not match the requirements (i.e. unrecognized names or invalid values for a setting), a
        pykwalify error is raised.
        """
        self._applyParams(paramsFile=paramsFile)

    def loadJSONParams(self, JSON_configuration):
        """
        Pars JSON structured configuration string and use it to update settings, enabled feature(Classes) and image types.
        For more information on the structure of the parameter file, see
        :ref:`Customizing the extraction <pathomics-customization-label>`.

        If supplied string does not match the requirements (i.e. unrecognized names or invalid values for a setting), a
        pykwalify error is raised.
        """
        parameter_data = json.loads(JSON_configuration)
        self._applyParams(paramsDict=parameter_data)

    def _applyParams(self, paramsFile=None, paramsDict=None):
        """
        Validates and applies a parameter dictionary. See :py:func:`loadParams` and :py:func:`loadJSONParams` for more info.
        """
        global logger

        # Ensure pykwalify.core has a log handler (needed when parameter validation fails)
        if len(pykwalify.core.log.handlers) == 0 and len(
                logging.getLogger().handlers) == 0:
            # No handler available for either pykwalify or root logger, provide first pathomics handler (outputs to stderr)
            pykwalify.core.log.addHandler(
                logging.getLogger('pathomics').handlers[0])

        schemaFile, schemaFuncs = getParameterValidationFiles()
        c = pykwalify.core.Core(source_file=paramsFile,
                                source_data=paramsDict,
                                schema_files=[schemaFile],
                                extensions=[schemaFuncs])
        params = c.validate()
        logger.debug('Parameters parsed, input is valid.')

        enabledImageTypes = params.get('imageType', {})
        enabledFeatures = params.get('featureClass', {})
        settings = params.get('setting', {})
        voxelSettings = params.get('voxelSetting', {})

        logger.debug("Applying settings")

        if len(enabledImageTypes) == 0:
            self.enabledImagetypes = {'Original': {}}
        else:
            self.enabledImagetypes = enabledImageTypes

        logger.debug("Enabled image types: %s", self.enabledImagetypes)

        if len(enabledFeatures) == 0:
            self.enabledFeatures = {}
            for featureClassName in self.featureClassNames:
                self.enabledFeatures[featureClassName] = []
        else:
            self.enabledFeatures = enabledFeatures

        logger.debug("Enabled features: %s", self.enabledFeatures)

        # Set default settings and update with and changed settings contained in kwargs
        self.settings = self._getDefaultSettings()
        self.settings.update(settings)
        self.settings.update(voxelSettings)

        logger.debug("Settings: %s", settings)

    def mergeResults(self, res):
        wsiFeatureVector = collections.OrderedDict()
        for featureVector in res:
            for featureName in featureVector.keys(
            ):  # the feature is still a dict
                feature = featureVector[featureName]
                if wsiFeatureVector.get(featureName) is None:
                    wsiFeatureVector[featureName] = feature
                else:
                    wsiFeatureVector[featureName] += feature

        return wsiFeatureVector

    def multiExecute(self, imageFilepath, maskFilepath):
        n_workers = self.settings.get('workers',
                                      multiprocessing.cpu_count() // 2)
        manager = Manager()
        res = manager.dict()
        if isinstance(imageFilepath,
                      six.string_types) and os.path.isfile(imageFilepath):
            image = readImage(imageFilepath, self.settings)
        if isinstance(maskFilepath,
                      six.string_types) and os.path.isfile(maskFilepath):
            image.setMask(maskFilepath)
            maskisImage = True
        else:
            maskisImage = False
        coords = image.genPatchCoords()
        patches = []
        masks = []
        for coord in coords:
            x, y = coord
            patch = image.getRegion(x, y)
            patch = sitk.GetImageFromArray(patch)
            if maskisImage:
                mask = image.getRegionOfMask(x, y)
                mask = sitk.GetImageFromArray(mask)
            else:
                mask = maskFilepath
            patches.append(patch)
            masks.append(mask)

        with Pool(processes=n_workers) as pool:
            res = pool.starmap(self.execute, zip(patches, masks))
        # for i in range(len(patches)):
        #     patch = patches[i]
        #     mask = masks[i]
        #     res.append(self.execute(patch, mask))
        return self.mergeResults(res)

    def execute(self,
                imageFilepath,
                maskFilepath,
                label=None,
                label_channel=None,
                voxelBased=False, 
                ifbinarization=True):
        """
        yuxin, delete some unused codes
        Compute pathomics signature for provide image and mask combination. It comprises of the following steps:

        1. Image and mask are loaded and normalized/resampled if necessary.
        2. The calculated features is returned as ``collections.OrderedDict``.

        Some params below are useless in our case.
        :param imageFilepath: SimpleITK Image, numpy.ndarray, or string pointing to image file location
        :param maskFilepath: SimpleITK Image, numpy.ndarray, or string pointing to labelmap file location
        :param label: Integer, value of the label for which to extract features. If not specified, last specified label
            is used. Default label is 1.
        :param label_channel: Integer, index of the channel to use when maskFilepath yields a SimpleITK.Image with a vector
            pixel type. Default index is 0.
        :param voxelBased: Boolean, default False. If set to true, a voxel-based extraction is performed, segment-based
            otherwise.
        :returns: dictionary containing calculated signature ("<imageType>_<featureClass>_<featureName>":value).
            In case of segment-based extraction, value type for features is float, if voxel-based, type is SimpleITK.Image.
            Type of diagnostic features differs, but can always be represented as a string.
        """

        global geometryTolerance, logger
        _settings = self.settings.copy()

        tolerance = _settings.get('geometryTolerance')
        additionalInfo = _settings.get('additionalInfo', False)
        resegmentShape = _settings.get('resegmentShape', False)

        if label is not None:
            _settings['label'] = label
        else:
            label = _settings.get('label', 1)

        if label_channel is not None:
            _settings['label_channel'] = label_channel

        if ifbinarization is not None:
            _settings['ifbinarization'] = ifbinarization

        if geometryTolerance != tolerance:
            self._setTolerance()

        if additionalInfo:
            generalInfo = generalinfo.GeneralInfo()
            generalInfo.addGeneralSettings(_settings)
            generalInfo.addEnabledImageTypes(self.enabledImagetypes)
        else:
            generalInfo = None

        if voxelBased:
            _settings['voxelBased'] = True
            kernelRadius = _settings.get('kernelRadius', 1)
            logger.info('Starting voxel based extraction')
        else:
            kernelRadius = 0

        logger.debug('Enabled features: %s', self.enabledFeatures)
        logger.debug('Current settings: %s', _settings)

        # 1. Load the image and mask
        featureVector = collections.OrderedDict()
        if isinstance(imageFilepath,
                      six.string_types) and os.path.isfile(imageFilepath):
            image = self.load2DImage(imageFilepath, generalInfo, **_settings)
        elif isinstance(imageFilepath, sitk.SimpleITK.Image):
            image = imageFilepath
        elif isinstance(imageFilepath, np.ndarray):
            image = sitk.GetImageFromArray(imageFilepath)
        else:
            raise ValueError('Unkown image Type', imageFilepath.__class__)
        if isinstance(maskFilepath, np.ndarray) or isinstance(
                maskFilepath,
                six.string_types) and os.path.isfile(maskFilepath):
            mask = self.load2DMask(image, maskFilepath, generalInfo,
                                   **_settings)
        elif isinstance(maskFilepath, six.string_types):
            # Generate machine masks depends on mask type, like hovernet nuclear mask
            mask = self.loadAutoMask(image, maskFilepath, generalInfo,
                                     **_settings)
        else:
            raise ValueError('Error reading Mask')

        # 2. Calculate other enabled feature classes using enabled image types
        # Make generators for all enabled image types
        logger.debug('Creating image type iterator')
        imageGenerators = []
        for imageType, customKwargs in six.iteritems(self.enabledImagetypes):
            args = _settings.copy()
            args.update(customKwargs)
            logger.info('Adding image type "%s" with custom settings: %s' %
                        (imageType, str(customKwargs)))
            imageGenerators = chain(
                imageGenerators,
                getattr(imageoperations, 'get%sImage' % imageType)(image, mask,
                                                                   **args))

        logger.debug('Extracting features')
        # Calculate features for all (filtered) images in the generator
        inputMask = mask
        for inputImage, imageTypeName, inputKwargs in imageGenerators:
            logger.info('Calculating features for %s image', imageTypeName)
            featureVector.update(
                self.computeFeatures(inputImage, inputMask, imageTypeName,
                                     **inputKwargs))

        logger.debug('Features extracted')

        return featureVector

    @staticmethod
    def load2DImage(ImageFilePath, generalInfo=None, **kwargs):
        # Quick 2d image load, and keep sitk format
        image = sitk.ReadImage(ImageFilePath)
        #print(sitk.GetArrayViewFromImage(image).shape)
        #print('image dimension:', image.GetDimension())
        return image

    @staticmethod
    def load2DMask(image, MaskFilePath, generalInfo=None, ifbinarization=True, **kwargs):
        # Quick 2d mask load, and keep sitk format
        if isinstance(MaskFilePath,
                      six.string_types) and os.path.isfile(MaskFilePath):
            mask = Image.open(MaskFilePath)
        else:
            mask = MaskFilePath
        np_mask = np.array(mask)
        np_image = sitk.GetArrayViewFromImage(image)
        if np_image.shape != np_mask.shape:
            logger.info('Mask size not match Image, Resizing Mask')

            tar_shape = np_image.shape
            mask = Image.fromarray(np_mask)
            mask = mask.resize(size=(tar_shape[1], tar_shape[0]),
                               resample=Image.NEAREST)   ##Image.BILINEAR is for image. Image.NEAREST for label

        np_mask = np.array(mask)
        np_mask = np_mask.astype(np.uint8)
        if len(np_mask.shape) < 3:
            new_mask = np.zeros((np_mask.shape[0], np_mask.shape[1], 3))
            new_mask[:, :, 0] = np_mask
            new_mask[:, :, 1] = np_mask
            new_mask[:, :, 2] = np_mask
            np_mask = new_mask

        if ifbinarization:  
            # in this case, .astype(bool) is proper. 
            # np.max(np_mask * 1.0) / 2 is not correct for extracting the nuclear-wise features, but it's OK for tissue features
            mask = np_mask.astype(bool)
        else:
            mask = np_mask

        # # pres = np.max(np_mask * 1.0) / 2
        # # mask = np_mask > pres
        # mask = np_mask 

        mask = mask.astype(np.uint8)
        mask = sitk.GetImageFromArray(mask)
        #print('mask dimension', mask.GetDimension())

        return mask

    @staticmethod
    def loadAutoMask(np_image, mask, generalInfo=None, **kwargs):
        # we can automatically generate masks
        # TODO
        if mask == 'hovernet':
            selectedMask = getHoverNetMask(np_image, **kwargs)
        elif mask == 'abc':
            selectedMask = getABC(np_image, **kwargs)
        elif mask == 'histoqc':
            selectedMask = getHistoQCMask(np_image, **kwargs)
        else:
            raise ValueError(f'Error generating Mask, unknown type {mask}')

        mask = sitk.GetImageFromArray(selectedMask)
        return mask

    def computeFeatures(self, image, mask, imageTypeName, **kwargs):
        r"""
        Compute signature using image, mask and \*\*kwargs settings.

        This function computes the signature for just the passed image (original or derived), it does not pre-process or
        apply a filter to the passed image. Features / Classes to use for calculation of signature are defined in
        ``self.enabledFeatures``. See also :py:func:`enableFeaturesByName`.

        :param image: The cropped (and optionally filtered) SimpleITK.Image object representing the image used
        :param mask: The cropped SimpleITK.Image object representing the mask used
        :param imageTypeName: String specifying the filter applied to the image, or "original" if no filter was applied.
        :param kwargs: Dictionary containing the settings to use for this particular image type.
        :return: collections.OrderedDict containing the calculated features for all enabled classes.
        If no features are calculated, an empty OrderedDict will be returned.

        .. note::

        shape descriptors are independent of gray level and therefore calculated separately (handled in `execute`). In
        this function, no shape features are calculated.
        """
        global logger
        featureVector = collections.OrderedDict()
        featureClasses = getFeatureClasses()

        enabledFeatures = self.enabledFeatures
        # debug

        # Calculate feature classes
        for featureClassName, featureNames in six.iteritems(enabledFeatures):

            if featureClassName in featureClasses:
                logger.info('Computing %s', featureClassName)

                featureClass = featureClasses[featureClassName](image, mask,
                                                                **kwargs)

                if featureNames is not None:
                    for feature in featureNames:
                        featureClass.enableFeatureByName(feature)

                for (featureName,
                     featureValue) in six.iteritems(featureClass.execute()):
                    # newFeatureName = '%s_%s_%s' % (
                    #     imageTypeName, featureClassName, featureName)
                    newFeatureName = '%s_%s' % (featureClassName, featureName)
                    featureVector[newFeatureName] = featureValue

        return featureVector

    def enableAllImageTypes(self):
        """
        Enable all possible image types without any custom settings.
        """
        global logger

        logger.debug('Enabling all image types')
        for imageType in getImageTypes():
            self.enabledImagetypes[imageType] = {}
        logger.debug('Enabled images types: %s', self.enabledImagetypes)

    def disableAllImageTypes(self):
        """
        Disable all image types.
        """
        global logger

        logger.debug('Disabling all image types')
        self.enabledImagetypes = {}

    def enableImageTypeByName(self, imageType, enabled=True, customArgs=None):
        r"""
        Enable or disable specified image type. If enabling image type, optional custom settings can be specified in
        customArgs.

        Current possible image types are:

        - Original: No filter applied
        - Wavelet: Wavelet filtering, yields 8 decompositions per level (all possible combinations of applying either
        a High or a Low pass filter in each of the three dimensions.
        See also :py:func:`~pathomics.imageoperations.getWaveletImage`
        - LoG: Laplacian of Gaussian filter, edge enhancement filter. Emphasizes areas of gray level change, where sigma
        defines how coarse the emphasised texture should be. A low sigma emphasis on fine textures (change over a
        short distance), where a high sigma value emphasises coarse textures (gray level change over a large distance).
        See also :py:func:`~pathomics.imageoperations.getLoGImage`
        - Square: Takes the square of the image intensities and linearly scales them back to the original range.
        Negative values in the original image will be made negative again after application of filter.
        - SquareRoot: Takes the square root of the absolute image intensities and scales them back to original range.
        Negative values in the original image will be made negative again after application of filter.
        - Logarithm: Takes the logarithm of the absolute intensity + 1. Values are scaled to original range and
        negative original values are made negative again after application of filter.
        - Exponential: Takes the the exponential, where filtered intensity is e^(absolute intensity). Values are
        scaled to original range and negative original values are made negative again after application of filter.
        - Gradient: Returns the gradient magnitude.
        - LBP2D: Calculates and returns a local binary pattern applied in 2D.
        - LBP3D: Calculates and returns local binary pattern maps applied in 3D using spherical harmonics. Last returned
        image is the corresponding kurtosis map.

        For the mathmetical formulas of square, squareroot, logarithm and exponential, see their respective functions in
        :ref:`imageoperations<pathomics-imageoperations-label>`
        (:py:func:`~pathomics.imageoperations.getSquareImage`,
        :py:func:`~pathomics.imageoperations.getSquareRootImage`,
        :py:func:`~pathomics.imageoperations.getLogarithmImage`,
        :py:func:`~pathomics.imageoperations.getExponentialImage`,
        :py:func:`~pathomics.imageoperations.getGradientImage`,
        :py:func:`~pathomics.imageoperations.getLBP2DImage` and
        :py:func:`~pathomics.imageoperations.getLBP3DImage`,
        respectively).
        """
        global logger

        if imageType not in getImageTypes():
            logger.warning('Image type %s is not recognized', imageType)
            return

        if enabled:
            if customArgs is None:
                customArgs = {}
                logger.debug(
                    'Enabling image type %s (no additional custom settings)',
                    imageType)
            else:
                logger.debug(
                    'Enabling image type %s (additional custom settings: %s)',
                    imageType, customArgs)
            self.enabledImagetypes[imageType] = customArgs
        elif imageType in self.enabledImagetypes:
            logger.debug('Disabling image type %s', imageType)
            del self.enabledImagetypes[imageType]
        logger.debug('Enabled images types: %s', self.enabledImagetypes)

    def enableImageTypes(self, **enabledImagetypes):
        """
        Enable input images, with optionally custom settings, which are applied to the respective input image.
        Settings specified here override those in kwargs.
        The following settings are not customizable:

        - interpolator
        - resampledPixelSpacing
        - padDistance

        Updates current settings: If necessary, enables input image. Always overrides custom settings specified
        for input images passed in inputImages.
        To disable input images, use :py:func:`enableInputImageByName` or :py:func:`disableAllInputImages`
        instead.

        :param enabledImagetypes: dictionary, key is imagetype (original, wavelet or log) and value is custom settings
        (dictionary)
        """
        global logger

        logger.debug('Updating enabled images types with %s',
                     enabledImagetypes)
        self.enabledImagetypes.update(enabledImagetypes)
        logger.debug('Enabled images types: %s', self.enabledImagetypes)

    def enableAllFeatures(self):
        """
        Enable all classes and all features.

        .. note::
        Individual features that have been marked "deprecated" are not enabled by this function. They can still be enabled
        manually by a call to :py:func:`~pathomics.base.PathomicsBase.enableFeatureByName()`,
        :py:func:`~pathomics.featureextractor.PathomicsFeaturesExtractor.enableFeaturesByName()`
        or in the parameter file (by specifying the feature by name, not when enabling all features).
        However, in most cases this will still result only in a deprecation warning.
        """
        global logger

        logger.debug('Enabling all features in all feature classes')
        for featureClassName in self.featureClassNames:
            self.enabledFeatures[featureClassName] = []
        logger.debug('Enabled features: %s', self.enabledFeatures)

    def disableAllFeatures(self):
        """
        Disable all classes.
        """
        global logger

        logger.debug('Disabling all feature classes')
        self.enabledFeatures = {}

    def enableFeatureClassByName(self, featureClass, enabled=True):
        """
        Enable or disable all features in given class.

        .. note::
        Individual features that have been marked "deprecated" are not enabled by this function. They can still be enabled
        manually by a call to :py:func:`~pathomics.base.PathomicsBase.enableFeatureByName()`,
        :py:func:`~pathomics.featureextractor.PathomicsFeaturesExtractor.enableFeaturesByName()`
        or in the parameter file (by specifying the feature by name, not when enabling all features).
        However, in most cases this will still result only in a deprecation warning.
        """
        global logger

        if featureClass not in self.featureClassNames:
            logger.warning('Feature class %s is not recognized', featureClass)
            return

        if enabled:
            logger.debug('Enabling all features in class %s', featureClass)
            self.enabledFeatures[featureClass] = []
        elif featureClass in self.enabledFeatures:
            logger.debug('Disabling feature class %s', featureClass)
            del self.enabledFeatures[featureClass]
        logger.debug('Enabled features: %s', self.enabledFeatures)

    def enableFeaturesByName(self, **enabledFeatures):
        """
        Specify which features to enable. Key is feature class name, value is a list of enabled feature names.

        To enable all features for a class, provide the class name with an empty list or None as value.
        Settings for feature classes specified in enabledFeatures.keys are updated, settings for feature classes
        not yet present in enabledFeatures.keys are added.
        To disable the entire class, use :py:func:`disableAllFeatures` or :py:func:`enableFeatureClassByName` instead.
        """
        global logger

        logger.debug('Updating enabled features with %s', enabledFeatures)
        self.enabledFeatures.update(enabledFeatures)
        logger.debug('Enabled features: %s', self.enabledFeatures)
