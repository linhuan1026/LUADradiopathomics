# 3.Radiomics features extraction

---

## How to use

---

Prepare:

```
pip install -r requirements.txt
```

Run:

```
python GetRadiomicsFeatures.py --imgDir FOLDER_DIR --saveDir FEATURES_FILE_DIR
```

Number of all features 4248, of which 17 (shape) * 5 (number of ROIs) + 19 (firstorder) * 9 (original image plus 8 wavelet transforms) * 5 (number of ROIs) + 24 (glcm) * 9 * 5 + 16 (glrlm) * 9 * 5 + 16 (glszm) * 9 * 5 + 5 (ngtdm) * 9 * 5 + 14 (gldm) * 9 * 5 + 1 (thickness)

features:

```
shape: #17
 - MeshVolume
 - VoxelVolume
 - SurfaceArea
 - SurfaceVolumeRatio
 - Sphericity
 - Compactness1
 - Compactness2
 - SphericalDisproportion
 - Maximum3DDiameter
 - Maximum2DDiameterSlice
 - Maximum2DDiameterColumn
 - Maximum2DDiameterRow
 - MajorAxisLength
 - MinorAxisLength
 - LeastAxisLength
 - Elongation
 - Flatness
 firstorder: #19
 - Energy
 - TotalEnergy
 - Entropy
 - Minimum
 - 10Percentile
 - 90Percentile
 - Maximum
 - Mean
 - Median
 - InterquartileRange
 - Range
 - MeanAbsoluteDeviation
 - RobustMeanAbsoluteDeviation
 - RootMeanSquared
 - StandardDeviation
 - Skewness
 - Kurtosis
 - Variance
 - Uniformity
 glcm: #24
 - Autocorrelation
 - JointAverage
 - ClusterProminence
 - ClusterShade
 - ClusterTendency
 - Contrast
 - Correlation
 - DifferenceAverage
 - DifferenceEntropy
 - DifferenceVariance
 - Dissimilarity
 - JointEnergy
 - JointEntropy
 - Imc1
 - Imc2
 - Idm
 - MCC
 - Idmn
 - Id
 - Idn
 - InverseVariance
 - MaximumProbability
 - SumAverage
 - SumVariance
 - SumEntropy
 - SumSquares
 glrlm: #16
 - ShortRunEmphasis
 - LongRunEmphasis
 - GrayLevelNonUniformity
 - GrayLevelNonUniformityNormalized
 - RunLengthNonUniformity
 - RunLengthNonUniformityNormalized
 - RunPercentage
 - GrayLevelVariance
 - RunVariance
 - RunEntropy
 - LowGrayLevelRunEmphasis
 - HighGrayLevelRunEmphasis
 - ShortRunLowGrayLevelEmphasis
 - ShortRunHighGrayLevelEmphasis
 - LongRunLowGrayLevelEmphasis
 - LongRunHighGrayLevelEmphasis
 glszm: #16
 - SmallAreaEmphasis
 - LargeAreaEmphasis
 - GrayLevelNonUniformity
 - GrayLevelNonUniformityNormalized
 - SizeZoneNonUniformity
 - SizeZoneNonUniformityNormalized
 - ZonePercentage
 - GrayLevelVariance
 - ZoneVariance
 - ZoneEntropy
 - LowGrayLevelZoneEmphasis
 - HighGrayLevelZoneEmphasis
 - SmallAreaLowGrayLevelEmphasis
 - SmallAreaHighGrayLevelEmphasis
 - LargeAreaLowGrayLevelEmphasis
 - LargeAreaHighGrayLevelEmphasis
 ngtdm: #5
 - Coarseness
 - Contrast
 - Busyness
 - Complexity
 - Strength
 gldm: #14
 - SmallDependenceEmphasis
 - LargeDependenceEmphasis
 - GrayLevelNonUniformity
 - DependenceNonUniformity
 - DependenceNonUniformityNormalized
 - GrayLevelVariance
 - DependenceVariance
 - DependenceEntropy
 - LowGrayLevelEmphasis
 - HighGrayLevelEmphasis
 - SmallDependenceLowGrayLevelEmphasis
 - SmallDependenceHighGrayLevelEmphasis
 - LargeDependenceLowGrayLevelEmphasis
 - LargeDependenceHighGrayLevelEmphasis
```
