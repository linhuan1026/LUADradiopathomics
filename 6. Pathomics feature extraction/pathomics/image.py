import os
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import large_image
import SimpleITK as sitk

Image.MAX_IMAGE_PIXELS = None


class AbstractImage(ABC):

    def __init__(self, path, params):
        self.path = path
        self.ext = '.' + self.path.split('.')[-1]
        self.name = path.split('/')[-1].replace(self.ext, '')
        self.params = params
        self.psize = params.get('patch_size', 256)
        self.stride = params.get('stride', self.psize)
        self.getScale()
        self.img = self.open()
        self.getSize()

    def getScale(self):
        scale = self.params.get('scale', 1)
        self.scale = float(scale)

    def setMask(self, mask_path):
        sitk_mask = sitk.ReadImage(mask_path)
        np_mask = sitk.GetArrayViewFromImage(sitk_mask)
        pres = np.max(np_mask * 1.0) / 2
        mask = mask > pres
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(np_mask)
        self.mask = mask.resize((self.width, self.height), Image.NEAREST)

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def getSize(self):
        pass

    @abstractmethod
    def getRegion(self, x, y, w=None, h=None):
        pass

    def getRegionOfMask(self, x, y, w=None, h=None):
        w = w if w != None else self.psize
        h = h if h != None else self.psize
        roi = self.img.crop((x, y, x + w, y + h))
        return np.array(roi).astype(np.uint8)

    def genPatchCoords(self):
        stride = self.stride
        psize = self.psize
        coords = []
        x = 0
        y = 0
        i = 0
        while True:
            if x + stride == self.width - psize + stride and y + stride == self.height - psize + stride:
                coords.append((x, y))
                break
            else:
                coords.append((x, y))
                x += stride
                if x == self.width - psize + stride:
                    x = 0
                    y += stride
                    if y > self.height - psize:
                        y = self.height - self.psize
                if x > self.width - psize:
                    x = self.width - self.psize
        return coords

    def setIterator(self, w, h=None, x_stride=None, y_stride=None):
        self.x = 0
        self.y = 0
        self.patchXStride = x_stride if x_stride != None else w
        self.patchYStride = y_stride if y_stride != None else w
        self.patchW = w
        self.patchH = h if h != None else w
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.x + self.patchXStride > self.width and self.y + self.patchYStride > self.height:
            raise StopIteration
        else:
            tmpX = self.x
            tmpY = self.y
            self.x += self.patchXStride
            if self.x > self.width:
                self.x = 0
                self.y += self.patchYStride
            roi = self.getRegion(tmpX, tmpY, self.patchW, self.patchH)
            return roi


class Region(AbstractImage):

    def __init__(self, path, params):
        super().__init__(path, params)

    def open(self):
        return Image.open(self.path)

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            w, h = self.img.size
            self.width = int(w / self.scale)
            self.height = int(h / self.scale)

    def getRegion(self, x, y, w=None, h=None):  # at curr mag
        w = w if w != None else self.psize
        h = h if h != None else self.psize
        x *= self.scale
        y *= self.scale
        W = w * self.scale
        H = h * self.scale
        x = int(x)
        y = int(y)
        W = int(W)
        H = int(H)
        roi = self.img.crop((x, y, x + W, y + H))
        roi = roi.resize((w, h), Image.BICUBIC)
        return np.array(roi).astype(np.uint8)


class WSI(AbstractImage):

    def __init__(self, path, params):
        super().__init__(path, params)

    def open(self):
        return large_image.getTileSource(self.path)

    def getSize(self):
        if hasattr(self, 'width') and hasattr(self, 'height'):
            return self.width, self.height
        else:
            w = self.img.sizeX
            h = self.img.sizeY
            self.width = int(w / self.scale)
            self.height = int(h / self.scale)

    def getRegion(self, x, y, w=None, h=None):  # at curr mag
        w = w if w != None else self.psize
        h = h if h != None else self.psize
        x *= self.scale
        y *= self.scale
        W = w * self.scale
        H = h * self.scale
        x = int(x)
        y = int(y)
        W = int(W)
        H = int(H)
        roi, _ = self.img.getRegion(
            region=dict(left=x, top=y, width=W, height=H),
            format=large_image.tilesource.TILE_FORMAT_PIL)
        roi = roi.resize((w, h), Image.BICUBIC)
        roi = roi.convert('RGB') 
        return np.array(roi).astype(np.uint8)


def readImage(img_path, params):
    name = os.path.basename(img_path)
    ext = name.split('.')[-1]
    if ext == 'tiff' or ext == 'tif' or ext == 'svs':
        img = WSI(img_path, params)
    else:
        img = Region(img_path, params)
    return img
