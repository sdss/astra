import numpy as np
import pickle
from peewee import (
    BitField as _BitField,
    VirtualField,
    ColumnBase
)
import h5py

from astropy.io import fits
from astra.utils import expand_path

class BitField(_BitField):

    """A binary bitfield field that allows for `help_text` to be specified in each `FlagDescriptor`."""
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('default', 0)
        super(_BitField, self).__init__(*args, **kwargs)
        self.__current_flag = 1

    def flag(self, value=None, help_text=None):
        if value is None:
            value = self.__current_flag
            self.__current_flag <<= 1
        else:
            self.__current_flag = value << 1

        class FlagDescriptor(ColumnBase):
            def __init__(self, field, value, help_text=None):
                self._field = field
                self._value = value
                self.help_text = help_text
                super(FlagDescriptor, self).__init__()
            def clear(self):
                return self._field.bin_and(~self._value)
            def set(self):
                return self._field.bin_or(self._value)
            def __get__(self, instance, instance_type=None):
                if instance is None:
                    return self
                value = getattr(instance, self._field.name) or 0
                return (value & self._value) != 0
            def __set__(self, instance, is_set):
                if is_set not in (True, False):
                    raise ValueError('Value must be either True or False')
                value = getattr(instance, self._field.name) or 0
                if is_set:
                    value |= self._value
                else:
                    value &= ~self._value
                setattr(instance, self._field.name, value)
            def __sql__(self, ctx):
                return ctx.sql(self._field.bin_and(self._value) != 0)
        return FlagDescriptor(self, value, help_text)


class BasePixelArrayAccessor(object):
    
    """A base pixel array accessor."""

    def __init__(self, model, field, name, ext, column_name, transform=None, help_text=None):
        self.model = model
        self.field = field
        self.name = name
        self.ext = ext
        self.column_name = column_name 
        self.transform = transform
        self.help_text = help_text
        return None

    def __set__(self, instance, value):
        self._initialise_pixel_array(instance)

        instance.__pixel_data__[self.name] = value
        return None

    def _initialise_pixel_array(self, instance):
        try:
            instance.__pixel_data__
        except AttributeError:
            instance.__pixel_data__ = {}
        return None
        
class PickledPixelArrayAccessor(BasePixelArrayAccessor):
    
    """A class to access pixel arrays stored in a pickle file."""

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            self._initialise_pixel_array(instance)

            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                # Load them all.
                instance.__pixel_data__ = {}
                with open(expand_path(instance.path), "rb") as fp:
                    instance.__pixel_data__.update(pickle.load(fp))
                
                return instance.__pixel_data__[self.name]

        return self.field


class PixelArrayAccessorFITS(BasePixelArrayAccessor):
    
    """A class to access pixel arrays stored in a FITS file."""

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            self._initialise_pixel_array(instance)
            
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                # Load them all.
                instance.__pixel_data__ = {}
                with fits.open(expand_path(instance.path)) as image:
                    for name, accessor in instance._meta.pixel_fields.items():

                        if callable(accessor.ext):
                            ext = accessor.ext(instance)
                        else:
                            ext = accessor.ext

                        if ext is None:
                            # non-FITSy looking thing
                            continue
                    
                        data = image[ext].data
                        
                        try:
                            value = data[accessor.column_name] # column acess
                        except:
                            value = data # image access
                        
                        value = np.copy(value)
                        
                        if accessor.transform is not None:
                            value = accessor.transform(value, image, instance)
                        
                        instance.__pixel_data__.setdefault(name, value)
                
                return instance.__pixel_data__[self.name]

        return self.field
    

class PixelArrayAccessorHDF(BasePixelArrayAccessor):

    """A class to access pixel arrays stored in a HDF-5 file."""

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]
            except KeyError:
                # Load them all.
                with h5py.File(instance.path, "r") as fp:
                    for name, accessor in instance._meta.pixel_fields.items():
                        value = fp[accessor.column_name][instance.row_index]
                        if accessor.transform is not None:
                            value = accessor.transform(value)
                        
                        instance.__pixel_data__.setdefault(name, value)            
            finally:
                return instance.__pixel_data__[self.name]

        return self.field

class LogLambdaArrayAccessor(BasePixelArrayAccessor):

    def __init__(self, model, field, name, ext, column_name, crval, cdelt, naxis, transform=None, help_text=None):
        self.model = model
        self.field = field
        self.name = name
        self.ext = ext
        self.column_name = column_name 
        self.transform = transform
        self.help_text = help_text
        self.naxis = naxis
        self.crval = crval
        self.cdelt = cdelt

    def __get__(self, instance, instance_type=None):
        if instance is not None:         
            # If we have a manually set pixel array, we don't want to clear that by accident
            # when we access .wavelength                
            self._initialise_pixel_array(instance)
            try:
                return instance.__pixel_data__[self.name]            
            except KeyError:
                instance.__pixel_data__[self.name] = 10**(self.crval + self.cdelt * np.arange(self.naxis))
            finally:
                return instance.__pixel_data__[self.name]
            
        return self.field


class PixelArray(VirtualField):

    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=PixelArrayAccessorFITS, help_text=None, accessor_kwargs=None, **kwargs):
        super(PixelArray, self).__init__(**kwargs)
        self.ext = ext
        self.column_name = column_name
        self.transform = transform
        self.accessor_class = accessor_class
        self.help_text = help_text
        self.accessor_kwargs = accessor_kwargs

    def bind(self, model, name, set_attribute=True):
        self.model
        self.name = self.safe_name = name
        self.column_name = self.column_name or name
        attr = self.accessor_class(
            model, self, name, 
            ext=self.ext, column_name=self.column_name, 
            transform=self.transform, help_text=self.help_text,
            **(self.accessor_kwargs or {})
        )        
        if set_attribute:
            setattr(model, name, attr)
        
        try:
            model._meta.pixel_fields
        except:
            model._meta.pixel_fields = {}
        finally:
            model._meta.pixel_fields[name] = attr
