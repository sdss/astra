from peewee import (
    BitField as _BitField,
    VirtualField,
    ColumnBase
)
from astropy.io import fits
from astra.utils import expand_path

class BitField(_BitField):
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
    def __init__(self, model, field, name, ext, column_name, transform=None):
        self.model = model
        self.field = field
        self.name = name
        self.ext = ext
        self.column_name = column_name 
        self.transform = transform

    def __set__(self, instance, value):
        try:
            instance.__pixel_data__
        except AttributeError:
            instance.__pixel_data__ = {}
        finally:
            instance.__pixel_data__[self.name] = value
        return None


class PixelArrayAccessorFITS(BasePixelArrayAccessor):

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except AttributeError:
                # Load them all.
                instance.__pixel_data__ = {}
                with fits.open(expand_path(instance.path)) as image:
                    for name, accessor in instance._meta.pixel_fields.items():
                        
                        if callable(accessor.ext):
                            ext = accessor.ext(instance)
                        else:
                            ext = accessor.ext
                    
                        data = image[ext].data
                        
                        try:
                            value = data[accessor.column_name] # column acess
                        except:
                            value = data # image access
                        
                        
                        if accessor.transform is not None:
                            value = accessor.transform(value)
                        
                        instance.__pixel_data__.setdefault(name, value)
                
                return instance.__pixel_data__[self.name]

        return self.field
    

class PixelArrayAccessorHDF(BasePixelArrayAccessor):

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            try:
                return instance.__pixel_data__[self.name]
            except AttributeError:
                # Load them all.
                instance.__pixel_data__ = {}
                import h5py
                with h5py.File(instance.path, "r") as fp:
                    for name, accessor in instance._meta.pixel_fields.items():
                        value = fp[accessor.column_name][instance.row_index]
                        if accessor.transform is not None:
                            value = accessor.transform(value)
                        
                        instance.__pixel_data__.setdefault(name, value)
                
                return instance.__pixel_data__[self.name]

        return self.field
    


class PixelArray(VirtualField):

    def __init__(self, ext=None, column_name=None, transform=None, accessor_class=PixelArrayAccessorFITS, **kwargs):
        super(PixelArray, self).__init__(**kwargs)
        self.ext = ext
        self.column_name = column_name
        self.transform = transform
        self.accessor_class = accessor_class

    def bind(self, model, name, set_attribute=True):
        self.model
        self.name = self.safe_name = name
        self.column_name = self.column_name or name
        attr = self.accessor_class(model, self, name, self.ext, self.column_name, self.transform)
        if set_attribute:
            setattr(model, name, attr)
        
        try:
            model._meta.pixel_fields
        except:
            model._meta.pixel_fields = {}
        finally:
            model._meta.pixel_fields[name] = attr
