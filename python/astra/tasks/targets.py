
import os
import collections
import datetime
import json
import luigi
import numpy as np
import sqlalchemy
from astra.utils import symlink_force
from shutil import copy2 as copyfile
from time import gmtime, strftime
from astropy.io import fits
from astropy.table import Table
from luigi.contrib import sqla
from luigi import LocalTarget
from luigi.event import Event
from luigi.mock import MockTarget
        

class BaseDatabaseTarget(luigi.Target):
    
    """ A mixin class for DatabaseTargets. """

    _engine_dict = {}
    Connection = collections.namedtuple("Connection", "engine pid")

    def __init__(self, connection_string, task_namespace, task_family, task_id, schema, echo=True):
        self.task_namespace = task_namespace
        self.task_family = task_family
        self.task_id = task_id

        self.schema = [
            sqlalchemy.Column("task_id", sqlalchemy.String(128), primary_key=True),
        ]
        self.schema.extend(schema)
        self.echo = echo
        self.connect_args = {}
        self.connection_string = connection_string
        return None       


    @property
    def engine(self):
        """
        Return an engine instance, creating it if it doesn't exist.

        Recreate the engine connection if it wasn't originally created
        by the current process.
        """
        pid = os.getpid()
        conn = BaseDatabaseTarget._engine_dict.get(self.connection_string)
        if not conn or conn.pid != pid:
            # create and reset connection
            engine = sqlalchemy.create_engine(
                self.connection_string,
                connect_args=self.connect_args,
                echo=self.echo
            )
            BaseDatabaseTarget._engine_dict[self.connection_string] = self.Connection(engine, pid)
        return BaseDatabaseTarget._engine_dict[self.connection_string].engine


    @property
    def __tablename__(self):
        """ The name of the table in the database. """        
        # Don't allow '.' in table names!
        return self.task_family.replace(".", "_")
    

    @property
    def table_bound(self):
        """ A bound reflection of the database table. """
        try:
            return self._table_bound
        except AttributeError:
            return self.create_table()


    def create_table(self):
        """
        Create a table if it doesn't exist.
        Use a separate connection since the transaction might have to be reset.
        """
        with self.engine.begin() as con:
            metadata = sqlalchemy.MetaData()
            if not con.dialect.has_table(con, self.__tablename__):
                self._table_bound = sqlalchemy.Table(
                    self.__tablename__, metadata, *self.schema
                )
                metadata.create_all(self.engine)

            else:
                #metadata.reflect(only=[self.results_table], bind=self.engine)
                #self._table_bound = metadata.tables[self.results_table]
                self._table_bound = sqlalchemy.Table(
                    self.__tablename__,
                    metadata,
                    autoload=True,
                    autoload_with=self.engine
                )

        return self._table_bound
        

    def exists(self):
        """ Returns True or False whether the database target exists. """
        r = self._read(self.table_bound, columns=[self.table_bound.c.task_id]) 
        return r is not None


    def read(self, as_dict=False, include_parameters=True):
        """ 
        Read the target from the database. 

        :param as_dict: (optional)
            Optionally return the result as a dictionary (default: False).        
        """
        return self._read(
            self.table_bound, 
            as_dict=as_dict, 
            include_parameters=include_parameters
        )
        

    def _read(self, table, columns=None, as_dict=False, include_parameters=True):
        """
        Read a target row from the database table.

        :param table:
            The bound table to read from.
        
        :param columns: (optional)
            Optionally return only these specific columns (default: entire table).
        
        :param as_dict: (optional)
            Optionally return the result as a dictionary (default: False).
        """
        if columns is None and not include_parameters:
            N = 0
            for key, value in self.__class__.__dict__.items():
                if isinstance(value, sqlalchemy.Column):
                    N += 1
            
            columns = [column for column in table.columns][-N:]
            column_names = [column.name for column in columns]
            
        else:
            columns = columns or [table]
            column_names = (column.name for column in table.columns)

        with self.engine.begin() as connection:
            s = sqlalchemy.select(columns).where(
                table.c.task_id == self.task_id
            ).limit(1)
            row = connection.execute(s).fetchone()
        if as_dict:
            
            return collections.OrderedDict(zip(column_names, row))
        return row



    def write(self, data=None):
        """
        Write a result to the database target.

        :param data: (optional)
            A dictionary where keys represent column names and values represent the result value.
        """
        
        exists = self.exists()
        table = self.table_bound
        data = data or dict()
        sanitised_data = dict()
        for key, value in data.items():
            # Don't sanitise booleans or date/datetime objects.
            if not isinstance(value, (datetime.datetime, datetime.date, bool)):
                if value is None:
                    continue
                
                if isinstance(getattr(table.c, key).type, sqlalchemy.ARRAY):
                    if not isinstance(value, (tuple, list, np.ndarray)):
                        value = [value]
                else:
                    try:
                        value = str(value)
                    except:
                        value = json.dumps(value)
                
            sanitised_data[key] = value
    
        with self.engine.begin() as connection:
            if not exists:
                insert = table.insert().values(
                    task_id=self.task_id,
                    **sanitised_data
                )
            else:
                insert = table.update().where(
                    table.c.task_id == self.task_id
                ).values(
                    task_id=self.task_id,
                    **sanitised_data
                )
            connection.execute(insert)
        return None


    def delete(self):
        """
        Delete a result target from the database. 
        """

        table = self.table_bound
        with self.engine.begin() as connection:
            connection.execute(
                table.delete().where(table.c.task_id == self.task_id)
            )

        return None




class DatabaseTarget(BaseDatabaseTarget):

    """ 
    A database target for outputs of task results. 
    
    This class should be sub-classed, where the sub-class has attributes that define the schema. For example::

    ```
    class MyDatabaseTarget(DatabaseTarget):

        teff = sqlalchemy.Column("effective_temperature", sqlalchemy.Float)
        logg = sqlalchemy.Column("surface_gravity", sqlalchemy.Float)
    ```

    The `task_id` of the task supplied will be added as a column by default.


    :param task:
        The task that this output will be the target for. This is necessary to reference the task ID, and to generate the table
        schema for the task parameters.

    :param echo: [optional]
        Echo the SQL queries that are supplied (default: False).
    
    :param only_significant: [optional]
        When storing the parameter values of the task in a database, only store the significant parameters (default: True).

    """

    def __init__(self, task, echo=False, only_significant=True):
        
        self.task = task
        self.only_significant = only_significant

        schema = generate_parameter_schema(task, only_significant)

        for key, value in self.__class__.__dict__.items():
            if isinstance(value, sqlalchemy.Column):
                schema.append(value)

        super(DatabaseTarget, self).__init__(
            task.connection_string,
            task.task_namespace,
            task.task_family,
            task.task_id,
            schema,
            echo=echo
        )
        return None


    def write(self, data=None, mark_complete=True):
        """
        Write a result to the database target row.

        :param data: (optional)
            A dictionary where keys represent column names and values represent the result value.

        :param mark_complete: (optional)
            Trigger the event as being completed successfully (default: True)
        """

        # Update with parameter keyword arguments.
        data = (data or dict()).copy()
        if self.only_significant:
            for parameter_name in self.task.get_param_names():
                data[parameter_name] = getattr(self.task, parameter_name)
        else:
            data.update(self.task.param_kwargs)

        super(DatabaseTarget, self).write(data)
        if mark_complete:
            self.task.trigger_event(Event.SUCCESS, self.task)
    

    def remove(self):
        """
        Delete the target from the database.
        """

        raise NotImplementedError()


    def copy_from(self, source):
        """ 
        Copy a result from another DatabaseTarget.

        :param source:
            The source result to copy from.
        """
        return self.write(source.read(as_dict=True, include_parameters=False))




def generate_parameter_schema(task, only_significant=True):
    """
    Generate SQLAlchemy table schema for a task's parameters.

    :param task:
        The task to generate the table schema for.
    
    :param only_significant: (optional)
        Only generate columns for parameters that are defined as significant (default: True).
    """
    jsonify = lambda _: json.dumps(_)

    mapping = {
        # TODO: Including sanitizers if they are useful in future, but may not be needed.
        luigi.parameter.Parameter: (sqlalchemy.String(1024), None),
        luigi.parameter.OptionalParameter: (sqlalchemy.String(1024), None),
        luigi.parameter.DateParameter: (sqlalchemy.Date(), None),
        luigi.parameter.IntParameter: (sqlalchemy.Integer(), None),
        luigi.parameter.FloatParameter: (sqlalchemy.Float(), None),
        luigi.parameter.BoolParameter: (sqlalchemy.Boolean(), None),
        luigi.parameter.DictParameter: (sqlalchemy.String(1024), jsonify),
        luigi.parameter.ListParameter: (sqlalchemy.String(1024), jsonify),
        luigi.parameter.TupleParameter: (sqlalchemy.String(1024), jsonify)
    }
    parameters_schema = []
    for parameter_name, parameter_type in task.get_params():
        if only_significant and not parameter_type.significant:
            continue
        
        try:
            column_type, sanitize = mapping[parameter_type.__class__]
        except KeyError:
            raise ValueError(f"Cannot find mapping to parameter class {mapping_type.__class__}")
        parameters_schema.append(
            sqlalchemy.Column(
                parameter_name,
                column_type
            )
        )

    return parameters_schema




def create_image_hdu(data, header, name=None, dtype=None, bunit=None, **kwargs):
    kwds = dict(do_not_scale_image_data=True)
    kwds.update(**kwargs)
    hdu = fits.ImageHDU(
        data=data.astype(dtype or data.dtype),
        header=header,
        name=name,
        **kwds
    )
    if bunit is not None:
        hdu.header["BUNIT1"] = bunit
    return hdu


class AstraSource(LocalTarget):
    
    """
    A class to represent an analysis product from an analysis task in Astra.

    :param task:
        The reference task that was used to analyse the spectrum.
    """

    def __init__(self, task, **kwargs):
        # Store a reference of the task since we will need it to determine the path.
        self.task = task
        super(AstraSource, self).__init__(path=self._path)
        return None


    def get_bitmask(self, spectrum):
        """
        Return a bitmask array, given the spectrum. 
        The reason for this is because the BHM pipeline produces AND/OR bitmasks and
        the MWM pipeline produces a single bitmask.
        """
        try:
            # BHM/BOSS bitmask
            return spectrum.meta["bitmask"]["or_mask"]

        except:
            # MWM
            return spectrum.meta["bitmask"]
    

    def _check_shapes(self, normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table):
        """
        Check that all input array shapes are consistent.
        """

        if model_flux is None:
            model_flux = np.nan * np.ones_like(normalized_flux)
        if model_ivar is None:
            model_ivar = np.nan * np.ones_like(normalized_flux)
        if continuum is None:
            continuum = np.nan * np.ones_like(normalized_flux)

        normalized_flux = np.atleast_2d(normalized_flux)
        normalized_ivar = np.atleast_2d(normalized_ivar)
        model_flux = np.atleast_2d(model_flux)
        model_ivar = np.atleast_2d(model_ivar)
        continuum = np.atleast_2d(continuum)

        N, P = shape = normalized_flux.shape

        if N > P:
            raise ValueError(f"I do not believe that you have more visits than pixels!")

        if shape != normalized_ivar.shape:
            raise ValueError(f"normalized_flux and normalized_ivar have different shapes ({shape} != {normalized_ivar.shape})")
        if shape != model_flux.shape:
            raise ValueError(f"normalized_flux and model_flux have different shapes ({shape} != {model_flux.shape})")
        if shape != model_ivar.shape:
            raise ValueError(f"normalized_flux and model_ivar have different shapes ({shape} != {model_ivar.shape}")
        if shape != continuum.shape:
            raise ValueError(f"normalized_flux and continuum have different shapes ({shape} != {continuum.shape})")
        if N != len(results_table):
            raise ValueError(f"results table should have the same number of rows as there are spectra ({N} != {len(data_table)})")

        results_table = results_table.copy()

        return (normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table)


    @property
    def _path(self):
        """
        Return the path of this astraSource target.
        """
        
        t = self.task
        # Check if we are SDSS-V or SDSS-IV.
        # In principle we should just have a switch here based on release and then use
        # sdss_access, and we will once we finalise the data model. For now this is the
        # only place where this path definition occurs.
        # TODO: Replace this path definition with that through sdss_access.
        is_sdss5 = getattr(t, "healpix", None) is not None
        
        # Check if we are APOGEE or BOSS.
        is_apogee = getattr(t, "apred", None) is not None
        reduction_version = t.apred if is_apogee else t.run2d

        # If the task has a short-hand descriptor, use that.
        task_short_id = getattr(t, "task_short_id", t.task_id)

        prefix = "AstraSource"

        if not is_sdss5:
            # SDSS-IV
            fork = f"{t.telescope}/{t.field}"
            basename = f"{prefix}-{reduction_version}-{t.telescope}-{t.field}-{t.obj}-{task_short_id}.fits"
        else:
            # SDSS-V
            fork = f"{t.telescope}/{int(t.healpix)/1000:.0f}/{t.healpix}"
            basename = f"{prefix}-{reduction_version}-{t.telescope}-{t.obj}-{task_short_id}.fits"

        path = os.path.join(
            t.output_base_dir,
            "star",
            fork,
            basename
        )
        # Check that the parent directory exists.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path
    
    
    def copy_from(self, source, use_symbolic_link=True):
        """
        Write an AstraSource object as a copy from an existing AstraSource object.

        :param source:
            An existing AstraSource object.
        """

        assert os.path.exists(source.path)
        f = symlink_force if use_symbolic_link else copyfile
        return f(source.path, self.path)
        
        

    def write(
            self,
            spectrum,
            normalized_flux,
            normalized_ivar,
            continuum,
            model_flux,
            model_ivar,
            results_table,
            **kwargs
        ):
        """
        Write the image to the appropriate path.

        :param spectrum:
            The spectrum that was used for analysis. Headers and relevant quality flags from this
            spectrum will be propagated.
            
        :param normalized_flux:
            A (N, P) shape array of the pseudo-continuum normalized observed flux, where N is
            the number of spectra and P is the number of pixels.
        
        :param normalized_ivar:
            A (N, P) shape array of the inverse variance of the pseudo-continuum normalized observed
            flux, where N is the number of spectra and P is the number of pixels. For example, 
            if the $1\sigma$ Gaussian uncertainty in pseudo-continuum normalized flux in a pixel is 
            $\sigma_n$, then the inverse variance of the pseudo-continuum normalized flux is $1/(\sigma_n^2)$. 
            
            Similarly if the $1\sigma$ Gaussian uncertainty in *un-normalized flux* $y$ is 
            $\sigma$ and the continuum is $C$, then the pseudo-continuum normalized flux is
            $y/C$ and the inverse variance of the pseudo-continuum normalized flux is 
            $(C/\sigma)^2$.
        
        :param continuum:
            A (N, P) shape array of the continuum value used to pseudo-normalize the observations,
            where N is the number of spectra and P is the number of pixels.

        :param model_flux:
            A (N, P) shape array of the pseudo-continuum normalized model flux, where N is the number
            of spectra and P is the number of pixels.
        
        :param model_ivar:
            A (N, P) shape array of the inverse variance of the pseudo-continuum normalized model flux,
            where N is the number of spectra and P is the number of pixels. This gives a measure of the
            inverse variance in the model predictions, which is often not known. If no model uncertainty
            is known then this array should be set to a constant positive value (e.g., 1).
        
        :param results_table:
            A :py:mod:`astropy.table.Table` of outputs from the analysis of the input spectra. This can
            contain any number of columns, but the naming convention of these columns should (as closely
            as possible) follow the conventions of other analysis tasks so that there is not great
            variation in names among identical columns (e.g., `R_CHI_SQ` and `RCHISQ` and `REDUCED_CHI_SQ`). 
            
            This table will be supplemented with the parameters and descriptions of the analysis task.        
        """

        # Check array shapes etc.
        normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table = self._check_shapes(
            normalized_flux, normalized_ivar, model_flux, model_ivar, continuum, results_table
        )
        
        # Build some HDUs.
        dtype = kwargs.pop("dtype", np.float32)

        # Copy the wavelength header cards we need.
        dispersion_map_keys = ("CRVAL1", "CDELT1", "CRPIX1", "CTYPE1", "DC-FLAG")
        header = fits.Header(cards=[(key, spectrum.meta["header"][key]) for key in dispersion_map_keys])
        
        flux_hdu = create_image_hdu(
            data=normalized_flux, 
            header=header, 
            name="NORMALIZED_FLUX", 
            bunit="Pseudo-continuum normalized flux (-)",
            dtype=dtype
        )
        ivar_hdu = create_image_hdu(
            data=normalized_ivar,
            header=header,
            name="NORMALIZED_IVAR",
            bunit="Inverse variance of pseudo-continuum normalized flux (-)",
            dtype=dtype
        )
        bitmask_hdu = create_image_hdu(
            data=self.get_bitmask(spectrum),
            header=header,
            name="BITMASK",
            bunit="Pixel bitmask",
        )
        continuum_hdu = create_image_hdu(
            data=continuum,
            header=header,
            name="CONTINUUM",
            bunit="Pseudo-continuum flux (10^-17 erg/s/cm^2/Ang)",
            dtype=dtype
        )        
        model_flux_hdu = create_image_hdu(
            data=model_flux,
            header=header,
            name="MODEL_FLUX",
            bunit="Model flux (-)",
            dtype=dtype
        )
        model_ivar_hdu = create_image_hdu(
            data=model_ivar,
            header=header,
            name="MODEL_IVAR",
            bunit="Inverse variance of model flux (-)",
            dtype=dtype
        )
        # TODO: Duplicate the visit information from headers to the results table?
        #       (E.g., fiber, mjd, field, RV information?)
        results_table_hdu = fits.BinTableHDU(
            data=results_table,
            name="RESULTS"
        )

        # Create a task parameter table.
        rows = []
        for name, parameter in self.task.get_params():
            rows.append((name, parameter.serialize(getattr(self.task, name))))
        
        parameter_table_hdu = fits.BinTableHDU(
            data=Table(rows=rows, names=("NAME", "VALUE")),
            name="TASK_PARAMETERS"
        )
        
        # Create a Primary HDU with the headers from the observation.
        primary_hdu = fits.PrimaryHDU(header=spectrum.meta["header"])
        astra_version = f"{self.task.astra_version_major}.{self.task.astra_version_minor}.{self.task.astra_version_micro}"
        cards = [
            ("ASTRA", astra_version, "Astra version"), 
            ("TASK_ID", self.task.task_id, "Astra task identifier"),
            ("CREATED", strftime("%Y-%m-%d %H:%M:%S", gmtime()), "GMT when this file was created"),
        ]
        # Delete reduction pipeline history comments and add what we need.
        del primary_hdu.header["HISTORY"]
        primary_hdu.header.extend(cards, end=True)
    
        # Add our own history comments.
        hdus = [
            (primary_hdu, "Primary header"),
            (flux_hdu, "Pseudo-continuum normalized flux"),
            (ivar_hdu, "Inverse variance of pseudo-continuum normalized flux"),
            (bitmask_hdu, "Pixel bitmask"),
            (continuum_hdu, "Pseudo-continuum used"),
            (model_flux_hdu, "Pseudo-continuum normalized model flux"),
            (model_ivar_hdu, "Inverse variance of pseudo-continuum normalized model flux"),
            (parameter_table_hdu, "Astra task parameters"),
            (results_table_hdu, "Results")
        ]

        hdu_list = []
        for i, (hdu, comment) in enumerate(hdus):
            hdu_list.append(hdu)
            hdu_list[0].header["HISTORY"] = f"HDU {i}: {comment}"
                
        image = fits.HDUList(hdu_list)

        kwds = dict(
            checksum=True, 
            overwrite=True,
            output_verify="silentfix"
        )
        kwds.update(kwargs)

        return image.writeto(self.path, **kwds)






if __name__ == "__main__":

    from astra.tasks.base import BaseTask
    from sqlalchemy import Column, Integer

    class MyTaskResultTarget(DatabaseTarget):
        
        a = Column("a", Integer)
        b = Column("b", Integer)
        c = Column("c", Integer)
        


    class MyTask(BaseTask):

        param_1 = luigi.FloatParameter()
        param_2 = luigi.IntParameter()
        param_3 = luigi.Parameter()
        
        def output(self):
            return MyTaskResultTarget(self)


        def run(self):
            self.output().write({"a": 5, "b": 3, "c": 2})
            print("Done")


    A = MyTask(param_1=3.5, param_2=4, param_3="what")

    A.run()
    print(A.output().read())
    print(A.output().read(as_dict=True))