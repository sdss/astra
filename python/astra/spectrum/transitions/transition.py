
import numpy as np
from astropy.constants import (e, m_e, c)
from astropy.io import registry

from grok.transitions.species import Species
from grok.transitions.connect import TransitionsRead, TransitionsWrite
from grok.transitions.utils import (air_to_vacuum, vacuum_to_air)



class Transition(object):

    """
    Represent an individual transition.
    """

    def __init__(
            self,
            lambda_vacuum,
            log_gf,
            species,
            E_lower,
            gamma_rad=None,
            gamma_stark=None,
            vdW=None,
            # All the extra possible stuff.
            lambda_air=None,
            E_upper=None,
            j_lower=None,
            j_upper=None,
            lande_factor_lower=None,
            lande_factor_upper=None,
            lande_factor_mean=None,
            # TODO: What is the translation between lande_factor_lower/upper/mean and lande_factor_depth?
            lande_factor=None,
            lande_depth=None,
            reference=None,
            comment=None,
            equivalent_width=None,
            equivalent_width_error=None,
            E_dissociation=None,
            lower_orbital_type=None,
            upper_orbital_type=None,
            **kwargs
        ):
                
        if lambda_vacuum is None:
            # Calculate it from lambda_air.
            if lambda_air is None:
                raise ValueError("Wavelength (lambda) must be given as lambda_vacuum or lambda_air.")
            lambda_vacuum = air_to_vacuum(lambda_air)
        
        if not isinstance(species, Species):
            species = Species(species)

        missing = lambda value: value is None or not np.isfinite(value)
        if any(map(missing, (gamma_stark, vdW))):
            gamma_stark_approx, vdW_approx = approximate_gammas(lambda_vacuum, species, E_lower)
            if missing(gamma_stark):
                gamma_stark = gamma_stark_approx
            if missing(vdW):
                raise a
                vdW = vdW_approx

        # Not a good idea to do this unless we have to.
        #if vdW < 0:
        #    # If the van der Waals constant is negative, we assume it is log(\Gamma_vdW)
        #    vdW = 10**vdW
        #
        #elif vdW > 20:
        #    # If the van der Waals constant is > 20 we assume that it's packed ABO
        #    #raise NotImplementedError("check to see how this is packed in MOOG vs others")
        #    None
            
        #gamma_rad = gamma_rad or approximate_radiative_gamma(lambda_vacuum, log_gf)
                    
        # Store everything.
        self._lambda_vacuum = lambda_vacuum
        self._lambda_air = lambda_air

        self.species = species
        self.log_gf = log_gf
        self.E_lower = E_lower # auto-apply units

        self.gamma_stark = gamma_stark # auto-apply units
        self.gamma_rad = gamma_rad # auto-apply units
        self.vdW = vdW # auto-apply units
        self.log_gf = log_gf
        
        self.E_upper = E_upper # auto-apply units
        self.j_lower = j_lower
        self.j_upper = j_upper
        if j_upper is not None:
            self.g_upper = 2 * j_upper + 1
        else:
            self.g_upper = None

        self.lande_factor_lower = lande_factor_lower
        self.lande_factor_upper = lande_factor_upper
        self.lande_factor_mean = lande_factor_mean
        # TODO: What is the translation between lande_factor_lower/upper/mean and lande_factor_depth?
        self.lande_factor = lande_factor
        self.lande_depth = lande_depth
        self.reference = reference
        self.comment = comment
        self.equivalent_width = equivalent_width # auto-apply units
        self.equivalent_width_error = equivalent_width_error # auto-apply units

        self.E_dissociation = E_dissociation # auto-apply units

        self.lower_orbital_type = lower_orbital_type
        self.upper_orbital_type = upper_orbital_type

        return None


    @property
    def lambda_vacuum(self):
        return self._lambda_vacuum or air_to_vacuum(self._lambda_air)

    @property
    def lambda_air(self):
        return self._lambda_air or vacuum_to_air(self._lambda_vacuum)
    
    def __repr__(self):
        return f"<{self} with χ = {self.E_lower:.2f}, log(gf) = {self.log_gf:.3f}>"

    def __str__(self):
        return f"{self.species} at λ = {self.lambda_vacuum:4.3f} (vacuum)"

    @property
    def vdW_compact(self):
        """
        Compact representation of van der Waals constant.
        """
        return np.log10(self.vdW) if (np.log10(self.vdW) < 0 and np.isfinite(np.log10(self.vdW))) else self.vdW


    def copy(self):
        data = dict([(k.lstrip("_"), v) for k, v in self.__dict__.items()])
        return self.__class__(**data)

    @property
    def is_molecule(self):
        """ A boolean flag to indicate whether this species is a molecule or not. """
        return (len(self.species.atoms) > 1)


    def __eq__(self, other):
        # Check formula.
        same_species = (self.species.compact == other.species.compact)
        check_properties = [
            ("lambda_vacuum", 1e-3, 1e-5),
            ("E_lower", 1e-3, 1e-5, ),
            ("log_gf", 1e-3, 1e-5)
        ]
        if same_species:
            for prop, atol, rtol in check_properties:
                A = getattr(self, prop)
                B = getattr(other, prop)
                try:
                    A = A.value
                    B = B.value
                except:
                    None
                if not np.isclose(A, B, rtol=rtol, atol=atol):
                    return False
            return True
        return False


            # check wavelength, excitation potential, loggf



def approximate_radiative_gamma(lambda_vacuum, log_gf):
    # TODO: assumes lambda_vacuum is in cm, or has a unit.
    e_ = e.value * 10 * c.to("m/s").value
    raise a
    return 8 * np.pi**2 * e_**2 / (m_e.cgs * c.cgs * lambda_vacuum**2) * 10**log_gf


def approximate_gammas(lambda_vacuum, species, ionization_energies=None):
    return (0, 0)



class Transitions(tuple):
    """A class to represent atomic and molecular transitions."""
    
    read = registry.UnifiedReadWriteMethod(TransitionsRead)
    write = registry.UnifiedReadWriteMethod(TransitionsWrite)

