

def test_help_text_inheritance_on_fields():
    from peewee import Model
    from astra.fields import FloatField
    from astra.glossary import Glossary

    dec_help_text = "Something I wrpte"

    class DummyModel(Model):
        ra = FloatField()
        some_field_that_is_not_in_glossary = FloatField()
        dec = FloatField(help_text=dec_help_text)
        
    assert DummyModel.ra.help_text == Glossary.ra
    assert DummyModel.some_field_that_is_not_in_glossary.help_text == None
    assert DummyModel.dec.help_text == dec_help_text


def test_help_text_inheritance_on_flags():

    from peewee import Model
    from astra.fields import BitField
    from astra.glossary import Glossary

    class DummyModel(Model):
        flags = BitField()
        flag_sdss4_apogee_faint = flags.flag()

    overwrite_help_text = "MOO"
    class DummyModel2(Model):
        flags = BitField()
        flag_sdss4_apogee_faint = flags.flag(help_text=overwrite_help_text)

    assert DummyModel2.flag_sdss4_apogee_faint.help_text == overwrite_help_text
