

def test_help_text_inheritance():
    from peewee import Model
    from astra.fields import FloatField
    from astra.glossary import Glossary

    dec_help_text = "Something I wrpte"

    class DummyModel(Model):
        ra = FloatField()
        some_field_that_is_not_in_glossary = FloatField()
        dec = FloatField(help_text=dec_help_text)
        
    assert DummyModel.ra.help_text == Glossary.ra
    assert DummyModel.some_field_that_is_not_in_glossary.help_text == ""
    assert DummyModel.dec.help_text == dec_help_text
