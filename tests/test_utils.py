

def test_version_handling():
    from astra import utils
    for major in [0, 9, 10, 99, 100, 999, 1000, 1001, 2147]:
        for minor in [0, 9, 10, 99, 100, 999, 483]:
            for patch in [0, 9, 10, 99, 100, 999, 647]:
                v = f"{major}.{minor}.{patch}"
                i = utils.version_string_to_integer(v)
                s = utils.version_integer_to_string(i)
                assert isinstance(i, int)
                assert s == v
