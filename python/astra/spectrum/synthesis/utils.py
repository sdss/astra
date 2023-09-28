def get_default_lambdas(transitions):
    raise DeprecatedError
    if not isinstance(transitions, (tuple, list)):
        transitions = (transitions,)

    lambda_min, lambda_max = (1e10, 0)
    for each in transitions:
        lambda_min = min(lambda_min, each["wavelength"].min())
        lambda_max = max(lambda_max, each["wavelength"].max())

    lambda_delta = 0.01
    return (lambda_min, lambda_max, lambda_delta)
