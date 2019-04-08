# !usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under a 3-clause BSD license.
#
# @Author: Brian Cherinka
# @Date:   2017-12-05 12:01:21
# @Last modified by:   Brian Cherinka
# @Last Modified time: 2017-12-05 12:19:32

from __future__ import print_function, division, absolute_import


class AstraError(Exception):
    """A custom core Astra exception"""

    def __init__(self, message=None):

        message = 'There has been an error' \
            if not message else message

        super(AstraError, self).__init__(message)


class AstraNotImplemented(AstraError):
    """A custom exception for not yet implemented features."""

    def __init__(self, message=None):

        message = 'This feature is not implemented yet.' \
            if not message else message

        super(AstraNotImplemented, self).__init__(message)


class AstraAPIError(AstraError):
    """A custom exception for API errors"""

    def __init__(self, message=None):
        if not message:
            message = 'Error with Http Response from Astra API'
        else:
            message = 'Http response error from Astra API. {0}'.format(message)

        super(AstraAPIError, self).__init__(message)


class AstraApiAuthError(AstraAPIError):
    """A custom exception for API authentication errors"""
    pass


class AstraMissingDependency(AstraError):
    """A custom exception for missing dependencies."""
    pass


class AstraWarning(Warning):
    """Base warning for Astra."""


class AstraUserWarning(UserWarning, AstraWarning):
    """The primary warning class."""
    pass


class AstraSkippedTestWarning(AstraUserWarning):
    """A warning for when a test is skipped."""
    pass


class AstraDeprecationWarning(AstraUserWarning):
    """A warning for deprecated features."""
    pass
