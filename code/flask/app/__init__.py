#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 25/04/2018 11:15 AM
# @Author  : sean10
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm

from flask import Flask, current_app


def create_app(config=None):
    app = Flask(__name__, static_folder="../static", template_folder="../templates")

    app.config.from_object(config)
    app.config.from_envvar('APP_SETTINGS', silent=True)

    app.app_context().push()

    return app

