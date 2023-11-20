#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/23
# @Author : jiang.hu
# @File : api.py 


from flask import Blueprint


blueprint = Blueprint(
    'latex',
    __name__,
    url_prefix='/latex'
)


@blueprint.route('/1.0.0.0', methods=['POST'])
def v_1_0_0_0():
    """

    :return:
    """
    return "ok"


if __name__ == '__main__':
    pass
