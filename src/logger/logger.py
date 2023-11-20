#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/23
# @Author : jiang.hu
# @File : logger.py
import logging
import os
from logging import Formatter
from pathlib import Path
import logging.handlers

BASE_DIR = Path(os.path.realpath(__file__)).parent.parent


def get_logger():
    _format = '%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] %(message)s'

    # log 路径
    log_path = BASE_DIR.joinpath('logs').__str__()
    log_level = "INFO"

    # 设置其余logger的日志水平
    root_logger = logging.getLogger()
    jaeger_tracing_logger = logging.getLogger('jaeger_tracing')
    tornado_access_log = logging.getLogger("tornado.access")
    tornado_app_log = logging.getLogger("tornado.application")
    tornado_gen_log = logging.getLogger("tornado.general")
    tornado_log = logging.getLogger('tornado')
    tornado_curl_log = logging.getLogger('tornado.curl_httpclient')
    other_loggers = [root_logger, jaeger_tracing_logger, tornado_access_log, tornado_app_log,
                     tornado_gen_log, tornado_log, tornado_curl_log]
    for other_logger in other_loggers:
        other_logger.setLevel(log_level)
        for handler in other_logger.handlers:
            handler.setFormatter(Formatter(_format))

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = logging.getLogger('mylogger')
    logger.setLevel(log_level)

    formatter = logging.Formatter(_format)

    rf_handler = logging.handlers.WatchedFileHandler('%s/all.log' % log_path)
    rf_handler.setLevel(log_level)
    rf_handler.setFormatter(formatter)

    f_handler = logging.handlers.WatchedFileHandler('%s/error.log' % log_path)
    f_handler.setLevel(logging.ERROR)
    f_handler.setFormatter(logging.Formatter(_format))

    ch = logging.StreamHandler()
    ch.setLevel(log_level)  # 输出到console的log等级的开关
    ch.setFormatter(formatter)

    root_logger.addHandler(rf_handler)
    root_logger.addHandler(f_handler)
    return logger


if __name__ == '__main__':
    print(BASE_DIR.joinpath('logs').__str__())
