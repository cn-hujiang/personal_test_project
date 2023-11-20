#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/23
# @Author : jiang.hu
# @File : app.py
from concurrent.futures import ThreadPoolExecutor

import pytz
import os

from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask
from flasgger import Swagger

from src.api import latex
from src.logger import logger

app = Flask(__name__)
Swagger(app)

# 注册蓝图，即多个模块
blueprints = [
    latex.blueprint
]
for blueprint in blueprints:
    app.register_blueprint(blueprint)
logger = logger.get_logger()


# 调度
logger.info('调度实例化.')
time_zone = pytz.timezone('Asia/Shanghai')
executors = {
    "default": ThreadPoolExecutor(2),
    # 用于订阅这线程执行
    "subscribe": ThreadPoolExecutor(1)
}
SCHEDULER = BackgroundScheduler(daemon=True, timezone=time_zone)
SCHEDULER.start()
logger.info('调度开始~')


if __name__ == '__main__':
    env_port = os.environ.get('APP_PORT', 8888)
    print('Server will run on port:', env_port)
    debug = os.environ.get('DEBUG', False)
    app.run(host='0.0.0.0', port=env_port)
