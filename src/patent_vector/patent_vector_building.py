#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/10/30
# @Author : jiang.hu
# @File : patent_vector_building.py
import time
from pathlib import Path

import numpy as np
import pickle
import gensim
import pandas as pd
import faiss
import os
import jieba
# from sklearn.decomposition import PCA


BASE_DIR = Path(os.path.realpath(__file__)).parent
__APP_NAME__ = "用户专利挖掘"


class PatentVectorBuilding:
    def __init__(self, dim=100):
        self.model = None
        self.index = faiss.IndexFlatL2(dim)
        self.index2key = dict()

    def main(self, file_name_csv, file_name_index: str, file_name_json: str, folder_name: str = None):
        """
        主函数
        """
        # self.transform_vectors(folder_name=folder_name)
        self.build_faiss_db(file_name_csv, file_name_index, file_name_json)

    def update(self):
        """
        加载模型文件
        """
        # 调度任务更新
        print("模型文件加载中！")
        self._load()
        print('模型文件加载完成！')

    def _load(self):
        """
        加载模型文件
        """
        model_dir = BASE_DIR.joinpath('data', 'model_data', 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt').__str__()
        print("正在加载腾讯词向量...")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=False)
        print("加载腾讯词向量完毕...")
        print(f'vocab size = {len(self.model)}, vector size = {self.model.vector_size}')

    def get_vector(self, word: str, dim=100):
        """
        获取模型向量
        """
        if not self.model:
            raise "模型文件未加载成功！"
        if word is np.nan:
            return np.zeros(dim)

        vec_words = []
        if word in self.model:
            vec_words.append(self.model[word])
        else:
            words = []
            try:
                words = jieba.lcut(word, cut_all=False)
            except Exception as e:
                print(e)
            for w in words:
                if w in self.model:
                    vec_words.append(self.model[w])
        if len(vec_words) == 0:
            return np.zeros(dim)
        return np.mean(vec_words, axis=0)

    def transform_vectors(self, folder_name: str = None, field: str = "title"):
        """
        向量转义 并保存文件
        folder_name：csv文件所在文件夹
        field：csv文件待构建向量的列名
        """
        folder_path = BASE_DIR.joinpath('data', folder_name)
        for file_name in os.listdir(folder_path):
            if 'patent_info' not in file_name:
                continue
            file_path = os.path.join(folder_path, file_name)
            if not os.path.isfile(file_path):
                continue

            print("-----开始转化文件向量:", file_name)
            start_time = time.time()
            per_df = pd.read_csv(file_path, encoding='utf-8')
            fields = per_df[field].values.tolist()
            vectors = []
            for _i, _w in enumerate(fields):
                words_embs = self.get_vector(_w)
                vectors.append(list(words_embs))
            datas = np.array(vectors)
            datas = datas / np.linalg.norm(datas, axis=1).reshape(-1, 1)
            per_df['vector'] = list([list(v) for v in datas])
            file_name2 = BASE_DIR.joinpath('data', f'word_vector_{file_name}.csv').__str__()
            per_df.to_csv(file_name2, index=False)
            print("-----结束转化文件向量:", file_name)
            print("-----耗时：", time.time() - start_time)
        print("所有文件向量转化完成！")

    def build_faiss_db(self, file_name_csv, file_name_index: str, file_name_json: str, folder_name: str = 'model_data'
                       , dim=100):
        """
        构建向量索引
        """
        # 所有向量csv文件
        print(file_name_csv)
        words_vectors_path = BASE_DIR.joinpath('data', folder_name)
        print(words_vectors_path)

        # 初始化索引
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, 200, faiss.METRIC_L2)
        # self.index = faiss.IndexIVFPQ(quantizer, dim, 100, 10, 8, faiss.METRIC_INNER_PRODUCT)

        # 加载第一个csv文件训练索引
        start_time = time.time()
        df_0 = pd.read_csv(file_name_csv, encoding='utf-8')
        print(f"word vector size = {len(df_0)}")
        print(f"第一个csv start init data ...: ", file_name_csv)
        datas = np.array(list(map(lambda x: eval(str(x).replace('nan', '0.0')), df_0['vector'].values.tolist())),
                         dtype='float32')
        print(f"第一个csv end init data ...：", file_name_csv)
        print("第一个csv 初始化数据耗时：", time.time() - start_time)

        print(f"start build faiss...")
        # 训练数据
        self.index.train(datas)
        self.index.add(datas)
        print(f"end build faiss...")

        # 循环其他数据
        df_total = df_0.copy()

        for vectors_path in os.listdir(words_vectors_path):
            print(vectors_path)
            # if vectors_path:
            #     break
            if 'word_vector' not in vectors_path or 'word_vector_patent_info_0.csv' in vectors_path:
                continue
            _vectors_path = BASE_DIR.joinpath('data', folder_name, vectors_path).__str__()
            df = pd.read_csv(_vectors_path, encoding='utf-8')
            print(f"word vector size = {len(df)}")

            start_time = time.time()
            print(f"start init data ...: ", vectors_path)
            datas = np.array(list(map(lambda x: eval(str(x).replace('nan', '0.0')), df['vector'].values.tolist())),
                             dtype='float32')
            print(f"end init data ...：", vectors_path)
            print("初始化数据耗时：", time.time() - start_time)

            self.index.add(datas)
            print(f"数据加入向量结束...:", vectors_path)

            df_total = pd.concat([df_total, df])
            print("mid total data len: ", df_total.shape)

        # 存json
        print(f"start save to json...")
        print("total data len: ", df_total.shape)
        self.index2key = dict(zip(range(0, len(df_total)), df_total["title"].values.tolist()))
        self.save(file_name_index, file_name_json)
        print(f"end build faiss...")
        print("构建向量索引耗时：", time.time() - start_time)
        print(f"end save to json...")

    def save(self, index_path, index2key_path):
        """
        保存索引
        """
        # with open(index_path, mode='wb') as file:
        #     pickle.dump(self.index, file)
        faiss.write_index(self.index, index_path)

        with open(index2key_path, mode='wb') as file:
            pickle.dump(self.index2key, file)

    def load_index(self, index_path, index2key_path):
        """
        加载索引
        """
        self.index = faiss.read_index(index_path)
        # with open(index_path, mode='rb') as file:
        #     self.index = pickle.load(file)
        with open(index2key_path, mode='rb') as file:
            self.index2key = pickle.load(file)

    def get_sim_words(self, word: str, dim=100, sim_topn=5, thr=0.8):
        results = {}
        word_vector = self.get_vector(word)
        datas = np.array(word_vector, dtype='float32')
        datas = datas / np.linalg.norm(datas).reshape(-1, 1)
        print(datas)
        print("开始查询")
        D, I = self.index.search(datas, sim_topn)
        print(D)
        print(I)
        for index, score in zip(I[0], D[0]):
            if results.get(word, -1) == -1:
                results[word] = {}
            if index == -1:
                continue
            sim_word = self.index2key[index]
            if sim_word != word and score >= thr:
                results[word][sim_word] = score
        print("结束查询")
        return results


patent_vector = PatentVectorBuilding()

if __name__ == '__main__':
    # 待构建向量的csv文件
    # hive_data_file = 'patent_info_0.csv'

    # 所有待构建向量的csv文件所在目录文件夹
    # folder_name = 'index_data'

    # 索引名称
    # 构建索引方式：IndexIVFFlat
    index_name = 'patent_title_vector_flat_l2'
    # 构建索引方式：IndexIVFPQ
    # index_name = 'patent_title_vector_pq_l2'

    # 向量文件所在路径下有多个向量时，传入第一个文件训练索引
    vector_name_csv = 'word_vector_patent_info_0.csv.csv'
    # 向量csv/索引index/索引json 路径
    file_name_csv = BASE_DIR.joinpath('data', 'model_data', vector_name_csv).__str__()
    index_path = BASE_DIR.joinpath('data', 'index_data', f'{index_name}.index').__str__()
    index2key_path = BASE_DIR.joinpath('data', 'index_data', f'{index_name}.json').__str__()

    # 加载腾讯词向量
    patent_vector.update()
    # 构建索引
    patent_vector.main(file_name_csv, index_path, index2key_path)

    # 加载索引
    # patent_vector.load_index(index_path, index2key_path)

    # 用户搜索数据
    # user_search_words_path = BASE_DIR.joinpath('data', 'user_data', f'user_search_key.csv').__str__()
    # df = pd.read_csv(user_search_words_path, encoding='utf-8')
    # sample_df = df.sample(300)
    # print(sample_df['word'].values.tolist())
    # sample_list = sample_df['word'].values.tolist()
    #
    # # 检索索引,单个搜索词单次检索
    # sim_words = patent_vector.get_sim_words(sample_list[0])
    # print(sim_words)
