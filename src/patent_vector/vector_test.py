#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/11/9
# @Author : jiang.hu
# @File : vector_test.py 

import time
from pathlib import Path
from typing import List, Union

import numpy as np
import pickle
import torch
import gensim
import pandas as pd
import faiss
import os
import jieba

BASE_DIR = Path(os.path.realpath(__file__)).parent


class PatentVectorBuildingOld:
    def __init__(self, dim=100):
        self.model = None
        self.index_2_words = dict()
        self.index = faiss.IndexFlatL2(dim)
        self.index2key = dict()

    def main(self, file_name_csv, file_name_json, file_name_index: str, folder_name: str = None):
        """
        主函数
        """
        self.transform_vectors(folder_name=folder_name)
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
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        print(f"device = {device}")
        print("正在加载腾讯词向量...")
        self.model = gensim.models.KeyedVectors.load_word2vec_format(model_dir, binary=False)
        print("加载腾讯词向量完毕...")
        print(f'vocab size = {len(self.model)}, vector size = {self.model.vector_size}')

    def get_all_candidate_words(self, file_name, batch_size=3000):
        """
        分批次
        获取所有候选词（此处为 专利标题）
        """
        data_df = pd.read_csv(open(file_name, encoding='utf-8'))
        # data_df = data_df.sample(12000)
        data_df['id'] = range(len(data_df))
        num_batchs = len(data_df) // batch_size
        num_batchs = (num_batchs + 1) if (num_batchs * batch_size) < len(data_df) else num_batchs
        for j in range(num_batchs):
            batch_start = j * batch_size
            batch_end = (j + 1) * batch_size
            print(f"split file -> batch_start = {batch_start}, batch_end = {batch_end}")
            file_name = BASE_DIR.joinpath('data', f'word_{batch_start}_{batch_end}.csv').__str__()
            data_df[(data_df['id'] >= batch_start) & (data_df['id'] < batch_end)].to_csv(file_name, index=False)
        return len(data_df)

    def get_vector(self, word: str, dim=100):
        """
        获取模型向量
        """
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
            file_name2 = BASE_DIR.joinpath('data', 'model_data', f'word_vector_{file_name}.csv').__str__()
            per_df.to_csv(file_name2, index=False)
            print("-----结束转化文件向量:", file_name)
            print("-----耗时：", time.time() - start_time)
        print("所有文件向量转化完成！")

    def build_faiss_db(self, file_name_csv, file_name_index: str, file_name_json: str, dim=100, field: str = "title"):
        """
        构建向量索引
        """
        df = pd.read_csv(file_name_csv, encoding='utf-8')
        print(f"word vector size = {len(df)}")
        self.index2key = dict(zip(range(0, len(df)), df[field].values.tolist()))
        datas = np.array(list(map(lambda x: eval(str(x).replace('nan', '0.0')), df['vector'].values.tolist())),
                         dtype='float32')
        print(f"start build faiss...")
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, 100, 10, 8)
        self.index.train(datas)
        self.index.add(datas)
        self.save(file_name_index, file_name_json)
        print(f"end build faiss...")

    def save(self, index_path, index2key_path):
        with open(index_path, mode='wb') as file:
            pickle.dump(self.index, file)
        with open(index2key_path, mode='wb') as file:
            pickle.dump(self.index2key, file)

    def load_index(self, index_path, index2key_path):
        """
        加载索引
        """
        with open(index_path, mode='rb') as file:
            self.index = pickle.load(file)
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
        # for cur_word, v in zip(words, list(datas)):
        #     D, I = index_db.search(np.array([v]), sim_topn)
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


class PatentVectorBuilding:
    def __init__(self, dim=100):
        self.model = None
        self.index = faiss.IndexFlatL2(dim)
        self.index2key = dict()

    def update(self):
        """
        加载模型文件
        """
        # 调度任务更新
        print("模型文件加载中！")
        self._load()
        print("模型文件加载完成！")

        #
        patent_title_index = BASE_DIR.joinpath('data', 'index_data', 'patent_title_vector_flat_l2_txt.bin').__str__()
        patent_title_json = BASE_DIR.joinpath('data', 'index_data', 'patent_title_vector_flat_l2_txt.json').__str__()
        print("加载专利标题向量开始~~~")
        print("索引文件加载中！")
        self.load_index(patent_title_index, patent_title_json)
        print("索引文件加载完成！")
        print("加载专利标题向量结束~~~")

    def _load(self):
        """
        加载模型文件
        """
        vector_file_v1 = BASE_DIR.joinpath('data', 'model_data',
                                           'tencent-ailab-embedding-zh-d100-v0.2.0-s.bin').__str__()
        print("正在加载腾讯词向量...")
        print(vector_file_v1)
        # self.model = gensim.models.KeyedVectors.load_word2vec_format(vector_file_v1, binary=False)
        self.model = gensim.models.KeyedVectors.load(vector_file_v1)
        print("加载腾讯词向量完毕...")

    def load_index(self, index_path: str = '', index2key_path: str = ''):
        """
        加载模型json文件
        """
        print("专利名称索引路径：%s", index_path)
        print("专利名称json文件路径：%s", index2key_path)
        if not os.path.exists(index_path):
            print("专利名称索引 文件不存在")
            raise "文件不存在"
        # with open(index_path, mode='rb') as file:
        #     self.index = pickle.load(file)
        self.index = faiss.read_index(index_path)
        with open(index2key_path, mode='rb') as file:
            self.index2key = pickle.load(file)

    def get_vector(self, word: str, dim=100):
        """
        获取模型向量
        """
        if not self.model:
            print("模型文件未加载成功！")
            return
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

    def get_sim_words(self, word: str, sim_topn=5, threshold=0.01, dim=100):
        result = {}
        word_vector = self.get_vector(word)
        if word_vector is None:
            return result
        datas = np.array(word_vector, dtype='float32')
        datas = datas / np.linalg.norm(datas).reshape(-1, 1)
        print("开始查询")
        D1, I1 = self.index.search(datas, sim_topn)
        sim_word_list = []
        for index, score in zip(I1[0], D1[0]):
            _sim_result = {}
            if index == -1:
                continue
            sim_word = self.index2key[index]
            score = round(float(score), 6)
            if sim_word != word and score >= threshold:
                result["key"] = word
                _sim_result["name"] = sim_word
                _sim_result["score"] = score
                sim_word_list.append(_sim_result)
        if sim_word_list:
            result["type"] = "IndexIVFFlat"
            result["key"] = word
            result["values"] = sim_word_list
        print("结束查询")
        return result

    def query_by_user_search_key(self, query_list: Union[str, List], top_n=5, threshold=0.01):
        """"""
        results = []
        if isinstance(query_list, str):
            query_list = [query_list]
        if not query_list:
            return results
        for query in query_list:
            result = self.get_sim_words(word=query, sim_topn=top_n, threshold=threshold)
            if result:
                results.append(result)
        return results


if __name__ == '__main__':
    hive_data_file = 'patent_info_001.csv'
    index_name = 'tencent_patent_words_test'

    file_name = BASE_DIR.joinpath('data', 'hive_data', hive_data_file).__str__()
    user_search_words_path = BASE_DIR.joinpath('data', 'user_data', f'user_search_key.csv').__str__()

    # 向量csv/索引index/索引json 路径
    file_name_csv = BASE_DIR.joinpath('data', 'model_data', f'word_vector_{hive_data_file}.csv').__str__()
    index_path = BASE_DIR.joinpath('data', 'index_data', f'{index_name}.index').__str__()
    index2key_path = BASE_DIR.joinpath('data', 'index_data', f'{index_name}.json').__str__()

    patent_vector = PatentVectorBuilding()

    # 构建索引并保存
    patent_vector.update()

    # 用户搜索数据
    df = pd.read_csv(user_search_words_path, encoding='utf-8')
    sample_df = df.sample(300)
    print(sample_df['word'].values.tolist())
    sample_list = sample_df['word'].values.tolist()

    # 检索索引
    sim_words = patent_vector.get_sim_words(sample_list[0])

    # sim_words = pd.DataFrame([(word1, word2) for word1, word2 in sim_words.items()], columns=['word', 'sim_words'])
    # sim_words.to_csv('test_sim_words.csv', index=False)
    print(sim_words)
    # print(BASE_DIR.joinpath('data', 'tencent-ailab-embedding-zh-d100-v0.2.0-s.txt').__str__())
