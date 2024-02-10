from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from cfg_generation import construct_ir_table
import logging
import gensim

import pickle
import os
import numpy as np
import pandas as pd

# Please modify your project dir
project_dir = "/media/ubuntu/My_Passport/PonziDetector"


def construct_documents(addr_list):
    documents = []
    k = 0

    for i, addr in enumerate(addr_list):
        b = construct_ir_table(addr)
        blocks = sorted(list(set(b["blockname"].values)))
        
        for bk in blocks:
            opcode = (b[b["blockname"] == bk].reset_index(drop=True))["op"].values.tolist()
            documents.append(gensim.models.doc2vec.TaggedDocument(opcode, [k]))
            k += 1
        print(i, addr)

    print("finish construct documents.\n")
    return documents


def train_doc2vec_model(documents, dim):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    doc2vec_model = gensim.models.Doc2Vec(vector_size=dim)
    doc2vec_model.build_vocab(documents)
    doc2vec_model.train(documents, total_examples=len(documents), epochs=10)
    doc2vec_model.save(project_dir + "/algorithm/model_files/doc2vec_{}dim".format(dim))



if __name__ == "__main__":
    addr_list = os.listdir(project_dir + "/data/.temp/")
    documents = construct_documents(addr_list)
    train_doc2vec_model(documents, dim=64)