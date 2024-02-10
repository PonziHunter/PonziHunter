import pandas as pd
import numpy as np
import csv
import re
import os
import shutil

import networkx as nx
import igraph as ig
import queue
import pygraphviz as pgv
import matplotlib.pyplot as plt

# Please modify your project dir
project_dir = "/media/ubuntu/My_Passport/PonziDetector"


def construct_ir_table(addr):
    b = pd.read_table(project_dir + "/data/.temp/" + addr + "/out/contract.tac")
    
    b.columns = ['3IR']
    b['blockname'] = '0'
    b['leftvariable'] = '0'
    b['rightvariable'] = '0'
    b['op'] = '0'
    b['functionname'] = '0'
    func_name = "__function_selector__"
    
    for i in range(0, len(b)):
        blockindex = b.iloc[i, 0].find('block')
        if blockindex > 0:
            k = b.iloc[i, 0][(blockindex + 6):]
            b.iloc[i, 1] = k
        b.iloc[i, 1] = k
        
        bsplit = re.split(r'(?:[, : \s ( )])', b.iloc[i, 0])
        
        if bsplit.count('=') > 0:
            eindex = bsplit.index('=')
            b.iloc[i, 4] = bsplit[eindex + 1]
            
            for p in range(0, eindex):
                if bsplit[p].find('v') > -1:
                    b.iloc[i, 2] = bsplit[p]
                    
            for p in range(eindex, len(bsplit)):
                if bsplit[p].find('v') > -1:
                    b.iloc[i, 3] = b.iloc[i, 3] + ',' + bsplit[p]
           
        else:
            if (len(bsplit) > 7 and b.iloc[i, 0].find('succ') < 0):
                b.iloc[i, 4] = bsplit[6]
                
                for p in range(0, len(bsplit)):
                    if bsplit[p].find('v') > -1:
                        b.iloc[i, 3] = b.iloc[i, 3] + ',' + bsplit[p]
        
        if bsplit[0] == "function":
            func_name = bsplit[1]
        b.iloc[i, 5] = func_name

    for i in range(0, len(b)):
        bsplit = re.split(r'(?:[, : \s ( )])', b.iloc[i, 0])
        if bsplit[0] == "function":
            b.iloc[i, 1] = b.iloc[i + 1, 1]
    
    return b


def find_state_var_dependency(b, addr):
    SD_edges = []
    
    for j in range(0, len(b)):
        sloadindex = b.iloc[j, 0].find('SLOAD')
        
        if sloadindex > 0:
            sloadstatement = b.iloc[j, 0][(sloadindex + 5):]
            addressindex = sloadstatement.find('(')
            
            if addressindex > 0:
                stateaddress = sloadstatement[(addressindex + 1):]
                stateaddressmodify = stateaddress[:(len(stateaddress) - 1)]
            
                state_var = 'stor_' + stateaddressmodify
                SD_edges.append([state_var, b.iloc[j, 1]])
            
            else:
                leftfind = [j]
                uloop = 1
                findSHA3 = 0
                
                while (findSHA3 < 1 and uloop < 10):
                    
                    findSHA3=0
                    SHAsplit=[]
                    for m in range(0, len(leftfind)):
                        SHAsplit = list(set(SHAsplit + re.split(r'(?:[,])', b.iloc[leftfind[m], 3])))
                    
                    leftfind = []
                    for n in range(0, len(SHAsplit)):
                        if (SHAsplit[n] != '0' and b.iloc[:, 2].tolist().count(SHAsplit[n]) > 0):
                            leftfind.append(b.iloc[:, 2].tolist().index(SHAsplit[n]))
                    
                    for m in range(0, len(leftfind)):
                        if (b.iloc[leftfind[m], 4].find('SHA3') < 0):
                            findSHA3 = findSHA3 or 0
                        if (b.iloc[leftfind[m], 4].find('SHA3') > -1):
                            findSHA3 = findSHA3 or 1
                            
                            if (b.iloc[leftfind[m], 0].count('(') > 0 and b.iloc[leftfind[m], 0].count(')') > 0):
                                SHAaddress1 = b.iloc[leftfind[m], 0][b.iloc[leftfind[m], 0].index('(') + 1: b.iloc[leftfind[m], 0].index(')')]
                                
                            if (b.iloc[leftfind[m], 0].count(')') > 0):
                                delete = b.iloc[leftfind[m], 0][b.iloc[leftfind[m], 0].index(')') + 1:]
                                
                            if (b.iloc[leftfind[m], 0].count('(') > 0 and b.iloc[leftfind[m], 0].count(')') > 0 and delete.count('(') > 0 and delete.count(')') > 0):
                                SHAaddress2 = delete[delete.index('(') + 1: delete.index(')')]
                                mapping = 'stor_' + SHAaddress1 + '_' + SHAaddress2
                                SD_edges.append([mapping, b.iloc[j, 1]])
                            
                    uloop = uloop + 1

    for j in range(0, len(b)):
        sstoreindex = b.iloc[j, 0].find('SSTORE')
    
        if sstoreindex > 0:   
            storesplit = re.split(r'(?:[, : \s])', b.iloc[j, 0])
        
            if (len(storesplit) > 8):
                storekey = storesplit[7]
                staddressindex = storekey.find('(')
            
                if (staddressindex > 0):
                    storeaddress = storekey[(staddressindex + 1):]
                    storeaddressmodify = storeaddress[:(len(storeaddress) - 1)]
                    
                    store_var = 'stor_' + storeaddressmodify
                    SD_edges.append([b.iloc[j, 1], store_var])
                
                else:
                    leftfind = [j]
                    uloop = 1
                    findSHA3 = 0
                    
                    while (findSHA3 < 1 and uloop < 10): 
                        
                        findSHA3 = 0
                        SHAsplit = []
                        for m in range(0, len(leftfind)):
                            SHAsplit = list(set(SHAsplit + re.split(r'(?:[,])', b.iloc[leftfind[m], 3])))
                        
                        leftfind = []
                        for n in range(0, len(SHAsplit)):
                            if (SHAsplit[n] != '0' and b.iloc[:, 2].tolist().count(SHAsplit[n]) > 0):
                                leftfind.append(b.iloc[:, 2].tolist().index(SHAsplit[n]))
                                
                        for m in range(0, len(leftfind)):
                            if (b.iloc[leftfind[m], 4].find('SHA3') < 0):
                                findSHA3 = findSHA3 or 0
                            if (b.iloc[leftfind[m], 4].find('SHA3') > -1):
                                findSHA3 = findSHA3 or 1
                                
                                if (b.iloc[leftfind[m], 0].count('(') > 0 and b.iloc[leftfind[m], 0].count(')') > 0):
                                    SHAaddress1 = b.iloc[leftfind[m], 0][b.iloc[leftfind[m], 0].index('(') + 1: b.iloc[leftfind[m], 0].index(')')]
                                    
                                if (b.iloc[leftfind[m], 0].count(')') > 0):
                                    delete = b.iloc[leftfind[m],0][b.iloc[leftfind[m],0].index(')')+1:]
                                
                                if (b.iloc[leftfind[m], 0].count('(') > 0 and b.iloc[leftfind[m], 0].count(')') > 0 and delete.count('(') > 0 and delete.count(')') > 0):
                                    SHAaddress2 = delete[delete.index('(') + 1: delete.index(')')]
                                    mapping = 'stor_' + SHAaddress1 + '_'+SHAaddress2
                                    SD_edges.append([b.iloc[j, 1], mapping])
                                
                        uloop = uloop + 1
    
    with open(project_dir + "/data/.temp/" + addr + "/out/State_dependency_edge.csv", "w", newline='', encoding='UTF-8') as f:
        writer = csv.writer(f)
        for m in range(0, len(SD_edges)):
            writer.writerow(SD_edges[m])
    
    return SD_edges


def search_blocks(G, start_idx, direction="prev"):
    que = queue.Queue()
    que.put(start_idx)
    
    search_idx = set()
    visited_idx = set()
    
    while not que.empty():
        idx = que.get()
        if idx in visited_idx: continue
        
        if direction == "prev":
            neigh_idx = G.neighbors(idx, mode="in")
        else:
            neigh_idx = G.neighbors(idx, mode="out") 
        
        for i in neigh_idx:
            if i not in visited_idx:
                que.put(i)
        
        visited_idx.add(idx)
        if idx != start_idx: search_idx.add(idx)
    
    return search_idx


def create_SD_edges(b, G, dependency_df, block2idx, idx2block, idx2func, show_stor=False):
    SD_edges = []

    stor_sstore = {}
    stor_sload = {}
        
    for i in range(len(dependency_df)):
        src = dependency_df.loc[i, "sload_block_idx"]
        trg = dependency_df.loc[i, "sstore_block_idx"]
        stor = dependency_df.loc[i, "stor"]

        if src in block2idx and trg in block2idx:
            if stor not in stor_sstore:
                stor_sstore[stor] = set([block2idx[trg]])
            else:
                stor_sstore[stor].add(block2idx[trg])
            
            if stor not in stor_sload:
                stor_sload[stor] = set([block2idx[src]])
            else:
                stor_sload[stor].add(block2idx[src])

    control_prev_block_idx = {}
    control_succ_block_idx = {}

    for i in range(len(dependency_df)):
        src = dependency_df.loc[i, "sload_block_idx"]
        trg = dependency_df.loc[i, "sstore_block_idx"]
        stor = dependency_df.loc[i, "stor"]
        
        if not (src in block2idx and trg in block2idx):
            continue
        else:
            src = block2idx[src]
            trg = block2idx[trg]

        if src not in control_prev_block_idx:
            control_prev_block_idx[src] = search_blocks(G, src, direction="prev")
        if trg not in control_succ_block_idx:
            control_succ_block_idx[trg] = search_blocks(G, trg, direction="succ")


        if idx2func[src] == idx2func[trg]:
            if trg not in control_prev_block_idx[src]:
                continue
            if (control_succ_block_idx[trg] & control_prev_block_idx[src] - set([src, trg])) & stor_sstore[stor]:
                continue
            
            if not show_stor:
                SD_edges.append((src, trg))
            else:
                SD_edges.append((src, trg, stor))

        else:
            if (control_succ_block_idx[trg] - set([trg])) & stor_sstore[stor]:
                continue
            if (control_prev_block_idx[src] - set([src])) & stor_sstore[stor]:
                continue
            
            if not show_stor:
                SD_edges.append((src, trg))
            else:
                SD_edges.append((src, trg, stor))
    
    return SD_edges


def contract_graph(b, addr, show_stor=False, show_impt=False):
    G = ig.Graph(directed=True)
    addr_dir = project_dir + "/data/.temp/" + addr + "/out/"

    if os.path.getsize(addr_dir + "LocalBlockEdge.csv") > 0:
        control_edges_df = pd.read_csv(addr_dir + "LocalBlockEdge.csv", delimiter='\t', header=None)
    else:
        control_edges_df = []
    
    if os.path.getsize(addr_dir + "IRFunctionCall.csv") > 0:
        call_edges_df = pd.read_csv(addr_dir + "IRFunctionCall.csv", delimiter='\t', header=None)
    else:
        call_edges_df = []
        
    if os.path.getsize(addr_dir + "IRFunction_Return.csv") > 0:
        return_edges_df = pd.read_csv(addr_dir + "IRFunction_Return.csv", delimiter='\t', header=None)
    else:
        return_edges_df = []
    
    if os.path.getsize(addr_dir + "State_dependency_edge.csv") > 0:
        SD_edges_df = pd.read_csv(addr_dir + "State_dependency_edge.csv", delimiter=',', header=None)
    else:
        SD_edges_df = []

    if not os.path.exists(project_dir + "/data/facts/" + addr + "/"):
        os.makedirs(project_dir + "/data/facts/" + addr + "/")
        shutil.copyfile(addr_dir + "LocalBlockEdge.csv", project_dir + "/data/facts/" + addr + "/LocalBlockEdge.csv")
        shutil.copyfile(addr_dir + "IRFunctionCall.csv", project_dir + "/data/facts/" + addr + "/IRfunctionCall.csv")
        shutil.copyfile(addr_dir + "IRFunction_Return.csv", project_dir + "/data/facts/" + addr + "/IRfunction_Return.csv")
        shutil.copyfile(addr_dir + "State_dependency_edge.csv", project_dir + "/data/facts/" + addr + "/State_dependency_edge.csv")
        b.to_csv(project_dir + "/data/facts/" + addr + "/tac_table.csv", index=False)

        if os.path.getsize(addr_dir + "IRFunctionCallReturn.csv") > 0:
            shutil.copyfile(addr_dir + "IRFunctionCallReturn.csv", project_dir + "/data/facts/" + addr + "/IRFunctionCallReturn.csv")
    
    if not os.path.exists(project_dir + "/data/tac/" + addr + ".tac"):
        shutil.copyfile(addr_dir + "contract.tac", project_dir + "/data/tac/" + addr + ".tac")
    

    blocks = sorted(list(set(b["blockname"].values)))
    block2idx = {}
    idx2block = {}
    
    for i in range(len(blocks)):
        block2idx[blocks[i]] = i  # block_idx->idx
        idx2block[i] = blocks[i]  # idx->block_idx
    
    G.add_vertices(len(blocks), attributes={
        "label": blocks,
        "shape": ["box"] * len(blocks),
        "style": ["filled"] * len(blocks),
        "fillcolor": ["white"] * len(blocks)
    })
    print("construct basic block nodes.")

    idx2func = {}
    for i in range(len(blocks)):
        func = (b[b["blockname"] == idx2block[i]].reset_index(drop=True))["functionname"].values[0]
        idx2func[i] = func

    call_blocks = set(b[b["op"] == "CALL"]["blockname"].tolist())
    call_block_idx = set([block2idx[bk] for bk in call_blocks])

    for idx in call_block_idx:
        G.vs[idx]["label"] = G.vs[idx]["label"] + "-CALL-" + idx2func[idx]
    

    control_edges = []
    for i in range(len(control_edges_df)):
        src = control_edges_df.iloc[i, 0]
        trg = control_edges_df.iloc[i, 1]
        
        if src in block2idx and trg in block2idx:
            control_edges.append((block2idx[src], block2idx[trg]))
    
    G.add_edges(control_edges, attributes={
        "type": ["control"] * len(control_edges),
        "arrowsize": [0.5] * len(control_edges),
        "color": ["black"] * len(control_edges)
    })
    print("construct control edges.")


    if len(SD_edges_df) > 0:
        sstore_df = SD_edges_df[SD_edges_df[0].str.startswith('0x')]
        sstore_df = sstore_df.drop_duplicates()
        sstore_df = sstore_df.reset_index(drop=True)
        sstore_df = sstore_df.rename(columns={0: 'sstore_block_idx', 1: 'stor'})
        
        sload_df = SD_edges_df[SD_edges_df[1].str.startswith('0x')]
        sload_df = sload_df.drop_duplicates()
        sload_df = sload_df.reset_index(drop=True)
        sload_df = sload_df.rename(columns={0: 'stor', 1: 'sload_block_idx'})
    
        dependency_df = pd.merge(sstore_df, sload_df, on='stor', how='inner')
        dependency_df = dependency_df.drop_duplicates()
        dependency_df = dependency_df.reset_index(drop=True)
        
        if not os.path.exists(project_dir + "/data/facts/" + addr + "/dependency.csv"):
            dependency_df.to_csv(project_dir + "/data/facts/" + addr + "/dependency.csv", index=False)

        SD_edges = create_SD_edges(b, G, dependency_df, block2idx, idx2block, idx2func, show_stor)
        SD_edges_attributes = {
            "type": ["dependency"] * len(SD_edges),
            "arrowsize": [0.5] * len(SD_edges),
            "color": ["red"] * len(SD_edges),
        }

        if show_stor:
            stor_list = [e[2] for e in SD_edges]
            SD_edges = [(e[0], e[1]) for e in SD_edges]
            SD_edges_attributes["label"] = stor_list
        
        G.add_edges(SD_edges, attributes=SD_edges_attributes)
        print("construct state dependency edges.")
    

    call_edges = []
    for i in range(len(call_edges_df)):
        src = call_edges_df.iloc[i, 0]
        trg = call_edges_df.iloc[i, 1]
        
        if src in block2idx and trg in block2idx:
            call_edges.append((block2idx[src], block2idx[trg]))
    
    G.add_edges(call_edges, attributes={
        "type": ["call"] * len(call_edges),
        "arrowsize": [0.5] * len(call_edges),
        "color": ["blue"] * len(call_edges)
    })
    print("construct call edges.")
    

    return_edges = []
    for i in range(len(call_edges_df)):
        for j in range(len(return_edges_df)):
            
            if call_edges_df.iloc[i, 1] == return_edges_df.iloc[j, 0]:
                src = return_edges_df.iloc[j, 1]
                trg = call_edges_df.iloc[i, 0]
                
                if src in block2idx and trg in block2idx:
                    return_edges.append((block2idx[src], block2idx[trg]))
    
    G.add_edges(return_edges, attributes={
        "type": ["return"] * len(return_edges),
        "arrowsize": [0.5] * len(return_edges),
        "color": ["green"] * len(return_edges)
    })
    print("construct return edges.")
    

    slice_block_idx = code_slicing(G, call_block_idx)
    print("finish code slicing.")

    ppr(addr, G, slice_block_idx, idx2block)
    print("finish personalized pagerank.")

    if show_impt:
        idx2impt = np.load(project_dir + "/data/facts/" + addr + "/idx2impt.npy", allow_pickle=True).item()
        for i in range(G.vcount()):
            G.vs[i]["label"] = G.vs[i]["label"] + " (" + str(round(idx2impt[i], 4)) + ")"
    

    if not os.path.exists(project_dir + "/data/facts/" + addr + "/block2idx.npy"):
        np.save(project_dir + "/data/facts/" + addr + "/block2idx.npy", block2idx)
        np.save(project_dir + "/data/facts/" + addr + "/idx2block.npy", idx2block)

    G.write_dot(project_dir + "/data/facts/" + addr + "/graph.dot")
    return G
    

def draw_contract_graph(addr):
    ag = pgv.AGraph(project_dir + "/data/facts/" + addr + "/graph.dot")
    ag.graph_attr["splines"] = "spline"
    ag.graph_attr["rankdir"] = "TB"
    ag.layout("dot")
    ag.draw(project_dir + "/data/graph_visualization/" + addr + ".png")


def code_slicing(G, call_block_idx):
    slice_block_idx = set()

    que = queue.Queue()
    visited_idx = set()
    
    for idx in call_block_idx:
        que.put(idx)
    
    while not que.empty():
        idx = que.get()
        if idx in visited_idx: continue
        
        for k in G.incident(idx, mode="out"):
            if G.es[k]["type"] == "dependency":
                continue
            if G.es[k].target not in visited_idx:
                que.put(G.es[k].target)
        
        visited_idx.add(idx)
        slice_block_idx.add(idx)

    que = queue.Queue()
    visited_idx = set()
    
    for idx in call_block_idx:
        que.put(idx)
    
    while not que.empty():
        idx = que.get()
        if idx in visited_idx: continue
        
        for k in G.incident(idx, mode="in"):
            if G.es[k]["type"] != "dependency":
                continue
            if G.es[k].source not in visited_idx:
                que.put(G.es[k].source)
        
        for k in G.incident(idx, mode="out"):
            if G.es[k]["type"] != "dependency":
                continue
            if G.es[k].target not in visited_idx:
                que.put(G.es[k].target)
        
        visited_idx.add(idx)
        slice_block_idx.add(idx)

    for idx in slice_block_idx:
        G.vs[idx]["fillcolor"] = "grey"
    
    return slice_block_idx


def ppr(addr, G, slice_block_idx, idx2block):
    undirected_G = G.as_undirected()

    if len(slice_block_idx) == 0:
        p = [float(1 / undirected_G.vcount()) for i in range(undirected_G.vcount())]
    else:
        p = undirected_G.personalized_pagerank(reset_vertices=slice_block_idx, damping=0.85)

    block2impt = {idx2block[i]: p[i] for i in range(G.vcount())}
    idx2impt = {i: p[i] for i in range(G.vcount())}
    
    if not os.path.exists(project_dir + "/data/facts/" + addr + "/block2impt.npy"):
        np.save(project_dir + "/data/facts/" + addr + "/block2impt.npy", block2impt)
        np.save(project_dir + "/data/facts/" + addr + "/idx2impt.npy", idx2impt)



if __name__ == "__main__":
    addr_list = os.listdir(project_dir + "/data/.temp/")
    
    for k, addr in enumerate(addr_list):
        print(k, addr)
        b = construct_ir_table(addr)
        find_state_var_dependency(b, addr)
        G = contract_graph(b, addr, show_stor=False, show_impt=True)
        draw_contract_graph(addr)