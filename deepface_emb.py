import os

from deepface import DeepFace
import pickle
import torch


def deepfaceCos(solo_file):
# def deepfaceCos():
    # solo_file = './output/0000000/'
    solo_list = os.listdir(solo_file)
    # if os.path.exists(str(solo_file) + '/' + "emb_data.pkl"):
    #     os.remove(str(solo_file) + '/' + "emb_data.pkl")
    solo_list.sort(key= lambda x:int(x[:-4]))
    embeddings = []
    for image in solo_list:
        img_path = str(solo_file) + '/' + image
        embeddings.append(DeepFace.represent(img_path=img_path, model_name='VGG-Face', enforce_detection=False))
        # embeddings.append(DeepFace.represent(img_path=img_path, model_name='ArcFace', enforce_detection=False))

    # 写
    # with open(str(solo_file) + '/' + "emb_data.pkl", 'wb') as fo:
    #     pickle.dump(embeddings, fo)

    # 读
    # with open(str(solo_file) + '/' + "emb_data.pkl", 'rb') as fo:
    #     embeddings = pickle.load(fo, encoding='bytes')

    # print(embeddings[0]) # embedding1

    # if os.path.exists(str(solo_file) + '/' + "emb_data.pkl"):
    #     os.remove(str(solo_file) + '/' + "emb_data.pkl")
    embeddings = torch.Tensor(embeddings).cuda()
    # print(embeddings)
    # print(embeddings.shape)
    # print(embeddings)
    # len_emb = len(embeddings)
    # print(len_emb)
    # print(embeddings.shape)

    similarity = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1).cuda()
    # print(similarity)

    return similarity


# deepfaceCos()