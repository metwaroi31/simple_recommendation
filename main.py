from flask import (
    Flask,
    request
)
from txtai.vectors import VectorsFactory
from txtai.embeddings import Embeddings
from txtai.ann import ANNFactory
import sqlite3
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# db = sqlite3.connect("queries.sqlite")
# sql = '''CREATE TABLE QUERY(
#    UID INTEGER PRIMARY KEY AUTOINCREMENT,
#    TEXT CHAR(256)
# )'''
# cur = db.cursor()
# cur.execute(sql)
# db.commit()
# db.close()
app = Flask(__name__)

model = VectorsFactory.create({"path": "keepitreal/vietnamese-sbert"}, None)

# Create Faiss index using normalized embeddings
ann = ANNFactory.create({"backend": "faiss"})
ann.index(model.encode(['03 tran nhan ton']))

def get_data(uid):
    db = sqlite3.connect("queries.sqlite")
    cur = db.cursor()
    cur.execute("SELECT TEXT FROM QUERY WHERE UID=" + str(uid))
    text = cur.fetchone()
    db.commit()
    db.close()
    return text

def save_new_data(text):
    db = sqlite3.connect("queries.sqlite")
    cur = db.cursor()
    print(text)
    cur.execute("INSERT INTO QUERY(TEXT) VALUES('" + text + "')")
    text = cur.fetchone()
    db.commit()
    db.close()
    return text

def re_index_data():
    db = sqlite3.connect("queries.sqlite")
    cur = db.cursor()
    cur.execute("SELECT TEXT FROM QUERY")
    res = cur.fetchall()
    db.commit()
    db.close()
    to_embeddings_list = []
    for row in res:
        to_embeddings_list.append(row[0])
    if len(to_embeddings_list) > 0:
        ann.index(model.encode(to_embeddings_list))
    return to_embeddings_list

re_index_data()

@app.route("/search", methods=['POST'])
def search():
    request_data = request.get_json()
    search_text = request_data['text']
    
    query = model.encode([search_text])
    return_list = []
    for uid, score in ann.search(query, 10)[0]:
        return_list.append(get_data(uid))
    save_new_data(search_text)
    re_index_data()
    return return_list
