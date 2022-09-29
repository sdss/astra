import sys
import sqlite3

db_path = sys.argv[1]

conn = sqlite3.connect(db_path)
curs = conn.cursor()

curs.executescript(
"""
CREATE TEMPORARY TABLE meta_bak(
res_id INTEGER NOT NULL,
path TEXT NOT NULL,
RA REAL,
DEC REAL,
FOREIGN KEY (res_id) REFERENCES RESULTS(res_id) ON DELETE CASCADE ON UPDATE CASCADE);


INSERT INTO meta_bak SELECT res_id, path, RA, DEC FROM metadata;

DROP TABLE METADATA;

CREATE TABLE METADATA(
res_id INTEGER NOT NULL,
path TEXT NOT NULL,
RA REAL,
DEC REAL,
FOREIGN KEY (res_id) REFERENCES RESULTS(res_id) ON DELETE CASCADE ON UPDATE CASCADE);

INSERT INTO METADATA SELECT res_id, path, RA, DEC FROM meta_bak;
DROP TABLE meta_bak;
VACUUM;
""")

conn.commit()




