import io
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import datetime


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def printf(row, meta):
    print('-'*50)
    print('row_id =', row[0])
    print('run_id =', row[1])
    print('date =', row[2])
    print('object =', row[3])
    print('full_path =', meta[1])
    print('RA =', meta[2])
    print('DEC =', meta[3])
    print('-'*25)
    print('SNR =', '%.1f'%row[4])
    print('TEFF =', '%.0f \u00b1 %.0f [K]'%(row[5], row[6]))
    print('LOG(G) =', '%.2f \u00b1 %.2f [cm/s^2]'%(row[7], row[8]))
    print('V*SIN(I) =', '%.2f \u00b1 %.2f [km/s]'%(row[9], row[10]))
    print('V_MICRO =', '%.2f \u00b1 %.2f [km/s]'%(row[11], row[12]))
    print('[M/H] =', '%.2f \u00b1 %.2f [dex]'%(row[13], row[14]))
    print('RV =', '%.2f \u00b1 %.2f [km/s]'%(row[15], row[16]))

    print('-'*50)

if len(sys.argv)<2:
    print('Use: '+sys.argv[0]+' <path to sqlite3 DB> [row id]')
    exit()

db_path = sys.argv[1]
res_id = None
if len(sys.argv)>2: res_id = sys.argv[2]

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
curs = conn.cursor()
curs2 = conn.cursor()

if res_id is None:
    res = curs.execute("SELECT * FROM RESULTS;")
else:
    res = curs.execute("SELECT * FROM RESULTS where res_id="+res_id+";")

for row in res:
    res_id = row[0]
    res2 = curs2.execute("SELECT * FROM METADATA WHERE res_id="+str(res_id)+";")
    meta = res2.fetchone()

    printf(row, meta)

    (wave, flux, model) = meta[4:7]
    plt.title(row[3])
    plt.plot(wave, flux, label='Data')
    plt.plot(wave, model, label='Model')
    plt.xlabel('Wavelength [A]')
    plt.ylabel('Flux')
    plt.legend()
    plt.show()

















