import io
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Lock
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

def multiplot(wave, flux, model, N, xlbl, ylbl):
    di = len(wave)//N
    fig, axes = plt.subplots(N, 1)
    for i in range(N):
        i1 = i*di
        i2 = (i+1)*di

        axes[i].plot(wave[i1:i2], flux[i1:i2], label='Observation')
        axes[i].plot(wave[i1:i2], model[i1:i2], label='Model')

        if ylbl!=None: axes[i].set_ylabel(ylbl)
        if i==N-1:
            if xlbl!=None: axes[i].set_xlabel(xlbl)
            axes[i].legend()


class FitLoggerDB:

    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.N_subplots = 5
        self.figure_width = 20
        self.figure_height = 10
        self.dpi = 300
        self.DB_name = 'LOG.sqlite3'
        self.lock = Lock()

    def save_plot(self, wave, flux, model, name):
        N = self.N_subplots

        multiplot(wave, flux, model, N, 'Wavelength [A]', 'Flux')
        
        fig = plt.gcf()
        fig.set_size_inches(self.figure_width, self.figure_height)
        plt.tight_layout()

        postfix = '_' + str(self.run_id) + '_' + str(self.lastrowid)
        path = os.path.join(self.log_dir, name)
        fig.savefig(path + postfix + '.pdf', dpi=self.dpi)
        fig.clf()

    def save_RV_P_plot(self, xx, P, name):
        plt.plot(xx, P)
        plt.xlabel('RV [km/s]')
        plt.ylabel('Probability density')
        plt.grid()

        fig = plt.gcf()
        fig.set_size_inches(8, 4)
        plt.tight_layout()

        postfix = '_' + str(self.run_id) + '_' + str(self.lastrowid)
        path = os.path.join(self.log_dir, name)
        fig.savefig(path + postfix + '_RV.png', dpi=self.dpi//3)
        fig.clf()

    def _DB_path(self):
        return os.path.join(self.log_dir, self.DB_name)

    def init_DB(self):
        params = ['TEFF','LOGG','VSINI','VMICRO','MH','RV']
        fields = ['SNR']
        for pn in params:
            fields.append(pn)
            fields.append(pn + '_ERR')
        self.fields = fields

        path = self._DB_path()
        exists = os.path.isfile(path)

        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)

        conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        curs = conn.cursor()

        if not exists:
            curs.execute("""
CREATE TABLE RUNS (
run_id INTEGER PRIMARY KEY NOT NULL,
date TEXT NOT NULL,
config TEXT NOT NULL);""")
            curs.execute("""
CREATE TABLE RESULTS (
res_id INTEGER PRIMARY KEY NOT NULL,
run_id INTEGER NOT NULL,
date TEXT NOT NULL,
object TEXT,""" + ','.join([f+' REAL' for f in fields])+""",
FOREIGN KEY (run_id) REFERENCES RUNS(run_id) ON DELETE CASCADE ON UPDATE CASCADE
);""")
            curs.execute("""
CREATE TABLE RESPONSE (
res_id INTEGER NOT NULL,
coeff_n INTEGER NOT NULL,
value REAL NOT NULL,
FOREIGN KEY (res_id) REFERENCES RESULTS(res_id) ON DELETE CASCADE ON UPDATE CASCADE
);""")
            curs.execute("""
CREATE TABLE METADATA (
res_id INTEGER NOT NULL,
path TEXT NOT NULL,
RA REAL,
DEC REAL,
wave array,
flux array,
model array,
FOREIGN KEY (res_id) REFERENCES RESULTS(res_id) ON DELETE CASCADE ON UPDATE CASCADE
);""")

            conn.commit()
            curs.execute("ANALYZE;")
            conn.commit()
        self.conn = conn
        self.curs = curs

    def new_run(self, config):
        date = str(datetime.datetime.now())
        self.curs.execute("INSERT INTO RUNS(date, config) VALUES (?,?);", (date, config))
        self.conn.commit()
        self.run_id = self.curs.lastrowid
        return self.run_id

    def add_record(self, obj_id, snr, st_params, cheb_coef):
        with self.lock:
            date = str(datetime.datetime.now())
            N_values = 12
            assert len(st_params)==N_values
            field_list = ['run_id','object','date'] + self.fields
            qest_marks = ['?' for i in range(N_values + 4)]
            value_list = [self.run_id, obj_id, date, snr] + st_params
            self.curs.execute("INSERT INTO RESULTS (" + ','.join(field_list) + ") VALUES (" + ','.join(qest_marks) + ");", value_list)
            res_id = self.curs.lastrowid
            self.lastrowid = res_id

            for i,v in enumerate(cheb_coef):
                self.curs.execute("INSERT INTO RESPONSE (res_id, coeff_n, value) VALUES (?,?,?);", (res_id, i, v))

            self.conn.commit()

    def add_metadata(self, full_path, ra_dec, arrays):
        res_id = self.lastrowid

        meta_fields = ['res_id', 'path']
        meta_values = [res_id, full_path]
        if ra_dec is not None:
            meta_fields.append('RA')
            meta_fields.append('DEC')
            meta_values.append(ra_dec[0])
            meta_values.append(ra_dec[1])

        if arrays is not None:
            assert len(arrays)==3
            meta_fields.extend(['wave', 'flux', 'model'])
            meta_values.extend(arrays)

        q_marks = ['?' for x in meta_fields]

        with self.lock:
            self.curs.execute("INSERT INTO METADATA (" + ','.join(meta_fields) + ") VALUES (" + ','.join(q_marks) + ");", meta_values)
            self.conn.commit()


    def close(self):
        self.curs.close()
        self.conn.close()

if __name__=='__main__':
    DB = FitLoggerDB('.')
    DB.init_DB()
    DB.new_run('test')
    stp = list(np.random.rand(12))
    stp[8] = None
    stp[9] = None
    che = list(np.random.rand(15))
    DB.add_record('star', 50.0, stp, che)
    DB.close()










