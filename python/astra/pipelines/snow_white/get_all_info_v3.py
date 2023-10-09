#out_names=open("out_names.csv","w")
import numpy as np
import get_line_info_v3
outfile=open("out_data.dat", "ab")
out_names=open("out_names.csv","w")
i=1
for line in open("classified_clean"):

    infile=(line.split(","))[0]
    spec=infile
    wave,flux,err=np.loadtxt(spec,usecols=(0,1,2),unpack=True)
    err[err==0.]=2e-18
    print(spec)
    print("DONE=",i)
    i+=1
    try:
        lines=get_line_info_v3.line_info(wave,flux,err)
    except:
        continue
    if np.size(lines)==1744:
        with open("out_data.dat", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, lines, delimiter=",",newline=" ")
            out_names.write(line)
    else:
        print("missing lines")
