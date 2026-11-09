import struct, numpy as np, glob, gzip
def zz(a):
    a=a.astype(np.int32); return (a>>1)^(-(a&1))
def read(fn):
    b=open(fn,'rb').read()
    if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
    hdr=struct.unpack('<3d2f4d3d',b[:88]); minh,maxh=hdr[3],hdr[4]
    n=struct.unpack('<I',b[88:92])[0]
    arr=np.frombuffer(b[92:92+6*n],dtype='<u2').reshape(3,n)
    u=np.cumsum(zz(arr[0]));v=np.cumsum(zz(arr[1]));h=np.cumsum(zz(arr[2]))
    return u/32767.,v/32767.,minh+h/32767.*(maxh-minh)
if __name__=='__main__':
    for fn in sorted(glob.glob('terrain/13_*'))[:3]:
        u,v,h=read(fn); print(fn,len(u),h.min(),h.max(),u.min(),u.max())
