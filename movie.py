import cv2 # to install in suite2p environment: conda install -c conda-forge opencv
import numpy as np
import suite2p
import os
import tifffile
from tifffile import imread,imsave
from IO import dialogMultiDir,getTifListsFull
from past.utils import old_div

def saveMovie(movie,
             file_name,
             to32=False,
             order='F',
             imagej=False,
             bigtiff=True, 
             compress=0,
             q_max=99.75,
             q_min=1):
        """
        Save the timeseries in single precision. Supported formats include
        TIFF,  AVI . from caiman github.
        Args:
            file_name: str
                name of file. Possible formats are tif, avi, npz, mmap and hdf5
            to32: Bool
                whether to transform to 32 bits
            order: 'F' or 'C'
                C or Fortran order 
            q_max, q_min: float in [0, 100]
                percentile for maximum/minimum clipping value if saving as avi
                (If set to None, no automatic scaling to the dynamic range [0, 255] is performed)
 
        """
        name, extension = os.path.splitext(file_name)[:2]
        extension = extension.lower()
        print("Parsing extension " + str(extension))

        if extension in ['.tif', '.tiff', '.btf']:
            with tifffile.TiffWriter(file_name, bigtiff=bigtiff, imagej=imagej) as tif:
                for i in range(movie.shape[0]):
                    if i % 200 == 0 and i != 0:
                        print(str(i) + ' frames saved')

                    curfr = movie[i].copy()
                    if to32 and not ('float32' in str(movie.dtype)):
                        curfr = curfr.astype(np.float32)
                    tif.save(curfr, compress=compress)

        elif extension == '.avi':
            codec = None
            try:
                codec = cv2.FOURCC('I', 'Y', 'U', 'V')
            except AttributeError:
                codec = cv2.VideoWriter_fourcc(*'IYUV')
            if q_max is None or q_min is None:
                data = movie.astype(np.uint8)
            else:
                if q_max < 100:
                    maxmov = np.nanpercentile(movie[::max(1, len(movie) // 100)], q_max)
                else:
                    maxmov = np.nanmax(movie)
                if q_min > 0:
                    minmov = np.nanpercentile(movie[::max(1, len(movie) // 100)], q_min)
                else:
                    minmov = np.nanmin(movie)
                data = 255 * (movie - minmov) / (maxmov - minmov)
                np.clip(data, 0, 255, data)
                data = data.astype(np.uint8)
                
            y, x = data[0].shape
            vw = cv2.VideoWriter(file_name, codec, movie.fr, (x, y), isColor=True)
            for d in data:
                vw.write(cv2.cvtColor(d, cv2.COLOR_GRAY2BGR))
            vw.release()
        print("finished")

def resizeMovie(movie, fx=1, fy=1, fz=1, interpolation=cv2.INTER_AREA):
    """ from caiman github"""
    #from caiman
    # todo: todocument
    T, d1, d2 = movie.shape
    print("shape: ",T, d1, d2 )
    d = d1 * d2
    elm = d * T
    max_els = 2**31 - 1
    print("elm > max_els",elm > max_els,elm,max_els)
    if elm > max_els:
        chunk_size = old_div((max_els), d)
        new_m:List = []
#         logging.debug('Resizing in chunks because of opencv bug')
        for chunk in range(0, T, chunk_size):
#             logging.debug([chunk, np.minimum(chunk + chunk_size, T)])
            m_tmp = movie[chunk:np.minimum(chunk + chunk_size, T)].copy()
            print("shape m_tmp ",m_tmp.shape)
            m_tmp = resizeMovie(m_tmp, fx=fx, fy=fy, fz=fz,
                                 interpolation=interpolation)
            if len(new_m) == 0:
                new_m = m_tmp
            else:
                new_m = np.concatenate([new_m, m_tmp], axis=0)
                print("new_m = np.concatenate shape: ",new_m.shape)

        return new_m
    else:
        if fx != 1 or fy != 1:
#             logging.debug("reshaping along x and y")
            t, h, w = movie.shape
            newshape = (int(w * fy), int(h * fx))
            mov = []
#             logging.debug("New shape is " + str(newshape))
            for frame in movie:
                mov.append(cv2.resize(frame, newshape, fx=fx,
                                      fy=fy, interpolation=interpolation))
            movie = np.asarray(mov) #movie(np.asarray(mov), **self.__dict__)
        if fz != 1:
#             logging.debug("reshaping along z")
            t, h, w = movie.shape
            movie = np.reshape(movie, (t, h * w))
            mov = cv2.resize(movie, (h * w, int(fz * t)),
                             fx=1, fy=fz, interpolation=interpolation)
            mov = np.reshape(mov, (np.maximum(1, int(fz * t)), h, w))
#             self = movie(mov, **self.__dict__)
#             self.fr = self.fr * fz
    del movie
    mov = np.asarray(mov)
#     print("downsampled "+str(mov.shape))
    return mov

def getFramesIdxSessions(sessions,idxs): 
    """ get all the frames number corresponding to the selected session(s)
    Args:
        sessions: np.ndarray
                    1D-array of selected sessions as integers
                    can be discontinuous e.g. [1,2,3,7,8,9] or [4,3,5]
        idxs: np.ndarray
                1D-array with all frame number starting and ending sessions
    Returns:
        sequence: np.ndarray
                    1D-array containing frame numbers
    """
        
    
    if sessions is not None:
        
        if isinstance(sessions,int) or isinstance(sessions,np.int32):
            return np.arange(idxs[sessions-1],idxs[sessions])
        if len(sessions)==1:
            return np.arange(idxs[sessions[0]-1],idxs[sessions[0]])
        if len(sessions)>1:
            sequence=np.arange(idxs[sessions[0]-1],idxs[sessions[0]])
            for i in range(1,len(sessions)):
                sequence=np.concatenate((sequence,np.arange(idxs[sessions[i]-1],idxs[sessions[i]])))
            return sequence
    else:
        return np.arange(0,idxs[-1])

def getMovie(ops,sessions=None,sequence=None):
    """ get movie as array from a registered binary file. optional: slice movie to the selected session(s) or sequence.
    Args:
        ops: dictionnary
                'Ly', 'Lx', 'nframes','reg_file'
        sessions: np.ndarray
                    1D-array containing selected sessions as integers
                    can be discontinuous e.g. [1,2,3,7,8,9] or [4,3,5]
        sequence: np.ndarray
                    1D-array containing frame numbers
    Returns:
        m: np.ndarray
            movie
    """
    
    if sequence is None:
        if  sessions is not None:
            idxs=np.load(ops['save_path']+"/idxs.npy")
            sequence=getFramesIdxSessions(sessions,idxs)
        else:
            sequence=np.arange(0,ops["nframes"])
        
    binFile=suite2p.io.BinaryFile(Ly=ops["Ly"],Lx=ops["Lx"],read_filename=ops['reg_file'])
    m=binFile.ix(sequence)
    binFile.close()
    del binFile
    return m

def generateDownsampledRawMovieFromOps(ops,changeLetterDrive=None):

    r=ops["filelist"]
    fx,fy,fz=ops['rescale_reg_tif']

    if changeLetterDrive is not None:
        for d in range(len(r)):
            r[d]=changeLetterDrive+r[d][1:]
            print(r[d])

    c=0
    print(c)
    m=resizeMovie(imread(r[0]),fx=fx,fy=fy,fz=fz)

    for i in r[1:]:
        c+=1
        print(c," / ", len(r[1:])-1)
        m=np.concatenate((m,resizeMovie(imread(i),fx=fx,fy=fy,fz=fz)),axis=0)
    print(m.shape,m.dtype)


    # imsave(ops["save_path"]+"/raw_ds.tiff",m)

    # resize m to reg_tiffs if different depth, 
    # difference of dimension Z length can occur because resize been done bits by bits
    regTiffZ=int(ops["nframes"]*fz)
    if m.shape[0] != regTiffZ:
        print(m.shape[0]," != ", regTiffZ)
        m=resizeMovie(m,fz=regTiffZ/m.shape[0])

    saveMovie(m,ops["save_path"]+"/raw_ds.tiff") 
    print(m.shape,m.dtype)
    
def generateDownsampledRawMovieFromSubdirsErrorProof(subdirs=None,fxy=0.5,fz=0.1,saveConcatenated=False):
    """ generate downsample movie for each group of tiff files present in the subdirs
        e.g. select A/X and A/Y -> A/X/[tiffs,..],A/Y/[tiffs,...] -> A/X_ds.tif, A/Y_ds.tif 
        Args:
            subdirs: list of string
                        list of subdir path, if not provided, dialog box to select them
            fxy: float 
                downsampling ratio in x and y
            fz: float 
                downsampling ratio in z
            saveConcatenated: boolean
                                to additionally save concatenation of all downsized movies"""
    
    fx,fy=fxy,fxy
    
    if subdirs is None:
        subdirs=dialogMultiDir( title="select tiff subdirs")
         
    path=('/').join(subdirs[0].split('/')[:-1])
#     print(path)
    for s in subdirs:
        name=s.split('/')[-1]
        print(path+"/"+name)
        
        try:
            r=getTifListsFull(s)

            c=0
            m=resizeMovie(imread(r[0]),fx=fx,fy=fy,fz=fz)
            print(c+1," / ", len(r))
            for i in r[1:]:
                c+=1
                print(c+1," / ", len(r))
                m=np.concatenate((m,resizeMovie(imread(i),fx=fx,fy=fy,fz=fz)),axis=0)
            print(m.shape,m.dtype)


            # imsave(ops["save_path"]+"/raw_ds.tiff",m)

            # resize m to reg_tiffs if different depth, 
            # difference of dimension Z length can occur because resize been done bits by bits
    #         regTiffZ=int(ops["nframes"]*fz)
    #         if m.shape[0] != regTiffZ:
    #             print(m.shape[0]," != ", regTiffZ)
    #             m=resizeMovie(m,fz=regTiffZ/m.shape[0])

            saveMovie(m,path+"/"+name+"_ds.tiff") 
            print(m.shape,m.dtype)

            if saveConcatenated:
                if s == subdirs[0]:
                    mm=m
                elif s== subdirs[-1]:
                    mm=np.concatenate((mm,m))
                    saveMovie(mm,path+"/raw_concatenated_ds.tiff") 
                else:
                    mm=np.concatenate((mm,m))
        except:
            print(name," failed")
    
    
def generateDownsampledRawMovieFromSubdirs(subdirs=None,fxy=0.5,fz=0.1,saveConcatenated=False):
    """ generate downsample movie for each group of tiff files present in the subdirs
        e.g. select A/X and A/Y -> A/X/[tiffs,..],A/Y/[tiffs,...] -> A/X_ds.tif, A/Y_ds.tif 
        Args:
            subdirs: list of string
                        list of subdir path, if not provided, dialog box to select them
            fxy: float 
                downsampling ratio in x and y
            fz: float 
                downsampling ratio in z
            saveConcatenated: boolean
                                to additionally save concatenation of all downsized movies"""
    
    fx,fy=fxy,fxy
    
    if subdirs is None:
        subdirs=dialogMultiDir( title="select tiff subdirs")
         
    path=('/').join(subdirs[0].split('/')[:-1])
#     print(path)
    for s in subdirs:
        name=s.split('/')[-1]
        print(path+"/"+name)
        r=getTifListsFull(s)

        c=0
        m=resizeMovie(imread(r[0]),fx=fx,fy=fy,fz=fz)
        print(c+1," / ", len(r))
        for i in r[1:]:
            c+=1
            print(c+1," / ", len(r))
            m=np.concatenate((m,resizeMovie(imread(i),fx=fx,fy=fy,fz=fz)),axis=0)
        print(m.shape,m.dtype)


        # imsave(ops["save_path"]+"/raw_ds.tiff",m)

        # resize m to reg_tiffs if different depth, 
        # difference of dimension Z length can occur because resize been done bits by bits
#         regTiffZ=int(ops["nframes"]*fz)
#         if m.shape[0] != regTiffZ:
#             print(m.shape[0]," != ", regTiffZ)
#             m=resizeMovie(m,fz=regTiffZ/m.shape[0])

        saveMovie(m,path+"/"+name+"_ds.tiff") 
        print(m.shape,m.dtype)
        
        if saveConcatenated:
            if s == subdirs[0]:
                mm=m
            elif s== subdirs[-1]:
                mm=np.concatenate((mm,m))
                saveMovie(mm,path+"/raw_concatenated_ds.tiff") 
            else:
                mm=np.concatenate((mm,m))