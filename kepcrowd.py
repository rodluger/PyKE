import kplr
import pyfits
import glob
import scipy
import time
import numpy as np
import matplotlib.pyplot as pl
import kepio, kepmsg, kepkey, kepplot, kepfit, keparray, kepstat, kepfunc
import re, urllib.request, urllib.parse, urllib.error
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy import interpolate, optimize, ndimage, stats
from scipy.optimize import fmin_powell
from scipy.interpolate import RectBivariateSpline, interp2d
from scipy.ndimage import interpolation
from scipy.ndimage.interpolation import shift, rotate

# plot ccd with interactive source selection
class kepfield(object):
  
    def __init__(self, infile, rownum = 0, imscale = 'linear', cmap = 'YlOrBr', 
                 lcolor = 'k', acolor = 'b', query = True, logfile = 'kepcrowd.log', **kwargs):
        
        self.colrow = []
        self.fluxes = []
        self._text = []
    
        # hide warnings
        np.seterr(all = "ignore") 
    
        # test log file
        logfile = kepmsg.test(logfile)
    
        # info
        hashline = '----------------------------------------------------------------------------'
        kepmsg.log(logfile,hashline,False)
        call = 'KEPFIELD -- '
        call += 'infile='+infile+' '
        call += 'rownum='+str(rownum)
        kepmsg.log(logfile,call+'\n',False)

        try:
            kepid, channel, skygroup, module, output, quarter, season, \
                ra, dec, column, row, kepmag, xdim, ydim, barytime, status = \
                kepio.readTPF(infile, 'TIME', logfile, False)
        except:
            message = 'ERROR -- KEPFIELD: is %s a Target Pixel File? ' % infile
            kepmsg.err(logfile, message, False)
            return "", "", "", None

        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, tcorr, status = \
            kepio.readTPF(infile,'TIMECORR',logfile,False)

        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, cadno, status = \
            kepio.readTPF(infile,'rownumNO',logfile,False)

        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, fluxpixels, status = \
            kepio.readTPF(infile,'FLUX',logfile,False)

        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, errpixels, status = \
            kepio.readTPF(infile,'FLUX_ERR',logfile,False)

        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, qual, status = \
            kepio.readTPF(infile,'QUALITY',logfile,False)

        # read mask defintion data from TPF file
        maskimg, pixcoord1, pixcoord2, status = kepio.readMaskDefinition(infile,logfile,False)

        # observed or simulated data?
        coa = False
        instr = pyfits.open(infile,mode='readonly',memmap=True)
        filever, status = kepkey.get(infile,instr[0],'FILEVER',logfile,False)
        if filever == 'COA': coa = True

        # is this a good row with finite timestamp and pixels?
        if not np.isfinite(barytime[rownum-1]) or not np.nansum(fluxpixels[rownum-1,:]):
            message = 'ERROR -- KEPFIELD: Row ' + str(rownum) + ' is a bad quality timestamp'
            kepmsg.err(logfile,message,True)
            return "", "", "", None

        # construct input pixel image
        flux = fluxpixels[rownum-1,:]

        # image scale and intensity limits of pixel data    
        flux_pl, zminfl, zmaxfl = kepplot.intScale1D(flux, imscale)
        n = 0
        imgflux_pl = np.empty((ydim + 2, xdim + 2))
        for i in range(ydim + 2):
            for j in range(xdim + 2):
                imgflux_pl[i,j] = np.nan
        for i in range(ydim):
            for j in range(xdim):
                imgflux_pl[i+1, j+1] = flux_pl[n]
                n += 1
        
        # cone search around target coordinates using the MAST target search form 
        dr = max([ydim + 2, xdim + 2]) * 4.0
        kepid, ra, dec, kepmag = MASTRADec(float(ra), float(dec), dr, query, logfile)

        # convert celestial coordinates to detector coordinates
        sx = np.array([])
        sy = np.array([])
        inf, status = kepio.openfits(infile, 'readonly', logfile, False)
        try:
            crpix1, crpix2, crval1, crval2, cdelt1, cdelt2, pc, status = \
                kepkey.getWCSs(infile,inf['APERTURE'],logfile,False) 
            crpix1p, crpix2p, crval1p, crval2p, cdelt1p, cdelt2p, status = \
                kepkey.getWCSp(infile,inf['APERTURE'],logfile,False)     
            for i in range(len(kepid)):
                dra = (ra[i] - crval1) * np.cos(np.radians(dec[i])) / cdelt1
                ddec = (dec[i] - crval2) / cdelt2
                if coa:
                    sx = np.append(sx, -(pc[0,0] * dra + pc[0,1] * ddec) + crpix1 + crval1p - 1.0)
                else:
                    sx = np.append(sx, pc[0,0] * dra + pc[0,1] * ddec + crpix1 + crval1p - 1.0) 
                sy = np.append(sy, pc[1,0] * dra + pc[1,1] * ddec + crpix2 + crval2p - 1.0)
        except:
            message = 'ERROR -- KEPFIELD: Non-compliant WCS information within file %s' % infile
            kepmsg.err(logfile,message,True)    
            return "", "", "", None

        # plot
        self.fig = pl.figure(figsize = [10, 10])
        pl.clf()
            
        # pixel limits of the subimage
        ymin = np.copy(float(row))
        ymax = ymin + ydim
        xmin = np.copy(float(column))
        xmax = xmin + xdim

        # plot limits for flux image
        ymin = float(ymin) - 1.5
        ymax = float(ymax) + 0.5
        xmin = float(xmin) - 1.5
        xmax = float(xmax) + 0.5

        # plot the image window
        ax = pl.axes([0.1,0.11,0.88,0.82])
        pl.title('Select sources for fitting (KOI first)', fontsize = 24)
        pl.imshow(imgflux_pl,aspect='auto',interpolation='nearest',origin='lower',
                     vmin=zminfl, vmax=zmaxfl, extent=(xmin,xmax,ymin,ymax), cmap=cmap)
        pl.gca().set_autoscale_on(False)
        labels = ax.get_yticklabels()
        pl.setp(labels, 'rotation', 90)
        pl.gca().xaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
        pl.gca().yaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
        pl.xlabel('Pixel Column Number', {'color' : 'k'}, fontsize = 24)
        pl.ylabel('Pixel Row Number', {'color' : 'k'}, fontsize = 24)

        # plot mask borders
        kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,1,lcolor,'--',0.5)

        # plot aperture borders
        kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,2,lcolor,'-',4.0)

        # list sources
        with open(logfile, 'a') as lf:
          print('Column    Row  RA J2000 Dec J2000    Kp    Kepler ID', file = lf)
          print('----------------------------------------------------', file = lf)
          for i in range(len(sx)-1,-1,-1):
              if sx[i] >= xmin and sx[i] < xmax and sy[i] >= ymin and sy[i] < ymax:
                  if kepid[i] != 0 and kepmag[i] != 0.0:
                      print('%6.1f %6.1f %9.5f  %8.5f %5.2f KIC %d' % \
                          (float(sx[i]),float(sy[i]),float(ra[i]),float(dec[i]),float(kepmag[i]),int(kepid[i])), file = lf)
                  elif kepid[i] != 0 and kepmag[i] == 0.0:
                      print('%6.1f %6.1f %9.5f  %8.5f       KIC %d' % \
                          (float(sx[i]),float(sy[i]),float(ra[i]),float(dec[i]),int(kepid[i])), file = lf)
                  else:
                      print('%6.1f %6.1f %9.5f  %8.5f' % (float(sx[i]),float(sy[i]),float(ra[i]),float(dec[i])), file = lf)

        # plot sources
        for i in range(len(sx)-1,-1,-1):
            if kepid[i] != 0 and kepmag[i] != 0.0:
                size = max(np.array([80.0,80.0 + (2.5**(18.0 - max(12.0,float(kepmag[i])))) * 250.0]))
                pl.scatter(sx[i],sy[i],s=size,facecolors='g',edgecolors='k',alpha=0.4)
            else:
                pl.scatter(sx[i],sy[i],s=80,facecolors='r',edgecolors='k',alpha=0.4)
    
        # Sizes
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        # render plot and activate source selection
        self.srcinfo = [kepid, sx, sy, kepmag]
        pl.connect('button_release_event', self.on_mouse_release)
        pl.show(block=True)
        pl.close()
  

    def on_mouse_release(self, event):
        if (event.inaxes is not None):
            colrow = [int(np.round(event.xdata)), int(np.round(event.ydata))]
            if not colrow in self.colrow:
                self.colrow.append(colrow)
                self.fluxes.append(1.)
                a = pl.annotate(len(self.colrow), xy = colrow, xycoords = 'data', 
                                ha = 'center', va = 'center', fontsize = 48,
                                color = 'b', alpha = 0.5)
                self._text.append(a)
            else:
                for i, cr in enumerate(self.colrow):
                  if cr == colrow:
                    self.colrow.pop(i)
                    self.fluxes.pop(i)
                    self._text.pop(i).remove()
                    break
            self.fig.canvas.draw()
      
# detector location retrieval based upon RA and Dec
def MASTRADec(ra,dec,darcsec,srctab,logfile):

# coordinate limits

    darcsec /= 3600.0
    ra1 = ra - darcsec / np.cos(dec * np.pi / 180)
    ra2 = ra + darcsec / np.cos(dec * np.pi / 180)
    dec1 = dec - darcsec
    dec2 = dec + darcsec
 
# build mast query

    url  = 'http://archive.stsci.edu/kepler/kepler_fov/search.php?'
    url += 'action=Search'
    url += '&masterRA=' + str(ra1) + '..' + str(ra2)
    url += '&masterDec=' + str(dec1) + '..' + str(dec2)
    url += '&max_records=10000'
    url += '&verb=3'
    url += '&outputformat=CSV'

# retrieve results from MAST

    if srctab:
        try:
            lines = urllib.request.urlopen(url)
        except:
            message = 'WARNING -- KEPFIELD: Cannot retrieve data from MAST'
            status = kepmsg.warn(logfile, message)
            lines = ''
    else:
        lines = ''

# collate nearby sources

    kepid = []
    kepmag = []
    ra = []
    dec = []
    for line in lines:
        line = line.strip().decode('ascii')
        
        if (len(line) > 0 and 
            'Kepler' not in line and 
            'integer' not in line and
            'no rows found' not in line):
            out = line.split(',')
            r,d = sex2dec(out[0],out[1])
            try:
                if out[-22] != 'Possible_artifact': kepid.append(int(out[2]))
            except:
                if out[-22] != 'Possible_artifact': kepid.append(0)
            try:
                if out[-22] != 'Possible_artifact': kepmag.append(float(out[42]))
            except:
                if out[-22] != 'Possible_artifact': kepmag.append(0.0)
            if out[-22] != 'Possible_artifact': ra.append(r)
            if out[-22] != 'Possible_artifact': dec.append(d)
    kepid = np.array(kepid)
    kepmag = np.array(kepmag)
    ra = np.array(ra)
    dec = np.array(dec)

    return kepid, ra, dec, kepmag

# convert sexadecimal hours to decimal degrees
def sex2dec(ra, dec):

    ra = re.sub('\s+','|',ra.strip())
    ra = re.sub(':','|',ra.strip())
    ra = re.sub(';','|',ra.strip())
    ra = re.sub(',','|',ra.strip())
    ra = re.sub('-','|',ra.strip())
    ra = ra.split('|')
    outra = (float(ra[0]) + float(ra[1]) / 60 + float(ra[2]) / 3600) * 15.0

    dec = re.sub('\s+','|',dec.strip())
    dec = re.sub(':','|',dec.strip())
    dec = re.sub(';','|',dec.strip())
    dec = re.sub(',','|',dec.strip())
    dec = re.sub('-','|',dec.strip())
    dec = dec.split('|')
    if float(dec[0]) > 0.0:
        outdec = float(dec[0]) + float(dec[1]) / 60 + float(dec[2]) / 3600
    else:
        outdec = float(dec[0]) - float(dec[1]) / 60 - float(dec[2]) / 3600

    return outra, outdec

# PRF fitting
def kepprf(infile, columns, rows, fluxes, rownum = 0, border = 0, background = 0,
           focus = 0, prfdir = '../KeplerPRF', xtol = 1.e-6, ftol = 1.e-6,
           imscale = 'linear', cmap = 'YlOrBr', lcolor = 'k', acolor = 'b',
           logfile = 'kepcrowd.log', CrowdTPF = np.nan, srcinfo = None, **kwargs): 

    # log the call 
    hashline = '----------------------------------------------------------------------------'
    kepmsg.log(logfile,hashline,True)
    call = 'KEPPRF -- '
    call += 'infile='+infile+' '
    call += 'rownum='+str(rownum)+' '
    call += 'columns='+columns+' '
    call += 'rows='+rows+' '
    call += 'fluxes='+fluxes+' '
    call += 'border='+str(border)+' '
    bground = 'n'
    if (background): bground = 'y'
    call += 'background='+bground+' '
    focs = 'n'
    if (focus): focs = 'y'
    call += 'focus='+focs+' '
    call += 'prfdir='+prfdir+' '
    call += 'xtol='+str(xtol)+' '
    call += 'ftol='+str(xtol)+' '
    call += 'logfile='+logfile
    kepmsg.log(logfile,call+'\n',True)

    guess = []
    try:
        f = fluxes.strip().split(',')
        x = columns.strip().split(',')
        y = rows.strip().split(',')
        for i in range(len(f)):
            f[i] = float(f[i])
    except:
        f = fluxes
        x = columns
        y = rows
    
    nsrc = len(f)
    for i in range(nsrc):
        try:
            guess.append(float(f[i]))
        except:
            message = 'ERROR -- KEPPRF: Fluxes must be floating point numbers'
            kepmsg.err(logfile,message,True)
            return None

    if len(x) != nsrc or len(y) != nsrc:
        message = 'ERROR -- KEPFIT:FITMULTIPRF: Guesses for rows, columns and '
        message += 'fluxes must have the same number of sources'
        kepmsg.err(logfile,message,True)
        return None
    
    for i in range(nsrc):
        try:
            guess.append(float(x[i]))
        except:
            message = 'ERROR -- KEPPRF: Columns must be floating point numbers'
            kepmsg.err(logfile,message,True)
            return None
        
    for i in range(nsrc):
        try:
            guess.append(float(y[i]))
        except:
            message = 'ERROR -- KEPPRF: Rows must be floating point numbers'
            kepmsg.err(logfile,message,True)
            return None
        
    if background:
        if border == 0:
            guess.append(0.0)
        else:
            for i in range((border+1)*2):
                guess.append(0.0)
            
    if focus:
        guess.append(1.0); guess.append(1.0); guess.append(0.0)

    # open TPF FITS file
    try:
        kepid, channel, skygroup, module, output, quarter, season, \
            ra, dec, column, row, kepmag, xdim, ydim, barytime, status = \
            kepio.readTPF(infile,'TIME',logfile,True)
    except:
        message = 'ERROR -- KEPPRF: is %s a Target Pixel File? ' % infile
        kepmsg.err(logfile,message,True)
        return None
    
    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, tcorr, status = \
        kepio.readTPF(infile,'TIMECORR',logfile,True)

    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, cadno, status = \
        kepio.readTPF(infile,'CADENCENO',logfile,True)

    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, fluxpixels, status = \
        kepio.readTPF(infile,'FLUX',logfile,True)

    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, errpixels, status = \
        kepio.readTPF(infile,'FLUX_ERR',logfile,True)

    kepid, channel, skygroup, module, output, quarter, season, \
        ra, dec, column, row, kepmag, xdim, ydim, qual, status = \
        kepio.readTPF(infile,'QUALITY',logfile,True)

    # read mask defintion data from TPF file
    maskimg, pixcoord1, pixcoord2, status = kepio.readMaskDefinition(infile,logfile,True)
    npix = np.size(np.nonzero(maskimg)[0])

    print('')
    print('      KepID: %s' % kepid)
    print('        BJD: %.2f' % (barytime[rownum-1] + 2454833.0))
    print(' RA (J2000): %s' % ra)
    print('Dec (J2000):  %s' % dec)
    print('     KepMag:  %s' % kepmag)
    print('   SkyGroup:   %2s' % skygroup)
    print('     Season:   %2s' % str(season))
    print('    Channel:   %2s' % channel)
    print('     Module:   %2s' % module)
    print('     Output:    %1s' % output)
    print('')

    # is this a good row with finite timestamp and pixels?
    if not np.isfinite(barytime[rownum-1]) or np.nansum(fluxpixels[rownum-1,:]) == np.nan:
        message = 'ERROR -- KEPFIELD: Row ' + str(rownum) + ' is a bad quality timestamp'
        status = kepmsg.err(logfile,message,True)

    # construct input pixel image
    flux = fluxpixels[rownum-1,:]
    ferr = errpixels[rownum-1,:]
    DATx = np.arange(column,column+xdim)
    DATy = np.arange(row,row+ydim)

    # image scale and intensity limits of pixel data
    n = 0
    DATimg = np.empty((ydim,xdim))
    ERRimg = np.empty((ydim,xdim))
    for i in range(ydim):
        for j in range(xdim):
            DATimg[i,j] = flux[n]
            ERRimg[i,j] = ferr[n]
            n += 1

    # determine suitable PRF calibration file
    if int(module) < 10:
        prefix = 'kplr0'
    else:
        prefix = 'kplr'
    prfglob = prfdir + '/' + prefix + str(module) + '.' + str(output) + '*' + '_prf.fits'
    try:
        prffile = glob.glob(prfglob)[0]
    except:
        message = 'ERROR -- KEPPRF: No PRF file found in ' + prfdir
        kepmsg.err(logfile,message,True)
        return None

    # read PRF images
    prfn = [0,0,0,0,0]
    crpix1p = np.zeros((5),dtype='float32')
    crpix2p = np.zeros((5),dtype='float32')
    crval1p = np.zeros((5),dtype='float32')
    crval2p = np.zeros((5),dtype='float32')
    cdelt1p = np.zeros((5),dtype='float32')
    cdelt2p = np.zeros((5),dtype='float32')
    for i in range(5):
        prfn[i], crpix1p[i], crpix2p[i], crval1p[i], crval2p[i], cdelt1p[i], cdelt2p[i], status \
            = kepio.readPRFimage(prffile,i+1,logfile,True) 
    prfn = np.array(prfn)
    PRFx = np.arange(0.5,np.shape(prfn[0])[1]+0.5)
    PRFy = np.arange(0.5,np.shape(prfn[0])[0]+0.5)
    PRFx = (PRFx - np.size(PRFx) / 2) * cdelt1p[0]
    PRFy = (PRFy - np.size(PRFy) / 2) * cdelt2p[0]

    # interpolate the calibrated PRF shape to the target position
    prf = np.zeros(np.shape(prfn[0]),dtype='float32')
    prfWeight = np.zeros((5),dtype='float32')
    for i in range(5):
        prfWeight[i] = np.sqrt((column - crval1p[i])**2 + (row - crval2p[i])**2)
        if prfWeight[i] == 0.0:
            prfWeight[i] = 1.0e-6
        prf = prf + prfn[i] / prfWeight[i]
    prf = prf / np.nansum(prf) / cdelt1p[0] / cdelt2p[0]

    # location of the data image centered on the PRF image (in PRF pixel units)
    prfDimY = int(ydim / cdelt1p[0])
    prfDimX = int(xdim / cdelt2p[0])
    PRFy0 = (np.shape(prf)[0] - prfDimY) / 2
    PRFx0 = (np.shape(prf)[1] - prfDimX) / 2

    # interpolation function over the PRF
    splineInterpolation = scipy.interpolate.RectBivariateSpline(PRFx,PRFy,prf)

    # construct mesh for background model
    if background:
        bx = np.arange(1.,float(xdim+1))
        by = np.arange(1.,float(ydim+1))
        xx, yy = np.meshgrid(np.linspace(bx.min(), bx.max(), xdim),
                                np.linspace(by.min(), by.max(), ydim))

    # fit PRF model to pixel data
    start = time.time()
    if focus and background:
        args = (DATx,DATy,DATimg,ERRimg,nsrc,border,xx,yy,splineInterpolation,float(x[0]),float(y[0]))
        ans = fmin_powell(kepfunc.PRFwithFocusAndBackground,guess,args=args,xtol=xtol,
                          ftol=ftol,disp=False)
    elif focus and not background:
        args = (DATx,DATy,DATimg,ERRimg,nsrc,splineInterpolation,float(x[0]),float(y[0]))
        ans = fmin_powell(kepfunc.PRFwithFocus,guess,args=args,xtol=xtol,
                          ftol=ftol,disp=False)                    
    elif background and not focus:
        args = (DATx,DATy,DATimg,ERRimg,nsrc,border,xx,yy,splineInterpolation,float(x[0]),float(y[0]))
        ans = fmin_powell(kepfunc.PRFwithBackground,guess,args=args,xtol=xtol,
                          ftol=ftol,disp=False)
    else:
        args = (DATx,DATy,DATimg,ERRimg,nsrc,splineInterpolation,float(x[0]),float(y[0]))
        ans = fmin_powell(kepfunc.PRF,guess,args=args,xtol=xtol,
                          ftol=ftol,disp=False)
    kepmsg.log(logfile,'Convergence time = %.2fs\n' % (time.time() - start),True)

    # pad the PRF data if the PRF array is smaller than the data array 
    flux = []; OBJx = []; OBJy = []
    PRFmod = np.zeros((prfDimY,prfDimX))
    if PRFy0 < 0 or PRFx0 < 0.0:
        PRFmod = np.zeros((prfDimY,prfDimX))
        superPRF = np.zeros((prfDimY+1,prfDimX+1))
        superPRF[np.abs(PRFy0):np.abs(PRFy0)+np.shape(prf)[0],np.abs(PRFx0):np.abs(PRFx0)+np.shape(prf)[1]] = prf
        prf = superPRF * 1.0
        PRFy0 = 0
        PRFx0 = 0

    # rotate the PRF model around its center
    if focus:
        angle = ans[-1]
        prf = rotate(prf,-angle,reshape=False,mode='nearest')

    # iterate through the sources in the best fit PSF model
    for i in range(nsrc):
        flux.append(ans[i])
        OBJx.append(ans[nsrc+i])
        OBJy.append(ans[nsrc*2+i]) 

        # calculate best-fit model
        y = (OBJy[i]-np.mean(DATy)) / cdelt1p[0]
        x = (OBJx[i]-np.mean(DATx)) / cdelt2p[0]
        prfTmp = shift(prf,[y,x],order=3,mode='constant')
        prfTmp = prfTmp[PRFy0:PRFy0+prfDimY,PRFx0:PRFx0+prfDimX]
        PRFmod = PRFmod + prfTmp * flux[i]
        wx = 1.0
        wy = 1.0
        angle = 0
        b = 0.0

        # write out best fit parameters
        txt = 'Flux = %10.2f e-/s ' % flux[i]
        txt += 'X = %9.4f pix ' % OBJx[i]
        txt += 'Y = %9.4f pix ' % OBJy[i]
        kepmsg.log(logfile,txt,True)
    
    if background:
        bterms = border + 1
        if bterms == 1:
            b = ans[nsrc*3]
        else:
            bcoeff = np.array([ans[nsrc*3:nsrc*3+bterms],ans[nsrc*3+bterms:nsrc*3+bterms*2]]) 
            bkg = kepfunc.polyval2d(xx,yy,bcoeff)
            b = nanmean(bkg.reshape(bkg.size))
        txt = '\n   Mean background = %.2f e-/s' % b
        kepmsg.log(logfile,txt,True)
    if focus:
        wx = ans[-3]
        wy = ans[-2]
        angle = ans[-1]
        if not background: kepmsg.log(logfile,'',True)
        kepmsg.log(logfile,' X/Y focus factors = %.3f/%.3f' % (wx,wy),True)
        kepmsg.log(logfile,'PRF rotation angle = %.2f deg' % angle,True)

    # measure flux fraction and contamination

    # LUGER: This looks horribly bugged. ``PRFall`` is certainly NOT the sum of the all the sources.
    # Check out my comments in ``kepfunc.py``.
    
    PRFall = kepfunc.PRF2DET(flux,OBJx,OBJy,DATx,DATy,wx,wy,angle,splineInterpolation)
    PRFone = kepfunc.PRF2DET([flux[0]],[OBJx[0]],[OBJy[0]],DATx,DATy,wx,wy,angle,splineInterpolation)
    
    # LUGER: Add up contaminant fluxes
    PRFcont = np.zeros_like(PRFone)
    for ncont in range(1, len(flux)):
      PRFcont += kepfunc.PRF2DET([flux[ncont]],[OBJx[ncont]],[OBJy[ncont]],DATx,DATy,wx,wy,angle,splineInterpolation)
    PRFcont[np.where(PRFcont < 0)] = 0

    FluxInMaskAll = np.nansum(PRFall)
    FluxInMaskOne = np.nansum(PRFone)
    FluxInAperAll = 0.0
    FluxInAperOne = 0.0
    FluxInAperAllTrue = 0.0

    for i in range(1,ydim):
        for j in range(1,xdim):
            if kepstat.bitInBitmap(maskimg[i,j],2):
                FluxInAperAll += PRFall[i,j]
                FluxInAperOne += PRFone[i,j]
                FluxInAperAllTrue += PRFone[i,j] + PRFcont[i,j]
    FluxFraction = FluxInAperOne / flux[0]
    try:
        Contamination = (FluxInAperAll - FluxInAperOne) / FluxInAperAll
    except:
        Contamination = 0.0

    # LUGER: Pixel crowding metrics
    Crowding = PRFone / (PRFone + PRFcont)
    Crowding[np.where(Crowding < 0)] = np.nan

    # LUGER: Optimal aperture crowding metric
    CrowdAper = FluxInAperOne / FluxInAperAllTrue

    kepmsg.log(logfile,'\n                Total flux in mask = %.2f e-/s' % FluxInMaskAll,True)
    kepmsg.log(logfile,'               Target flux in mask = %.2f e-/s' % FluxInMaskOne,True)
    kepmsg.log(logfile,'            Total flux in aperture = %.2f e-/s' % FluxInAperAll,True)
    kepmsg.log(logfile,'           Target flux in aperture = %.2f e-/s' % FluxInAperOne,True)
    kepmsg.log(logfile,'  Target flux fraction in aperture = %.2f%%' % (FluxFraction * 100.0),True)
    kepmsg.log(logfile,'Contamination fraction in aperture = %.2f%%' % (Contamination * 100.0),True)
    kepmsg.log(logfile,'       Crowding metric in aperture = %.4f' % (CrowdAper),True)
    kepmsg.log(logfile,'          Crowding metric from TPF = %.4f' % (CrowdTPF),True)
    
    # constuct model PRF in detector coordinates
    PRFfit = PRFall + 0.0
    if background and bterms == 1:
        PRFfit = PRFall + b
    if background and bterms > 1:
        PRFfit = PRFall + bkg

    # calculate residual of DATA - FIT
    PRFres = DATimg - PRFfit
    FLUXres = np.nansum(PRFres) / npix

    # calculate the sum squared difference between data and model
    Pearson = np.abs(np.nansum(np.square(DATimg - PRFfit) / PRFfit))
    Chi2 = np.nansum(np.square(DATimg - PRFfit) / np.square(ERRimg))
    DegOfFreedom = npix - len(guess) - 1
    try:
        kepmsg.log(logfile,'\n       Residual flux = %.2f e-/s' % FLUXres,True)
        kepmsg.log(logfile,'Pearson\'s chi^2 test = %d for %d dof' % (Pearson,DegOfFreedom),True)
    except:
        pass
    kepmsg.log(logfile,'          Chi^2 test = %d for %d dof' % (Chi2,DegOfFreedom),True)

    # image scale and intensity limits for plotting images
    imgdat_pl, zminfl, zmaxfl = kepplot.intScale2D(DATimg,imscale)
    imgprf_pl, zminpr, zmaxpr = kepplot.intScale2D(PRFmod,imscale)
    imgfit_pl, zminfi, zmaxfi = kepplot.intScale2D(PRFfit,imscale)
    imgres_pl, zminre, zmaxre = kepplot.intScale2D(PRFres,'linear')
    if imscale == 'linear':
        zmaxpr *= 0.9
    elif imscale == 'logarithmic':
        zmaxpr = np.max(zmaxpr)
        zminpr = zmaxpr / 2
    
    # plot
    pl.figure(figsize=[12,10])
    pl.clf()
    
    # data
    plotimage(imgdat_pl,zminfl,zmaxfl,1,row,column,xdim,ydim,0.07,0.58,'observation',cmap,lcolor)
    pl.text(0.05, 0.05,'CROWDSAP: %.4f' % CrowdTPF,horizontalalignment='left',verticalalignment='center',
            fontsize=18,fontweight=500,color=lcolor,transform=pl.gca().transAxes)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,1,acolor,'--',0.5)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,2,acolor,'-',3.0)
    
    # model
    plotimage(imgprf_pl,zminpr,zmaxpr,2,row,column,xdim,ydim,0.445,0.58,'model',cmap,lcolor)
    pl.text(0.05, 0.05,'Crowding: %.4f' % CrowdAper,horizontalalignment='left',verticalalignment='center',
            fontsize=18,fontweight=500,color=lcolor,transform=pl.gca().transAxes)
    for x,y in zip(OBJx, OBJy):
      pl.scatter(x, y, marker = 'x', color = 'w')
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,1,acolor,'--',0.5)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,2,acolor,'-',3.0)
    
    if srcinfo is not None:
        kepid, sx, sy, kepmag = srcinfo
        for i in range(len(sx)-1,-1,-1):
            if kepid[i] != 0 and kepmag[i] != 0.0:
                size = max(np.array([80.0,80.0 + (2.5**(18.0 - max(12.0,float(kepmag[i])))) * 250.0]))
                pl.scatter(sx[i],sy[i],s=size,facecolors='g',edgecolors='k',alpha=0.1)
            else:
                pl.scatter(sx[i],sy[i],s=80,facecolors='r',edgecolors='k',alpha=0.1)
    
    # binned model            
    plotimage(imgfit_pl,zminfl,zmaxfl,3,row,column,xdim,ydim,0.07,0.18,'fit',cmap,lcolor,crowd=Crowding)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,1,acolor,'--',0.5)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,2,acolor,'-',3.0)
    
    # residuals
    reslim = max(np.abs(zminre), np.abs(zmaxre))
    plotimage(imgres_pl,-reslim,reslim,4,row,column,xdim,ydim,0.445,0.18,'residual','coolwarm',lcolor)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,1,acolor,'--',0.5)
    kepplot.borders(maskimg,xdim,ydim,pixcoord1,pixcoord2,2,acolor,'-',3.0)
        
    # plot data color bar
    barwin = pl.axes([0.84,0.18,0.03,0.8])
    if imscale == 'linear':
        brange = np.arange(zminfl,zmaxfl,(zmaxfl-zminfl)/1000)
    elif imscale == 'logarithmic':
        brange = np.arange(10.0**zminfl,10.0**zmaxfl,(10.0**zmaxfl-10.0**zminfl)/1000)
    elif imscale == 'squareroot':
        brange = np.arange(zminfl**2,zmaxfl**2,(zmaxfl**2-zminfl**2)/1000)
    if imscale == 'linear':
        barimg = np.resize(brange,(1000,1))
    elif imscale == 'logarithmic':
        barimg = np.log10(np.resize(brange,(1000,1)))        
    elif imscale == 'squareroot':
        barimg = np.sqrt(np.resize(brange,(1000,1)))        
    try:
        nrm = len(str(int(np.nanmax(brange))))-1
    except:
        nrm = 0
    brange = brange / 10**nrm
    pl.imshow(barimg,aspect='auto',interpolation='nearest',origin='lower',
              vmin=np.nanmin(barimg),vmax=np.nanmax(barimg),
              extent=(0.0,1.0,brange[0],brange[-1]),cmap=cmap)
    barwin.yaxis.tick_right()
    barwin.yaxis.set_label_position('right')
    barwin.yaxis.set_major_locator(MaxNLocator(7))
    pl.gca().yaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
    pl.gca().set_autoscale_on(False)
    pl.setp(pl.gca(),xticklabels=[],xticks=[])
    pl.ylabel('Flux (10$^%d$ e$^-$ s$^{-1}$)' % nrm)
    pl.setp(barwin.get_yticklabels(), 'rotation', 90)
    barwin.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # plot residual color bar
    barwin = pl.axes([0.07,0.08,0.75,0.03])
    brange = np.arange(-reslim,reslim,reslim/500)
    barimg = np.resize(brange,(1,1000))       
    pl.imshow(barimg,aspect='auto',interpolation='nearest',origin='lower',
              vmin=np.nanmin(barimg),vmax=np.nanmax(barimg),
              extent=(brange[0],brange[-1],0.0,1.0),cmap='coolwarm')
    barwin.xaxis.set_major_locator(MaxNLocator(7))
    pl.gca().xaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
    pl.gca().set_autoscale_on(False)
    pl.setp(pl.gca(),yticklabels=[],yticks=[])
    pl.xlabel('Residuals (e$^-$ s$^{-1}$)')
    barwin.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # render plot
    pl.show(block=True)
    pl.close()
    
    # stop time
    kepmsg.clock('\nKEPPRF ended at',logfile,True)

    return Crowding

# plot prf image
def plotimage(imgflux_pl,zminfl,zmaxfl,plmode,row,column,xdim,ydim,winx,winy,tlabel,colmap,labcol,crowd = None):

# pixel limits of the subimage

    ymin = row
    ymax = ymin + ydim
    xmin = column
    xmax = xmin + xdim

# plot limits for flux image

    ymin = float(ymin) - 0.5
    ymax = float(ymax) - 0.5
    xmin = float(xmin) - 0.5
    xmax = float(xmax) - 0.5

# plot the image window

    ax = pl.axes([winx,winy,0.375,0.4])
    pl.imshow(imgflux_pl,aspect='auto',interpolation='nearest',origin='lower',
              vmin=zminfl,vmax=zmaxfl,extent=(xmin,xmax,ymin,ymax),cmap=colmap)
    pl.gca().set_autoscale_on(False)
    labels = ax.get_yticklabels()
    pl.setp(labels, 'rotation', 90)
    pl.gca().xaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
    pl.gca().yaxis.set_major_formatter(pl.ScalarFormatter(useOffset=False))
    pl.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    pl.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    if plmode == 1:
        pl.setp(pl.gca(),xticklabels=[])
    if plmode == 2:
        pl.setp(pl.gca(),xticklabels=[],yticklabels=[])
    if plmode == 4:
        pl.setp(pl.gca(),yticklabels=[])
    if plmode == 3 or plmode == 4:
        pl.xlabel('Pixel Column Number', {'color' : 'k'}, fontsize = 14)
    if plmode == 1 or plmode == 3:
        pl.ylabel('Pixel Row Number', {'color' : 'k'}, fontsize = 14)
    pl.text(0.05, 0.93,tlabel,horizontalalignment='left',verticalalignment='center',
            fontsize=36,fontweight=500,color=labcol,transform=ax.transAxes)

    if crowd is not None:
      for i in range(crowd.shape[0]):
        for j in range(crowd.shape[1]):
          y = (i + 0.5) / crowd.shape[0]
          x = (j + 0.5) / crowd.shape[1]
          pl.text(x, y, '%.2f' % crowd[i][j],horizontalalignment='center',verticalalignment='center',
                  fontsize=14,fontweight=500,color=labcol,transform=ax.transAxes)
    return

def GetPixelCrowding(KIC, quarter, short_cadence = False, **kwargs):
  
  client = kplr.API()
  kic = client.star(KIC)
  tpf = kic.get_target_pixel_files(short_cadence = short_cadence, fetch = False)
  quarters = np.array([f.sci_data_quarter for f in tpf])
  fnum = np.argmax(quarters == quarter)
  if quarters[fnum] != quarter:
    raise Exception('Invalid quarter!')
  
  with tpf[fnum].open(clobber = False) as f:
    fitsfile = f.filename()
    CrowdTPF = f[1].header['CROWDSAP']
  
  field = kepfield(fitsfile, **kwargs)
  strcols = ",".join([str(x[0]) for x in field.colrow])
  strrows = ",".join([str(x[1]) for x in field.colrow])
  strflxs = ",".join([str(x) for x in field.fluxes])
  srcinfo = field.srcinfo
  
  if strcols != "":
    return kepprf(fitsfile, strcols, strrows, strflxs, CrowdTPF = CrowdTPF, srcinfo = srcinfo, **kwargs)
  else:
    return None