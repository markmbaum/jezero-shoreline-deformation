from numpy import *
from copy import copy
from os.path import join, isfile
from pandas import read_csv, read_excel
from geopandas import read_file
from scipy.interpolate import RegularGridInterpolator
from orthopoly.legendre import *
from orthopoly.spherical_harmonic import *
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

π = pi

#-------------------------------------------------------------------------------
# INPUT

#-----------------
#general variables

#path to directory for saving plots
dirplots = 'plots'

#whether to save and close plots instead of showing them
saveplots = False

#path to global topography grid downsampled from MOLA data here:
#   https://pds-geosciences.wustl.edu/missions/mgs/megdr.html
fngtopo = join('topo', 'topo_modern.csv')

#path to csv with approximate lat-lon of Arabia level
#    thanks to Robert Citron for providing these coordinates
fnarabia = join('cite', 'shorelines', 'arabia_coordinates.csv')

#mean radius of Mars (m)
R = 3.39e6

#gravity on Mars (m/s^2)
g = 3.7

#----------------
#Jezero variables

#path to Jezero regional topography grid, stored as a BSQ file
#more info on file format at:
#   https://www.harrisgeospatial.com/docs/ENVIHeaderFiles.html
fnjtopo = join('topo', 'jezero_regional_topography.bsq')

#folder with shapefiles for Jezero structures, from:
#  Goudge, T. A., Fassett, C. I. & Mohrig, D. Incision of paleolake outlet canyons on Mars from overflow flooding. Geology 47, 7–10 (2019).
dirjshp = join('cite', 'jezero-shapefiles', 'geographic-coordinates')

#name of shapefile for outlet canyon
fnjoutlet = 'Goudge_et_al_2018_Jezero_Outlet_Canyon.shp'

#name of shapefile for lake outline after breach
fnjlake = 'Goudge_et_al_2018_Jezero_Post_Breach_Lake_Outline.shp'

#Jezero location in degrees and radians
jezlocd = 18.38, 77.68
jezlocr = 1.25, 4.5

#limits for extra cropping of Jezero elevation grid
jezlatlim = 16.5, 20.5
jezlonlim = 75, 80

#resolution reduction interval for jezero topo
fjezres = 2

#-----------------------------
#Tharsis deformation variables

#path to excel workbook with spherical harmonic coeffs of Tharsis deformation
#workbook modified slightly from original at:
#   https://www.nature.com/articles/nature26144
fnthar = join('cite', 'correction-parameters', 'nature26144-s1.xlsx')

#lithospheric elastic thickness column in Tharsis deformation workbook
colthar = 'T_e = 58 km'

#tharsis reduction fraction between 0 (no reduction) and 1 (full reduction)
tharfrac = 1.0

#sea level for Tharsis deformation scenario (m)
sealevthar = -2.3e3

#---------------------------
#true polar wander variables

#file with TPW results from Perron et al. 2007
fntpw = join('cite', 'correction-parameters', 'tpw_results.csv')

#elastic lithospheric thickness to use for main plot
Te = 200

#planetary rotation rate (radians/s)
ω = 2*π/((24 + 37.0/60)*60*60)

#-------------------------------------------------------------------------------
# FUNCTIONS

#flattens one level of nested containers
flatten = lambda l: [item for sublist in l for item in sublist]

def sph_arclength(θ1, ϕ1, θ2, ϕ2):
    """Computes the orthodromic (great-circle) arclength between points on a sphere

    :param float θ1: colatitude of point 1 in [0,π]
    :param float ϕ1: azimuth of point 1 in [0,2*π]
    :param float θ2: colatitude of point 2 in [0,π]
    :param float ϕ2: azimuth of point 2 in [0,2*π]

    :return: arclength in radians
    :rtype: float"""

    return( arccos(dot(sph2cart(1.0, θ1, ϕ1), sph2cart(1.0, θ2, ϕ2))) )

def true_polar_wander(θ, ϕ, ω, a, g, h2, k2, θp, ϕp):
    """Computes deformation due to true polar wander (TPW)

    :param float/array θ: input colatitude(s) [0,π]
    :param float/array ϕ: input azimuth(s) [0,2*π], same shape as θ
    :param float ω: planetary rotation rate (radians/sec)
    :param float a: planetary radius (m)
    :param float g: gravity (m/s^2)
    :param float h2: secular (fluid-limit) degree-2 Love number (?)
    :param float k2: secular (fluid-limit) degree-2 Love number (?)
    :param float θp: paleopole colatitude [0,π]
    :param float ϕp: paleopole azimuth [0,2*π]

    :return: topographic deformation/response with same shape as θ & ϕ"""

    #distance between input locations and paleopole
    gamma = sph_arclength(θ, ϕ, θp, ϕp)
    #legendre polynomial degree & order
    n, m = 2, 0
    #normalization factor
    norm = legen_norm(n, m)
    #legendre polynomials
    P = (legen_theta(gamma, n, m) - legen_theta(θ, n, m))/norm
    #deformation
    delz = ((ω**2*a**2)/(3*g))*P*(h2 - (1 + k2))

    return(delz)

def ginterp(θ, ϕ, Z, sθ, sϕ):
    """Interpolates a grid along a profile

    :param array θ: colatitude coordiantes of grid [0,π]
    :param array ϕ: azimuth coordinates of grid [0,2*π]
    :param array Z: 2D array to interpolate
    :param array sθ: colatitude coordinates of profile to interpolate on
    :param array sϕ: azimuth coordinates of profile to interpolate on

    :return array: interpolated profile"""

    #make an interpolating object
    f = RegularGridInterpolator((θ, ϕ), Z)
    #arrange the interpolation points properly
    pts = array(list(zip(sθ, sϕ)))
    #interpolate
    z = f(pts)
    return(z)

#-------------------------------------------------------------------------------
# PLOTTING FUNCTIONS

def spines(b, *axs):
    """Makes all spines of Axes visible"""

    for ax in axs:
        for s in ax.spines:
            ax.spines[s].set_visible(True)

def annotate_jezero(ax, x, y, label=False):
    """Draws a star on Jezero and optionally label it

    :param Axes ax: the Axes object to draw in
    :param float x: the x coordinate of Jezero
    :param float y: the y coordinate of Jezero
    :param bool label: whether to write the word Jezero """

    ax.plot(x, y, '*',
            color='w',
            markersize=11,
            zorder=100,
            markeredgecolor='k')
    if label:
        text = ax.text(x, y, 'Jezero',
                color='yellow',
                ha='right',
                va='top',
                fontsize=16,
                zorder=100)
        text.set_path_effects([path_effects.Stroke(linewidth=1.6, foreground='k'), path_effects.Normal()])

def draw_shp(ax, shp, color='k', linewidth=0.75, linestyle='solid'):
    """Draw the lines from input list of shapely multilines

    :param Axes ax: the Axes object to draw in
    :param list shp: list of shapely multilines
    :param str color: line color
    :param float linewidth: line width
    :param str linestyle: line style"""

    for i in range(len(shp)):
        lon, lat = zip(*shp[i].coords)
        ax.plot(lon, lat, color=color, linestyle=linestyle, linewidth=linewidth)

def draw_contours(ax, x, y, Z, onlyzero=False, color='grey'):
    """Draws contours in an Axes

    :param Axes ax: the Axes to draw in
    :param x: the x coordinates
    :param y: the y coordinates
    :param array Z: 2D array to contour
    :param bool onlyzero: whether to contour the zero level only
    :param str color: color of contour lines"""

    if onlyzero:
        ax.contour(x, y, Z, colors=color, levels=[0], linewidths=1.5)
    else:
        ax.contour(x, y, Z, colors=color, linewidths=0.5)

def label_panel(ax, lab, xlab, ylab):
    """Labels a panel with a letter

    :param Axes ax: the Axes object to draw in
    :param str lab: label to put in the top left corner or None
    :param str xlab: horizontal axis label or None
    :param str ylab: vertical axis label or None"""

    if lab is not None:
        ax.text(0.03, 0.96, lab,
                ha='left', va='top',
                bbox=dict(fc='w', ec='k', alpha=0.8, boxstyle='round'),
                fontsize=12,
                transform=ax.transAxes,
                zorder=100)
    if xlab is not None:
        ax.set_xlabel(xlab)
    if ylab is not None:
        ax.set_ylabel(ylab)

def handle_colorbar(ax, r, clab=None):
    """Draws a colorbar or not

    :param Axes ax: Axes to put colorbar on
    :param r: collection (the thing that was plotted) to make colorscale from
    :param str clab: label for colorbar"""

    if clab is not None:
        cb = plt.colorbar(r, ax=ax)
        if type(clab) is str:
            cb.set_label(clab, rotation=270, va='bottom')

def handle_subax(fig, subax, projection=None):
    """If subax is an integer, create a new subplot in the figure and return
    the Axes. Otherwise, if subax is already an Axes, do nothing and return it.

    :param Figure fig: the Figure object being plotted in
    :param subax: an integer code for subplot or an Axes
    :param str projection: plotting projection to use or None

    :return: Axes object"""

    if type(subax) is int:
        ax = fig.add_subplot(subax, projection=projection)
    else:
        ax = subax
    return(ax)

def plot_global_def(fig, subax, gΘ, gΦ, gTopo, gDef,
        cθ=None, cϕ=None, axlab=None, clab=None, title=None):
    """Make a global, projected plot of deformation field

    :param Figure fig: Figure object to plot in
    :param subax: subplot code or Axes object
    :param array gΘ: colatitude coordinate grid (radians)
    :param array gΦ: azimuth coordinate grid (radians)
    :param array gTopo: topography grid for contours
    :param array gDef: deformation grid
    :param array cθ: colatitude coordinates of shoreline/contact
    :param array cϕ: azimuth coordinates of shoreline/contact
    :param str axlab: label for top left corner of axes
    :param str clab: label for colorbar or None for no colorbar
    :param str title: title for axes or None

    :return: Axes object"""

    #make subplotting decision
    ax = handle_subax(fig, subax, projection='mollweide')
    #get total range of values
    v = abs(gDef/1e3).max()
    #plot global deformation
    r = ax.pcolormesh(gΦ-π, π/2-gΘ, gDef/1e3,
            cmap='PuOr',
            vmin=-v, vmax=v,
            shading='gouraud')
    #plot topo contours for context
    ax.contour(gΦ-π, π/2-gΘ, gTopo/1e3,
            colors='k',
            linewidths=0.5,
            levels=arange(-10, 20, 2))
    #draw shoreline/contact
    if cθ is not None and cϕ is not None:
        ax.plot(cϕ-π, π/2-cθ, 'r')
    #make a colorbar if desired
    handle_colorbar(ax, r, clab)
    #label the location of Jezero
    annotate_jezero(ax, jezlocr[1]-π, π/2-jezlocr[0])
    #other formatting
    ax.set_xticks([])
    ax.set_yticks([])
    label_panel(ax, axlab, None, None)
    if title is not None:
        ax.set_title(title)
    return(ax)

def plot_jezero_def(fig, subax, jlon, jlat, jTopo, jDef, shps,
        xlab=None, ylab=None, axlab=None, clab=None, title=''):
    """Make a plot of deformation in the Jezero region

    :param Figure fig: Figure object to plot in
    :param subax: subplot code or Axes object
    :param array jlon: longitude coordinates
    :param array jlat: latitude coordinates
    :param array jTopo: topography grid for contours
    :param array jDef: deformation grid
    :param list shps: list of shapely multiline objects
    :param str xlab: x axis label
    :param str ylab: y axis label
    :param str axlab: label for top left corner of axes
    :param str clab: label for colorbar or None for no colorbar
    :param str title: title for axes or None

    :return: Axes object"""

    #make subplotting decision
    ax = handle_subax(fig, subax)
    #plot the deformation field
    r = ax.pcolormesh(jlon, jlat, jDef/1e3, cmap='Purples', shading='gouraud')
    #include topo contours
    draw_contours(ax, jlon, jlat, jTopo, color='k')
    #make a colorbar if desired
    handle_colorbar(ax, r, clab)
    #formatting
    label_panel(ax, axlab, xlab, ylab)
    if title:
        ax.set_title(title)
    spines(True, ax)
    return(ax)

def plot_jezero_ocean(fig, subax, jlon, jlat, jTopo, jDef, sealev, shps,
        xlab=None, ylab=None, axlab=None, clab=None, title=''):
    """Make a plot showing ocean in the Jezero region

    :param Figure fig: Figure object to plot in
    :param subax: subplot code or Axes object
    :param array jlon: longitude coordinates
    :param array jlat: latitude coordinates
    :param array jTopo: topography grid for contours
    :param array jDef: deformation grid
    :param float sealev: sea level
    :param list shps: list of shapely multiline objects
    :param str xlab: x axis label
    :param str ylab: y axis label
    :param str axlab: label for top left corner of axes
    :param str clab: label for colorbar or None for no colorbar
    :param str title: title for axes or None

    :return: Axes object"""

    #make subplotting decision
    ax = handle_subax(fig, subax)
    #specify coloring
    cmap = copy(plt.get_cmap('Greys'))
    cmap.set_bad('royalblue')
    h = jTopo - jDef
    h[h < sealev] = nan
    r = ax.pcolormesh(jlon, jlat, h/1e3, cmap=cmap, shading='auto')
    #make a colorbar if desired
    handle_colorbar(ax, r, clab)
    #draw Jezero shapefile info
    for shp in shps:
        draw_shp(ax, shp)
    label_panel(ax, axlab, xlab, ylab)
    if title:
        ax.set_title(title)
    spines(True, ax)
    return(ax)

def plot_jezero_sealev(fig, subax, jlon, jlat, jTopo, jDef, sealev, shps,
        div=True, xlab=None, ylab=None, axlab=None, clab=None, title='',
        cmap='RdBu_r'):
    """Make a global, projected plot of deformation field

    :param Figure fig: Figure object to plot in
    :param subax: subplot code or Axes object
    :param array jlon: longitude coordinates
    :param array jlat: latitude coordinates
    :param array jTopo: topography grid for contours
    :param array jDef: deformation grid
    :param float sealev: sea level
    :param list shps: list of shapely multiline objects
    :param bool div: whether to use a centered diverging colorscale
    :param str xlab: x axis label
    :param str ylab: y axis label
    :param str axlab: label for top left corner of axes
    :param str clab: label for colorbar or None for no colorbar
    :param str title: title for axes or None
    :param str cmap: the colorscale to use

    :return: Axes object"""

    #make subplotting decision
    ax = handle_subax(fig, subax)
    #compute level from sea level
    h = (jTopo - jDef) - sealev
    #determine colorscale range
    if div is True:
        v = abs(h).max()
        vmin = -v
        vmax = v
    elif div is False:
        vmin = h.min()
        vmax = h.max()
    else:
        vmin = -div
        vmax = div
    #plot topography from sea level
    r = ax.pcolormesh(jlon, jlat, h/1e3,
            vmin=vmin/1e3,
            vmax=vmax/1e3,
            cmap=cmap,
            shading='auto')
    #make a colorbar if desired
    handle_colorbar(ax, r, clab)
    #include topo contours
    draw_contours(ax, jlon, jlat, h/1e3, True)
    #draw Jezero shapefile info
    for shp in shps:
        draw_shp(ax, shp)
    label_panel(ax, axlab, xlab, ylab)
    if title:
        ax.set_title(title)
    spines(True, ax)
    return(ax, r)

def save_close(fig, d, fn):
    """Saves and closes a figure, printing a message on the way

    :param Figure fig: the Figure object to save
    :param str d: the directory to save in
    :param stf fn: the name of the file to save"""

    if saveplots:
        p = join(d, fn)
        fig.savefig(p)
        print('figure saved: %s' % p)
        plt.close(fig)

#-------------------------------------------------------------------------------
# MAIN

if __name__ == "__main__":

    #read in global topography
    print('loading global topography')
    gTopo = genfromtxt(fngtopo, delimiter=',', dtype=float)
    glon, glat, gTopo = gTopo[0,1:], gTopo[1:,0], gTopo[1:,1:]
    gθ, gϕ = latlon2sph(glat, glon)

    #read in Jezero topography
    print('loading Jezero regional topography')
    nx, ny = 3301, 2580 #from the .hdr file
    dx, dy = 0.00337412083062016, 0.00337412083065738 #from the .hdr file
    x0, y0 =  70.778156617, 23.651287986 #from the .hdr file
    jTopo = fromfile(fnjtopo, dtype='int16').reshape(ny, nx)[::fjezres,::fjezres]
    print('Jezero regional topography resolution reduced by factor of %g' % fjezres)
    jlon = (x0 + dx*arange(nx))[::fjezres]
    jlat = (y0 - dy*arange(ny))[::fjezres]

    #extra cropping
    mlat = (jlat >= jezlatlim[0]) & (jlat <= jezlatlim[1])
    mlon = (jlon >= jezlonlim[0]) & (jlon <= jezlonlim[1])
    jlat, jlon = jlat[mlat], jlon[mlon]
    jTopo = jTopo[mlat,:]
    jTopo = jTopo[:,mlon]

    #convert to radians
    jθ, jϕ = latlon2sph(jlat, jlon)

    #make a grid
    jΦ, jΘ = meshgrid(jϕ, jθ)

    #read Jezero shapefile info, which is in latitude-longitude coordinates
    print('reading Jezero shapefiles from Goudge et al. 2018')
    joutlet = read_file(join(dirjshp, fnjoutlet)).geometry[0]
    jlake = read_file(join(dirjshp, fnjlake)).geometry[0]
    jshps = [joutlet, jlake]

    #read approximate Arabia level coordinates and store spherical coordinates
    print('loading coordinates of Arabia contact')
    arabia = read_csv(fnarabia, skiprows=1)
    arabia['theta'], arabia['phi'] = latlon2sph(arabia['latitude'], arabia['longitude'])

    #-------------------
    #Tharsis deformation

    #read in the expansion coefficients for the shape
    print('loading Tharsis correction model of Citron et al. 2018')
    df = read_excel(fnthar, sheet_name='shape', skiprows=2)
    yn = df['degree'].values.astype('int')
    ym = df['order'].values.astype('int')

    #create a spherical harmonic expansion object from orthopoly
    ex_s = Expansion(df[colthar].values, yn, ym)

    #read in the expansion coefficients for the geoid
    df = read_excel(fnthar, sheet_name='gravity', skiprows=2)

    #create a spherical harmonic expansion object from orthopoly
    ex_g = Expansion(df[colthar].values, yn, ym)

    #coefficients are for unnormalized harmonics, need to unnormalize them
    norms = array([sph_har_norm(n,abs(m)) for n,m in zip(yn, ym)])
    ex_s /= norms
    ex_g /= norms

    #evaluate the Tharsis contribution on the topography grids
    print('evaluating Tharsis correction over global and Jezero topography')
    jThar = ex_s(jΘ, jΦ-π) - R*ex_g(jΘ, jΦ-π) #subtract π b/c of different origin
    gΦ, gΘ = meshgrid(gϕ, gθ)
    gThar = ex_s(gΘ, gΦ) - R*ex_g(gΘ, gΦ)
    gThar = roll(gThar, gThar.shape[1]//2, axis=1) #roll b/c of different origin

    #reduce the tharsis contributions by some fraction, if needed
    jThar *= tharfrac
    gThar *= tharfrac

    #print the deformation in the center of Jezero
    z = ginterp(gθ, gϕ, gThar, [jezlocr[0]], [jezlocr[1]])
    print("Tharsis correction at center of Jezero = %g m" % z)

    #-----------------------------
    #true polar wander deformation

    #read in the TPW parameters
    print('loading TPW parameters of Perron et al. 2007')
    df = read_csv(fntpw, skiprows=1, index_col=0)

    #pull out sea levels for each Te
    sealevtpw = df['Z']
    #paleopole position in radians
    df['theta'], df['phi'] = latlon2sph(df['lat'], df['lon'])

    #tpw function isn't fast and vectorized. whatever.
    true_polar_wander = vectorize(true_polar_wander)

    #compute global TPW effect with Te = 200
    print('evaluating TPW correction over global and Jezero topography')
    gTPW = true_polar_wander(gΘ, gΦ, ω, R, g,
            df.at[Te,'hf'], df.at[Te,'kf'],
            df.at[Te,'theta'], df.at[Te,'phi'])

    #compute Jezero TPW effect for all elastic thicknesses
    jTPW = dict()
    for idx in df.index:
        jTPW[idx] = true_polar_wander(jΘ, jΦ, ω, R, g,
                df.at[idx,'hf'], df.at[idx,'kf'],
                df.at[idx,'theta'], df.at[idx,'phi'])

    #print the deformation in the center of Jezero
    z = ginterp(gθ, gϕ, gTPW, [jezlocr[0]], [jezlocr[1]])
    print("TPW correction at center of Jezero = %g m" % z)

    #-------------------------------------------------------------------------------
    #make some plots

    print('\nmaking plots')

    #elevation w/r/t sea level for Tharsis and TPW with Te = 200 km
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(8.5,3.25), constrained_layout=True)
    plot_jezero_sealev(fig, axa, jlon, jlat, jTopo,
            jTPW[Te], sealevtpw[Te], jshps,
            xlab='Longitude',
            ylab='Latitude',
            axlab='a')
    plot_jezero_sealev(fig, axb, jlon, jlat, jTopo,
            jThar, sealevthar, jshps,
            clab='Elevation Relative to Sea Level (km)',
            xlab='Longitude',
            axlab='b')
    axb.set_yticklabels([])
    axa.set_title('TPW Scenario, $T_e=200$ km')
    axb.set_title('Tharsis Growth Scenario')
    save_close(fig, dirplots, 'jezero_sealev_both')

    #global correction for both scenarios together
    fig = plt.figure(figsize=(8.5,2.5))
    plot_global_def(fig, 121,
            gΘ, gΦ, gTopo, gTPW,
            arabia['theta'], arabia['phi'],
            axlab='a',
            clab='',
            title='TPW Scenario, $T_e=200$ km')
    plot_global_def(fig, 122,
            gΘ, gΦ, gTopo, gThar,
            arabia['theta'], arabia['phi'],
            axlab='b',
            clab='Topographic Correction (km)',
            title='Tharsis Growth Scenario')
    fig.tight_layout()
    save_close(fig, dirplots, 'global_def_both')

    #everything for Tharsis growth scenario together
    fig = plt.figure(figsize=(8.5,5.5))
    plot_global_def(fig, 221,
            gΘ, gΦ, gTopo, gThar,
            arabia['theta'], arabia['phi'],
            axlab='a',
            clab='Tharsis Contribution (km)')
    plot_jezero_def(fig, 222,
            jlon, jlat, jTopo, jThar, jshps,
            xlab='Longitude',
            ylab='Latitude',
            axlab='b',
            clab='Tharsis Contribution (km)')
    plot_jezero_ocean(fig, 223,
            jlon, jlat, jTopo, jThar, sealevthar, jshps,
            xlab='Longitude',
            ylab='Latitude',
            axlab='c',
            clab='Pre-Tharsis Topography (km)')
    plot_jezero_sealev(fig, 224,
            jlon, jlat, jTopo, jThar, sealevthar, jshps,
            div=True,
            xlab='Longitude',
            ylab='Latitude',
            axlab='d',
            clab='Elevation From Sea Level (km)')
    fig.tight_layout()
    save_close(fig, dirplots, 'all_tharsis')

    #all tpw plots
    fig = plt.figure(figsize=(8.5,5.5))
    plot_global_def(fig, 221,
            gΘ, gΦ, gTopo, gTPW,
            arabia['theta'], arabia['phi'],
            axlab='a',
            clab='TPW Contribution (km)')
    plot_jezero_def(fig, 222,
            jlon, jlat, jTopo, jTPW[Te], jshps,
            xlab='Longitude',
            ylab='Latitude',
            axlab='b',
            clab='TPW Contribution (km)')
    plot_jezero_ocean(fig, 223,
            jlon, jlat, jTopo, jTPW[Te], sealevtpw[Te], jshps,
            xlab='Longitude',
            ylab='Latitude',
            axlab='c',
            clab='Pre-TPW Topography (km)')
    plot_jezero_sealev(fig, 224,
            jlon, jlat, jTopo, jTPW[Te], sealevtpw[Te], jshps,
            div=True,
            xlab='Longitude',
            ylab='Latitude',
            axlab='d',
            clab='Elevation From Sea Level (km)')
    fig.tight_layout()
    save_close(fig, dirplots, 'all_tpw')

    #All the T_e values for TPW scenarios
    fig, axs = plt.subplots(2, 2, figsize=(7.5,5.5), constrained_layout=True)
    axs = flatten(axs)
    v = max([abs(jTopo - jTPW[k] - sealevtpw[k]).max() for k in jTPW])
    keys = sorted(jTPW.keys())
    for i,k,lab in zip(range(len(keys)), keys, 'abcd'):
        _ , r = plot_jezero_sealev(fig, axs[i],
                    jlon, jlat, jTopo, jTPW[k], sealevtpw[k], jshps,
                    div=v, axlab=lab)
        axs[i].set_title('$T_e = %g$ km' % k)
    cb = fig.colorbar(r, ax=axs)
    cb.set_label('Elevation Relative to Sea Level (km)', rotation=270, va='bottom')
    axs[0].set_ylabel('Latitude')
    axs[0].set_xticks([])
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[2].set_ylabel('Latitude')
    axs[2].set_xlabel('Longitude')
    axs[3].set_yticks([])
    axs[3].set_xlabel('Longitude')
    save_close(fig, dirplots, 'tpw_Te')

    #arabia contact interpolations to validate the reproduced corrections
    fig, ax = plt.subplots(1, 1, figsize=(4.1,2.8))
    dist = arabia['distance (m)'].values/1e3
    #elevation of Arabia contact coordinates
    ax.plot(dist, arabia['elevation (m)'].values/1e3, 'k.',
        label='Arabia Feature',
        markersize=3)
    #TPW deflection
    z = ginterp(gθ, gϕ, gTPW, arabia['theta'], arabia['phi']) + sealevtpw[Te]
    ax.plot(dist, z/1e3, label='TPW Correction')
    #Tharsis deflection
    z = ginterp(gθ, gϕ, gThar, arabia['theta'], arabia['phi']) + sealevthar
    ax.plot(dist, z/1e3, label='Tharsis Correction')
    #format
    ax.set_xlabel('Distance Along Feature (km)')
    ax.set_ylabel('Elevation (km)')
    ax.legend()
    fig.tight_layout()
    save_close(fig, dirplots, 'arabia_corrections')

    plt.show()
