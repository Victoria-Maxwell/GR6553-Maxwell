import numpy as np
import matplotlib.pyplot as plt
#import tropycal
import tropycal.tracks as tracks
import tropycal.recon
from tropycal.recon import dropsondes
import cartopy.feature as cf
import cartopy.crs as ccrs
import pygrib
#from datetime import datetime
import pandas as pd
#from metpy.plots import SkewT
#from metpy.units import pandas_dataframe_to_unit_arrays, units
import matplotlib.gridspec as gridspec
from metpy.calc import azimuth_range_to_lat_lon
#from metpy.cbook import get_test_data
from metpy.io import Level2File
from metpy.plots import add_metpy_logo, add_timestamp, USCOUNTIES
from metpy.units import units



#read in HURDAT and data for Cristobal
    #plot the track for Cristobal
hdat = tracks.TrackDataset(basin='north_atlantic',source='hurdat')
cristobal=hdat.get_storm(('cristobal',2020))
df_cristobal=cristobal.to_dataframe()
print(df_cristobal)
cristobal.plot()
plt.savefig('Tropical Storm Cristobal Track.png')


#### Make surface analysis maps for Mexico ####
        #### CAN USE GFS DATA THANK GOD #####

#### Plot 850 mb pressure heights, wind speed
## open the file
grbGFS1=pygrib.open('gfs_4_20200603_1200_000.grb2') 
## search for index for each variable and set the variable name to the index

grbGFS1.select(name='U component of wind')
grbGFS1.select(name='V component of wind')
grbGFS1.select(name='Geopotential height')

uwnd850 = grbGFS1[320]; uwnd = uwnd850['values']
vwnd850 = grbGFS1[321]; vwnd = vwnd850['values']
gpm850 = grbGFS1[314]; gpm = gpm850['values']
lats,lons = gpm850.latlons()

## set projection and figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons,lats,np.sqrt(uwnd**2 + vwnd**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons, lats, gpm/10, np.arange(np.min(gpm/10),np.max(gpm/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

## add title and save image
plt.title ('850mb Heights (dm) / Wind Speed (knots) June 3-12z')
plt.savefig('850mb June 3-12z.png')
plt.show()
plt.close()

##### Plot 850 mb heights and wind speeds 24 in advance
    #### make sure to change lats, lons and variable names, if not it will continue to plot the like the first map
## open the file
grbGFS2=pygrib.open('gfs_4_20200603_1200_024.grb2')
## search for index for each variable and set the variable name to the index
grbGFS2.select(name='U component of wind')
grbGFS2.select(name='V component of wind')
grbGFS2.select(name='Geopotential height')

Uwnd850 = grbGFS2[335]; Uwnd = Uwnd850['values']
Vwnd850 = grbGFS2[336]; Vwnd = Vwnd850['values']
Gpm850 = grbGFS2[329]; Gpm = Gpm850['values']
lats2,lons2 = gpm850.latlons()
## set projection and figure size
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)
## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons2,lats2,np.sqrt(Uwnd**2 + Vwnd**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())
cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons2, lats2, Gpm/10, np.arange(np.min(Gpm/10),np.max(Gpm/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

## add title and save image
plt.title ('850mb Heights (dm) / Wind Speed (knots) Forecast 24 hours, June 3-12z-24')
plt.savefig('850mb June 3-12z-24.png')
plt.show()
plt.close()
#### Plot 850 mb Heights and Winds at the observed 24 hours (day of run)
    #### make sure to change lats, lons and variable names, if not it will continue to plot the like the first map

#open the file
grbGFS3=pygrib.open('gfs_4_20200604_1200_000.grb2')

## search for index for each variable and set the variable name to the index
grbGFS3.select(name='U component of wind')
grbGFS3.select(name='V component of wind')
grbGFS3.select(name='Geopotential height')

UWnd850 = grbGFS3[320]; UWnd = UWnd850['values']
VWnd850 = grbGFS3[321]; VWnd = VWnd850['values']
GPm850 = grbGFS3[314]; GPm = GPm850['values']
lats3,lons3 = gpm850.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons3,lats3,np.sqrt(UWnd**2 + VWnd**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())
cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons3, lats3, GPm/10, np.arange(np.min(GPm/10),np.max(GPm/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add title and save figure
plt.title ('850mb Heights (dm) / Wind Speed (knots) Observed 24 hours, June 4-12z')
plt.savefig('850mb June 4-12z.png')
plt.show()
plt.close()

##### Plot 500 mb heights and wind speeds
    ### Change variable names
#open file
grbGFS1=pygrib.open('gfs_4_20200603_1200_000.grb2') 

## search for index for each variable and set the variable name to the index
grbGFS1.select(name='U component of wind')
grbGFS1.select(name='V component of wind')
grbGFS1.select(name='Geopotential height')

uwnd500 = grbGFS1[220]; uwnd1 = uwnd500['values']
vwnd500 = grbGFS1[221]; vwnd1 = vwnd500['values']
gpm500 = grbGFS1[214]; gpm1 = gpm500['values']
lats4,lons4 = gpm500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons4,lats4,np.sqrt(uwnd1**2 + vwnd1**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons4, lats4, gpm1/10, np.arange(np.min(gpm1/10),np.max(gpm1/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add title and save image
plt.title ('500mb Heights (dm) / Wind Speed (knots) June 3-12z')
plt.savefig('500mb June 3-12z.png')
plt.show()
plt.close()

##### Plot 500 mb heights and wind speeds 24 in advance
    #### make sure to change lats, lons and variable names, if not it will continue to plot the like the first 500 map
##open file
grbGFS2=pygrib.open('gfs_4_20200603_1200_024.grb2')

## search for index for each variable and set the variable name to the index
grbGFS2.select(name='U component of wind')
grbGFS2.select(name='V component of wind')
grbGFS2.select(name='Geopotential height')

uwnd500 = grbGFS2[228]; uwnd2 = uwnd500['values']
vwnd500 = grbGFS2[229]; vwnd2 = vwnd500['values']
gpm500 = grbGFS2[222]; gpm2 = gpm500['values']
lats5,lons5 = gpm500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')

gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons2,lats2,np.sqrt(uwnd2**2 + vwnd2**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())
cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons5, lats5, gpm2/10, np.arange(np.min(gpm2/10),np.max(gpm2/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add title and save image
plt.title ('500b Heights (dm) / Wind Speed (knots) Forecast 24 hours, June 4-12z')
plt.savefig('500mb June 3-12z-24.png')
plt.show()
plt.close()

#### Plot 500mb Heights and Winds observed
    #### make sure to change lats, lons and variable names, if not it will continue to plot the like the first 500 map
##open file
grbGFS3=pygrib.open('gfs_4_20200604_1200_000.grb2')

#### search for index for each variable and set the variable name to the index
grbGFS3.select(name='U component of wind')
grbGFS3.select(name='V component of wind')
grbGFS3.select(name='Geopotential height')

uwnd500 = grbGFS3[220]; uwnd3 = uwnd500['values']
vwnd500 = grbGFS3[221]; vwnd3 = vwnd500['values']
gpm500 = grbGFS3[214]; gpm3 = gpm500['values']
lats6,lons6 = gpm500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')

gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60,65,70]
plt.contourf(lons6,lats6,np.sqrt(uwnd3**2 + vwnd3**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())
cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons6, lats6, gpm3/10, np.arange(np.min(gpm3/10),np.max(gpm3/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add tile and save image
plt.title ('500b Heights (dm) / Wind Speed (knots) Observed 24 hours, June 4-12z')
plt.savefig('500mb June 4-12z.png')
plt.show()
plt.close()
###### Dropsonde profile before landfall in Mexico #######
    #### Tropycal #####

##import TrackDataset to be able to pull data for storm of choice
storm = tracks.TrackDataset(basin='north_atlantic',source='hurdat')
Cristobal=storm.get_storm(('Cristobal',2020))
##search for the dropsondes for the storm of choice, import datetime to select the time for the dropsonde
dropsondes_obj=dropsondes(Cristobal)
import datetime
drop_time = datetime.datetime(2020,6,3,12,45)
#plot the dropsonde
dropsondes_obj.plot_skewt(time=drop_time)
### FOR SOME REASON I CANNOT GET IT TO SAVE AS A PNG, IT SHOWS UP BLANK, IF USING SPYDER IT WILL PLOT IN THE KERNEL AND CAN BE SAVED FROM THERE
#plt.savefig('June 3 Dropsonde 1245.png')
#plt.show()
#plt.close()

##### plot GFS precip for pseudo-radar

##open file, find index and set variable to index
grbGFS=pygrib.open('gfs_4_20200603_1200_003.grb2')
grbGFS.select(name='Precipitation rate')

precip = grbGFS[447]; precip = precip['values']
lats,lons = gpm850.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,10.,25.])
ax.add_feature(cf.LAND,color='white')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='saddlebrown')
ax.add_feature(cf.STATES,edgecolor='saddlebrown')
ax.add_feature(cf.BORDERS,edgecolor='saddlebrown',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='black', alpha=0.5, linestyle='--')

gl.top_labels = False
gl.left_labels = False

#bounds = [0.00250,0.00750,.01250,0.01750,0.02500,0.03000]

bounds =np.arange(0.1,2.2,0.1)
### conversion found to convert kgm^2 of water into inches
con=plt.contourf(lons, lats, ((precip*0.03937)*3600),levels=bounds, cmap=plt.cm.gist_ncar, transform=ccrs.PlateCarree())
cbar=plt.colorbar(location='bottom')

#plot title and save figure
plt.title ('Precipitation Rate ( Inches hr$^-$$^1$ )')
plt.savefig('Precipitiation Rate.png')
plt.show()
plt.close()


#### Time Series Pressure vs Wind Speed June 3 Flight Level Data ######


flight_level=pd.read_csv('20200603_flight-level.csv')
time=flight_level['GMT'].tolist()
wspd=flight_level['WSpd'].tolist()
pres=flight_level['GA'].tolist()

#create plot
fig=plt.figure(figsize=(8,8))

#assign data to axes
x=np.array(time)
y2=np.array(pres)
y=np.array(wspd)

#two axes
fig,ax1=plt.subplots()
ax2=ax1.twinx()
#plot data
#bounds=[8:22:24,14:49:02,10]
ax1.plot(x,y, color='blue', label='wspd')
ax2.plot(x,y2, color = 'orange', label='pressure')

#add elements to plot
ax1.set_xlabel("time (HHMMSS)")
ax1.set_ylabel("m/s")
ax2.set_ylabel("hPa")
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')

##add title and save
plt.title("Wind Speed vs Pressure Tropical Storm Cristobal June 3,2020 Flight")
plt.savefig('June 3 Flight Level.png')
plt.show()
plt.close()



#-------------------PLOT DATA FOR LOUISIANA --------------------------------------------------------------------------------------------------------

##### Plot timeseries
flight_level=pd.read_csv('20200607_flight-level.csv')
time=flight_level['GMT'].tolist()
wspd=flight_level['WSpd'].tolist()
pres=flight_level['GA'].tolist()

#create plot
fig=plt.figure(figsize=(8,8))

#assign data to axes
x=np.array(time)
y2=np.array(pres)
y=np.array(wspd)

#two axes
fig,ax1=plt.subplots()
ax2=ax1.twinx()
#plot data
ax1.plot(x,y, color='blue', label='wspd')
ax2.plot(x,y2, color = 'orange', label='pressure')

#add elements to plot
ax1.set_xlabel("time (HHMMSS)")
ax1.set_ylabel("m/s")
ax2.set_ylabel("hPa")
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')

##add title and save
plt.title("Wind Speed vs Pressure Tropical Storm Cristobal June 7,2020 Flight")
plt.savefig("June 7 Flight Level.png")
plt.show()
plt.close()

#### Plot dropsonde data

storm = tracks.TrackDataset(basin='north_atlantic',source='hurdat')
Cristobal=storm.get_storm(('Cristobal',2020))
dropsondes_obj=dropsondes(Cristobal)
import datetime
drop_time = datetime.datetime(2020,6,7,3)

dropsondes_obj.plot_skewt(time=drop_time)

### FOR SOME REASON I CANNOT GET IT TO SAVE AS A PNG, IT SHOWS UP BLANK
#plt.savefig('June 7 Dropsonde.png')
#plt.show()
#plt.close()

##### Plot Radar Data
# Open the file

f = Level2File('KLIX20200607_180330_V06')

print(f.sweeps[0][0])
###########################################

# Pull data out of the file
sweep = 0

# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = f.sweeps[0][0][1].lon
cent_lat = f.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    ax.pcolormesh(xlocs, ylocs, data, cmap='nipy_spectral', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')

plt.title('1803 Radar Image KLIX')
plt.savefig('1803 Radar.png')
plt.show()
plt.close()

#####plot at 20
f = Level2File('KLIX20200607_200401_V06')

print(f.sweeps[0][0])

# Pull data out of the file
sweep = 0

# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = f.sweeps[0][0][1].lon
cent_lat = f.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    ax.pcolormesh(xlocs, ylocs, data, cmap='nipy_spectral', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')

plt.title('2004 Radar Image KLIX')
plt.savefig('2004 Radar.png')
plt.show()
plt.close()

##### plot 2159 radar image
f = Level2File('KLIX20200607_220647_V06')

print(f.sweeps[0][0])

# Pull data out of the file
sweep = 0

# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = f.sweeps[0][0][1].lon
cent_lat = f.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    ax.pcolormesh(xlocs, ylocs, data, cmap='nipy_spectral', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')

plt.title('2206 Radar Image KLIX')
plt.savefig('2206 Radar.png')
plt.show()
plt.close()
##### plot 0004
f = Level2File('KLIX20200608_000457_V06')

print(f.sweeps[0][0])

# Pull data out of the file
sweep = 0

# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])

###########################################
# We need to take the single azimuth (nominally a mid-point) we get in the data and
# convert it to be the azimuth of the boundary between rays of data, taking care to handle
# where the azimuth crosses from 0 to 360.
diff = np.diff(az)
crossed = diff < -180
diff[crossed] += 360.
avg_spacing = diff.mean()

# Convert mid-point to edge
az = (az[:-1] + az[1:]) / 2
az[crossed] += 180.

# Concatenate with overall start and end of data we calculate using the average spacing
az = np.concatenate(([az[0] - avg_spacing], az, [az[-1] + avg_spacing]))
az = units.Quantity(az, 'degrees')

###########################################
# Calculate ranges for the gates from the metadata

# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
ref_hdr = f.sweeps[sweep][0][4][b'REF'][0]
ref_range = (np.arange(ref_hdr.num_gates + 1) - 0.5) * ref_hdr.gate_width + ref_hdr.first_gate
ref_range = units.Quantity(ref_range, 'kilometers')
ref = np.array([ray[4][b'REF'][1] for ray in f.sweeps[sweep]])

rho_hdr = f.sweeps[sweep][0][4][b'RHO'][0]
rho_range = (np.arange(rho_hdr.num_gates + 1) - 0.5) * rho_hdr.gate_width + rho_hdr.first_gate
rho_range = units.Quantity(rho_range, 'kilometers')
rho = np.array([ray[4][b'RHO'][1] for ray in f.sweeps[sweep]])

# Extract central longitude and latitude from file
cent_lon = f.sweeps[0][0][1].lon
cent_lat = f.sweeps[0][0][1].lat
###########################################
#spec=gridspec.GridSpec(1, 2)
spec = gridspec.GridSpec(1, 1)
#fig = plt.figure(figsize=(15, 8))
fig = plt.figure(figsize=(8,8))

for var_data, var_range, ax_rect in zip((ref, rho), (ref_range, rho_range), spec):
    # Turn into an array, then mask
    data = np.ma.array(var_data)
    data[np.isnan(data)] = np.ma.masked

    # Convert az,range to x,y
    xlocs, ylocs = azimuth_range_to_lat_lon(az, var_range, cent_lon, cent_lat)

    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=cent_lon, central_latitude=cent_lat)
    ax = fig.add_subplot(ax_rect, projection=crs)
    ax.add_feature(USCOUNTIES, linewidth=0.5)
    ax.pcolormesh(xlocs, ylocs, data, cmap='nipy_spectral', transform=ccrs.PlateCarree())
    ax.set_extent([cent_lon - 2, cent_lon + 2, cent_lat - 2, cent_lat + 2])
    ax.set_aspect('equal', 'datalim')
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='grey', alpha=0.5, linestyle='--')

plt.title('0004 Radar Image KLIX')
plt.savefig('0004 Radar.png')
plt.show()
plt.close()

#### Plot 850 mb heights, winds
##open file
grbGFS4=pygrib.open('gfs_4_20200607_1800_003.grb2')

## search for index for each variable and set the variable name to the index
grbGFS4.select(name='U component of wind')
grbGFS4.select(name='V component of wind')
grbGFS4.select(name='Geopotential height')

Uwnd850 = grbGFS4[335]; Uwnd = Uwnd850['values']
Vwnd850 = grbGFS4[336]; Vwnd = Vwnd850['values']
GPM850 = grbGFS4[329]; GPM = GPM850['values']
lats7,lons7 = GPM850.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons7,lats7,np.sqrt(Uwnd**2 + Vwnd**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons7, lats7, GPM/10, np.arange(np.min(GPM/10),np.max(GPM/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add tile and save figure
plt.title ('850mb Heights (dm) / Wind Speed (knots')
plt.savefig('850mb June 7-03z.png')
plt.close()


#### Plot 850 mb heights and wind speeds Forecast 24 hours

##open file
grbGFS5=pygrib.open('gfs_4_20200607_1800_027.grb2')
## search for index for each variable and set the variable name to the index

grbGFS5.select(name='U component of wind')
grbGFS5.select(name='V component of wind')
grbGFS5.select(name='Geopotential height')

Uwnd850 = grbGFS5[335]; Uwnd1 = Uwnd850['values']
Vwnd850 = grbGFS5[336]; Vwnd1 = Vwnd850['values']
GPM850 = grbGFS5[329]; GPM1 = GPM850['values']
lats8,lons8 = GPM850.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons8,lats8,np.sqrt(Uwnd1**2 + Vwnd1**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons8, lats8, GPM1/10, np.arange(np.min(GPM1/10),np.max(GPM1/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

plt.title ('850mb Heights (dm) / Wind Speed (knots) Forecast 24 hours, June 7')
plt.savefig('850mb June 7-03z-24.png')
plt.show()
plt.close()

#### Plot 850 mb heights at model time 24 hours time
##open file
grbGFS6=pygrib.open('gfs_4_20200608_1800_003.grb2')

## search for index for each variable and set the variable name to the index
grbGFS6.select(name='U component of wind')
grbGFS6.select(name='V component of wind')
grbGFS6.select(name='Geopotential height')

Uwnd850 = grbGFS6[335]; Uwnd2 = Uwnd850['values']
Vwnd850 = grbGFS6[336]; Vwnd2 = Vwnd850['values']
GPM850 = grbGFS6[329]; GPM2 = GPM850['values']
lats9,lons9 = GPM850.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons9,lats9,np.sqrt(Uwnd2**2 + Vwnd2**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons9, lats9, GPM2/10, np.arange(np.min(GPM2/10),np.max(GPM2/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add tile and save figure
plt.title ('850mb Heights (dm) / Wind Speed (knots) Observed 24 Hours, June 8')
plt.savefig('850mb June 8-03z.png')
plt.show()
plt.close()

#### Plot 500 mb heights and wind speeds

##open file
grbGFS4=pygrib.open('gfs_4_20200607_1800_003.grb2')

## search for index for each variable and set the variable name to the index
grbGFS4.select(name='U component of wind')
grbGFS4.select(name='V component of wind')
grbGFS4.select(name='Geopotential height')

Uwnd500 = grbGFS4[228]; Uwnd3 = Uwnd500['values']
Vwnd500 = grbGFS4[229]; Vwnd3 = Vwnd500['values']
GPM500 = grbGFS4[222]; GPM3 = GPM500['values']
lats10,lons10 = GPM500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons10,lats10,np.sqrt(Uwnd3**2 + Vwnd3**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons10, lats10, GPM3/10, np.arange(np.min(GPM3/10),np.max(GPM3/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add title and save figure
plt.title ('500mb Heights (dm) / Wind Speed (knots')
plt.savefig('500mb June 7-03.png')
plt.show()
plt.close()


#### Plot 500 mb heights and wind speeds forecast 24 hours
##open file
grbGFS5=pygrib.open('gfs_4_20200607_1800_027.grb2')

## search for index for each variable and set the variable name to the index
grbGFS5.select(name='U component of wind')
grbGFS5.select(name='V component of wind')
grbGFS5.select(name='Geopotential height')

Uwnd500 = grbGFS5[228]; Uwnd4 = Uwnd500['values']
Vwnd500 = grbGFS5[229]; Vwnd4 = Vwnd500['values']
GPM500 = grbGFS5[222]; GPM4 = GPM500['values']
lats11,lons11 = GPM500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons11,lats11,np.sqrt(Uwnd4**2 + Vwnd4**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons11, lats11, GPM4/10, np.arange(np.min(GPM4/10),np.max(GPM4/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##open file
plt.title ('500mb Heights (dm) / Wind Speed (knots) Forecast 24 Hours, June 7')
plt.savefig('500mb June 7-03z-24.png')
plt.show()
plt.close()

#### Plot actaul 24 hours
##open file
grbGFS6=pygrib.open('gfs_4_20200608_1800_003.grb2')

## search for index for each variable and set the variable name to the index
grbGFS6.select(name='U component of wind')
grbGFS6.select(name='V component of wind')
grbGFS6.select(name='Geopotential height')

Uwnd500 = grbGFS6[228]; Uwnd5 = Uwnd500['values']
Vwnd500 = grbGFS6[229]; Vwnd5 = Vwnd500['values']
GPM500 = grbGFS6[222]; GPM5 = GPM500['values']
lats12,lons12 = GPM500.latlons()

## set extent, set feature colors, add gridlines, labels, set bounds and plot variables
fig = plt.figure (figsize=(8,8))
proj=ccrs.LambertConformal(central_longitude=-96.,central_latitude=40.,standard_parallels=(40.,40.))
ax=plt.axes(projection=proj)

ax.set_extent([-100.,-80.,20.,35.])
ax.add_feature(cf.LAND,color='lightgray')
ax.add_feature(cf.OCEAN, color='white',)
ax.add_feature(cf.COASTLINE,edgecolor='gray')
ax.add_feature(cf.STATES,edgecolor='gray')
ax.add_feature(cf.BORDERS,edgecolor='gray',linestyle='-')
ax.add_feature(cf.LAKES, color='white', alpha=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='white', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
bounds = [20,25,30,35,40,45,50,55,60]
plt.contourf(lons12,lats12,np.sqrt(Uwnd5**2 + Vwnd5**2)*1.94, bounds, cmap=plt.cm.BuPu,transform=ccrs.PlateCarree())

cbar=plt.colorbar(location='bottom')
cbar.set_label ('knots')
height=plt.contour (lons12, lats12, GPM5/10, np.arange(np.min(GPM5/10),np.max(GPM5/10),3), linewidths=2, linestyles='-', colors='black', transform=ccrs.PlateCarree())

##add tile and save figure
plt.title ('500mb Heights (dm) / Wind Speed (knots) Actaul 24 hours, June 8')
plt.savefig('500mb June 8-03z')
plt.show()
plt.close()
